"""
Multi-agent orchestration for the Kairos repo.

Roles
-----
ClaudeBuilder      : proposes a unified diff for a given task
GeminiCritic        } adversarial review of the diff, independent of
ChatGPTCritic       } each other (no shared context)
MistralController   : reads both critiques + diff, decides:
                       "approve" | "revise" | "escalate_to_human"
GitManager          : plain subprocess git ops (branch/commit/push).
                       Never touches main. Never auto-merges.
OrchestrationSession: ties the loop together.

Setup
-----
Env vars required:
    ANTHROPIC_API_KEY, GOOGLE_API_KEY, OPENAI_API_KEY, MISTRAL_API_KEY

This file only defines classes / a CLI entry point. Nothing runs on import.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal
from context_builder import CodingStandards, RepoContextSelector, ContextBuilder
import time
from google.genai.errors import ServerError, APIError
from openai import OpenAI
#from groq import Groq
Verdict = Literal["approve", "revise", "escalate_to_human"]


# --------------------------------------------------------------------------- #
# Agents
# --------------------------------------------------------------------------- #

class BaseAgent(ABC):
    """Common interface every LLM agent implements."""

    def __init__(self, model: str):
        self.model = model

    @abstractmethod
    def respond(self, system: str, user: str) -> str:
        """Send one system+user turn, return raw text response."""
        raise NotImplementedError

class OpenRouterAgent(BaseAgent):
    """Generic agent routing through OpenRouter (supports standard models like Google Gemini, Claude, etc.)"""

    def __init__(self, model: str = "google/gemini-2.5-flash"):
        super().__init__(model)
        self._client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ["OPENROUTER_API_KEY"],
        )

    def respond(self, system: str, user: str) -> str:
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return resp.choices[0].message.content

class ClaudeBuilder(BaseAgent):
    """Proposes a unified diff implementing the task, given prior critiques."""

    def __init__(self, model: str = "claude-sonnet-4-6"):
        super().__init__(model)
        import anthropic
        self._client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    def respond(self, system: str, user: str) -> str:
        msg = self._client.messages.create(
            model=self.model,
            max_tokens=4000,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(b.text for b in msg.content if b.type == "text")

    def propose_patch(self, task: str, repo_context: str, prior_critiques: str = "") -> str:
        system = (
            "You are a senior engineer. Output ONLY a valid unified diff "
            "(git apply-able), no prose, no markdown fences. If no change "
            "is needed, output an empty string."
        )
        user = (
            f"Task:\n{task}\n\n"
            f"Relevant repo context:\n{repo_context}\n\n"
            f"Prior critiques to address (may be empty):\n{prior_critiques}"
        )
        return self.respond(system, user)

FALLBACK_MODELS = ["gemini-3.5-flash-lite", "gemini-3.1-flash-lite"]
class GeminiCritic(BaseAgent):
    def __init__(self, model: str = "gemini-3.6-flash"):
        super().__init__(model)
        from google import genai
        self._client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

    def respond(self, system: str, user: str) -> str:
        max_retries = 5
        base_delay = 2  # seconds
        models_to_try = [self.model] + [m for m in FALLBACK_MODELS if m != self.model]

        for attempt in range(max_retries):
            for model_name in models_to_try:
                try:
                    resp = self._client.models.generate_content(
                        model=self.model,
                        contents=user,
                        config={"system_instruction": system},
                    )
                    return resp.text
                except (ServerError, APIError) as e:
                    print(f"[Gemini Error] Model {model_name} failed: {e}")
                    continue  # Try next fallback model in the list

            # If all models failed for this attempt, wait and retry the whole cycle
            if attempt < max_retries - 1:
                sleep_time = base_delay * (2 ** attempt)
                print(f"[Gemini Overloaded] All models busy. Retrying attempt {attempt + 1}/{max_retries} in {sleep_time}s...")
                time.sleep(sleep_time)
        raise RuntimeError("All configured Gemini models are currently overloaded.")


class ChatGPTCritic(BaseAgent):
    def __init__(self, model: str = "gpt-4.1"):
        super().__init__(model)
        from openai import OpenAI
        self._client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    def respond(self, system: str, user: str) -> str:
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return resp.choices[0].message.content


class Critic:
    """Shared adversarial-review prompt, wraps any BaseAgent."""

    SYSTEM = (
        "You are an adversarial code reviewer. Find bugs, security issues, "
        "edge cases, and redundancy with existing code. Be concise. "
        "End with a one-line verdict: SEVERITY=none|minor|major."
    )

    def __init__(self, agent: BaseAgent):
        self._agent = agent

    def review(self, task: str, diff: str) -> str:
        user = f"Task:\n{task}\n\nProposed diff:\n{diff}"
        return self._agent.respond(self.SYSTEM, user)


class MistralController(BaseAgent):
    """Aggregates critiques, decides whether to approve/revise/escalate."""

    def __init__(self, model: str = "mistral-large-latest"):
        super().__init__(model)
        from mistralai.client import Mistral
        self._client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])

    def respond(self, system: str, user: str) -> str:
        resp = self._client.chat.complete(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return resp.choices[0].message.content

    def decide(self, task: str, diff: str, critiques: dict[str, str]) -> tuple[Verdict, str]:
        system = (
            "You are a workflow controller. Read the diff and both critiques. "
            "Reply with strict JSON: "
            '{"verdict": "approve"|"revise"|"escalate_to_human", "reason": "..."}. '
            "Escalate only for genuine ambiguity or conflicting critiques you "
            "cannot resolve yourself."
        )
        user = (
            f"Task:\n{task}\n\nDiff:\n{diff}\n\n"
            f"Gemini critique:\n{critiques.get('gemini', '')}\n\n"
            f"ChatGPT critique:\n{critiques.get('chatgpt', '')}"
        )
        raw = self.respond(system, user)
        try:
            parsed = json.loads(raw)
            return parsed["verdict"], parsed.get("reason", "")
        except (json.JSONDecodeError, KeyError):
            return "escalate_to_human", f"Controller returned unparseable output: {raw}"


# --------------------------------------------------------------------------- #
# Git operations (no LLM involved)
# --------------------------------------------------------------------------- #

class GitManager:
    """Thin wrapper around subprocess git calls. Operates on a feature branch only."""

    def __init__(self, repo_path: str | Path, protected_branches: tuple[str, ...] = ("main", "master")):
        self.repo_path = Path(repo_path)
        self._protected = protected_branches

    def _run(self, *args: str) -> str:
        result = subprocess.run(
            ["git", *args], cwd=self.repo_path,
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"git {' '.join(args)} failed: {result.stderr}")
        return result.stdout.strip()

    def current_branch(self) -> str:
        return self._run("rev-parse", "--abbrev-ref", "HEAD")

    def ensure_feature_branch(self, branch_name: str) -> None:
        if self.current_branch() in self._protected:
            self._run("checkout", "-b", branch_name)
        elif self.current_branch() != branch_name:
            self._run("checkout", "-B", branch_name)

    def apply_patch(self, diff_text: str) -> None:
        if not diff_text.strip():
            return
        patch_path = self.repo_path / ".orchestrator_patch.diff"
        patch_path.write_text(diff_text)
        try:
            self._run("apply", "--whitespace=fix", str(patch_path))
        finally:
            patch_path.unlink(missing_ok=True)

    def commit(self, message: str) -> None:
        if self.current_branch() in self._protected:
            raise RuntimeError("Refusing to commit directly on a protected branch.")
        self._run("add", "-A")
        self._run("commit", "-m", message)

    def push(self, branch_name: str) -> None:
        if branch_name in self._protected:
            raise RuntimeError("Refusing to push directly to a protected branch.")
        self._run("push", "-u", "origin", branch_name)


# --------------------------------------------------------------------------- #
# Orchestration loop
# --------------------------------------------------------------------------- #

@dataclass
class RoundLog:
    round_num: int
    diff: str
    critiques: dict[str, str]
    verdict: Verdict
    reason: str


@dataclass
class OrchestrationSession:
    task: str
    repo_context: str
    git: GitManager
    builder: ClaudeBuilder
    gemini: Critic
    chatgpt: Critic
    controller: MistralController
    branch_name: str = field(default_factory=lambda: f"orch/{datetime.now():%Y%m%d-%H%M%S}")
    max_rounds: int = 3
    log: list[RoundLog] = field(default_factory=list)
    cache_path: Path = field(default_factory=lambda: Path(".orchestrator_cache.json"))

    def run(self) -> Literal["pushed", "escalated", "gave_up"]:
        self.git.ensure_feature_branch(self.branch_name)
        prior_critiques = ""

        # Load cache if available
        cache = {}
        if self.cache_path.exists():
            try:
                cache = json.loads(self.cache_path.read_text())
                print(f"[Cache] Found cached session state for task.")
            except json.JSONDecodeError:
                pass

        for round_num in range(1, self.max_rounds + 1):
            print(f"\n--- Round {round_num}/{self.max_rounds} ---", flush=True)
            cache_key = f"round_{round_num}_diff"
            
            # Use cached patch if it exists for this round
            if cache_key in cache:
                print(f"[Cache] Loading round {round_num} diff from cache...", flush=True)
                diff = cache[cache_key]
            else:
                print(f"[LLM] Requesting diff from ClaudeBuilder...", flush=True)
                diff = self.builder.propose_patch(self.task, self.repo_context, prior_critiques)
                if diff.strip():
                    cache[cache_key] = diff
                    self.cache_path.write_text(json.dumps(cache, indent=2))

            if not diff.strip():
                print("[Orchestrator] Builder returned empty diff. Giving up.", flush=True)
                return "gave_up"

            print("[LLM] Requesting critique from GeminiCritic...", flush=True)
            gemini_critique = self.gemini.review(self.task, diff)

            print("[LLM] Requesting critique from ChatGPTCritic...", flush=True)
            chatgpt_critique = self.chatgpt.review(self.task, diff)

            critiques = {
                "gemini": gemini_critique,
                "chatgpt": chatgpt_critique,
            }

            print("[LLM] Requesting verdict from MistralController...", flush=True)

            verdict, reason = self.controller.decide(self.task, diff, critiques)
            print(f"[Controller] Verdict: {verdict} | Reason: {reason}", flush=True)

            self.log.append(RoundLog(round_num, diff, critiques, verdict, reason))

            if verdict == "approve":
                self.git.apply_patch(diff)
                self.git.commit(f"[orchestrator] {self.task[:72]}")
                self.git.push(self.branch_name)
                # Cleanup cache after success
                if self.cache_path.exists():
                    self.cache_path.unlink()
                return "pushed"

            if verdict == "escalate_to_human":
                return "escalated"

            # verdict == "revise": feed critiques back into the next round
            prior_critiques = json.dumps(critiques, indent=2)

        return "gave_up"

    def summary(self) -> str:
        lines = [f"Session on branch '{self.branch_name}' — task: {self.task}"]
        for entry in self.log:
            lines.append(f"  Round {entry.round_num}: verdict={entry.verdict} ({entry.reason})")
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser(description="Run one orchestrated change.")
    parser.add_argument("--repo", required=True, help="Path to local git repo")
    parser.add_argument("--task", required=True, help="Description of the change to make")
    parser.add_argument("--max-rounds", type=int, default=3)
    args = parser.parse_args()

    context = ContextBuilder(
        standards=CodingStandards(oop=True, numba_when_numeric=True),
        selector=RepoContextSelector(args.repo),
    ).build(args.task)

    session = OrchestrationSession(
        task=args.task,
        repo_context=context,
        git=GitManager(args.repo),
        builder=ClaudeBuilder(),
        #gemini=Critic(GeminiCritic()),
        gemini=Critic(OpenRouterAgent(model="openrouter/free")),
        #chatgpt=Critic(ChatGPTCritic()),
        chatgpt=Critic(OpenRouterAgent(model="openrouter/free")),
        controller=MistralController(),
        max_rounds=args.max_rounds,
    )
    outcome = session.run()
    print(session.summary())
    print(f"\nOutcome: {outcome}")
    sys.exit(0 if outcome == "pushed" else 1)


if __name__ == "__main__":
    main()
