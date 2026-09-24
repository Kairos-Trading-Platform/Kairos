r"""
Claude-Centric Multi-Agent Orchestrator for the Kairos / Flask Repo.

Architecture
------------
ClaudeOrchestrator : High-level manager directing task execution across sub-agents.
CoderSubAgent      : Generates and refactors Flask routes, models, and features.
TesterSubAgent     : Executes test suite (pytest) and reports pass/fail stack traces.
SecuritySubAgent   : Scans Flask code/diffs for web security vulnerabilities (SQLi, auth, CORS, secrets).
WriterSubAgent     : Generates and updates documentation (README, API routes, OpenAPI specs).
GitManager         : Plain subprocess git ops (branch/commit/push). Never touches protected branches directly.

Env vars required:
    ANTHROPIC_API_KEY

Features
--------
- Deterministic Branching: Reuses the same feature branch for a given task if incomplete.
- Branch-Scoped Caching: Stores sub-agent results inside .git/ to survive failures
  without dirtying the working directory.
- Test & Security Loops: Automatically re-runs failed checks using cached or updated state.

Usage (CMD):
    python orchestrator.py --repo "C:\path\to\app" --task "Add health check endpoint"
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from abc import ABC
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal
from context_builder import ContextBuilder, CodingStandards, RepoContextSelector

import anthropic

# --------------------------------------------------------------------------- #
# Base Agent Setup (Anthropic Claude 3.5 / 3.7)
# --------------------------------------------------------------------------- #

class ClaudeAgent(ABC):
    """Base wrapper for Claude-powered sub-agents."""

    def __init__(self, model: str = "claude-fable-5-1"):
        self.model = model
        self._client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    def respond(self, system: str, user: str, max_tokens: int = 100) -> str:
        msg = self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(b.text for b in msg.content if b.type == "text")

    @staticmethod
    def _strip_markdown_fences(text: str) -> str:
        """Removes a leading/trailing ``` or ```lang fence, if present."""
        text = re.sub(r"^```(?:\w+)?\n", "", text.strip())
        text = re.sub(r"\n?```$", "", text)
        return text.strip()

# --------------------------------------------------------------------------- #
# Specialised Sub-Agents
# --------------------------------------------------------------------------- #

class CoderSubAgent(ClaudeAgent):
    """Generates unified diffs to implement features or fix bugs in the Flask app."""

    SYSTEM_PROMPT = (
        "You are a Senior Python & Flask Software Engineer.\n"
        "Your goal is to implement requested feature changes, refactors, or bug fixes.\n"
        "Focus on OOP and, when possible, Numba.\n"
        "Output ONLY a valid unified diff (git apply-able), with no conversational prose and no markdown formatting.\n"
        "If no changes are required, return an empty string."
    )

    def propose_patch(self, task: str, repo_context: str, feedback: str = "") -> str:
        user_content = (
            f"Task Specification:\n{task}\n\n"
            f"Repository Context:\n{repo_context}\n\n"
            f"Feedback / Test Results / Security Audit to address:\n{feedback}"
        )
        raw_response = self.respond(self.SYSTEM_PROMPT, user_content)
        print(f"[Coder Raw Response]\n{raw_response!r}\n", flush=True)
        return self._clean_diff(raw_response)

    def _clean_diff(self, text: str) -> str:
        """Strips markdown code blocks and trailing prose from LLM response."""
        # Strip markdown fences if present
        text = self._strip_markdown_fences(text)

        # Extract only from the first unified diff header ('--- ' or 'diff --git')
        diff_match = re.search(r"^(?:--- |diff --git ).*", text, re.DOTALL | re.MULTILINE)
        if diff_match:
            return diff_match.group(0).strip()

        return text.strip()


class SecuritySubAgent(ClaudeAgent):
    """Audits proposed diffs specifically for Flask / Web Security vulnerabilities."""

    SYSTEM_PROMPT = (
        "You are a Web Application Security Engineer specialising in Python and Flask applications.\n"
        "Review the proposed patch for common vulnerabilities including:\n"
        "- SQL Injection / ORM misuse\n"
        "- Hardcoded secrets, keys, or endpoints\n"
        "- Broken authentication or session management\n"
        "- Insecure CORS settings, CSRF risks, or unvalidated redirects/uploads\n\n"
        "Respond strictly with JSON in the following format:\n"
        '{"status": "PASS" | "FAIL", "findings": ["list of findings or recommendation"]}'
    )

    def audit(self, diff: str) -> dict:
        if not diff.strip():
            return {"status": "PASS", "findings": ["No code changes to audit."]}
        
        raw = self.respond(self.SYSTEM_PROMPT, f"Proposed Diff:\n{diff}")
        print(f"[Security Raw Response]\n{raw!r}\n", flush=True)
        try:
            return json.loads(self._strip_markdown_fences(raw))
        except json.JSONDecodeError:
            return {
                "status": "FAIL",
                "findings": [f"Security Agent returned invalid JSON response: {raw}"],
            }


class WriterSubAgent(ClaudeAgent):
    """Generates and updates documentation (docstrings, README, OpenAPI/Swagger routes)."""
    # def __init__(self, model: str = "claude-opus-5-5"):
    #     super().__init__(model=model)

    SYSTEM_PROMPT = (
        "You are a Technical Writer specialising in REST APIs and Python Flask application documentation.\n"
        "Given a task and code diff, generate clear Markdown documentation updates, "
        "including updated route parameters, request/response payload examples, and env vars."
    )

    def write_documentation(self, task: str, diff: str) -> str:
        user_content = f"Task:\n{task}\n\nApplied Diff:\n{diff}"
        return self.respond(self.SYSTEM_PROMPT, user_content)


class TesterSubAgent:
    """Executes local testing suites (pytest) against applied patches to get deterministic feedback."""

    def __init__(self, repo_path: str | Path):
        self.repo_path = Path(repo_path)

    def run_tests(self) -> tuple[bool, str]:
        """Runs pytest on the local workspace and captures stderr/stdout output."""
        try:
            result = subprocess.run(
                ["testenv", "-v"],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                timeout=60,
            )
            passed = result.returncode == 0
            output = result.stdout if passed else f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
            return passed, output
        except FileNotFoundError:
            return False, "pytest executable not found in current environment."
        except subprocess.TimeoutExpired:
            return False, "Test suite execution timed out after 60 seconds."


# --------------------------------------------------------------------------- #
# Git Operations
# --------------------------------------------------------------------------- #

class GitManager:
    """Subprocess interface for feature branch git operations."""

    def __init__(self, repo_path: str | Path, protected_branches: tuple[str, ...] = ("main", "master")):
        self.repo_path = Path(repo_path)
        self._protected = protected_branches

    def _run(self, *args: str) -> str:
        result = subprocess.run(
            ["git", *args], cwd=self.repo_path, capture_output=True, text=True
        )
        if result.returncode != 0:
            raise RuntimeError(f"git {' '.join(args)} failed: {result.stderr}")
        return result.stdout.strip()

    def current_branch(self) -> str:
        return self._run("rev-parse", "--abbrev-ref", "HEAD")

    def ensure_branch(self, branch_name: str) -> None:
        """Switches to the branch if it exists, or creates it if it doesn't."""
        curr = self.current_branch()
        if curr == branch_name:
            return
        
        # Check if branch exists locally
        branches = self._run("branch", "--list", branch_name)
        if branches:
            self._run("checkout", branch_name)
        else:
            self._run("checkout", "-b", branch_name)

    def apply_patch(self, diff_text: str) -> None:
        if not diff_text.strip():
            return
        if not diff_text.endswith("\n"):
            diff_text += "\n"
        patch_path = self.repo_path / ".orchestrator_patch.diff"
        patch_path.write_text(diff_text, encoding="utf-8")
        try:
            self._run("apply", "--whitespace=fix", str(patch_path))
        finally:
            patch_path.unlink(missing_ok=True)

    def revert_uncommitted(self) -> None:
        self._run("checkout", "--", ".")
        self._run("clean", "-fd")

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
# Branch Caching Engine
# --------------------------------------------------------------------------- #

class BranchCache:
    """Stores agent results inside .git/ folder to keep cache tied to local branch state."""

    def __init__(self, repo_path: Path, branch_name: str):
        safe_branch = re.sub(r"[^\w\-]", "_", branch_name)
        self.cache_file = repo_path / ".git" / f"orchestrator_cache_{safe_branch}.json"
        self._data = self._load()

    def _load(self) -> dict:
        if self.cache_file.exists():
            try:
                return json.loads(self.cache_file.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                pass
        return {}

    def get(self, key: str):
        return self._data.get(key)

    def set(self, key: str, value):
        self._data[key] = value
        self.cache_file.write_text(json.dumps(self._data, indent=2), encoding="utf-8")

    def clear(self):
        if self.cache_file.exists():
            self.cache_file.unlink()

# --------------------------------------------------------------------------- #
# Claude Orchestrator
# --------------------------------------------------------------------------- #

@dataclass
class RoundLog:
    round_num: int
    diff: str
    security_status: str
    test_passed: bool
    feedback: str


@dataclass
class ClaudeOrchestrationSession:
    task: str
    repo_context: str
    repo_path: Path
    branch_name: str | None = None
    max_rounds: int = 3
    log: list[RoundLog] = field(default_factory=list)

    def __post_init__(self):
        if not self.branch_name:
            # Deterministic branch name slugified from task
            slug = re.sub(r"[^\w]+", "-", self.task.lower()).strip("-")[:30]
            self.branch_name = f"orch/{slug}"

        self.git = GitManager(self.repo_path)
        self.coder = CoderSubAgent()
        self.security = SecuritySubAgent()
        self.tester = TesterSubAgent(self.repo_path)
        self.writer = WriterSubAgent()
        self.cache = BranchCache(self.repo_path, self.branch_name)

    def run(self) -> Literal["pushed", "escalated", "gave_up"]:
        # Ensure we stay on or reuse the same feature branch
        self.git.ensure_branch(self.branch_name)
        feedback = ""

        for round_num in range(1, self.max_rounds + 1):
            print(f"\n--- Orchestrator Round {round_num}/{self.max_rounds} [Branch: {self.branch_name}] ---", flush=True)

            # Step 1: Coder agent proposes patch
            diff_key = f"round_{round_num}_diff"
            diff = self.cache.get(diff_key)
            if diff:
                print("[Cache] Loaded Coder diff from branch cache.")
            else:
                print("[Sub-Agent: Coder] Generating diff...")
                diff = self.coder.propose_patch(self.task, self.repo_context, feedback)
                if diff.strip():
                    self.cache.set(diff_key, diff)

            if not diff.strip():
                print("[Orchestrator] Empty diff produced. Escalating.")
                return "escalated"

            # Step 2: AppSec Agent performs security audit
            sec_key = f"round_{round_num}_sec"
            sec_report = self.cache.get(sec_key)
            if sec_report:
                print("[Cache] Loaded Security audit from branch cache.")
            else:
                print("[Sub-Agent: Security] Auditing patch...")
                sec_report = self.security.audit(diff)
                self.cache.set(sec_key, sec_report)

            print(f"[Security Result] {sec_report}", flush=True)

            if sec_report.get("status") == "FAIL":
                findings = "\n".join(sec_report.get("findings", []))
                feedback = f"SECURITY AUDIT FAILED:\n{findings}"
                continue

            # Step 3: Apply patch temporarily to test
            self.git.apply_patch(diff)

            # Step 4: Tester agent executes test suite
            print("[Sub-Agent: Tester] Running pytest suite...", flush=True)
            test_passed, test_output = self.tester.run_tests()
            print(f"[Tester Result] Passed: {test_passed}", flush=True)

            if not test_passed:
                print(f"[Tester Output]\n{test_output}\n", flush=True)
                self.git.revert_uncommitted()
                feedback = f"UNIT TESTS FAILED:\n{test_output}"
                self.log.append(RoundLog(round_num, diff, "PASS", False, feedback))
                continue

            # Step 5: Generate documentation update
            doc_key = f"round_{round_num}_docs"
            doc_update = self.cache.get(doc_key)
            if doc_update:
                print("[Cache] Loaded Writer docs from branch cache.")
            else:
                print("[Sub-Agent: Writer] Generating docs...")
                doc_update = self.writer.write_documentation(self.task, diff)
                self.cache.set(doc_key, doc_update)

            doc_path = self.repo_path / "CHANGELOG_ORCHESTRATOR.md"
            doc_path.write_text(f"## Task: {self.task}\n\n{doc_update}\n", encoding="utf-8")
            
            # Step 6: Commit and push changes
            print("[Git] Committing and pushing feature changes...", flush=True)
            self.git.commit(f"[orchestrator] {self.task[:72]}")
            self.git.push(self.branch_name)

            self.log.append(RoundLog(round_num, diff, "PASS", True, "Successfully verified and documented."))
            # Clean up branch cache after success
            self.cache.clear()
            return "pushed"

        return "gave_up"

    def summary(self) -> str:
        lines = [f"Session on branch '{self.branch_name}' — task: {self.task}"]
        for entry in self.log:
            lines.append(
                f"  Round {entry.round_num}: Security={entry.security_status} | "
                f"TestsPassed={entry.test_passed}"
            )
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# CLI Entry Point
# --------------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser(description="Run Claude Orchestration Session.")
    parser.add_argument("--repo", required=True, help="Path to local git repo")
    parser.add_argument("--task", required=True, help="Description of task to implement")
    parser.add_argument("--branch", default=None, help="Optional explicit branch name")
    parser.add_argument("--max-rounds", type=int, default=3)
    args = parser.parse_args()

    repo_context = ContextBuilder(
        standards=CodingStandards(oop=True),
        selector=RepoContextSelector(args.repo),
    ).build(args.task)

    session = ClaudeOrchestrationSession(
        task=args.task,
        repo_context=repo_context,
        repo_path=Path(args.repo),
        branch_name=args.branch,
        max_rounds=args.max_rounds,
    )
    
    outcome = session.run()
    print("\n" + session.summary())
    print(f"Final Outcome: {outcome}")
    sys.exit(0 if outcome == "pushed" else 1)


if __name__ == "__main__":
    main()