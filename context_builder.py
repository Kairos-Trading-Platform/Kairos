"""
Context-selection step for the orchestrator.

Two independent concerns, composed:
  CodingStandards     — the fixed constraints every builder prompt must carry
                         (OOP, Numba-where-numeric, etc). Not derived from the
                         repo; this is "who you are" context.
  RepoContextSelector  — task-relevant file snippets pulled from the repo,
                         keyword-scored, budget-capped. This is "what you're
                         working on" context.
  ContextBuilder       — combines both into the single string ClaudeBuilder
                         consumes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

STOPWORDS = {
    "the", "a", "an", "to", "for", "of", "in", "on", "and", "or", "is",
    "fix", "add", "make", "update", "change", "with", "that", "this",
}

DEFAULT_EXTENSIONS = (".py", ".html", ".js")
EXCLUDED_DIRS = {".git", "__pycache__", "node_modules", "venv", ".venv", "data", "instance"}


@dataclass
class CodingStandards:
    """Constraints the builder must follow, independent of any one task."""

    oop: bool = True
    numba_when_numeric: bool = True
    language: str = "Python"
    extra_notes: list[str] = field(default_factory=list)

    def to_prompt(self) -> str:
        rules = [f"- Write {self.language}."]
        if self.oop:
            rules.append("- Prefer classes over free-floating functions; keep responsibilities single-purpose."
                         "Find the less redundant solution, i.e. additions should not duplicate existing code."
                         )
        if self.numba_when_numeric:
            rules.append(
                "- If the change involves a numeric hot loop (array math, "
                "simulation, per-row computation), use Numba (@njit) instead "
                "of plain Python loops. Skip Numba entirely for I/O, string, "
                "or Flask/Jinja code — it doesn't apply there."
            )
        rules.extend(f"- {note}" for note in self.extra_notes)
        return "\n".join(rules)


@dataclass
class ScoredFile:
    path: Path
    score: int
    snippet: str


class RepoContextSelector:
    """Keyword-ranks repo files against a task description, returns top snippets."""

    def __init__(
        self,
        repo_path: str | Path,
        extensions: tuple[str, ...] = DEFAULT_EXTENSIONS,
        max_files: int = 6,
        max_chars_per_file: int = 2000,
    ):
        self.repo_path = Path(repo_path)
        self.extensions = extensions
        self.max_files = max_files
        self.max_chars_per_file = max_chars_per_file

    def select(self, task: str) -> str:
        keywords = self._extract_keywords(task)
        candidates = [
            self._score_file(path, keywords)
            for path in self._iter_files()
        ]
        top = sorted((c for c in candidates if c.score > 0), key=lambda c: -c.score)[: self.max_files]

        if not top:
            return "(no files matched task keywords — builder should search the repo itself)"

        return "\n\n".join(
            f"--- {f.path.relative_to(self.repo_path)} (score={f.score}) ---\n{f.snippet}"
            for f in top
        )

    def _extract_keywords(self, task: str) -> set[str]:
        tokens = re.findall(r"[a-zA-Z_]+", task.lower())
        return {t for t in tokens if t not in STOPWORDS and len(t) > 2}

    def _iter_files(self):
        for path in self.repo_path.rglob("*"):
            if not path.is_file() or path.suffix not in self.extensions:
                continue
            if any(part in EXCLUDED_DIRS for part in path.parts):
                continue
            yield path

    def _score_file(self, path: Path, keywords: set[str]) -> ScoredFile:
        try:
            text = path.read_text(errors="ignore")
        except OSError:
            return ScoredFile(path, 0, "")

        lower = text.lower()
        path_score = sum(2 for kw in keywords if kw in str(path).lower())
        body_score = sum(lower.count(kw) for kw in keywords)
        score = path_score + body_score

        snippet = self._extract_snippet(text, lower, keywords) if score else ""
        return ScoredFile(path, score, snippet)

    def _extract_snippet(self, text: str, lower: str, keywords: set[str]) -> str:
        # Centre the snippet on the first keyword hit rather than always
        # taking the file's head, so the relevant function is more likely
        # to survive the char budget.
        first_hit = min(
            (lower.find(kw) for kw in keywords if kw in lower),
            default=0,
        )
        start = max(0, first_hit - 200)
        end = min(len(text), start + self.max_chars_per_file)
        return text[start:end]


class ContextBuilder:
    """Combines coding standards + repo-relevant snippets into one context string."""

    def __init__(self, standards: CodingStandards, selector: RepoContextSelector):
        self.standards = standards
        self.selector = selector

    def build(self, task: str) -> str:
        return (
            f"## Coding standards\n{self.standards.to_prompt()}\n\n"
            f"## Relevant repo excerpts\n{self.selector.select(task)}"
        )
