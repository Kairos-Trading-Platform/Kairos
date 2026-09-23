"""
Context-selection step for the Claude Orchestrator.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

STOPWORDS = {
    "the", "a", "an", "to", "for", "of", "in", "on", "and", "or", "is",
    "fix", "add", "make", "update", "change", "with", "that", "this",
}

DEFAULT_EXTENSIONS = (".py", ".html", ".js", ".json", ".md")
EXCLUDED_DIRS = {".git", "__pycache__", "node_modules", "venv", ".venv", "instance"}


@dataclass
class CodingStandards:
    """Constraints the builder must follow."""

    oop: bool = True
    language: str = "Python"
    extra_notes: list[str] = field(default_factory=list)

    def to_prompt(self) -> str:
        rules = [f"- Write clean, idiomatic {self.language}."]
        if self.oop:
            rules.append("- Prefer classes and modular blueprints over monolithic script blocks.")
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
        max_chars_per_file: int = 4000,
    ):
        self.repo_path = Path(repo_path)
        self.extensions = extensions
        self.max_files = max_files
        self.max_chars_per_file = max_chars_per_file

    def select(self, task: str) -> str:
        keywords = self._extract_keywords(task)
        candidates = [
            self._score_file(path, keywords, task)
            for path in self._iter_files()
        ]
        
        # Sort by score descending
        top = sorted((c for c in candidates if c.score > 0), key=lambda c: -c.score)[: self.max_files]

        # FALLBACK: If no keyword matches, return files present in repo so Claude isn't blind
        if not top:
            all_files = list(self._iter_files())[: self.max_files]
            if not all_files:
                return "(Repository is empty)"
            
            snippets = []
            for path in all_files:
                try:
                    text = path.read_text(encoding="utf-8", errors="ignore")[: self.max_chars_per_file]
                    snippets.append(f"--- {path.relative_to(self.repo_path)} (Fallback Preview) ---\n{text}")
                except OSError:
                    continue
            return "\n\n".join(snippets)

        return "\n\n".join(
            f"--- {f.path.relative_to(self.repo_path)} (score={f.score}) ---\n{f.snippet}"
            for f in top
        )

    def _extract_keywords(self, task: str) -> set[str]:
        tokens = re.findall(r"[a-zA-Z_0-9\.]+", task.lower())
        return {t for t in tokens if t not in STOPWORDS and len(t) > 1}

    def _iter_files(self):
        for path in self.repo_path.rglob("*"):
            if not path.is_file() or path.suffix not in self.extensions:
                continue
            if any(part in EXCLUDED_DIRS for part in path.parts):
                continue
            yield path

    def _score_file(self, path: Path, keywords: set[str], raw_task: str) -> ScoredFile:
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            return ScoredFile(path, 0, "")

        lower = text.lower()
        file_name = path.name.lower()

        # Direct file name match in task prompt gets massive score boost
        path_score = 0
        if file_name in raw_task.lower():
            path_score += 100

        path_score += sum(10 for kw in keywords if kw in str(path).lower())
        body_score = sum(lower.count(kw) for kw in keywords)
        score = path_score + body_score

        snippet = self._extract_snippet(text, lower, keywords) if score else ""
        return ScoredFile(path, score, snippet)

    def _extract_snippet(self, text: str, lower: str, keywords: set[str]) -> str:
        # If file is within max character budget, send entire file
        if len(text) <= self.max_chars_per_file:
            return text

        first_hit = min(
            (lower.find(kw) for kw in keywords if kw in lower),
            default=0,
        )
        start = max(0, first_hit - 200)
        end = min(len(text), start + self.max_chars_per_file)
        return text[start:end]


class ContextBuilder:
    def __init__(self, standards: CodingStandards, selector: RepoContextSelector):
        self.standards = standards
        self.selector = selector

    def build(self, task: str) -> str:
        return (
            f"## Coding Standards\n{self.standards.to_prompt()}\n\n"
            f"## Relevant Repo Excerpts\n{self.selector.select(task)}"
        )