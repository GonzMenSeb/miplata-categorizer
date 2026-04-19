from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass(frozen=True)
class Category:
    slug: str
    name: str
    parent_slug: str | None
    implemented: bool
    examples: tuple[str, ...] = field(default_factory=tuple)


class Taxonomy:
    """Loaded once at startup; immutable afterwards.

    Use `all_slugs` for JSON-schema enum generation, `implemented_slugs` for
    gating LLM outputs (we never emit a carcass slug as a prediction).
    """

    def __init__(self, categories: list[Category]) -> None:
        self._by_slug: dict[str, Category] = {c.slug: c for c in categories}
        self._children: dict[str | None, list[Category]] = {}
        for c in categories:
            self._children.setdefault(c.parent_slug, []).append(c)

        self._all_slugs: tuple[str, ...] = tuple(c.slug for c in categories)
        self._implemented_slugs: tuple[str, ...] = tuple(
            c.slug for c in categories if c.implemented
        )
        # LLM output enum: only implemented LEAVES (no parent/root slugs).
        # Parents are structural; emitting one would be meaningless for
        # spending analytics and skews kNN training when labels flow back.
        self._emittable_slugs: tuple[str, ...] = tuple(
            c.slug
            for c in categories
            if c.implemented and self._children.get(c.slug) is None
        )

    # ── Accessors ──────────────────────────────────────────────────────────
    @property
    def all_slugs(self) -> tuple[str, ...]:
        return self._all_slugs

    @property
    def implemented_slugs(self) -> tuple[str, ...]:
        return self._implemented_slugs

    @property
    def emittable_slugs(self) -> tuple[str, ...]:
        """Slugs valid as LLM / cascade outputs: implemented leaves only."""
        return self._emittable_slugs

    def get(self, slug: str) -> Category | None:
        return self._by_slug.get(slug)

    def roots(self) -> Iterator[Category]:
        yield from self._children.get(None, ())

    def children_of(self, parent_slug: str | None) -> Iterator[Category]:
        yield from self._children.get(parent_slug, ())

    def parent_of(self, slug: str) -> Category | None:
        child = self._by_slug.get(slug)
        if not child or child.parent_slug is None:
            return None
        return self._by_slug.get(child.parent_slug)

    def path(self, slug: str) -> tuple[Category, ...]:
        """Root-to-leaf path. Useful for hierarchical metrics."""
        out: list[Category] = []
        cur = self._by_slug.get(slug)
        while cur is not None:
            out.append(cur)
            cur = self._by_slug.get(cur.parent_slug) if cur.parent_slug else None
        return tuple(reversed(out))

    def is_valid(self, slug: str) -> bool:
        return slug in self._by_slug

    def __contains__(self, slug: str | None) -> bool:
        """Nullable-friendly membership check.

        Callers frequently test ``candidate_slug in taxonomy`` where the
        candidate is ``str | None`` (LLM parsed output, merchant default
        category). Accepting ``None`` and returning ``False`` for it keeps
        those call-sites type-clean without per-site guards.
        """
        if slug is None:
            return False
        return slug in self._by_slug

    def __len__(self) -> int:
        return len(self._by_slug)


def load_taxonomy(path: Path) -> Taxonomy:
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    cats: list[Category] = []
    for entry in raw.get("categories", []):
        cats.append(
            Category(
                slug=entry["slug"],
                name=entry["name"],
                parent_slug=entry.get("parent"),
                implemented=bool(entry.get("implemented", False)),
                examples=tuple(entry.get("examples", [])),
            )
        )
    return Taxonomy(cats)
