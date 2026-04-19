from __future__ import annotations

from pathlib import Path

import pytest

from categorizer.taxonomy import Taxonomy, load_taxonomy

TAXONOMY_YAML = Path(__file__).resolve().parents[1] / "config" / "taxonomy.yaml"


@pytest.fixture(scope="module")
def taxonomy() -> Taxonomy:
    return load_taxonomy(TAXONOMY_YAML)


def test_emittable_slugs_excludes_parents(taxonomy: Taxonomy) -> None:
    parent_slugs = {c.slug for c in taxonomy._by_slug.values() if c.parent_slug is None}
    for slug in taxonomy.emittable_slugs:
        assert slug not in parent_slugs, (
            f"Emittable slug {slug!r} is a top-level parent; LLM must emit leaves only."
        )


def test_emittable_slugs_are_all_leaves(taxonomy: Taxonomy) -> None:
    for slug in taxonomy.emittable_slugs:
        assert not list(taxonomy.children_of(slug)), (
            f"Emittable slug {slug!r} has children; it is a parent, not a leaf."
        )


def test_emittable_slugs_includes_pendiente(taxonomy: Taxonomy) -> None:
    assert "sin_clasificar.pendiente" in taxonomy.emittable_slugs


def test_emittable_slugs_are_implemented(taxonomy: Taxonomy) -> None:
    for slug in taxonomy.emittable_slugs:
        cat = taxonomy.get(slug)
        assert cat is not None and cat.implemented


def test_implemented_slugs_still_includes_parents(taxonomy: Taxonomy) -> None:
    assert "comida" in taxonomy.implemented_slugs
    assert "educacion" in taxonomy.implemented_slugs
    assert "comida" not in taxonomy.emittable_slugs
    assert "educacion" not in taxonomy.emittable_slugs
