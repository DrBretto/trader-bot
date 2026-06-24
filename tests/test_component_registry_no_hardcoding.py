"""PKT-TB-BV-01 — the no-hard-coding CI for the component registry.

Two rules (DESIGN_DOSSIER §5.4), enforced as failing tests so a hard-coded
component id breaks `scripts/check.sh` / CI:

  (a) A literal component-id ARRAY or component->label MAP in a frontend render
      module fails the build. The registry is the single source of truth; the
      brittleness this replaces is exactly an enumeration of component ids living
      in the render code (the old `LADDER_RUNGS` + `COMPONENT_LABEL`).

  (b) A shared test asserts:
        - writer-methods            ⊆ registry  (the attribution writer's
                                                 component keys are all declared)
        - live-engine ladder        ⊆ registry  (the live ladder's rung
                                                 components are all declared, with
                                                 matching book_pairs)
        - the registry carries NO measured numbers (existence + how-measured only)
        - the legacy->ladder migration map is derived from the registry, and
          matches the engine's ladder
        - zero hard-coded component-id enumerations anywhere in the frontend.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SHADOW_DIR = REPO / "runs" / "pkt_tb_007_orthogonal_brain" / "shadow"
FRONTEND_SRC = REPO / "frontend" / "src"
REGISTRY_DIR = FRONTEND_SRC / "registry"

# the Python registry loader is the single source the writer + frontend share
sys.path.insert(0, str(REPO))
from src.brain.component_registry import (          # noqa: E402
    load_registry, component_ids, ladder_migration_plan, ladder_books)


# ----------------------------------------------------------------- fixtures
def _registry():
    return load_registry()


def _shadow_lib():
    """Import shadow_lib (stdlib-only at module top) to read the live ladder /
    the attribution writer's component keys."""
    if str(SHADOW_DIR) not in sys.path:
        sys.path.insert(0, str(SHADOW_DIR))
    import shadow_lib  # noqa: E402
    return shadow_lib


def _frontend_modules():
    """Every frontend .ts/.tsx module EXCEPT the registry source dir (the one
    place ids legitimately live) and type-declaration files."""
    out = []
    for p in FRONTEND_SRC.rglob("*.ts*"):
        if REGISTRY_DIR in p.parents:
            continue
        if p.name.endswith(".d.ts"):
            continue
        out.append(p)
    return out


_COMMENT_BLOCK = re.compile(r"/\*.*?\*/", re.S)
_COMMENT_LINE = re.compile(r"//[^\n]*")


def _strip_comments(src: str) -> str:
    return _COMMENT_LINE.sub("", _COMMENT_BLOCK.sub("", src))


def _id_string_literals(src: str, ids: set) -> set:
    """Distinct registry ids that appear as an EXACT quoted string literal."""
    found = set()
    for m in re.finditer(r"""(['"])([^'"]+)\1""", src):
        if m.group(2) in ids:
            found.add(m.group(2))
    return found


# ----------------------------------------------------------------- rule (b)
def test_registry_loads_and_ids_are_unique():
    reg = _registry()
    ids = component_ids(reg)
    assert ids, "registry has no components"
    assert len(ids) == len(set(ids)), "registry ids must be unique"
    # the inventory must include the dark, the comparison, and the retired
    statuses = {c["expected_status"] for c in reg["components"]}
    assert {"live", "comparison", "retired"}.issubset(statuses), (
        f"registry must enumerate live + comparison + retired; saw {statuses}")
    assert any(c["instrumentation_status"] == "retired" for c in reg["components"])


def test_writer_methods_subset_of_registry():
    """The attribution writer's component keys (shadow_lib RUNGS) are all
    declared in the registry — a writer that emits an undeclared component is a
    build failure."""
    SL = _shadow_lib()
    writer_components = {r["component"] for r in SL.RUNGS}
    ids = set(component_ids())
    missing = writer_components - ids
    assert not missing, f"writer components not in registry: {sorted(missing)}"


def test_live_ladder_components_subset_of_registry():
    """Every live-engine ladder rung is a registry component, with a matching
    book + book_pair (the join the frontend relies on)."""
    SL = _shadow_lib()
    reg = _registry()
    by_id = {c["id"]: c for c in reg["components"]}
    for rung in SL.RUNGS:
        comp = rung["component"]
        assert comp in by_id, f"ladder component {comp!r} missing from registry"
        d = by_id[comp]
        book_pair = f"{rung['book']}-{rung['parent']}"
        assert d["attribution"]["book"] == rung["book"], (
            f"{comp}: registry book {d['attribution']['book']} != engine {rung['book']}")
        assert d["attribution"]["book_pair"] == book_pair, (
            f"{comp}: registry book_pair {d['attribution']['book_pair']} != {book_pair}")


def test_registry_carries_no_measured_numbers():
    """The registry declares existence + how-measured only. The only numeric
    field allowed is the structural `order`; nothing that looks like a measured
    value (bp/day, IC, counts, CIs) may appear."""
    reg = _registry()
    allowed_numeric_keys = {"order"}

    def _walk(obj, path=""):
        if isinstance(obj, dict):
            for k, v in obj.items():
                _walk(v, f"{path}.{k}")
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                _walk(v, f"{path}[{i}]")
        elif isinstance(obj, bool):
            return                              # booleans are fine (strippable, etc.)
        elif isinstance(obj, (int, float)):
            key = path.rsplit(".", 1)[-1]
            assert key in allowed_numeric_keys, (
                f"registry carries a measured number at {path} = {obj!r}; "
                f"numbers join from the contribution ledger, never the registry")

    _walk(reg)


def test_migration_map_is_derived_from_registry():
    """The legacy->ladder migration map is data-described from the registry's
    ladder_books, and matches the engine's ladder (shadow_lib.BOOKS)."""
    SL = _shadow_lib()
    plan = ladder_migration_plan()
    assert tuple(plan["books"]) == SL.BOOKS, (
        f"registry ladder {plan['books']} != engine BOOKS {list(SL.BOOKS)}")
    # every legacy rename target and seed source is a real ladder book
    for legacy, new in plan["rename_map"].items():
        assert new in plan["books"], f"rename target {new} not a ladder book"
    for new, parent in plan["seed_map"].items():
        assert new in plan["books"] and parent in plan["books"]
    # the registry's ladder_books cover exactly the engine's LADDER components
    reg_components = {b.get("component") for b in ladder_books()}
    eng_components = {r["component"] for r in SL.LADDER}   # incl. None baseline
    assert reg_components == eng_components, (
        f"registry ladder components {reg_components} != engine {eng_components}")


# ----------------------------------------------------------------- rule (a)
def test_no_hardcoded_component_id_enumeration_in_frontend():
    """No frontend render/data module (outside the registry source) may
    enumerate component ids — i.e. carry >= 2 distinct registry-id string
    literals (an id array or a component->label map). A single targeted id
    reference (e.g. a lone equality guard) is allowed; an ENUMERATION is the
    brittleness the registry replaces."""
    ids = set(component_ids())
    offenders = {}
    for p in _frontend_modules():
        src = _strip_comments(p.read_text())
        hits = _id_string_literals(src, ids)
        if len(hits) >= 2:
            offenders[str(p.relative_to(REPO))] = sorted(hits)
    assert not offenders, (
        "hard-coded component-id enumeration found in the frontend (use the "
        f"registry instead): {offenders}")


# ----------------------------------------------------------------- rule (a) demo
def test_a_hardcoded_id_array_would_fail_the_build():
    """Demonstration that rule (a) actually bites: a synthetic render module
    carrying a literal component-id array / label map is detected as an
    enumeration (>= 2 registry ids)."""
    ids = set(component_ids())
    bad_render_module = (
        "const LADDER_RUNGS = ['regime', 'forecast', 'event', 'universe'];\n"
        "const COMPONENT_LABEL = { regime: 'Regime', forecast: 'Forecast' };\n"
    )
    hits = _id_string_literals(_strip_comments(bad_render_module), ids)
    assert len(hits) >= 2, "the scanner must flag a hard-coded component-id array/map"
