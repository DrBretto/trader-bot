"""Component registry loader (Python side) — PKT-TB-BV-01.

The declarative component registry is the single source of truth for what
components the brain has and how each is measured. The canonical document is
``frontend/src/registry/component_registry.json``; the frontend renders off it
(``componentRegistry.ts``) and the nightly attribution/migration code reads the
SAME file here. It carries NO measured numbers — existence + how-it-would-be-
measured only; numbers join from the contribution ledger (BV-03) on the stable
``id``.

This module is intentionally dependency-free (stdlib only) and loads the JSON
lazily, so importing it (or ``shadow_lib``, which uses it) never requires the
frontend tree to be present until a function actually needs the registry.
"""
from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

_THIS = Path(__file__).resolve()
# src/brain/component_registry.py -> parents[2] == repo root
_REPO_ROOT = _THIS.parents[3]
_DEFAULT_REL = "frontend/src/registry/component_registry.json"


def registry_path() -> Path:
    """Resolve the canonical registry JSON. ``BRAIN_COMPONENT_REGISTRY`` overrides
    (e.g. inside a packaged image where the frontend tree is relocated)."""
    env = os.environ.get("BRAIN_COMPONENT_REGISTRY")
    if env:
        return Path(env)
    return _REPO_ROOT / _DEFAULT_REL


@lru_cache(maxsize=1)
def load_registry() -> Dict[str, Any]:
    """Load + cache the registry. Refuses to guess: a missing/garbled registry
    raises (the same forward-only/refuse-to-guess discipline the migration uses),
    never silently returns an empty set."""
    p = registry_path()
    if not p.exists():
        raise FileNotFoundError(
            f"component registry not found at {p}; set BRAIN_COMPONENT_REGISTRY "
            f"or restore {_DEFAULT_REL}")
    reg = json.loads(p.read_text())
    if not isinstance(reg.get("components"), list) or not reg["components"]:
        raise ValueError(f"component registry at {p} has no components")
    return reg


def components(registry: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    return (registry or load_registry())["components"]


def component_ids(registry: Optional[Dict[str, Any]] = None) -> List[str]:
    return [c["id"] for c in components(registry)]


def ladder_books(registry: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    return (registry or load_registry()).get("ladder_books", [])


# ----------------------------------------------------------------- migration map
def ladder_migration_plan(registry: Optional[Dict[str, Any]] = None
                          ) -> Dict[str, Any]:
    """Derive the legacy->ladder migration map from the registry's ``ladder_books``
    (data-described), replacing the old hand-written Python table in
    ``shadow_lib.migrate_state_books``.

    Each ladder book carries ``parent`` (its seed source) and ``history_of``
    (prior book ids it inherits). From those:

      - ``books``        — the full ladder, in order (== shadow_lib.BOOKS).
      - ``rename_map``   — {legacy_id -> ladder_book} for every id listed in any
                           ``history_of`` (incl. the identity I->I).
      - ``seed_map``     — {new_book -> parent_book} for every ladder book with an
                           empty ``history_of`` (a genuinely-new rung seeded from
                           its parent, e.g. R<-I, U<-E).
      - ``legacy_books`` — the set of all ``history_of`` ids (== the pre-migration
                           book set, e.g. {I, A, B}).
      - ``baseline_book``— the ladder book with no organ (component is null), i.e.
                           the incumbent book I; seeding FROM it contributes no
                           extra actions (n_actions == 0), matching the original.
    """
    lbs = ladder_books(registry)
    if not lbs:
        raise ValueError("registry has no ladder_books — cannot derive migration map")

    books = [b["book"] for b in lbs]
    rename_map: Dict[str, str] = {}
    seed_map: Dict[str, str] = {}
    legacy: set = set()
    baseline_book: Optional[str] = None

    for b in lbs:
        book = b["book"]
        if b.get("component") is None:
            baseline_book = book
        hist = b.get("history_of") or []
        for h in hist:
            rename_map[h] = book
            legacy.add(h)
        if not hist:
            parent = b.get("parent")
            if parent is None:
                raise ValueError(
                    f"ladder book {book!r} has empty history_of and no parent — "
                    f"cannot determine how to seed it (refuse to guess)")
            seed_map[book] = parent

    return {"books": books, "rename_map": rename_map, "seed_map": seed_map,
            "legacy_books": sorted(legacy), "baseline_book": baseline_book,
            "ladder_books": lbs}
