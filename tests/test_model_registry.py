"""FP-08-6 — model socket + registry dispatch (G-REGISTRY-DISPATCH)."""
import pytest

from src.brain.model_registry import (
    Model, TiltSelectAdapter, resolve_live_engine,
    EngineRegistryError, REGISTERED_ENGINES,
)


def test_registered_engines_resolve(monkeypatch):
    # no engine_sha declared -> registry membership is enough
    assert resolve_live_engine({"engine": "native_two_stage"}) == "native_two_stage"
    assert resolve_live_engine({"engine": "tilt_adapter"}) == "tilt_adapter"


def test_unregistered_engine_aborts():
    with pytest.raises(EngineRegistryError):
        resolve_live_engine({"engine": "totally_made_up"})
    with pytest.raises(EngineRegistryError):
        resolve_live_engine({"engine": ""})


def test_cold_start_hash_mismatch_aborts(monkeypatch):
    import src.brain.model_registry as mr
    monkeypatch.setattr(mr, "_freeze_engine_sha", lambda: "frozensha123")
    # declared engine_sha disagrees with the freeze -> hard stop
    with pytest.raises(EngineRegistryError):
        resolve_live_engine({"engine": "native_two_stage", "engine_sha": "differentsha"})
    # matching -> resolves
    assert resolve_live_engine(
        {"engine": "native_two_stage", "engine_sha": "frozensha123"}) == "native_two_stage"


def test_tilt_adapter_conforms_to_model_protocol():
    t = TiltSelectAdapter()
    assert isinstance(t, Model)
    assert t.name == "tilt_adapter"
    with pytest.raises(NotImplementedError):
        t.select(None, None)  # intents-first comparison model, no select seam
