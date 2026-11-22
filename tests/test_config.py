"""Tests de la configuration centrale."""

import dataclasses

import pytest

from config import Thresholds, load_settings

ENV_VARS = ("UDP_HOST", "UDP_PORT", "OLLAMA_URL", "OLLAMA_MODEL", "WHISPER_MODEL", "ELEVENLABS_API_KEY")


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    """Retire les variables du projet pour tester les valeurs par défaut."""
    for name in ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


def test_defaults(clean_env: pytest.MonkeyPatch) -> None:
    s = load_settings()
    assert s.udp_host == "127.0.0.1"
    assert s.udp_port == 20777
    assert s.ollama_model == "llama3.1:8b"
    assert s.elevenlabs_api_key is None


def test_env_override(clean_env: pytest.MonkeyPatch) -> None:
    clean_env.setenv("UDP_PORT", "20778")
    clean_env.setenv("OLLAMA_MODEL", "qwen2.5:7b")
    s = load_settings()
    assert s.udp_port == 20778
    assert s.ollama_model == "qwen2.5:7b"


def test_default_thresholds() -> None:
    t = Thresholds()
    assert t.lap_delta_s == 0.5
    assert t.tyre_age_laps == 20
    assert t.gap_danger_behind_s == 2.0
    assert t.non_critical_cooldown_s == 15.0
    assert t.tyre_wear_tiers_pct == (50.0, 75.0, 90.0)


def test_thresholds_are_immutable() -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        Thresholds().lap_delta_s = 1.0  # type: ignore[misc]
