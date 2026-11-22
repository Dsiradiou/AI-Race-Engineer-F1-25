"""Configuration centrale du projet.

Deux sources :
- les variables d'environnement (fichier `.env`, voir `.env.example`) pour ce qui dépend
  de la machine ou est secret : ports, URL, noms de modèles, clés API ;
- les seuils de détection ci-dessous, définis dans le code et ajustés en Phase 7.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Thresholds:
    """Seuils de déclenchement des experts (section 6 du cadrage)."""

    lap_delta_s: float = 0.5
    """Pace : alerte si le tour est plus lent que le meilleur tour perso de plus de X s."""

    tyre_age_laps: int = 20
    """Strategy : avertissement si les pneus ont plus de X tours."""

    gap_danger_behind_s: float = 2.0
    """Strategy : danger si la voiture derrière est à moins de X s."""

    damage_pct: float = 0.0
    """Meca : alerte dès que les dégâts dépassent X %."""

    tyre_wear_tiers_pct: tuple[float, ...] = (50.0, 75.0, 90.0)
    """Paliers d'usure pneus : info, avertissement, critique (90 % = valeur initiale à affiner)."""

    non_critical_cooldown_s: float = 15.0
    """Délai minimum entre deux messages non critiques."""


@dataclass(frozen=True)
class Settings:
    """Paramètres dépendant de l'environnement d'exécution."""

    udp_host: str
    udp_port: int
    ollama_url: str
    ollama_model: str
    whisper_model: str
    elevenlabs_api_key: str | None
    thresholds: Thresholds


def load_settings() -> Settings:
    """Construit les paramètres à partir des variables d'environnement (avec valeurs par défaut)."""
    return Settings(
        udp_host=os.getenv("UDP_HOST", "127.0.0.1"),
        udp_port=int(os.getenv("UDP_PORT", "20777")),
        ollama_url=os.getenv("OLLAMA_URL", "http://localhost:11434"),
        ollama_model=os.getenv("OLLAMA_MODEL", "llama3.1:8b"),
        whisper_model=os.getenv("WHISPER_MODEL", "small"),
        elevenlabs_api_key=os.getenv("ELEVENLABS_API_KEY") or None,
        thresholds=Thresholds(),
    )


settings = load_settings()
