"""Runtime configuration, loaded from environment variables with the ``SV_`` prefix."""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="SV_", env_file=".env", extra="ignore")

    # Below this model confidence, an assessment is routed to manual review.
    min_confidence: float = 0.55

    # Deterministic rule-layer thresholds for the unambiguous cases (see detector.py).
    hotspot_contrast_z: float = 6.0
    dead_cell_dark_frac: float = 0.020
    crack_min_elongation: float = 4.0


def get_settings() -> Settings:
    return Settings()
