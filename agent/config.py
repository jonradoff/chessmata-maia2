"""Configuration loading for the Maia2 agent."""

import os
from dataclasses import dataclass, field
from typing import List

import yaml


@dataclass
class AgentVariant:
    """A Maia2 agent variant with a specific Elo."""
    name: str
    elo: int


@dataclass
class ChessmataConfig:
    """Chessmata server connection settings."""
    server_url: str = "https://chessmata.metavert.io"
    api_key: str = ""


@dataclass
class AgentConfig:
    """Agent behavior settings."""
    max_concurrent_games: int = 50
    max_concurrent_per_variant: int = 0  # 0 = unlimited
    poll_interval_seconds: float = 1.5
    matchmaking_poll_interval_seconds: float = 3.0
    client_software: str = "Chessmata-Maia2/1.0"
    variants: List[AgentVariant] = field(default_factory=lambda: [
        AgentVariant("Maia2-400", 400),
        AgentVariant("Maia2-600", 600),
        AgentVariant("Maia2-800", 800),
        AgentVariant("Maia2-1000", 1000),
        AgentVariant("Maia2-1200", 1200),
        AgentVariant("Maia2-1500", 1500),
        AgentVariant("Maia2-1800", 1800),
        AgentVariant("Maia2-2100", 2100),
    ])


@dataclass
class EngineConfig:
    """Maia2 engine settings."""
    model_type: str = "rapid"
    device: str = "auto"


@dataclass
class LoggingConfig:
    """Logging settings."""
    level: str = "INFO"
    file: str = "maia2_agent.log"


@dataclass
class Config:
    """Top-level configuration."""
    chessmata: ChessmataConfig = field(default_factory=ChessmataConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    engine: EngineConfig = field(default_factory=EngineConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


def load_config(path: str = "config.yaml") -> Config:
    """Load configuration from a YAML file.

    Falls back to CHESSMATA_CONFIG env var, then default path.
    """
    config_path = os.environ.get("CHESSMATA_CONFIG", path)

    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Config file not found: {config_path}\n"
            "Copy config.example.prod.yaml to config.yaml and fill in your API key."
        )

    with open(config_path, "r") as f:
        raw = yaml.safe_load(f)

    cfg = Config()

    # Chessmata section
    cm = raw.get("chessmata", {})
    cfg.chessmata.server_url = cm.get("server_url", cfg.chessmata.server_url)
    cfg.chessmata.api_key = cm.get("api_key", cfg.chessmata.api_key)

    # Environment variable overrides (for containerized deployments)
    if os.environ.get("CHESSMATA_API_KEY"):
        cfg.chessmata.api_key = os.environ["CHESSMATA_API_KEY"]

    # Agent section
    ag = raw.get("agent", {})
    cfg.agent.max_concurrent_games = ag.get("max_concurrent_games", cfg.agent.max_concurrent_games)
    cfg.agent.max_concurrent_per_variant = ag.get("max_concurrent_per_variant", cfg.agent.max_concurrent_per_variant)
    cfg.agent.poll_interval_seconds = ag.get("poll_interval_seconds", cfg.agent.poll_interval_seconds)
    cfg.agent.matchmaking_poll_interval_seconds = ag.get(
        "matchmaking_poll_interval_seconds", cfg.agent.matchmaking_poll_interval_seconds
    )
    cfg.agent.client_software = ag.get("client_software", cfg.agent.client_software)

    variants_raw = ag.get("variants")
    if variants_raw:
        cfg.agent.variants = [
            AgentVariant(name=v["name"], elo=v["elo"])
            for v in variants_raw
        ]

    # Engine section
    eng = raw.get("engine", {})
    cfg.engine.model_type = eng.get("model_type", cfg.engine.model_type)
    cfg.engine.device = eng.get("device", cfg.engine.device)

    # Logging section
    log = raw.get("logging", {})
    cfg.logging.level = log.get("level", cfg.logging.level)
    cfg.logging.file = log.get("file", cfg.logging.file)

    return cfg
