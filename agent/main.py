"""Entry point for the Chessmata Maia2 agent."""

import asyncio
import logging
import logging.handlers
import os
import signal
import sys

from .config import load_config
from .chessmata_client import ChessmataClient
from .maia2_engine import Maia2Engine, BatchingEngine
from .game_manager import GameManager


def setup_logging(level: str, log_file: str):
    """Configure logging to both console and a rotating file."""
    fmt = "%(asctime)s %(levelname)-7s %(name)s  %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file:
        rotating = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10 MB per file
            backupCount=5,              # keep 5 old files
        )
        handlers.append(rotating)

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=fmt,
        datefmt=datefmt,
        handlers=handlers,
    )


async def run_agent():
    """Load everything and run the game manager loop."""
    # Resolve config path relative to project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.environ.get("CHESSMATA_CONFIG", os.path.join(project_root, "config.yaml"))

    config = load_config(config_path)
    setup_logging(config.logging.level, config.logging.file)

    logger = logging.getLogger("agent")
    logger.info("=" * 60)
    logger.info("  Chessmata Maia2 Agent starting")
    logger.info("  Server: %s", config.chessmata.server_url)
    logger.info("  Variants: %s", ", ".join(v.name for v in config.agent.variants))
    logger.info("  Max concurrent games: %d (per-variant: %s)",
                config.agent.max_concurrent_games,
                config.agent.max_concurrent_per_variant or "unlimited")
    logger.info("  Engine model: %s (device: %s)", config.engine.model_type, config.engine.device)
    logger.info("=" * 60)

    # Load and warm up the Maia2 model
    engine = Maia2Engine(model_type=config.engine.model_type, device=config.engine.device)
    engine.load()
    engine.warmup()

    # Start the batching engine
    batching = BatchingEngine(engine)
    batching.start()

    # Create the API client
    client = ChessmataClient(
        server_url=config.chessmata.server_url,
        api_key=config.chessmata.api_key,
        client_software=config.agent.client_software,
    )
    await client.start()

    # Verify authentication
    try:
        user = await client.get_current_user()
        display = user.get("displayName", "?")
        elo = user.get("eloRating", "?")
        logger.info("Authenticated as: %s (Elo %s)", display, elo)
    except Exception as e:
        logger.error("Authentication failed: %s", e)
        await client.close()
        sys.exit(1)

    # Run the game manager
    manager = GameManager(config, batching, client)

    # Resume any in-progress games from a previous session
    user_id = user.get("id", "")
    if user_id:
        await manager.resume_active_games(user_id)

    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, manager.stop)

    try:
        await manager.run()
    finally:
        batching.stop()
        await client.close()
        logger.info("Agent shut down.")


def main():
    asyncio.run(run_agent())


if __name__ == "__main__":
    main()
