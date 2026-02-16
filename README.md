# Chessmata Maia2 Agent

An autonomous chess agent for the [Chessmata](https://github.com/jonradoff/chessmata) platform, powered by the [Maia2](https://github.com/CSSLab/maia2) neural network.

Maia2 is a unified model that predicts human-like chess moves conditioned on player Elo. This agent runs eight variants simultaneously (400, 600, 800, 1000, 1200, 1500, 1800, 2100 Elo), each playing at a human-realistic skill level.

## About Chessmata

[Chessmata](https://github.com/jonradoff/chessmata) is an open-source chess platform designed for both human and AI players. It provides a matchmaking system, real-time game play over WebSocket, and an API for building autonomous chess agents. Any chess engine or neural network can be connected as an agent through the platform's REST + WebSocket API.

This project is a **reference implementation** of a Chessmata agent. While it uses Maia2 for move generation, the architecture is engine-agnostic. You can fork this repo and swap in any chess engine -- Stockfish, Leela Chess Zero, your own neural network, or any other move generator -- by replacing the inference layer in `agent/maia2_engine.py`. The rest of the agent (matchmaking, game management, WebSocket handling, reconnection logic) works with any engine.

## Dependencies

- **Python 3.10+**
- **PyTorch 2.0+** (CPU or CUDA)
- **[Maia2 engine](https://github.com/CSSLab/maia2)** -- cloned automatically by `run.sh`, or manually:
  ```bash
  git clone https://github.com/CSSLab/maia2.git maia2-engine
  ```
- Python packages listed in `requirements.txt`

The Maia2 model weights (~90 MB) are downloaded automatically on first run via `gdown`.

## Setup

```bash
# Clone this repo
git clone https://github.com/jonradoff/chessmata-maia2.git
cd chessmata-maia2

# Clone the Maia2 engine
git clone https://github.com/CSSLab/maia2.git maia2-engine

# Install Python dependencies
pip install -r requirements.txt

# Create your config (API key required)
cp config.example.prod.yaml config.yaml
# Edit config.yaml and set your chessmata API key
```

## Configuration

Copy one of the example configs and fill in your API key:

| File | Use case |
|------|----------|
| `config.example.prod.yaml` | Production (chessmata.metavert.io, 50 games) |
| `config.example.dev.yaml` | Development (localhost, 5 games, DEBUG logging) |

Key settings in `config.yaml`:

```yaml
chessmata:
  api_key: "cmk_YOUR_API_KEY_HERE"

agent:
  max_concurrent_games: 50       # global limit
  max_concurrent_per_variant: 0  # 0 = unlimited, or set per-variant cap

engine:
  device: "auto"  # "auto" detects GPU, falls back to CPU
```

## Running

```bash
# One-shot
python3 -m agent.main

# With auto-restart wrapper (recommended for production)
./run.sh
```

The `run.sh` wrapper will:
1. Clone `maia2-engine` if not present
2. Start the agent
3. Automatically restart on crash (5-second delay)

To stop: kill the wrapper process (PID is printed on startup).

## Deployment

The agent includes a Fly.io configuration for cloud deployment:

```bash
fly apps create chessmata-maia2 --org your-org
fly volumes create maia2_models --region ord --size 1
fly secrets set CHESSMATA_API_KEY="cmk_YOUR_KEY"
fly deploy
```

Model weights are cached on a persistent volume to avoid re-downloading on restarts. A health check endpoint runs on `:8080/health` with live stats (active games, W/L/D, inference latency).

## Architecture

- **Batched inference** -- Concurrent game requests are collected and run through the model in a single forward pass for high throughput
- **WebSocket-based** -- Real-time game updates via WebSocket; REST only for matchmaking joins and move submissions
- **Resilient** -- Automatic reconnection with exponential backoff on server restarts, session recreation on persistent connection failures
- **Game resumption** -- On restart, detects and resumes any in-progress games from the previous session
- **Observability** -- Health endpoint with JSON stats, periodic stats logging (W/L/D, inference p50/p95)

## Adapting to Other Engines

To use a different chess engine:

1. Fork this repo
2. Replace `Maia2Engine` in `agent/maia2_engine.py` with your engine's inference logic
3. Ensure your engine's `get_move()` returns `(move_probs: Dict[str, float], win_prob: float)` where moves are in UCI format (e.g., `e2e4`, `e7e8q`)
4. Update `BatchingEngine` if your engine handles batching differently, or remove it if batching isn't needed

The `GameManager`, `ChessmataClient`, and all matchmaking/WebSocket logic remain unchanged regardless of the engine.

## Maia2

This project depends on the Maia2 neural network by [CSSLab](https://github.com/CSSLab/maia2). Maia2 is a 23M-parameter model that predicts human chess moves conditioned on both player and opponent Elo ratings. See the [Maia2 paper](https://arxiv.org/abs/2409.20553) for details.

## License

MIT -- see [LICENSE](LICENSE).
