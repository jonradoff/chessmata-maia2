"""Wrapper around the Maia2 neural network for chess move prediction.

Supports both single-position and batched inference.  The BatchingEngine
collects pending requests from concurrent asyncio tasks and runs them
through the model in a single forward pass for much higher throughput.
"""

import asyncio
import logging
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Add the maia2-engine directory to the path so we can import from the cloned repo
_ENGINE_DIR = os.path.join(os.path.dirname(__file__), "..", "maia2-engine")
if os.path.isdir(_ENGINE_DIR) and _ENGINE_DIR not in sys.path:
    sys.path.insert(0, os.path.abspath(_ENGINE_DIR))


@dataclass
class _InferenceRequest:
    """A single position waiting to be batched."""
    fen: str
    elo_self: int
    elo_oppo: int
    future: asyncio.Future = field(default_factory=lambda: None)  # type: ignore[assignment]


class Maia2Engine:
    """Loads the Maia2 model and provides single-position move prediction.

    For batched inference, use BatchingEngine which wraps this class.
    """

    def __init__(self, model_type: str = "rapid", device: str = "auto"):
        self.model_type = model_type
        self.device = self._resolve_device(device)
        self._model = None
        self._prepared = None
        self._torch_device: Optional[str] = None  # actual torch device string

    @staticmethod
    def _resolve_device(device: str) -> str:
        """Resolve 'auto' to the best available device, with diagnostics."""
        if device == "auto":
            try:
                import torch

                if torch.cuda.is_available():
                    gpu_name = torch.cuda.get_device_name(0)
                    vram_mb = torch.cuda.get_device_properties(0).total_mem / (1024 * 1024)
                    logger.info("GPU detected: %s (%.0f MB VRAM)", gpu_name, vram_mb)
                    return "gpu"

                if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    logger.info("Apple MPS (Metal) GPU detected")
                    # Maia2's from_pretrained only recognises "gpu" and "cpu";
                    # MPS isn't wired up in the upstream code, so fall back.
                    logger.info("MPS not supported by Maia2 upstream — using CPU")
                    return "cpu"

                logger.info("No GPU detected — using CPU")
                return "cpu"
            except ImportError:
                return "cpu"
        if device == "cuda":
            return "gpu"
        return device

    def load(self):
        """Load the model and prepare inference lookup tables."""
        from maia2 import model as maia2_model_mod
        from maia2 import inference as maia2_inference

        logger.info("Loading Maia2 model (type=%s, device=%s)...", self.model_type, self.device)
        self._model = maia2_model_mod.from_pretrained(type=self.model_type, device=self.device)
        self._prepared = maia2_inference.prepare()

        import torch
        self._torch_device = str(next(self._model.parameters()).device)
        logger.info("Maia2 model loaded on device: %s", self._torch_device)

    def warmup(self):
        """Run a throwaway inference to warm up the model."""
        logger.info("Warming up Maia2 model...")
        start_fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
        self.get_move(start_fen, elo_self=1500, elo_oppo=1500)
        logger.info("Maia2 model warmed up.")

    def get_move(
        self, fen: str, elo_self: int, elo_oppo: int
    ) -> Tuple[Dict[str, float], float]:
        """Single-position inference (blocking)."""
        from maia2 import inference as maia2_inference

        move_probs, win_prob = maia2_inference.inference_each(
            self._model, self._prepared, fen, elo_self, elo_oppo,
        )
        return move_probs, win_prob

    def get_moves_batch(
        self, requests: List[Tuple[str, int, int]]
    ) -> List[Tuple[Dict[str, float], float]]:
        """Batched inference for multiple positions (blocking).

        Args:
            requests: list of (fen, elo_self, elo_oppo) tuples.

        Returns:
            list of (move_probs, win_prob) in the same order.
        """
        import torch
        from maia2.inference import preprocessing
        from maia2.utils import mirror_move

        all_moves_dict, elo_dict, all_moves_dict_reversed = self._prepared
        device = self._torch_device
        n = len(requests)

        # Pre-process all positions
        board_inputs = []
        elos_self = []
        elos_oppo = []
        legal_moves_list = []
        fens = []

        for fen, elo_s, elo_o in requests:
            board_input, es, eo, legal_moves = preprocessing(
                fen, elo_s, elo_o, elo_dict, all_moves_dict
            )
            board_inputs.append(board_input)
            elos_self.append(es)
            elos_oppo.append(eo)
            legal_moves_list.append(legal_moves)
            fens.append(fen)

        # Stack into batch tensors
        boards_t = torch.stack(board_inputs).to(device)
        elos_self_t = torch.tensor(elos_self, dtype=torch.long).to(device)
        elos_oppo_t = torch.tensor(elos_oppo, dtype=torch.long).to(device)
        legal_moves_t = torch.stack(legal_moves_list).to(device)

        # Forward pass
        self._model.eval()
        with torch.no_grad():
            logits_maia, _, logits_value = self._model(boards_t, elos_self_t, elos_oppo_t)

        logits_maia_legal = logits_maia * legal_moves_t
        probs = logits_maia_legal.softmax(dim=-1).cpu().tolist()
        values = (logits_value / 2 + 0.5).clamp(0, 1).cpu().tolist()

        # Post-process each position
        results: List[Tuple[Dict[str, float], float]] = []
        for i in range(n):
            fen = fens[i]
            black_flag = fen.split(" ")[1] == "b"

            win_prob = round(1 - values[i] if black_flag else values[i], 4)

            move_probs: Dict[str, float] = {}
            legal_indices = legal_moves_t[i].nonzero().flatten().cpu().numpy().tolist()
            for idx in legal_indices:
                move = all_moves_dict_reversed[idx]
                if black_flag:
                    move = mirror_move(move)
                move_probs[move] = round(probs[i][idx], 4)

            move_probs = dict(sorted(move_probs.items(), key=lambda item: item[1], reverse=True))
            results.append((move_probs, win_prob))

        return results


class BatchingEngine:
    """Collects inference requests and runs them in batches.

    Asyncio tasks call `get_move()` which returns a future.  A background
    thread collects pending requests every `max_wait_ms` milliseconds (or
    when `max_batch_size` is reached) and runs a single batched forward pass.
    """

    def __init__(
        self,
        engine: Maia2Engine,
        max_batch_size: int = 32,
        max_wait_ms: float = 50,
        inference_timeout: float = 30.0,
    ):
        self.engine = engine
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms
        self.inference_timeout = inference_timeout

        self._queue: List[_InferenceRequest] = []
        self._lock = threading.Lock()
        self._event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._running = False

    def start(self):
        """Start the background batch-processing thread."""
        self._running = True
        self._thread = threading.Thread(target=self._batch_loop, daemon=True, name="maia2-batch")
        self._thread.start()
        logger.info(
            "Batching engine started (max_batch=%d, max_wait=%dms, timeout=%.0fs)",
            self.max_batch_size, self.max_wait_ms, self.inference_timeout,
        )

    def stop(self):
        """Signal the batch thread to stop."""
        self._running = False
        self._event.set()
        if self._thread:
            self._thread.join(timeout=5)

    async def get_move(
        self, fen: str, elo_self: int, elo_oppo: int
    ) -> Tuple[Dict[str, float], float]:
        """Submit a position for inference. Returns when the batch completes."""
        loop = asyncio.get_event_loop()
        future = loop.create_future()
        req = _InferenceRequest(fen=fen, elo_self=elo_self, elo_oppo=elo_oppo, future=future)

        with self._lock:
            self._queue.append(req)
            if len(self._queue) >= self.max_batch_size:
                self._event.set()  # wake up the batch thread immediately

        return await asyncio.wait_for(future, timeout=self.inference_timeout)

    # ── Background thread ─────────────────────────────────

    def _batch_loop(self):
        """Runs in a daemon thread: drains the queue and runs batched inference."""
        while self._running:
            # Wait until there's work or max_wait expires
            self._event.wait(timeout=self.max_wait_ms / 1000.0)
            self._event.clear()

            # Drain the queue
            with self._lock:
                batch = self._queue[:self.max_batch_size]
                self._queue = self._queue[self.max_batch_size:]

            if not batch:
                continue

            # Run batched inference
            try:
                inputs = [(r.fen, r.elo_self, r.elo_oppo) for r in batch]
                results = self.engine.get_moves_batch(inputs)

                for req, result in zip(batch, results):
                    if not req.future.done():
                        req.future.get_loop().call_soon_threadsafe(req.future.set_result, result)

            except Exception as exc:
                logger.exception("Batched inference failed for %d positions", len(batch))
                for req in batch:
                    if not req.future.done():
                        req.future.get_loop().call_soon_threadsafe(req.future.set_exception, exc)

            # If there are more queued items, loop immediately
            with self._lock:
                if self._queue:
                    self._event.set()
