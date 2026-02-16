"""Manages concurrent matchmaking and game play via WebSocket."""

import asyncio
import json
import logging
import uuid
from typing import Dict, List, Optional

import aiohttp

from .chessmata_client import ChessmataClient, APIError
from .config import AgentVariant, Config
from .maia2_engine import BatchingEngine

logger = logging.getLogger(__name__)

ENGINE_NAME = "Maia2"

# If no WS message arrives for this long during an active game, re-fetch
# game state via REST to check for missed events.
WS_IDLE_TIMEOUT = 300  # 5 minutes


class GameManager:
    """Orchestrates matchmaking and concurrent game play over WebSocket.

    Architecture:
      - One long-lived asyncio task per variant runs _matchmaking_loop().
        It joins the queue (REST), then listens on the matchmaking WebSocket
        for a match_found event.  When matched it spawns a _play_game() task
        and immediately loops back to queue the next match for that variant.
      - Each _play_game() task opens a game WebSocket, makes moves via REST
        when it is our turn, and exits when the game ends.
    """

    def __init__(self, config: Config, batching_engine: BatchingEngine, client: ChessmataClient):
        self.config = config
        self.batching_engine = batching_engine
        self.client = client

        self._game_tasks: Dict[str, asyncio.Task] = {}    # session_id -> task
        self._variant_game_counts: Dict[str, int] = {}     # variant_name -> active count
        self._mm_tasks: List[asyncio.Task] = []
        self._running = False

    @property
    def total_active(self) -> int:
        return len(self._game_tasks)

    def _variant_active(self, variant_name: str) -> int:
        return self._variant_game_counts.get(variant_name, 0)

    def _can_start_game(self, variant: AgentVariant) -> bool:
        """Check both global and per-variant capacity."""
        if self.total_active >= self.config.agent.max_concurrent_games:
            return False
        per_variant = self.config.agent.max_concurrent_per_variant
        if per_variant > 0 and self._variant_active(variant.name) >= per_variant:
            return False
        return True

    def _register_game(self, session_id: str, variant_name: str, task: asyncio.Task):
        """Track a game task and its variant."""
        self._game_tasks[session_id] = task
        self._variant_game_counts[variant_name] = self._variant_game_counts.get(variant_name, 0) + 1

        def _on_done(_t, sid=session_id, vname=variant_name):
            self._game_tasks.pop(sid, None)
            self._variant_game_counts[vname] = max(0, self._variant_game_counts.get(vname, 0) - 1)

        task.add_done_callback(_on_done)

    # ── Public ───────────────────────────────────────────────

    async def resume_active_games(self, user_id: str):
        """Find and resume any in-progress games from a previous session."""
        try:
            active_games = await self.client.list_active_games()
        except APIError as e:
            logger.warning("[Resume] Could not list active games: %s", e)
            return

        # Build a lookup of variant name -> AgentVariant
        variant_map = {v.name: v for v in self.config.agent.variants}
        resumed = 0

        for game_data in active_games:
            players = game_data.get("players", [])

            # Find our player entry by userId
            our_player = None
            opponent = None
            for p in players:
                if p.get("userId") == user_id:
                    our_player = p
                else:
                    opponent = p

            if not our_player:
                continue  # not our game

            session_id = game_data.get("sessionId", "")
            agent_name = our_player.get("agentName", "")
            variant = variant_map.get(agent_name)

            if not variant:
                logger.warning(
                    "[Resume] Unknown agent '%s' in game %s, skipping",
                    agent_name, session_id[:8],
                )
                continue

            player_id = our_player["id"]
            our_color = our_player["color"]
            elo_oppo = (opponent.get("eloRating") or 1500) if opponent else 1500

            logger.info(
                "[Resume] Resuming %s | session=%s color=%s elo_oppo=%d",
                variant.name, session_id[:8], our_color, elo_oppo,
            )

            task = asyncio.create_task(
                self._play_game(session_id, player_id, variant, our_color, elo_oppo)
            )
            self._register_game(session_id, variant.name, task)
            resumed += 1

        if resumed:
            logger.info("[Resume] Resumed %d active game(s)", resumed)
        else:
            logger.info("[Resume] No active games to resume")

    async def run(self):
        self._running = True
        logger.info(
            "Starting game manager [WebSocket mode] (max_concurrent=%d, per_variant=%s, variants=%d, already_active=%d)",
            self.config.agent.max_concurrent_games,
            self.config.agent.max_concurrent_per_variant or "unlimited",
            len(self.config.agent.variants),
            self.total_active,
        )

        for variant in self.config.agent.variants:
            task = asyncio.create_task(self._matchmaking_loop(variant))
            self._mm_tasks.append(task)

        # Block until all matchmaking loops end (they run forever unless stopped)
        await asyncio.gather(*self._mm_tasks, return_exceptions=True)
        logger.info("Game manager stopped. Active games: %d", self.total_active)

    def stop(self):
        self._running = False
        for t in self._mm_tasks:
            t.cancel()
        for t in list(self._game_tasks.values()):
            t.cancel()

    # ── Matchmaking loop (one per variant) ───────────────────

    async def _matchmaking_loop(self, variant: AgentVariant):
        """Continuously find matches for *variant* and spawn game tasks."""
        conn_id: Optional[str] = None

        while self._running:
            try:
                # Back off if at capacity
                while not self._can_start_game(variant):
                    if not self._running:
                        return
                    await asyncio.sleep(5)

                conn_id = str(uuid.uuid4())

                # 1. Join the queue (REST)
                result = await self.client.join_matchmaking(
                    connection_id=conn_id,
                    display_name=variant.name,
                    agent_name=variant.name,
                    engine_name=ENGINE_NAME,
                    is_ranked=True,
                    opponent_type="either",
                )
                logger.info("[MM] Queued %s (conn=%s)", variant.name, conn_id[:8])

                # Immediate match?
                if result.get("match"):
                    sid = result["match"].get("sessionId")
                    if sid:
                        logger.info("[MM] %s matched immediately! session=%s", variant.name, sid)
                        await self._start_game(sid, conn_id, variant)
                        continue

                # 2. Wait for match via WebSocket
                session_id = await self._wait_for_match_ws(conn_id, variant)
                if session_id:
                    await self._start_game(session_id, conn_id, variant)
                else:
                    # WS closed without a match (timeout / server restart) → retry
                    logger.info("[MM] %s WS ended without match, re-queuing", variant.name)

            except asyncio.CancelledError:
                # Graceful shutdown: leave the queue
                if conn_id:
                    try:
                        await self.client.leave_matchmaking(conn_id)
                    except Exception:
                        pass
                return
            except Exception:
                logger.exception("[MM] Error in %s loop, retrying in 5s", variant.name)
                await asyncio.sleep(5)

    async def _wait_for_match_ws(self, conn_id: str, variant: AgentVariant) -> Optional[str]:
        """Connect to the matchmaking WebSocket and block until match_found."""
        try:
            async with self.client.ws_connect_matchmaking(conn_id) as ws:
                logger.debug("[MM] WS connected for %s (conn=%s)", variant.name, conn_id[:8])
                async for msg in ws:
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        data = json.loads(msg.data)
                        if data.get("type") == "match_found" or "sessionId" in data:
                            session_id = data.get("sessionId") or data.get("matchedSessionId")
                            logger.info("[MM] %s matched via WS! session=%s", variant.name, session_id)
                            return session_id
                    elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                        break
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning("[MM] WS error for %s, will retry", variant.name, exc_info=True)
        return None

    # ── Game lifecycle ───────────────────────────────────────

    async def _start_game(self, session_id: str, conn_id: str, variant: AgentVariant):
        """Fetch initial game state, then spawn a _play_game task."""
        try:
            game_data = await self.client.get_game(session_id)
        except APIError as e:
            logger.error("[Game] Failed to fetch %s: %s", session_id, e)
            return

        our_color: Optional[str] = None
        elo_oppo = 1500

        for p in game_data.get("players", []):
            if p["id"] == conn_id:
                our_color = p["color"]
            else:
                elo_oppo = p.get("eloRating") or 1500

        if not our_color:
            logger.error("[Game] Cannot find our player in %s", session_id)
            return

        logger.info(
            "[Game] Started %s | session=%s color=%s elo_self=%d elo_oppo=%d",
            variant.name, session_id, our_color, variant.elo, elo_oppo,
        )

        task = asyncio.create_task(
            self._play_game(session_id, conn_id, variant, our_color, elo_oppo)
        )
        self._register_game(session_id, variant.name, task)

    # ── Playing a single game over WebSocket ─────────────────

    async def _play_game(
        self,
        session_id: str,
        player_id: str,
        variant: AgentVariant,
        our_color: str,
        elo_oppo: int,
    ):
        """Play one game to completion, reconnecting on WS/server drops.

        The outer loop handles reconnection: if the WebSocket dies (server
        restart, network blip, etc.) we back off, re-check the game via
        REST, and reconnect.  The game only exits this method when it is
        complete or the manager is shutting down.
        """
        move_count = 0
        backoff = 3  # seconds, doubles on consecutive failures, caps at 30

        while self._running:
            try:
                async with self.client.ws_connect_game(session_id, player_id) as ws:
                    backoff = 3  # reset on successful connect

                    # Fetch state *after* WS is open so we don't miss events
                    game_data = await self._get_game_retry(session_id)

                    if game_data.get("status") == "complete":
                        self._log_result(variant, session_id, our_color, game_data, move_count)
                        return

                    # If it is already our turn, move immediately
                    if game_data.get("currentTurn") == our_color:
                        move_count = await self._make_move(
                            session_id, player_id, game_data["boardState"],
                            variant, elo_oppo, move_count, our_color=our_color,
                        )

                    # Listen for server-pushed events with idle timeout
                    async for msg in self._ws_iter_with_timeout(ws, WS_IDLE_TIMEOUT):
                        if msg is None:
                            # Idle timeout — re-check game state via REST
                            game_data = await self._get_game_retry(session_id)
                            if game_data.get("status") == "complete":
                                self._log_result(variant, session_id, our_color, game_data, move_count)
                                return
                            if (
                                game_data.get("currentTurn") == our_color
                                and game_data.get("status") == "active"
                            ):
                                fen = game_data.get("boardState", "")
                                if fen:
                                    move_count = await self._make_move(
                                        session_id, player_id, fen,
                                        variant, elo_oppo, move_count, our_color=our_color,
                                    )
                            continue

                        if not self._running:
                            return

                        if msg.type == aiohttp.WSMsgType.TEXT:
                            data = json.loads(msg.data)
                            msg_type = data.get("type", "")

                            if msg_type == "game_over":
                                self._log_result(variant, session_id, our_color, data, move_count)
                                return

                            if msg_type in ("move", "game_update", "player_joined"):
                                game = data.get("game", {})
                                if game.get("status") == "complete":
                                    full = await self._get_game_retry(session_id)
                                    self._log_result(variant, session_id, our_color, full, move_count)
                                    return
                                if (
                                    game.get("currentTurn") == our_color
                                    and game.get("status") == "active"
                                ):
                                    fen = game.get("boardState", "")
                                    if fen:
                                        move_count = await self._make_move(
                                            session_id, player_id, fen,
                                            variant, elo_oppo, move_count, our_color=our_color,
                                        )

                        elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                            break  # fall through to reconnection logic

                # WS closed — check game status before reconnecting
                if not self._running:
                    return

                final = await self._get_game_retry(session_id)
                if final.get("status") == "complete":
                    self._log_result(variant, session_id, our_color, final, move_count)
                    return

                logger.info(
                    "[Game] WS lost for %s, game still active — reconnecting in %ds",
                    session_id[:8], backoff,
                )
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30)

            except asyncio.CancelledError:
                return
            except Exception:
                if not self._running:
                    return
                logger.warning(
                    "[Game] Connection error for %s, retrying in %ds",
                    session_id[:8], backoff, exc_info=True,
                )
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30)

    @staticmethod
    async def _ws_iter_with_timeout(ws, timeout: float):
        """Yield WS messages, yielding None on idle timeout instead of blocking forever."""
        while True:
            try:
                msg = await asyncio.wait_for(ws.receive(), timeout=timeout)
                if msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR, aiohttp.WSMsgType.CLOSING):
                    yield msg
                    return
                yield msg
            except asyncio.TimeoutError:
                yield None

    async def _get_game_retry(self, session_id: str, max_attempts: int = 5) -> dict:
        """Fetch game state with retries (survives brief server outages)."""
        delay = 2
        for attempt in range(max_attempts):
            try:
                return await self.client.get_game(session_id)
            except Exception:
                if attempt == max_attempts - 1:
                    raise
                logger.debug("[Game] REST fetch failed for %s, retry %d/%d in %ds",
                             session_id[:8], attempt + 1, max_attempts, delay)
                await asyncio.sleep(delay)
                delay = min(delay * 2, 15)
        return {}  # unreachable

    # ── Inference + move submission ──────────────────────────

    async def _make_move(
        self,
        session_id: str,
        player_id: str,
        fen: str,
        variant: AgentVariant,
        elo_oppo: int,
        move_count: int,
        our_color: str = "",
        _is_retry: bool = False,
    ) -> int:
        """Run Maia2 via the batching engine, submit the move, return updated move_count."""
        if not fen:
            return move_count

        try:
            move_probs, win_prob = await self.batching_engine.get_move(
                fen, variant.elo, elo_oppo,
            )
        except asyncio.TimeoutError:
            logger.error("[Engine] Inference timed out for %s", session_id[:8])
            return move_count
        except Exception:
            logger.exception("[Engine] Inference failed for %s", session_id[:8])
            return move_count

        best_move = next(iter(move_probs))
        from_sq = best_move[:2]
        to_sq = best_move[2:4]
        promotion = best_move[4] if len(best_move) > 4 else None

        top3 = list(move_probs.items())[:3]
        top3_str = ", ".join(f"{m}={p:.3f}" for m, p in top3)

        try:
            await self.client.make_move(session_id, player_id, from_sq, to_sq, promotion)
            move_count += 1
            logger.info(
                "[Move] %s game=%s move=%s (win=%.2f) top=[%s] #%d",
                variant.name, session_id[:8], best_move,
                win_prob, top3_str, move_count,
            )
        except APIError as e:
            if not _is_retry and e.status_code in (400, 409, 422):
                # Move rejected (stale state, not our turn, illegal move) — re-fetch and retry once
                logger.warning("[Move] Rejected for %s (%s), re-fetching state", session_id[:8], e)
                try:
                    fresh = await self.client.get_game(session_id)
                    fresh_fen = fresh.get("boardState", "")
                    if (
                        fresh.get("status") == "active"
                        and fresh.get("currentTurn") == our_color
                        and fresh_fen
                        and fresh_fen != fen  # state actually changed
                    ):
                        return await self._make_move(
                            session_id, player_id, fresh_fen,
                            variant, elo_oppo, move_count,
                            our_color=our_color, _is_retry=True,
                        )
                except Exception:
                    logger.exception("[Move] Retry also failed for %s", session_id[:8])
            else:
                logger.error("[Move] Failed for %s: %s", session_id[:8], e)

        return move_count

    # ── Result logging ───────────────────────────────────────

    def _log_result(self, variant: AgentVariant, session_id: str, our_color: str, data: dict, move_count: int):
        winner = data.get("winner", "")
        reason = data.get("winReason", "")
        elo_changes = data.get("eloChanges") or {}
        our_change = elo_changes.get(f"{our_color}Change", 0)
        our_new = elo_changes.get(f"{our_color}NewElo", "?")

        if winner == our_color:
            result_str = "WIN"
        elif winner:
            result_str = "LOSS"
        else:
            result_str = "DRAW"

        logger.info(
            "[GameOver] %s game=%s result=%s reason=%s elo_change=%+d new_elo=%s moves=%d",
            variant.name, session_id[:8], result_str, reason,
            our_change, our_new, move_count,
        )
