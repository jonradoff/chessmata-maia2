"""Async HTTP + WebSocket client for the Chessmata API."""

import logging
from typing import Any, Dict, List, Optional

import aiohttp

logger = logging.getLogger(__name__)

# After this many consecutive connection failures across all callers,
# the session is torn down and recreated.
_SESSION_FAILURE_THRESHOLD = 10


class APIError(Exception):
    """Chessmata API error."""

    def __init__(self, message: str, status_code: int = 0):
        super().__init__(message)
        self.status_code = status_code


class ChessmataClient:
    """Async client for the Chessmata REST API and WebSocket endpoints."""

    def __init__(
        self,
        server_url: str,
        api_key: str,
        client_software: str = "Chessmata-Maia2/1.0",
        connection_limit: int = 200,
    ):
        self.base_url = server_url.rstrip("/")
        self.api_key = api_key
        self.client_software = client_software
        self._connection_limit = connection_limit
        self._session: Optional[aiohttp.ClientSession] = None
        self._consecutive_failures = 0

    @property
    def ws_base_url(self) -> str:
        """Derive the WebSocket base URL from the HTTP base URL."""
        return self.base_url.replace("https://", "wss://").replace("http://", "ws://")

    def _make_session(self) -> aiohttp.ClientSession:
        connector = aiohttp.TCPConnector(limit=self._connection_limit)
        return aiohttp.ClientSession(
            connector=connector,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
                "User-Agent": self.client_software,
            },
            timeout=aiohttp.ClientTimeout(total=30),
        )

    async def start(self):
        """Create the aiohttp session."""
        self._session = self._make_session()

    async def close(self):
        """Close the aiohttp session."""
        if self._session:
            await self._session.close()
            self._session = None

    async def _recreate_session(self):
        """Tear down and recreate the HTTP session."""
        logger.warning("Recreating HTTP session after %d consecutive failures", self._consecutive_failures)
        old = self._session
        self._session = self._make_session()
        self._consecutive_failures = 0
        if old:
            try:
                await old.close()
            except Exception:
                pass

    # ── REST helpers ─────────────────────────────────────────

    async def _request(self, method: str, endpoint: str, data: Optional[Dict] = None) -> Dict[str, Any]:
        url = f"{self.base_url}/api{endpoint}"
        kwargs: Dict[str, Any] = {}
        if data is not None:
            kwargs["json"] = data
        try:
            async with self._session.request(method, url, **kwargs) as resp:
                body = await resp.text()
                if resp.status >= 400:
                    raise APIError(f"HTTP {resp.status}: {body}", resp.status)
                self._consecutive_failures = 0  # success resets counter
                if body:
                    return await resp.json(content_type=None)
                return {}
        except APIError:
            raise  # API errors are not connection failures
        except aiohttp.ClientError as e:
            self._consecutive_failures += 1
            if self._consecutive_failures >= _SESSION_FAILURE_THRESHOLD:
                await self._recreate_session()
            raise APIError(f"Connection error: {e}")

    # ── WebSocket helpers ────────────────────────────────────

    def ws_connect_matchmaking(self, connection_id: str):
        """Return an async context manager for a matchmaking WebSocket."""
        url = f"{self.ws_base_url}/ws/matchmaking/{connection_id}"
        return self._session.ws_connect(url, heartbeat=30)

    def ws_connect_game(self, session_id: str, player_id: str):
        """Return an async context manager for a game WebSocket."""
        url = f"{self.ws_base_url}/ws/games/{session_id}?playerId={player_id}&token={self.api_key}"
        return self._session.ws_connect(url, heartbeat=30)

    # ── Matchmaking REST ─────────────────────────────────────

    async def join_matchmaking(
        self,
        connection_id: str,
        display_name: str,
        agent_name: str,
        engine_name: str,
        is_ranked: bool = True,
        opponent_type: str = "either",
        time_controls: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        payload = {
            "connectionId": connection_id,
            "displayName": display_name,
            "agentName": agent_name,
            "engineName": engine_name,
            "isRanked": is_ranked,
            "opponentType": opponent_type,
            "clientSoftware": self.client_software,
            "timeControls": time_controls or [
                "unlimited", "casual", "standard", "quick", "blitz", "tournament"
            ],
        }
        return await self._request("POST", "/matchmaking/join", payload)

    async def get_queue_status(self, connection_id: str) -> Dict[str, Any]:
        return await self._request("GET", f"/matchmaking/status?connectionId={connection_id}")

    async def leave_matchmaking(self, connection_id: str) -> Dict[str, Any]:
        return await self._request("POST", f"/matchmaking/leave?connectionId={connection_id}")

    # ── Game REST ────────────────────────────────────────────

    async def get_game(self, session_id: str) -> Dict[str, Any]:
        return await self._request("GET", f"/games/{session_id}")

    async def make_move(
        self,
        session_id: str,
        player_id: str,
        from_sq: str,
        to_sq: str,
        promotion: Optional[str] = None,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "playerId": player_id,
            "from": from_sq,
            "to": to_sq,
        }
        if promotion:
            payload["promotion"] = promotion
        return await self._request("POST", f"/games/{session_id}/move", payload)

    async def resign_game(self, session_id: str, player_id: str) -> Dict[str, Any]:
        return await self._request("POST", f"/games/{session_id}/resign", {"playerId": player_id})

    async def get_current_user(self) -> Dict[str, Any]:
        return await self._request("GET", "/auth/me")

    async def list_active_games(self, limit: int = 50, inactive_mins: int = 1440) -> List[Dict[str, Any]]:
        """List active games on the server (filter client-side for ours)."""
        data = await self._request(
            "GET", f"/games/active?limit={limit}&inactiveMins={inactive_mins}"
        )
        if isinstance(data, list):
            return data
        return []
