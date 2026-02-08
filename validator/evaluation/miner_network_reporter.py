"""
Background task to periodically report miner network observations to the Challenge API.

Reads the current metagraph (IP, port, coldkey, ASN) for every serving miner
and POSTs it to ``POST /analytics/miner-network/bulk``.  The Challenge API uses
this data to compute IP-cluster, coldkey-concentration, and ASN-cluster signals
for the composite sybil score.

Runs asynchronously without blocking main processing.
"""

import asyncio
from typing import Callable, List, Optional

import httpx
from fiber.chain.models import Node
from loguru import logger

from validator.config import get_validator_config
from validator.internal_config import INTERNAL_CONFIG


class MinerNetworkReporter:
    """
    Periodically sends miner network observations to the Challenge API.

    Uses a user-supplied ``get_nodes`` callback so it can read the latest
    metagraph snapshot without importing chain-fetching logic itself.
    """

    def __init__(
        self,
        validator_hotkey_ss58: str,
        get_nodes: Callable[[], List[Node]],
        report_interval_seconds: float = 300.0,
    ):
        """
        Args:
            validator_hotkey_ss58: This validator's SS58 hotkey.
            get_nodes: Callable that returns the current list of ``Node``
                       objects from the metagraph (e.g. from the availability
                       worker or a chain query).
            report_interval_seconds: Interval between reports (default 5 min).
                Aligned with metagraph refresh so observations stay fresh.
        """
        self.validator_hotkey_ss58 = validator_hotkey_ss58
        self._get_nodes = get_nodes
        self.report_interval_seconds = report_interval_seconds
        self._config = get_validator_config()
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._http_client: Optional[httpx.AsyncClient] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start the background reporting task."""
        if self._running:
            logger.warning("MinerNetworkReporter already running")
            return

        self._running = True
        self._http_client = httpx.AsyncClient(
            timeout=30.0,
            limits=httpx.Limits(max_connections=5, max_keepalive_connections=2),
        )
        self._task = asyncio.create_task(self._report_loop())
        logger.info(
            f"MinerNetworkReporter started — interval: {self.report_interval_seconds}s"
        )

    async def stop(self) -> None:
        """Stop the background reporting task."""
        self._running = False

        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None

        logger.info("MinerNetworkReporter stopped")

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    async def _report_loop(self) -> None:
        """Main loop — waits, then sends a report, repeat."""
        # Small initial delay so the first metagraph fetch has time to complete
        await asyncio.sleep(30.0)

        while self._running:
            try:
                await self._send_report()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in MinerNetworkReporter loop: {e}", exc_info=True)

            try:
                await asyncio.sleep(self.report_interval_seconds)
            except asyncio.CancelledError:
                break

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    async def _send_report(self) -> None:
        """Build observation list from metagraph and POST to Challenge API."""
        if not self._http_client:
            return

        nodes: List[Node] = self._get_nodes()
        if not nodes:
            logger.debug("[miner-network] No nodes to report")
            return

        # Build observations list — one entry per node.
        # We exclude our own hotkey (we are a validator, not a miner).
        observations = []
        for node in nodes:
            if node.hotkey == self.validator_hotkey_ss58:
                continue
            # Skip nodes with no advertised endpoint
            if not node.ip or node.ip in ("0", "0.0.0.0") or node.port == 0:
                continue

            # Fields aligned with MinerNetworkObservation Pydantic model in
            # loosh-challenge-api/app/models.py
            observations.append(
                {
                    "miner_hotkey": node.hotkey,
                    "ip_address": str(node.ip),
                    "miner_uid": node.node_id,        # informational / transient
                    "coldkey": node.coldkey or "",
                    "port": node.port,
                    # ASN is not available from the chain — leave null.
                    # A future enhancement could do an external ASN lookup.
                    "asn": None,
                    "asn_name": None,
                }
            )

        if not observations:
            logger.debug("[miner-network] All nodes filtered — nothing to report")
            return

        url = (
            f"{self._config.challenge_api_url.rstrip('/')}"
            "/analytics/miner-network/bulk"
        )
        # Payload must conform to MinerNetworkBulkReport schema
        payload = {
            "validator_hotkey": self.validator_hotkey_ss58,
            "observations": observations,
        }

        try:
            resp = await self._http_client.post(
                url,
                json=payload,
                headers={
                    "X-API-Key": self._config.challenge_api_key,
                    "Content-Type": "application/json",
                },
            )
            if resp.status_code == 201:
                data = resp.json()
                logger.info(
                    f"[miner-network] Reported {len(observations)} observations "
                    f"(upserted: {data.get('upserted', '?')})"
                )
            else:
                logger.warning(
                    f"[miner-network] Report failed: HTTP {resp.status_code} — "
                    f"{resp.text[:200]}"
                )
        except httpx.RequestError as e:
            logger.warning(f"[miner-network] Network error reporting observations: {e}")
        except Exception as e:
            logger.error(
                f"[miner-network] Unexpected error reporting observations: {e}",
                exc_info=True,
            )
