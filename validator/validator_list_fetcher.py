"""
Fetches and caches the list of validators from Challenge API.

Periodically queries the Challenge API to get the list of registered validators
and filters them out when selecting miners for challenges.
"""

import asyncio
from typing import Set, Optional
from datetime import datetime, timedelta
import httpx
from loguru import logger

from validator.config import get_validator_config


class ValidatorListFetcher:
    """
    Fetches and caches validator hotkeys from Challenge API.
    
    Periodically queries the Challenge API to get registered validators
    and maintains a cache of validator hotkeys to filter out.
    """
    
    def __init__(
        self,
        challenge_api_url: str,
        challenge_api_key: str,
        refresh_interval_seconds: float = 300.0  # Refresh every 5 minutes
    ):
        """
        Initialize validator list fetcher.
        
        Args:
            challenge_api_url: Base URL of Challenge API
            challenge_api_key: API key for Challenge API
            refresh_interval_seconds: How often to refresh the validator list (default: 5 minutes)
        """
        self.challenge_api_url = challenge_api_url.rstrip('/')
        self.challenge_api_key = challenge_api_key
        self.refresh_interval_seconds = refresh_interval_seconds
        
        self._validator_hotkeys: Set[str] = set()
        self._last_refresh: Optional[datetime] = None
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._http_client: Optional[httpx.AsyncClient] = None
        self._refresh_lock = asyncio.Lock()
    
    async def start(self) -> None:
        """Start the background refresh task."""
        if self._running:
            logger.warning("Validator list fetcher already running")
            return
        
        self._running = True
        self._http_client = httpx.AsyncClient(
            timeout=30.0,
            limits=httpx.Limits(max_connections=5, max_keepalive_connections=2)
        )
        
        # Do initial fetch
        await self._fetch_validators()
        
        # Start background refresh task
        self._task = asyncio.create_task(self._refresh_loop())
        logger.info(
            f"Validator list fetcher started - refresh interval: {self.refresh_interval_seconds}s, "
            f"initial validators: {len(self._validator_hotkeys)}"
        )
    
    async def stop(self) -> None:
        """Stop the background refresh task."""
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
        
        logger.info("Validator list fetcher stopped")
    
    async def _refresh_loop(self) -> None:
        """Background loop to periodically refresh validator list."""
        while self._running:
            try:
                await asyncio.sleep(self.refresh_interval_seconds)
                if self._running:
                    await self._fetch_validators()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in validator list refresh loop: {e}", exc_info=True)
    
    async def _fetch_validators(self) -> None:
        """Fetch validators from Challenge API."""
        if not self._http_client:
            return
        
        async with self._refresh_lock:
            try:
                url = f"{self.challenge_api_url}/validators"
                headers = {
                    "Authorization": f"Bearer {self.challenge_api_key}",
                    "Content-Type": "application/json"
                }
                
                response = await self._http_client.get(url, headers=headers)
                
                if response.status_code == 200:
                    validators = response.json()
                    new_hotkeys = {v.get('hotkey_ss58') for v in validators if v.get('hotkey_ss58')}
                    
                    old_count = len(self._validator_hotkeys)
                    self._validator_hotkeys = new_hotkeys
                    self._last_refresh = datetime.utcnow()
                    
                    logger.info(
                        f"Refreshed validator list: {len(self._validator_hotkeys)} validators "
                        f"(was {old_count}, last refresh: {self._last_refresh.isoformat()})"
                    )
                elif response.status_code == 503:
                    logger.warning(
                        "Challenge API database not available - validator list not refreshed. "
                        "Will retry on next refresh cycle."
                    )
                else:
                    logger.warning(
                        f"Failed to fetch validators: HTTP {response.status_code} - {response.text[:200]}"
                    )
                    
            except httpx.RequestError as e:
                logger.warning(f"Network error fetching validators: {e}")
            except Exception as e:
                logger.error(f"Error fetching validators: {e}", exc_info=True)
    
    def is_validator(self, hotkey_ss58: str) -> bool:
        """
        Check if a hotkey belongs to a validator.
        
        Args:
            hotkey_ss58: SS58 address to check
        
        Returns:
            True if the hotkey is a registered validator, False otherwise
        """
        return hotkey_ss58 in self._validator_hotkeys
    
    def get_validator_count(self) -> int:
        """Get the number of validators in the cache."""
        return len(self._validator_hotkeys)
    
    def get_last_refresh_time(self) -> Optional[datetime]:
        """Get the timestamp of the last successful refresh."""
        return self._last_refresh
    
    async def refresh_now(self) -> None:
        """Manually trigger a refresh of the validator list."""
        await self._fetch_validators()
