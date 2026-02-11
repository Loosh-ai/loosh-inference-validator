"""
Background task to periodically sync sybil detection records to Challenge API.

Runs asynchronously without blocking main processing.
"""

import asyncio
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
import httpx
from loguru import logger

from validator.db.operations import DatabaseManager
from validator.db.schema import SybilDetectionResult
from validator.config import get_validator_config
from validator.network.challenge_api_auth import merge_auth_headers


class SybilSyncTask:
    """
    Background task to periodically sync sybil detection records to Challenge API.
    
    Queries the local database for unsent records and sends them in batches.
    """
    
    def __init__(
        self,
        db_manager: DatabaseManager,
        validator_hotkey_ss58: str,
        sync_interval_seconds: float = 60.0,
        batch_size: int = 10
    ):
        """
        Initialize sybil sync task.
        
        Args:
            db_manager: Database manager for querying records
            validator_hotkey_ss58: Validator hotkey for identification
            sync_interval_seconds: How often to sync (default: 60 seconds)
            batch_size: Maximum records to send per batch (default: 10)
        """
        self.db_manager = db_manager
        self.validator_hotkey_ss58 = validator_hotkey_ss58
        self.sync_interval_seconds = sync_interval_seconds
        self.batch_size = batch_size
        self.config = get_validator_config()
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._http_client: Optional[httpx.AsyncClient] = None
        self._sent_record_ids: set = set()  # Track sent records to avoid duplicates
    
    async def start(self) -> None:
        """Start the background sync task."""
        if self._running:
            logger.warning("Sybil sync task already running")
            return
        
        self._running = True
        self._http_client = httpx.AsyncClient(
            timeout=30.0,
            limits=httpx.Limits(max_connections=10, max_keepalive_connections=5)
        )
        self._task = asyncio.create_task(self._sync_loop())
        logger.info(
            f"Sybil sync task started - interval: {self.sync_interval_seconds}s, "
            f"batch_size: {self.batch_size}"
        )
    
    async def stop(self) -> None:
        """Stop the background sync task."""
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
        
        logger.info("Sybil sync task stopped")
    
    async def _sync_loop(self) -> None:
        """Main sync loop - runs periodically."""
        while self._running:
            try:
                await self._sync_batch()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in sybil sync loop: {e}", exc_info=True)
            
            # Wait before next sync
            try:
                await asyncio.sleep(self.sync_interval_seconds)
            except asyncio.CancelledError:
                break
    
    async def _sync_batch(self) -> None:
        """Sync a batch of unsent records to Challenge API."""
        if not self._http_client:
            return
        
        # Get unsent records from database
        records = self._get_unsent_records()
        
        if not records:
            return
        
        logger.debug(f"Syncing {len(records)} sybil detection records to Challenge API")
        
        # Send records in batches
        for i in range(0, len(records), self.batch_size):
            batch = records[i:i + self.batch_size]
            await self._send_batch(batch)
    
    def _get_unsent_records(self) -> List[SybilDetectionResult]:
        """Get unsent sybil detection records from database."""
        try:
            session = self.db_manager.get_session()
            try:
                # Get records that haven't been sent yet
                # We'll track sent records by ID in memory (could be improved with a sent flag in DB)
                all_records = session.query(SybilDetectionResult).order_by(
                    SybilDetectionResult.created_at.asc()
                ).limit(100).all()  # Limit to prevent memory issues
                
                # Filter out already sent records
                unsent = [
                    record for record in all_records
                    if record.id not in self._sent_record_ids
                ]
                
                return unsent
            finally:
                session.close()
        except Exception as e:
            logger.error(f"Error getting unsent records: {e}", exc_info=True)
            return []
    
    async def _send_batch(self, records: List[SybilDetectionResult]) -> None:
        """Send a batch of records to Challenge API in bulk."""
        if not self._http_client or not records:
            return
        
        challenge_api_url = self.config.challenge_api_url
        api_key = self.config.challenge_api_key
        endpoint = f"{challenge_api_url.rstrip('/')}/analytics/sybil-detection/bulk"
        
        try:
            # Convert all records to API format
            records_data = []
            for record in records:
                record_data = {
                    "validator_hotkey_ss58": self.validator_hotkey_ss58,
                    "challenge_id": str(record.challenge_id) if record.challenge_id else None,
                    "suspicious_pairs_count": record.suspicious_pairs_count,
                    "suspicious_groups_count": record.suspicious_groups_count,
                    "suspicious_pairs": record.suspicious_pairs if record.suspicious_pairs else None,
                    "suspicious_groups": record.suspicious_groups if record.suspicious_groups else None,
                    "analysis_report": record.analysis_report,
                    "high_similarity_threshold": record.high_similarity_threshold,
                    "very_high_similarity_threshold": record.very_high_similarity_threshold
                }
                records_data.append(record_data)
            
            # Send bulk request
            bulk_request = {"records": records_data}
            body_bytes = json.dumps(bulk_request).encode()
            headers = merge_auth_headers(
                {"Content-Type": "application/json"},
                body=body_bytes,
                api_key=api_key,
            )
            response = await self._http_client.post(
                endpoint,
                content=body_bytes,
                headers=headers,
            )
            
            if response.status_code == 201:
                response_data = response.json()
                inserted_count = response_data.get("inserted_count", len(records))
                
                # Delete all successfully submitted records
                deleted_count = 0
                for record in records:
                    deleted = self.db_manager.delete_sybil_detection_result(record.id)
                    if deleted:
                        deleted_count += 1
                        self._sent_record_ids.add(record.id)
                
                logger.info(
                    f"Successfully synced {inserted_count} sybil detection records in bulk "
                    f"(deleted {deleted_count} from local database)"
                )
            else:
                logger.warning(
                    f"Failed to sync sybil detection records in bulk: "
                    f"HTTP {response.status_code} - {response.text[:200]}"
                )
                # Don't delete records if bulk submission failed
                
        except httpx.RequestError as e:
            logger.warning(
                f"Network error syncing sybil detection records in bulk: {e}"
            )
        except Exception as e:
            logger.error(
                f"Error syncing sybil detection records in bulk: {e}",
                exc_info=True
            )
