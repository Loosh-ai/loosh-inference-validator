"""
Background task to periodically sync sybil detection records to Challenge API.

Runs asynchronously without blocking main processing.

Uses delete-before-send to prevent restart duplication: records are removed
from the local DB before the API call.  On failure they are re-inserted so
the next sync cycle retries.  Worst-case on crash between delete and send
is data loss (preferred over double-counting on the API side).
"""

import asyncio
import json
from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Any, Optional
import httpx
from loguru import logger

from validator.db.operations import DatabaseManager
from validator.db.schema import SybilDetectionResult
from validator.config import get_validator_config
from validator.network.challenge_api_auth import merge_auth_headers


@dataclass(frozen=True)
class _SybilRecordSnapshot:
    """Detached, immutable copy of a SybilDetectionResult row.
    
    Created before the row is deleted from the local DB so that all fields
    survive the delete-before-send cycle without depending on a live
    SQLAlchemy session.
    """
    id: int
    challenge_id: Optional[int]
    suspicious_pairs_count: int
    suspicious_groups_count: int
    suspicious_pairs: Optional[Any]
    suspicious_groups: Optional[Any]
    analysis_report: Optional[str]
    high_similarity_threshold: float
    very_high_similarity_threshold: float


class SybilSyncTask:
    """
    Background task to periodically sync sybil detection records to Challenge API.
    
    Queries the local database for unsent records and sends them in batches.
    Uses delete-before-send to eliminate restart duplication.
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
    
    def _get_unsent_records(self) -> List[_SybilRecordSnapshot]:
        """Get unsent sybil detection records from database.
        
        Returns detached snapshots so they survive session close and can be
        used for delete-before-send (restart-safe dedup).
        """
        try:
            session = self.db_manager.get_session()
            try:
                all_records = session.query(SybilDetectionResult).order_by(
                    SybilDetectionResult.created_at.asc()
                ).limit(100).all()
                
                # Detach snapshots -- we need fields to survive after session close
                # and after the rows are deleted (delete-before-send pattern).
                snapshots = []
                for r in all_records:
                    snapshots.append(_SybilRecordSnapshot(
                        id=r.id,
                        challenge_id=r.challenge_id,
                        suspicious_pairs_count=r.suspicious_pairs_count,
                        suspicious_groups_count=r.suspicious_groups_count,
                        suspicious_pairs=r.suspicious_pairs,
                        suspicious_groups=r.suspicious_groups,
                        analysis_report=r.analysis_report,
                        high_similarity_threshold=r.high_similarity_threshold,
                        very_high_similarity_threshold=r.very_high_similarity_threshold,
                    ))
                return snapshots
            finally:
                session.close()
        except Exception as e:
            logger.error(f"Error getting unsent records: {e}", exc_info=True)
            return []
    
    async def _send_batch(self, records: List[_SybilRecordSnapshot]) -> None:
        """Send a batch of records to Challenge API in bulk.
        
        Uses delete-before-send to eliminate restart duplication:
        1. Delete records from local DB (they are already in ``records`` list)
        2. Send to API
        3. On failure, re-insert so they are retried next cycle
        
        Worst-case on crash between step 1 and 2: records are lost locally
        but were never sent.  This is preferred over the alternative (records
        sent twice and counted twice by the Challenge API).
        """
        if not self._http_client or not records:
            return
        
        challenge_api_url = self.config.challenge_api_url
        api_key = self.config.challenge_api_key
        endpoint = f"{challenge_api_url.rstrip('/')}/analytics/sybil-detection/bulk"
        
        # Step 1: Delete from local DB BEFORE sending (restart-safe)
        record_ids = [r.id for r in records]
        deleted_count = 0
        for rid in record_ids:
            if self.db_manager.delete_sybil_detection_result(rid):
                deleted_count += 1
        
        try:
            # Step 2: Build and send bulk request
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
                logger.info(
                    f"Successfully synced {inserted_count} sybil detection records in bulk "
                    f"(deleted {deleted_count} from local database)"
                )
            else:
                logger.warning(
                    f"Failed to sync sybil detection records in bulk: "
                    f"HTTP {response.status_code} - {response.text[:200]}. "
                    f"Re-inserting {len(records)} records for retry."
                )
                self._reinsert_records(records)
                
        except httpx.RequestError as e:
            logger.warning(
                f"Network error syncing sybil detection records in bulk: {e}. "
                f"Re-inserting {len(records)} records for retry."
            )
            self._reinsert_records(records)
        except Exception as e:
            logger.error(
                f"Error syncing sybil detection records in bulk: {e}. "
                f"Re-inserting {len(records)} records for retry.",
                exc_info=True
            )
            self._reinsert_records(records)
    
    def _reinsert_records(self, records: List[_SybilRecordSnapshot]) -> None:
        """Re-insert records into local DB after a failed API submission.
        
        Records were deleted before send (restart-safe dedup). If the send
        fails, we put them back so the next sync cycle retries.
        """
        reinserted = 0
        for record in records:
            try:
                self.db_manager.log_sybil_detection_result(
                    challenge_id=record.challenge_id,
                    suspicious_pairs=record.suspicious_pairs or [],
                    suspicious_groups=record.suspicious_groups or [],
                    analysis_report=record.analysis_report or "",
                    high_similarity_threshold=record.high_similarity_threshold,
                    very_high_similarity_threshold=record.very_high_similarity_threshold,
                )
                reinserted += 1
            except Exception as e:
                logger.error(f"Failed to re-insert sybil record (original id={record.id}): {e}")
        
        if reinserted:
            logger.info(f"Re-inserted {reinserted}/{len(records)} sybil records for retry")
