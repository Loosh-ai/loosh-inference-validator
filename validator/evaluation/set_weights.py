"""
Weight setting module for the Loosh Inference Validator.

Sets weights on-chain based on miner performance using EMA (Exponential Moving Average)
scores calculated from evaluation emissions.

Reference: https://docs.learnbittensor.org/tutorials/ocr-subnet-tutorial#step-4-set-weights

NOTE: This module uses the Bittensor SDK for weight setting because:
- Commit Reveal v3 (CRv3) has been REMOVED from the chain (SDK v10 migration)
- Fiber uses the deprecated bittensor-commit-reveal package which implements CRv3
- The Bittensor SDK v10+ handles CRv4 automatically via commit_timelocked_weights_extrinsic
- See: https://docs.learnbittensor.org/sdk/migration-guide

We still use Fiber for other chain operations (node fetching, etc.) where it works fine.
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import httpx
from bittensor import Subtensor, Wallet
from fiber.chain.fetch_nodes import get_nodes_for_netuid
from fiber.chain.interface import get_substrate
from fiber.chain.models import Node
from loguru import logger

from validator.db.operations import DatabaseManager
from validator.config import get_validator_config
from validator.internal_config import (
    WEIGHTS_INTERVAL_SECONDS,
    MAX_MINER_STAKE,
    WEIGHT_FRESHNESS_HOURS,
    WEIGHT_FRESHNESS_HOURS_DEGRADED,
    WEIGHT_MIN_SERVING_NODES,
    DEREGISTRATION_BLOCK_LIMIT,
    DEGRADED_MODE_THRESHOLD,
    EMERGENCY_MODE_THRESHOLD,
    SYBIL_PENALTY_ENABLED,
    SYBIL_PENALTY_MAX,
    SYBIL_PENALTY_THRESHOLD,
    SYBIL_SAFETY_MAX_PENALIZED_FRACTION,
    SYBIL_SCORE_CACHE_TTL_SECONDS,
)

if TYPE_CHECKING:
    from validator.validator_list_fetcher import ValidatorListFetcher


# ---------------------------------------------------------------------------
# Sybil score caching
# ---------------------------------------------------------------------------
# In-memory hot cache (fast path; survives across set_weights calls)
_sybil_score_cache: Dict[str, float] = {}
_sybil_score_cache_time: float = 0.0

# Full breakdown cache (for logging / penalty reports)
_sybil_breakdown_cache: Dict[str, Dict[str, Any]] = {}

# File-based cold cache (survives validator restart)
_SYBIL_CACHE_FILE = Path("sybil_score_cache.json")


def _filter_serving_miners(
    nodes: List[Node],
    validator_hotkey: str,
    validator_list_fetcher: Optional["ValidatorListFetcher"] = None,
) -> tuple[List[Node], Dict[str, int]]:
    """
    Filter nodes to only include serving miners.
    
    Filters out (in order of priority):
    - Validator's own node
    - Nodes without advertised endpoints (no IP or port) - fast pre-filter
    - Nodes in the validator database (via ValidatorListFetcher) - authoritative
    - Nodes with stake >= MAX_MINER_STAKE (fallback heuristic when no validator_list)
    
    Filter ordering rationale:
    - Self/endpoint checks are fast and definitive
    - validator_list_fetcher is authoritative when available
    - Stake threshold is only used as fallback to avoid excluding high-performing miners
    
    Args:
        nodes: List of all nodes from chain
        validator_hotkey: The validator's own hotkey to exclude
        validator_list_fetcher: Optional fetcher for registered validators
    
    Returns:
        Tuple of (filtered_miners, filter_stats) where filter_stats tracks
        how many nodes were filtered for each reason
    """
    filtered_miners: List[Node] = []
    stats = {
        "total": len(nodes),
        "self_excluded": 0,
        "high_stake_excluded": 0,
        "no_endpoint_excluded": 0,
        "validator_db_excluded": 0,
        "included": 0,
    }
    
    for node in nodes:
        # Skip validator's own node
        if node.hotkey == validator_hotkey:
            stats["self_excluded"] += 1
            continue
        
        # Skip nodes without advertised endpoints (fast pre-filter)
        # Check for empty IP, "0" (unset on chain), "0.0.0.0", or port 0
        if not node.ip or node.ip in ("0", "0.0.0.0") or node.port == 0:
            stats["no_endpoint_excluded"] += 1
            continue
        
        # Skip nodes registered as validators in Challenge API (authoritative)
        # This is checked BEFORE stake to avoid excluding high-stake miners
        if validator_list_fetcher and validator_list_fetcher.is_validator(node.hotkey):
            stats["validator_db_excluded"] += 1
            continue
        
        # Skip high-stake nodes ONLY as fallback when validator_list unavailable
        # This prevents excluding high-performing miners when we have authoritative data
        if not validator_list_fetcher and node.stake >= MAX_MINER_STAKE:
            stats["high_stake_excluded"] += 1
            continue
        
        # Node passes all filters - it's a serving miner
        filtered_miners.append(node)
        stats["included"] += 1
    
    return filtered_miners, stats


def _apply_freshness_gate(
    miner_ema_scores: Dict[str, float],
    last_success_times: Dict[str, datetime],
    freshness_hours: int,
) -> tuple[Dict[str, float], int]:
    """
    Apply freshness gate to EMA scores.
    
    Miners without a successful response within freshness_hours get zero weight,
    regardless of their EMA score. This prevents stale miners from receiving
    weight based on old performance.
    
    UID COMPRESSION SAFETY: All dicts are keyed by miner HOTKEY (persistent SS58 address),
    NOT by UID. UIDs are transient indices that change on UID compression/trimming.
    
    IMPORTANT: Assumes all last_success timestamps are stored as UTC naive datetimes.
    Database schema (db/schema.py) uses DateTime columns without timezone, and all
    timestamps are inserted via datetime.utcnow(). Do not mix timezone-aware and
    naive datetimes. For future migration to timezone-aware datetimes, see:
    docs/SET_WEIGHTS_REVIEW_ANALYSIS.md section 3.
    
    Args:
        miner_ema_scores: EMA scores keyed by miner hotkey (persistent SS58 address)
        last_success_times: Last success timestamps keyed by miner hotkey (UTC naive)
        freshness_hours: Hours threshold for considering a miner stale
    
    Returns:
        Tuple of (updated_scores, stale_count) where updated_scores has
        stale miners zeroed out
    """
    # Use UTC naive datetime to match DB storage format
    cutoff_time = datetime.utcnow() - timedelta(hours=freshness_hours)
    updated_scores = {}
    stale_count = 0
    
    for miner_hotkey, ema_score in miner_ema_scores.items():
        last_success = last_success_times.get(miner_hotkey)
        
        if last_success is None or last_success < cutoff_time:
            # Miner is stale - zero out their score
            updated_scores[miner_hotkey] = 0.0
            if ema_score > 0:
                stale_count += 1
        else:
            # Miner is fresh - keep their score
            updated_scores[miner_hotkey] = ema_score
    
    return updated_scores, stale_count


def _load_file_cache() -> Dict[str, float]:
    """Load sybil scores from the JSON file cache (cold cache)."""
    try:
        if _SYBIL_CACHE_FILE.exists():
            raw = json.loads(_SYBIL_CACHE_FILE.read_text())
            scores = {k: float(v) for k, v in raw.get("scores", {}).items()}
            age = time.time() - raw.get("timestamp", 0)
            logger.info(
                f"[sybil] Loaded file cache: {len(scores)} entries, age={age:.0f}s"
            )
            return scores
    except Exception as e:
        logger.warning(f"[sybil] Could not load file cache: {e}")
    return {}


def _save_file_cache(scores: Dict[str, float]) -> None:
    """Persist sybil scores to JSON file (cold cache) using atomic write."""
    try:
        payload = {
            "scores": scores,
            "timestamp": time.time(),
            "saved_at": datetime.utcnow().isoformat(),
        }
        # Atomic write: write to temp file then rename to prevent corruption
        # on crash mid-write.  rename() is atomic on POSIX when src and dst
        # are on the same filesystem.
        tmp_path = _SYBIL_CACHE_FILE.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(payload, indent=2))
        tmp_path.replace(_SYBIL_CACHE_FILE)
    except Exception as e:
        logger.warning(f"[sybil] Could not save file cache: {e}")


def get_sybil_breakdown(hotkey: str) -> Optional[Dict[str, Any]]:
    """
    Return the full breakdown dict for *hotkey* from the last fetch.

    Useful for penalty reports and diagnostics.  Returns ``None`` when no
    breakdown has been cached yet.
    """
    return _sybil_breakdown_cache.get(hotkey)


async def _fetch_sybil_scores(
    miner_hotkeys: List[str],
    challenge_api_url: str,
    challenge_api_key: str,
) -> Dict[str, float]:
    """
    Fetch aggregated sybil scores from the Challenge API with caching.

    Cache hierarchy (fastest → slowest):
      1. In-memory hot cache (TTL = SYBIL_SCORE_CACHE_TTL_SECONDS)
      2. JSON file cold cache (survives validator restart)
      3. Live fetch from Challenge API ``POST /analytics/sybil-scores/batch``

    Returns hotkey → sybil_score (0.0 = clean, 1.0 = certain sybil).

    On network errors the best available cache is returned; if nothing is
    available an empty dict is returned (fail-open: no penalty on fetch failure).
    """
    global _sybil_score_cache, _sybil_score_cache_time, _sybil_breakdown_cache

    # 1. Hot cache (in-memory)
    if _sybil_score_cache and (time.monotonic() - _sybil_score_cache_time) < SYBIL_SCORE_CACHE_TTL_SECONDS:
        logger.debug(
            f"[sybil] Using hot cache ({len(_sybil_score_cache)} entries, "
            f"age={time.monotonic() - _sybil_score_cache_time:.0f}s)"
        )
        return _sybil_score_cache

    url = f"{challenge_api_url.rstrip('/')}/analytics/sybil-scores/batch"

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(
                url,
                json={"miner_hotkeys": miner_hotkeys},
                headers={
                    "X-API-Key": challenge_api_key,
                    "Content-Type": "application/json",
                },
            )
            response.raise_for_status()
            data = response.json()

        scores: Dict[str, float] = {}
        breakdown: Dict[str, Dict[str, Any]] = {}

        for hotkey, score_data in data.get("scores", {}).items():
            if isinstance(score_data, dict):
                scores[hotkey] = float(score_data.get("sybil_score", 0.0))
                # Store full breakdown for logging / penalty reports
                breakdown[hotkey] = score_data
            else:
                scores[hotkey] = float(score_data)

        # Update hot cache
        _sybil_score_cache = scores
        _sybil_score_cache_time = time.monotonic()
        _sybil_breakdown_cache = breakdown

        # Persist to file (cold cache)
        _save_file_cache(scores)

        flagged = sum(1 for s in scores.values() if s >= SYBIL_PENALTY_THRESHOLD)
        logger.info(
            f"[sybil] Fetched sybil scores for {len(scores)} miners "
            f"({flagged} flagged above threshold {SYBIL_PENALTY_THRESHOLD})"
        )
        return scores

    except Exception as e:
        logger.warning(f"[sybil] Failed to fetch sybil scores from Challenge API: {e}")

        # 2. Return stale hot cache if available
        if _sybil_score_cache:
            logger.info(
                f"[sybil] Returning stale hot cache ({len(_sybil_score_cache)} entries, "
                f"age={time.monotonic() - _sybil_score_cache_time:.0f}s)"
            )
            return _sybil_score_cache

        # 3. Try file (cold) cache
        file_scores = _load_file_cache()
        if file_scores:
            _sybil_score_cache = file_scores
            _sybil_score_cache_time = time.monotonic()
            return file_scores

        # 4. Fail open: no penalty when we have no data at all
        logger.warning("[sybil] No cached data available — skipping sybil penalty (fail-open)")
        return {}


def _apply_sybil_penalty(
    miner_ema_scores: Dict[str, float],
    sybil_scores: Dict[str, float],
    serving_miners_count: int,
) -> tuple[Dict[str, float], int, bool]:
    """
    Apply graduated sybil penalty to miner EMA scores.

    Penalty formula (per miner):
        penalty = min(sybil_score, SYBIL_PENALTY_MAX)   (clamped to max)
        penalized_ema = ema_score * (1 - penalty)

    Safety valve with **fallback-K restoration**:
        If more than ``SYBIL_SAFETY_MAX_PENALIZED_FRACTION`` of serving miners
        would be penalized, we do NOT skip penalties entirely.  Instead we
        penalize only the **top-K worst offenders** (sorted by descending
        sybil_score, K = max_allowed).  This prevents sybils from gaming the
        system by mass-flooding detections to trigger a blanket skip.

    Args:
        miner_ema_scores: EMA scores keyed by hotkey (not mutated)
        sybil_scores: Sybil scores keyed by hotkey (0.0 – 1.0)
        serving_miners_count: Total number of serving miners (for safety valve)

    Returns:
        (updated_scores, penalized_count, safety_triggered)
        ``safety_triggered`` is True when fallback-K was used (some miners
        were spared that otherwise would have been penalized).
    """
    if not SYBIL_PENALTY_ENABLED:
        return miner_ema_scores, 0, False

    if not sybil_scores:
        return miner_ema_scores, 0, False

    # Identify candidates: miners above threshold with non-zero EMA
    candidates = [
        (hk, sybil_scores[hk])
        for hk in miner_ema_scores
        if sybil_scores.get(hk, 0.0) >= SYBIL_PENALTY_THRESHOLD
        and miner_ema_scores[hk] > 0
    ]

    max_allowed = max(1, int(serving_miners_count * SYBIL_SAFETY_MAX_PENALIZED_FRACTION))
    safety_triggered = len(candidates) > max_allowed

    if safety_triggered:
        # Fallback-K: sort by descending sybil_score, keep only worst K
        candidates.sort(key=lambda x: x[1], reverse=True)
        spared = len(candidates) - max_allowed
        candidates = candidates[:max_allowed]
        logger.critical(
            f"[sybil] SAFETY VALVE (fallback-K): {len(candidates) + spared} miners "
            f"would be penalized, limit is {max_allowed} "
            f"({SYBIL_SAFETY_MAX_PENALIZED_FRACTION*100:.0f}% of "
            f"{serving_miners_count} serving). Penalizing top-{max_allowed} worst "
            f"offenders, sparing {spared} lower-score miners."
        )

    # Build set of hotkeys to penalize
    penalize_set = {hk for hk, _ in candidates}

    # Apply graduated penalty
    updated = {}
    penalized_count = 0

    for hotkey, ema_score in miner_ema_scores.items():
        if hotkey in penalize_set:
            sybil_score = sybil_scores.get(hotkey, 0.0)
            penalty = min(sybil_score, SYBIL_PENALTY_MAX)
            new_score = ema_score * (1.0 - penalty)

            logger.info(
                f"[sybil] Penalizing miner {hotkey[:16]}...: "
                f"sybil_score={sybil_score:.3f}, penalty={penalty:.3f}, "
                f"ema={ema_score:.6f} → {new_score:.6f}"
            )
            updated[hotkey] = new_score
            penalized_count += 1
        else:
            updated[hotkey] = ema_score

    return updated, penalized_count, safety_triggered


async def _submit_penalty_report(
    validator_hotkey: str,
    penalty_actions: List[Dict[str, Any]],
    challenge_api_url: str,
    challenge_api_key: str,
) -> None:
    """
    Fire-and-forget: POST the penalty report to the Challenge API.

    This is called via ``asyncio.create_task`` so failures are logged but
    never block or crash the weight-setting pipeline.
    """
    url = f"{challenge_api_url.rstrip('/')}/analytics/penalty-report"
    payload = {
        "validator_hotkey": validator_hotkey,
        "timestamp": datetime.utcnow().isoformat(),
        "penalties": penalty_actions,
    }

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.post(
                url,
                json=payload,
                headers={
                    "X-API-Key": challenge_api_key,
                    "Content-Type": "application/json",
                },
            )
        if resp.status_code == 201:
            logger.debug(
                f"[sybil] Penalty report submitted ({len(penalty_actions)} actions)"
            )
        else:
            logger.warning(
                f"[sybil] Penalty report failed: HTTP {resp.status_code} — "
                f"{resp.text[:200]}"
            )
    except Exception as e:
        logger.warning(f"[sybil] Could not submit penalty report: {e}")


async def _set_weights_with_sdk(
    subtensor: Subtensor,
    wallet: Wallet,
    netuid: int,
    uids: List[int],
    weights: List[float],
    timeout: float = 120.0
) -> bool:
    """
    Set weights using the Bittensor SDK.
    
    The SDK handles CRv4 automatically based on subnet configuration.
    Per the SDK v10 migration guide:
    - CRv3 logic has been removed
    - Use commit_timelocked_weights_extrinsic with commit_reveal_version=4
    - set_weights() handles this automatically based on subnet settings
    
    Args:
        subtensor: Bittensor Subtensor connection
        wallet: Bittensor Wallet with hotkey
        netuid: Subnet UID
        uids: List of node UIDs to set weights for
        weights: List of weights (will be normalized by SDK)
        timeout: Timeout in seconds
    
    Returns:
        True if weights were set successfully, False otherwise
    """
    try:
        # Run in executor since SDK calls are blocking
        loop = asyncio.get_running_loop()
        
        def _set_weights_sync():
            # SDK v10 set_weights returns ExtrinsicResponse object
            # It handles CRv4 automatically based on subnet commit_reveal settings
            response = subtensor.set_weights(
                wallet=wallet,
                netuid=netuid,
                uids=uids,
                weights=weights,
                wait_for_inclusion=True,
                wait_for_finalization=True,
            )
            return response
        
        response = await asyncio.wait_for(
            loop.run_in_executor(None, _set_weights_sync),
            timeout=timeout
        )
        
        # SDK v10 returns ExtrinsicResponse with success attribute
        if hasattr(response, 'success'):
            if response.success:
                logger.info("Successfully set weights on chain via Bittensor SDK")
                return True
            else:
                error_msg = getattr(response, 'message', 'Unknown error')
                logger.error(f"Failed to set weights: {error_msg}")
                return False
        else:
            # Fallback for older SDK versions that return bool
            if response:
                logger.info("Successfully set weights on chain via Bittensor SDK")
                return True
            else:
                logger.error("Failed to set weights on chain")
                return False
        
    except asyncio.TimeoutError:
        logger.error(f"set_weights timed out after {timeout} seconds")
        return False
    except Exception as e:
        logger.error(f"Error in set_weights: {str(e)}")
        logger.exception("Full traceback:")
        return False


async def set_weights(
    db_manager: DatabaseManager,
    validator_list_fetcher: Optional["ValidatorListFetcher"] = None,
    ema_lookback_hours: int = 24,
    ema_alpha: float = 0.3
) -> None:
    """
    Set weights for miners based on their EMA performance scores.
    
    This function uses a tiered fallback strategy to prevent validator deregistration:
    
    NORMAL MODE (blocks_since_update < 4000):
    - Use standard EMA + 3-hour freshness gate
    - Skip weight setting if all weights are zero (preserve on-chain weights)
    
    DEGRADED MODE (4000 <= blocks_since_update < 4500):
    - Use EMA + relaxed 24-hour freshness gate
    - Set weights even if Challenge API had issues
    - Warn operators of degraded operation
    
    EMERGENCY MODE (blocks_since_update >= 4500):
    - Use ANY available emissions (no freshness gate)
    - Prevents deregistration at 5000-block threshold
    - Critical warnings for operator intervention
    
    Process:
    1. Connects to the Bittensor chain via SDK (for weight setting)
    2. Uses Fiber to fetch registered nodes on the subnet
    3. Filters to serving miners only (excludes validators, non-serving nodes)
    4. Determines operation mode based on blocks since last update
    5. Calculates EMA scores from evaluation emissions in the database
    6. Applies appropriate freshness gate based on mode
    7. Falls back to emergency weights if necessary
    8. Normalizes weights to sum to 1.0
    9. Sets weights on-chain via Bittensor SDK (handles CRv4 automatically)
    
    Args:
        db_manager: DatabaseManager instance for querying scores
        validator_list_fetcher: Optional fetcher for registered validators to exclude
        ema_lookback_hours: Hours of history to consider for EMA (default: 24)
        ema_alpha: EMA smoothing factor (default: 0.3)
    
    Raises:
        Exception: If weight setting fails (caller should handle retries)
    """
    config = get_validator_config()
    
    subtensor = None
    substrate = None
    
    try:
        # 1. Initialize Bittensor SDK Subtensor connection
        network = config.subtensor_network
        chain_endpoint = config.subtensor_address if config.subtensor_address else None
        
        logger.info(
            f"[set_weights] Connecting to chain via Bittensor SDK: network={network}, "
            f"endpoint={chain_endpoint}"
        )
        
        # Create Subtensor instance
        if chain_endpoint and chain_endpoint.startswith(("ws://", "wss://")):
            subtensor = Subtensor(network=chain_endpoint)
        else:
            subtensor = Subtensor(network=network)
        
        # 2. Load wallet using Bittensor SDK
        wallet = Wallet(
            name=config.wallet_name,
            hotkey=config.hotkey_name
        )
        
        hotkey_ss58 = wallet.hotkey.ss58_address
        logger.info(f"[set_weights] Loaded validator wallet: {hotkey_ss58[:16]}...")
        
        # 3. Get validator UID from chain
        validator_uid = subtensor.get_uid_for_hotkey_on_subnet(
            hotkey_ss58=hotkey_ss58,
            netuid=config.netuid
        )
        
        if validator_uid is None:
            logger.error(
                f"[set_weights] Validator {hotkey_ss58[:16]}... is not registered on subnet "
                f"{config.netuid}. Cannot set weights without registration."
            )
            raise ValueError(f"Validator not registered on subnet {config.netuid}")
        
        logger.info(f"[set_weights] Validator UID: {validator_uid}")
        
        # 4. Check rate limit EARLY to avoid expensive operations if not eligible
        # Also determine operation mode based on blocks since last update
        operation_mode = "NORMAL"  # Default
        blocks_since_update = None
        
        try:
            blocks_since_update = subtensor.blocks_since_last_update(
                netuid=config.netuid,
                uid=validator_uid
            )
            weights_rate_limit = subtensor.weights_rate_limit(netuid=config.netuid)
            
            # Determine operation mode based on deregistration risk
            if blocks_since_update >= EMERGENCY_MODE_THRESHOLD:
                operation_mode = "EMERGENCY"
                logger.critical(
                    f"[set_weights] ⚠️ EMERGENCY MODE: {blocks_since_update} blocks since last update "
                    f"({DEREGISTRATION_BLOCK_LIMIT - blocks_since_update} blocks until deregistration at {DEREGISTRATION_BLOCK_LIMIT}). "
                    f"Will use ANY available emissions to prevent deregistration!"
                )
            elif blocks_since_update >= DEGRADED_MODE_THRESHOLD:
                operation_mode = "DEGRADED"
                logger.warning(
                    f"[set_weights] ⚠️ DEGRADED MODE: {blocks_since_update} blocks since last update "
                    f"({DEREGISTRATION_BLOCK_LIMIT - blocks_since_update} blocks until deregistration at {DEREGISTRATION_BLOCK_LIMIT}). "
                    f"Will use relaxed freshness gate ({WEIGHT_FRESHNESS_HOURS_DEGRADED}h instead of {WEIGHT_FRESHNESS_HOURS}h)."
                )
            else:
                logger.info(
                    f"[set_weights] NORMAL MODE: {blocks_since_update} blocks since last update "
                    f"({DEREGISTRATION_BLOCK_LIMIT - blocks_since_update} blocks until deregistration threshold)."
                )
            
            if blocks_since_update < weights_rate_limit:
                logger.warning(
                    f"[set_weights] Cannot set weights yet - rate limit not met. "
                    f"Blocks since update: {blocks_since_update}, required: {weights_rate_limit}. "
                    f"Skipping this cycle (no expensive operations performed)."
                )
                return
            
            logger.info(
                f"[set_weights] Rate limit check passed: {blocks_since_update} blocks since update "
                f"(required: {weights_rate_limit})"
            )
        except Exception as e:
            logger.warning(
                f"[set_weights] Could not check rate limit: {e}. "
                "Proceeding with weight setting (will fail at submission if rate limit not met)."
            )
        
        # 5. Get version key from chain (for logging/info purposes)
        try:
            version_key = subtensor.substrate.query(
                "SubtensorModule", "WeightsVersionKey", [config.netuid]
            ).value
            logger.info(f"[set_weights] Subnet WeightsVersionKey: {version_key}")
        except Exception as e:
            logger.warning(f"[set_weights] Could not query WeightsVersionKey: {e}")
            version_key = None
        
        # 6. Get all registered nodes using Fiber
        substrate = get_substrate(
            subtensor_network=config.subtensor_network,
            subtensor_address=config.subtensor_address
        )
        all_nodes = get_nodes_for_netuid(substrate=substrate, netuid=config.netuid)
        
        if not all_nodes:
            logger.warning(f"[set_weights] No nodes found on subnet {config.netuid}. Skipping weight setting.")
            return
        
        logger.info(f"[set_weights] Found {len(all_nodes)} registered nodes on subnet {config.netuid}")
        
        # 7. Filter to serving miners only
        serving_miners, filter_stats = _filter_serving_miners(
            nodes=all_nodes,
            validator_hotkey=hotkey_ss58,
            validator_list_fetcher=validator_list_fetcher,
        )
        
        logger.info(
            f"[set_weights] Node filtering results: "
            f"total={filter_stats['total']}, "
            f"self_excluded={filter_stats['self_excluded']}, "
            f"high_stake_excluded={filter_stats['high_stake_excluded']}, "
            f"no_endpoint_excluded={filter_stats['no_endpoint_excluded']}, "
            f"validator_db_excluded={filter_stats['validator_db_excluded']}, "
            f"serving_miners={filter_stats['included']}"
        )
        
        if len(serving_miners) < WEIGHT_MIN_SERVING_NODES:
            logger.warning(
                f"[set_weights] Only {len(serving_miners)} serving miners found, "
                f"minimum required is {WEIGHT_MIN_SERVING_NODES}. Skipping weight setting."
            )
            return
        
        # 8. Get EMA scores from database
        miner_ema_scores = db_manager.get_miner_ema_scores(
            lookback_hours=ema_lookback_hours,
            alpha=ema_alpha
        )
        logger.info(f"[set_weights] Retrieved EMA scores for {len(miner_ema_scores)} miners from DB")
        
        # 9. Get last success times and apply freshness gate based on operation mode
        last_success_times = db_manager.get_miner_last_success_times()
        logger.info(f"[set_weights] Retrieved last_success times for {len(last_success_times)} miners")
        
        # Choose freshness threshold based on operation mode
        if operation_mode == "EMERGENCY":
            # EMERGENCY: No freshness gate - use ALL available emissions
            logger.warning(
                f"[set_weights] {operation_mode} MODE: Skipping freshness gate to prevent deregistration. "
                f"Using ALL available emissions from DB."
            )
            stale_count = 0
            # Don't apply freshness gate - keep all scores
        elif operation_mode == "DEGRADED":
            # DEGRADED: Relaxed freshness gate (24 hours)
            logger.info(
                f"[set_weights] {operation_mode} MODE: Applying relaxed freshness gate "
                f"({WEIGHT_FRESHNESS_HOURS_DEGRADED}h instead of {WEIGHT_FRESHNESS_HOURS}h)"
            )
            miner_ema_scores, stale_count = _apply_freshness_gate(
                miner_ema_scores=miner_ema_scores,
                last_success_times=last_success_times,
                freshness_hours=WEIGHT_FRESHNESS_HOURS_DEGRADED,
            )
        else:
            # NORMAL: Standard freshness gate (3 hours)
            miner_ema_scores, stale_count = _apply_freshness_gate(
                miner_ema_scores=miner_ema_scores,
                last_success_times=last_success_times,
                freshness_hours=WEIGHT_FRESHNESS_HOURS,
            )
        
        if stale_count > 0:
            logger.info(
                f"[set_weights] Freshness gate: zeroed out {stale_count} stale miners "
                f"in {operation_mode} mode"
            )
        
        # 9.5. Fetch sybil scores and apply penalty
        serving_hotkeys = [node.hotkey for node in serving_miners]
        pre_penalty_scores = dict(miner_ema_scores)  # snapshot for summary
        sybil_scores = await _fetch_sybil_scores(
            miner_hotkeys=serving_hotkeys,
            challenge_api_url=config.challenge_api_url,
            challenge_api_key=config.challenge_api_key,
        )
        if sybil_scores:
            logger.info(
                f"[set_weights] Fetched sybil scores for {len(sybil_scores)} miners "
                f"from Challenge API."
            )
            miner_ema_scores, penalized_count, safety_triggered = _apply_sybil_penalty(
                miner_ema_scores=miner_ema_scores,
                sybil_scores=sybil_scores,
                serving_miners_count=len(serving_miners),
            )

            # --- Detailed penalty summary logging ---
            graduated_count = 0
            zeroed_count = 0
            penalty_actions: List[Dict[str, Any]] = []

            if penalized_count > 0:
                # Build a UID lookup for informational logging
                hotkey_to_uid = {n.hotkey: n.node_id for n in serving_miners}

                for hk in miner_ema_scores:
                    old_ema = pre_penalty_scores.get(hk, 0.0)
                    new_ema = miner_ema_scores[hk]
                    if old_ema > 0 and new_ema < old_ema:
                        if new_ema > 0:
                            graduated_count += 1
                            p_type = "graduated"
                        else:
                            zeroed_count += 1
                            p_type = "zeroed"
                        penalty_actions.append({
                            "miner_hotkey": hk,
                            "miner_uid": hotkey_to_uid.get(hk),
                            "penalty_type": p_type,
                            "sybil_score": sybil_scores.get(hk, 0.0),
                        })

                logger.info(
                    f"[set_weights] Sybil penalty summary: "
                    f"penalized={penalized_count}, graduated={graduated_count}, "
                    f"zeroed={zeroed_count}, safety_valve={'YES' if safety_triggered else 'no'}"
                )

            if safety_triggered:
                logger.critical(
                    "[set_weights] Sybil safety valve (fallback-K) was triggered — "
                    "only the worst offenders were penalized this round."
                )

            # Fire-and-forget: submit penalty report to Challenge API
            if penalty_actions:
                asyncio.create_task(
                    _submit_penalty_report(
                        validator_hotkey=hotkey_ss58,
                        penalty_actions=penalty_actions,
                        challenge_api_url=config.challenge_api_url,
                        challenge_api_key=config.challenge_api_key,
                    )
                )
        else:
            logger.debug("[set_weights] No sybil scores available — skipping penalty step.")
        
        # 10. Build weight vectors for serving miners only
        # UID COMPRESSION SAFETY: EMA scores are keyed by HOTKEY (persistent SS58 address).
        # We look up each node's score by hotkey, then build the UID-based weight vector
        # that the chain requires for set_weights().
        uids: List[int] = []
        weights_list: List[float] = []
        miners_with_weight = 0
        
        for node in serving_miners:
            node_id = node.node_id
            # Look up EMA score by HOTKEY (persistent identity), not by UID
            score = miner_ema_scores.get(node.hotkey, 0.0)
            
            uids.append(node_id)
            weights_list.append(score)
            
            if score > 0:
                miners_with_weight += 1
        
        # 11. Check if we have any non-zero weights and handle based on operation mode
        total_weight = sum(weights_list)
        
        if total_weight <= 0:
            if operation_mode == "EMERGENCY":
                # EMERGENCY: Distribute minimal uniform weights to prevent deregistration
                # This is only done as absolute last resort
                logger.critical(
                    f"[set_weights] {operation_mode} MODE: All EMA scores are zero but must set weights "
                    f"to prevent deregistration. Distributing uniform minimal weights to all {len(serving_miners)} "
                    f"serving miners as absolute last resort."
                )
                # Set uniform minimal weights
                uniform_weight = 1.0 / len(serving_miners)
                weights_list = [uniform_weight for _ in serving_miners]
                miners_with_weight = len(serving_miners)
            elif operation_mode == "DEGRADED":
                # DEGRADED: Skip but warn more urgently
                blocks_until_emergency = EMERGENCY_MODE_THRESHOLD - (blocks_since_update or 0)
                logger.error(
                    f"[set_weights] {operation_mode} MODE: All EMA scores are zero even with relaxed freshness gate. "
                    f"SKIPPING weight setting. WARNING: Only {blocks_until_emergency} blocks until EMERGENCY MODE. "
                    f"If Challenge API issues persist, validator may face deregistration!"
                )
                return
            else:
                # NORMAL: Standard skip behavior
                logger.warning(
                    f"[set_weights] {operation_mode} MODE: All EMA scores are zero (after freshness gate). "
                    f"SKIPPING weight setting to preserve previous on-chain weights. "
                    f"This can happen during startup, DB outage, or if all miners are stale."
                )
                return
        else:
            # We have non-zero weights - normalize them
            weights_list = [w / total_weight for w in weights_list]
        
        # 12. Calculate weight statistics for logging
        non_zero_weights = [w for w in weights_list if w > 0]
        if non_zero_weights:
            min_weight = min(non_zero_weights)
            max_weight = max(non_zero_weights)
            avg_weight = sum(non_zero_weights) / len(non_zero_weights)
        else:
            min_weight = max_weight = avg_weight = 0.0
        
        logger.info(
            f"[set_weights] [{operation_mode} MODE] Weight distribution: "
            f"miners_with_weight={miners_with_weight}/{len(serving_miners)}, "
            f"min={min_weight:.6f}, max={max_weight:.6f}, avg={avg_weight:.6f}"
        )
        
        # Log detailed weight information (at debug level to avoid spam)
        # NOTE: EMA scores and last_success_times are keyed by hotkey (persistent identity)
        for uid, weight, node in zip(uids, weights_list, serving_miners):
            ema_score = miner_ema_scores.get(node.hotkey, 0.0)
            last_success = last_success_times.get(node.hotkey)
            # Format datetime as ISO for better log readability and parseability
            freshness_str = f", last_success={last_success.isoformat()}" if last_success else ", no_success_recorded"
            logger.debug(
                f"[set_weights]   Node {uid} ({node.hotkey[:16]}...): "
                f"ema_score={ema_score:.6f}, weight={weight:.6f}{freshness_str}"
            )
        
        # 13. Set weights using Bittensor SDK (rate limit already checked at step 4)
        logger.info(
            f"[set_weights] [{operation_mode} MODE] Submitting weights for {len(uids)} miners "
            f"via Bittensor SDK (handles CRv4 automatically)..."
        )
        
        success = await _set_weights_with_sdk(
            subtensor=subtensor,
            wallet=wallet,
            netuid=config.netuid,
            uids=uids,
            weights=weights_list
        )
        
        if success:
            logger.info(
                f"[set_weights] [{operation_mode} MODE] SUCCESS: Set weights on chain for {miners_with_weight} "
                f"miners with non-zero weight out of {len(uids)} total (netuid={config.netuid})"
            )
        else:
            logger.error(
                f"[set_weights] [{operation_mode} MODE] FAILED: Could not set weights on chain "
                f"(netuid={config.netuid}). Check chain connection and validator registration."
            )
            raise RuntimeError("Failed to set weights on chain")
        
    except Exception as e:
        logger.error(f"[set_weights] Error setting weights: {str(e)}")
        logger.exception("[set_weights] Full error traceback:")
        raise
    finally:
        # Ensure chain connections are closed to prevent resource leaks
        if substrate is not None:
            try:
                substrate.close()
            except Exception:
                pass
        if subtensor is not None:
            try:
                # Subtensor wraps a SubstrateInterface; close it explicitly
                if hasattr(subtensor, 'substrate') and subtensor.substrate is not None:
                    subtensor.substrate.close()
            except Exception:
                pass