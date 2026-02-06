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
from datetime import datetime, timedelta
from typing import List, Optional, Dict, TYPE_CHECKING

from bittensor import Subtensor, Wallet
from fiber.chain.fetch_nodes import get_nodes_for_netuid
from fiber.chain.interface import get_substrate
from fiber.chain.models import Node
from loguru import logger

from validator.db.operations import DatabaseManager
from validator.config import get_validator_config

if TYPE_CHECKING:
    from validator.validator_list_fetcher import ValidatorListFetcher

# Hard-coded to ensure all validators set weights on the same schedule
# 4320 seconds = 72 minutes
WEIGHTS_INTERVAL_SECONDS = 4320

# Maximum stake for a node to be considered a miner (not a validator)
# Matches the challenge selection logic in main.py
MAX_MINER_STAKE = 999

# Freshness gate: miners without successful responses within this window get zero weight
# Hard-coded to ensure consistent behavior across all validators
WEIGHT_FRESHNESS_HOURS = 3

# Minimum number of serving miners required to set weights
# If fewer serving miners are found, weight setting is skipped to avoid bad state
WEIGHT_MIN_SERVING_NODES = 1


def _filter_serving_miners(
    nodes: List[Node],
    validator_hotkey: str,
    validator_list_fetcher: Optional["ValidatorListFetcher"] = None,
) -> tuple[List[Node], Dict[str, int]]:
    """
    Filter nodes to only include serving miners.
    
    Filters out:
    - Validator's own node
    - Nodes with stake >= MAX_MINER_STAKE (high stake = likely validator)
    - Nodes without advertised endpoints (no IP or port)
    - Nodes in the validator database (via ValidatorListFetcher)
    
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
        
        # Skip high-stake nodes (likely validators)
        if node.stake >= MAX_MINER_STAKE:
            stats["high_stake_excluded"] += 1
            continue
        
        # Skip nodes without advertised endpoints
        # Check for empty IP, "0" (unset on chain), "0.0.0.0", or port 0
        if not node.ip or node.ip in ("0", "0.0.0.0") or node.port == 0:
            stats["no_endpoint_excluded"] += 1
            continue
        
        # Skip nodes registered as validators in Challenge API
        if validator_list_fetcher and validator_list_fetcher.is_validator(node.hotkey):
            stats["validator_db_excluded"] += 1
            continue
        
        # Node passes all filters - it's a serving miner
        filtered_miners.append(node)
        stats["included"] += 1
    
    return filtered_miners, stats


def _apply_freshness_gate(
    miner_ema_scores: Dict[int, float],
    last_success_times: Dict[int, datetime],
    freshness_hours: int,
) -> tuple[Dict[int, float], int]:
    """
    Apply freshness gate to EMA scores.
    
    Miners without a successful response within freshness_hours get zero weight,
    regardless of their EMA score. This prevents stale miners from receiving
    weight based on old performance.
    
    Args:
        miner_ema_scores: EMA scores keyed by node_id
        last_success_times: Last success timestamps keyed by node_id
        freshness_hours: Hours threshold for considering a miner stale
    
    Returns:
        Tuple of (updated_scores, stale_count) where updated_scores has
        stale miners zeroed out
    """
    cutoff_time = datetime.utcnow() - timedelta(hours=freshness_hours)
    updated_scores = {}
    stale_count = 0
    
    for node_id, ema_score in miner_ema_scores.items():
        last_success = last_success_times.get(node_id)
        
        if last_success is None or last_success < cutoff_time:
            # Miner is stale - zero out their score
            updated_scores[node_id] = 0.0
            if ema_score > 0:
                stale_count += 1
        else:
            # Miner is fresh - keep their score
            updated_scores[node_id] = ema_score
    
    return updated_scores, stale_count


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
    
    This function:
    1. Connects to the Bittensor chain via SDK (for weight setting)
    2. Uses Fiber to fetch registered nodes on the subnet
    3. Filters to serving miners only (excludes validators, non-serving nodes)
    4. Calculates EMA scores from evaluation emissions in the database
    5. Applies freshness gate (stale miners get zero weight)
    6. If all weights are zero, SKIPS setting weights (keeps previous on-chain weights)
    7. Normalizes weights to sum to 1.0
    8. Sets weights on-chain via Bittensor SDK (handles CRv4 automatically)
    
    Args:
        db_manager: DatabaseManager instance for querying scores
        validator_list_fetcher: Optional fetcher for registered validators to exclude
        ema_lookback_hours: Hours of history to consider for EMA (default: 24)
        ema_alpha: EMA smoothing factor (default: 0.3)
    
    Raises:
        Exception: If weight setting fails (caller should handle retries)
    """
    config = get_validator_config()
    
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
        
        # 4. Get version key from chain (for logging/info purposes)
        try:
            version_key = subtensor.substrate.query(
                "SubtensorModule", "WeightsVersionKey", [config.netuid]
            ).value
            logger.info(f"[set_weights] Subnet WeightsVersionKey: {version_key}")
        except Exception as e:
            logger.warning(f"[set_weights] Could not query WeightsVersionKey: {e}")
            version_key = None
        
        # 5. Get all registered nodes using Fiber
        substrate = get_substrate(
            subtensor_network=config.subtensor_network,
            subtensor_address=config.subtensor_address
        )
        all_nodes = get_nodes_for_netuid(substrate=substrate, netuid=config.netuid)
        
        if not all_nodes:
            logger.warning(f"[set_weights] No nodes found on subnet {config.netuid}. Skipping weight setting.")
            return
        
        logger.info(f"[set_weights] Found {len(all_nodes)} registered nodes on subnet {config.netuid}")
        
        # 6. Filter to serving miners only
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
        
        # 7. Get EMA scores from database
        miner_ema_scores = db_manager.get_miner_ema_scores(
            lookback_hours=ema_lookback_hours,
            alpha=ema_alpha
        )
        logger.info(f"[set_weights] Retrieved EMA scores for {len(miner_ema_scores)} miners from DB")
        
        # 8. Get last success times and apply freshness gate
        last_success_times = db_manager.get_miner_last_success_times()
        logger.info(f"[set_weights] Retrieved last_success times for {len(last_success_times)} miners")
        
        miner_ema_scores, stale_count = _apply_freshness_gate(
            miner_ema_scores=miner_ema_scores,
            last_success_times=last_success_times,
            freshness_hours=WEIGHT_FRESHNESS_HOURS,
        )
        
        if stale_count > 0:
            logger.info(
                f"[set_weights] Freshness gate: zeroed out {stale_count} stale miners "
                f"(no success in {WEIGHT_FRESHNESS_HOURS}h)"
            )
        
        # 9. Build weight vectors for serving miners only
        uids: List[int] = []
        weights_list: List[float] = []
        miners_with_weight = 0
        
        for node in serving_miners:
            node_id = node.node_id
            score = miner_ema_scores.get(node_id, 0.0)
            
            uids.append(node_id)
            weights_list.append(score)
            
            if score > 0:
                miners_with_weight += 1
        
        # 10. Check if we have any non-zero weights
        total_weight = sum(weights_list)
        
        if total_weight <= 0:
            # CRITICAL: Do NOT distribute weights evenly - this would reward
            # untested, down, or adversarial miners. Instead, skip this cycle
            # and keep the previous on-chain weights intact.
            logger.warning(
                f"[set_weights] All EMA scores are zero (after freshness gate). "
                f"SKIPPING weight setting to preserve previous on-chain weights. "
                f"This can happen during startup, DB outage, or if all miners are stale."
            )
            return
        
        # 11. Normalize weights to sum to 1.0
        weights_list = [w / total_weight for w in weights_list]
        
        # Calculate weight statistics for logging
        non_zero_weights = [w for w in weights_list if w > 0]
        if non_zero_weights:
            min_weight = min(non_zero_weights)
            max_weight = max(non_zero_weights)
            avg_weight = sum(non_zero_weights) / len(non_zero_weights)
        else:
            min_weight = max_weight = avg_weight = 0.0
        
        logger.info(
            f"[set_weights] Weight distribution: "
            f"miners_with_weight={miners_with_weight}/{len(serving_miners)}, "
            f"min={min_weight:.6f}, max={max_weight:.6f}, avg={avg_weight:.6f}"
        )
        
        # Log detailed weight information (at debug level to avoid spam)
        for uid, weight, node in zip(uids, weights_list, serving_miners):
            ema_score = miner_ema_scores.get(uid, 0.0)
            last_success = last_success_times.get(uid)
            freshness_str = f", last_success={last_success}" if last_success else ", no_success_recorded"
            logger.debug(
                f"[set_weights]   Node {uid} ({node.hotkey[:16]}...): "
                f"ema_score={ema_score:.6f}, weight={weight:.6f}{freshness_str}"
            )
        
        # 12. Check rate limit
        try:
            blocks_since_update = subtensor.blocks_since_last_update(
                netuid=config.netuid,
                uid=validator_uid
            )
            weights_rate_limit = subtensor.weights_rate_limit(netuid=config.netuid)
            
            if blocks_since_update < weights_rate_limit:
                logger.warning(
                    f"[set_weights] Cannot set weights yet - rate limit not met. "
                    f"Blocks since update: {blocks_since_update}, required: {weights_rate_limit}. "
                    f"Skipping this cycle."
                )
                return
            
            logger.info(
                f"[set_weights] Rate limit check passed: {blocks_since_update} blocks since update "
                f"(required: {weights_rate_limit})"
            )
        except Exception as e:
            logger.warning(f"[set_weights] Could not check rate limit: {e}. Proceeding with weight setting.")
        
        # 13. Set weights using Bittensor SDK
        logger.info(
            f"[set_weights] Submitting weights for {len(uids)} miners via Bittensor SDK "
            f"(handles CRv4 automatically)..."
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
                f"[set_weights] SUCCESS: Set weights on chain for {miners_with_weight} miners "
                f"with non-zero weight out of {len(uids)} total (netuid={config.netuid})"
            )
        else:
            logger.error(
                f"[set_weights] FAILED: Could not set weights on chain (netuid={config.netuid}). "
                "Check chain connection and validator registration."
            )
            raise RuntimeError("Failed to set weights on chain")
        
    except Exception as e:
        logger.error(f"[set_weights] Error setting weights: {str(e)}")
        logger.exception("[set_weights] Full error traceback:")
        raise
