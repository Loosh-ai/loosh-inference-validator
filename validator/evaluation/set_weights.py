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
from typing import List

from bittensor import Subtensor, Wallet
from fiber.chain.fetch_nodes import get_nodes_for_netuid
from fiber.chain.interface import get_substrate
from loguru import logger

from validator.db.operations import DatabaseManager
from validator.config import get_validator_config


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
        loop = asyncio.get_event_loop()
        
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
                logger.info("✅ Successfully set weights on chain via Bittensor SDK")
                return True
            else:
                error_msg = getattr(response, 'message', 'Unknown error')
                logger.error(f"❌ Failed to set weights: {error_msg}")
                return False
        else:
            # Fallback for older SDK versions that return bool
            if response:
                logger.info("✅ Successfully set weights on chain via Bittensor SDK")
                return True
            else:
                logger.error("❌ Failed to set weights on chain")
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
    ema_lookback_hours: int = 24,
    ema_alpha: float = 0.3
) -> None:
    """
    Set weights for miners based on their EMA performance scores.
    
    This function:
    1. Connects to the Bittensor chain via SDK (for weight setting)
    2. Uses Fiber to fetch registered nodes on the subnet
    3. Calculates EMA scores from evaluation emissions in the database
    4. Normalizes weights to sum to 1.0
    5. Sets weights on-chain via Bittensor SDK (handles CRv4 automatically)
    
    Args:
        db_manager: DatabaseManager instance for querying scores
        ema_lookback_hours: Hours of history to consider for EMA (default: 24)
        ema_alpha: EMA smoothing factor (default: 0.3)
    
    Raises:
        Exception: If weight setting fails (caller should handle retries)
    """
    config = get_validator_config()
    
    try:
        # 1. Initialize Bittensor SDK Subtensor connection
        # Map network names to SDK expected values
        network = config.subtensor_network
        chain_endpoint = config.subtensor_address if config.subtensor_address else None
        
        logger.info(
            f"Connecting to chain via Bittensor SDK: network={network}, "
            f"endpoint={chain_endpoint}"
        )
        
        # Create Subtensor instance
        # For custom endpoints, use the endpoint directly
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
        logger.info(f"Loaded validator wallet: {hotkey_ss58}")
        
        # 3. Get validator UID from chain
        validator_uid = subtensor.get_uid_for_hotkey_on_subnet(
            hotkey_ss58=hotkey_ss58,
            netuid=config.netuid
        )
        
        if validator_uid is None:
            logger.error(
                f"Validator {hotkey_ss58} is not registered on subnet {config.netuid}. "
                "Cannot set weights without registration."
            )
            raise ValueError(f"Validator not registered on subnet {config.netuid}")
        
        logger.info(f"Validator UID: {validator_uid}")
        
        # 4. Get version key from chain (for logging/info purposes)
        # SDK handles this internally during set_weights
        try:
            version_key = subtensor.substrate.query(
                "SubtensorModule", "WeightsVersionKey", [config.netuid]
            ).value
            logger.info(f"Subnet WeightsVersionKey: {version_key}")
        except Exception as e:
            logger.warning(f"Could not query WeightsVersionKey: {e}")
            version_key = None
        
        # 5. Get all registered nodes using Fiber (this still works fine)
        # Create a substrate connection for Fiber's node fetching
        substrate = get_substrate(
            subtensor_network=config.subtensor_network,
            subtensor_address=config.subtensor_address
        )
        nodes = get_nodes_for_netuid(substrate=substrate, netuid=config.netuid)
        
        if not nodes:
            logger.warning(f"No nodes found on subnet {config.netuid}. Skipping weight setting.")
            return
        
        logger.info(f"Found {len(nodes)} registered nodes on subnet {config.netuid}")
        
        # 6. Get EMA scores from database
        miner_ema_scores = db_manager.get_miner_ema_scores(
            lookback_hours=ema_lookback_hours,
            alpha=ema_alpha
        )
        logger.info(f"Retrieved EMA scores for {len(miner_ema_scores)} miners")
        
        # 7. Build weight vectors - must include ALL nodes on subnet
        uids: List[int] = []
        weights_list: List[float] = []
        
        for node in nodes:
            node_id = node.node_id
            # Get score for this node, default to 0.0 if not found
            score = miner_ema_scores.get(node_id, 0.0)
            
            uids.append(node_id)
            weights_list.append(score)
        
        # 8. Normalize weights to sum to 1.0
        total_weight = sum(weights_list)
        if total_weight > 0:
            weights_list = [w / total_weight for w in weights_list]
        else:
            # If no scores at all, distribute evenly
            logger.warning(
                "No EMA scores available for any miners. Distributing weights evenly."
            )
            weights_list = [1.0 / len(nodes)] * len(nodes)
        
        # Log detailed weight information
        logger.info(f"Setting weights for {len(nodes)} nodes:")
        for uid, weight, node in zip(uids, weights_list, nodes):
            ema_score = miner_ema_scores.get(uid, 0.0)
            logger.info(
                f"  Node {uid} ({node.hotkey[:16]}...): "
                f"ema_score={ema_score:.6f}, weight={weight:.6f}"
            )
        
        # 9. Check rate limit (using Subtensor)
        # The SDK's set_weights handles this internally, but we can pre-check
        try:
            blocks_since_update = subtensor.blocks_since_last_update(
                netuid=config.netuid,
                uid=validator_uid
            )
            weights_rate_limit = subtensor.weights_rate_limit(netuid=config.netuid)
            
            if blocks_since_update < weights_rate_limit:
                logger.warning(
                    f"Cannot set weights yet - rate limit not met. "
                    f"Blocks since update: {blocks_since_update}, required: {weights_rate_limit}. "
                    f"Skipping this cycle."
                )
                return
            
            logger.info(
                f"Rate limit check passed: {blocks_since_update} blocks since update "
                f"(required: {weights_rate_limit})"
            )
        except Exception as e:
            # If we can't check rate limit, proceed anyway - SDK will handle the error
            logger.warning(f"Could not check rate limit: {e}. Proceeding with weight setting.")
        
        # 10. Set weights using Bittensor SDK
        # SDK handles:
        # - Weight normalization and quantization
        # - CRv4 commit-reveal if enabled on subnet
        # - Version key validation
        logger.info("Submitting weights via Bittensor SDK (handles CRv4 automatically)...")
        
        success = await _set_weights_with_sdk(
            subtensor=subtensor,
            wallet=wallet,
            netuid=config.netuid,
            uids=uids,
            weights=weights_list
        )
        
        if success:
            logger.info(
                f"Successfully set weights on chain for {len(uids)} nodes "
                f"(netuid={config.netuid})"
            )
        else:
            logger.error(
                f"Failed to set weights on chain (netuid={config.netuid}). "
                "Check chain connection and validator registration."
            )
            raise RuntimeError("Failed to set weights on chain")
        
    except Exception as e:
        logger.error(f"Error setting weights: {str(e)}")
        logger.exception("Full error traceback:")
        raise
