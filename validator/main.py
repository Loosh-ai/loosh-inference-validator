# VALIDATOR MAIN

import asyncio
import sys
import time
import random
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any

import httpx
from loguru import logger

from validator.challenge.challenge_types import InferenceChallenge
from validator.challenge.send_challenge import send_challenge

from validator.miner_api.check_miner_availability import check_miner_availability, get_available_nodes
from validator.miner_api.availability_worker import AvailabilityWorker

# Pull-based challenge fetching removed - only push mode (Fiber) is supported
from validator.challenge_api.update_challenge_response import process_challenge_results
from validator.challenge_api.models import Challenge as ChallengeAPIRequest
from validator.challenge_api.models import ChallengeResponse as ChallengeAPIResponse
from validator.endpoints.challenges import get_next_challenge as get_next_challenge_from_queue
from validator.endpoints.challenges import ChallengeCreate

from validator.db.operations import DatabaseManager
from validator.evaluation.sybil_sync import SybilSyncTask
from validator.validator_list_fetcher import ValidatorListFetcher
from validator.evaluation.evaluation import InferenceValidator

from fiber.chain.models import Node
from fiber.validator.client import construct_server_address
from fiber.chain.chain_utils import load_hotkey_keypair, load_coldkeypub_keypair
from fiber.chain.fetch_nodes import get_nodes_for_netuid
from fiber.chain.interface import get_substrate

from pathlib import Path
from datetime import timedelta

from validator.config import get_validator_config


def convert_challenge_create_to_api_response(challenge_create: ChallengeCreate) -> ChallengeAPIResponse:
    """
    Convert ChallengeCreate (from push queue) to ChallengeAPIResponse format.
    
    Args:
        challenge_create: Challenge from push queue
        
    Returns:
        ChallengeAPIResponse compatible with existing processing logic
    """
    return ChallengeAPIResponse(
        id=challenge_create.id,
        correlation_id=challenge_create.correlation_id,
        prompt=challenge_create.prompt,
        temperature=challenge_create.temperature,
        top_p=challenge_create.top_p,
        max_tokens=challenge_create.max_tokens,
        metadata=challenge_create.metadata,
        created_at=challenge_create.created_at or datetime.now(timezone.utc),
        status=challenge_create.status,
        requester=challenge_create.requester,
        response_time_ms=0  # Not applicable for pushed challenges
    )


async def main_loop():
    """Main validator loop."""
    # Load configuration
    try:
        config = get_validator_config()
    except ValueError as e:
        # Configuration error already logged with helpful message
        logger.error("Validator startup aborted due to configuration error")
        return
    except Exception as e:
        error_msg = (
            f"\n{'='*70}\n"
            f"CONFIGURATION ERROR: Failed to load validator configuration\n"
            f"{'='*70}\n"
            f"Error: {str(e)}\n"
            f"\nPlease check your environment variables or .env file.\n"
            f"See environments/env.validator.example for valid configuration options.\n"
            f"{'='*70}\n"
        )
        logger.error(error_msg)
        return
    
    logger.info(f"Starting validator with configuration:")
    logger.info(f"Network: {config.subtensor_network}")
    logger.info(f"Subnet: {config.netuid}")
    logger.info(f"Wallet: {config.wallet_name}")
    logger.info(f"Hotkey: {config.hotkey_name}")
    logger.info(f"Challenge Interval: {config.challenge_interval}")
    logger.info(f"Challenge Timeout: {config.challenge_timeout}")
    logger.info(f"Challenge API URL: {config.challenge_api_url}")
    logger.info(f"Challenge API Key: {'*' * (len(config.challenge_api_key) - 4) + config.challenge_api_key[-4:] if len(config.challenge_api_key) > 4 else '***'}")
    test_mode = getattr(config, 'test_mode', False)
    if test_mode:
        logger.info("[TEST MODE] Test mode enabled - will pick first response without evaluation or heatmap generation")
    
    # Load validator keys
    # Note: Fiber only supports wallets in ~/.bittensor/wallets
    from pathlib import Path
    
    wallet_path = Path.home() / ".bittensor" / "wallets" / config.wallet_name
    hotkey_path = wallet_path / "hotkeys" / config.hotkey_name
    coldkey_path = wallet_path / "coldkey"
    
    logger.info(f"Wallet path: {wallet_path}")
    logger.info(f"Expected hotkey path: {hotkey_path}")
    logger.info(f"Expected coldkey path: {coldkey_path}")
    
    # Verify the paths exist before trying to load
    if not hotkey_path.exists():
        logger.error(f"Hotkey file not found at: {hotkey_path}")
        logger.error(f"Please ensure the hotkey file exists at the expected path")
    if not coldkey_path.exists():
        logger.error(f"Coldkey file not found at: {coldkey_path}")
        logger.error(f"Please ensure the coldkey file exists at the expected path")
    
    try:
        hotkey = load_hotkey_keypair(config.wallet_name, config.hotkey_name)
        coldkey = load_coldkeypub_keypair(config.wallet_name)
    except FileNotFoundError as e:
        error_msg = (
            f"\n{'='*70}\n"
            f"VALIDATOR STARTUP FAILED: Wallet not found\n"
            f"{'='*70}\n"
            f"The validator requires a valid Bittensor wallet to start.\n"
            f"\nWallet: {config.wallet_name}\n"
            f"Hotkey: {config.hotkey_name}\n"
            f"Expected paths:\n"
            f"  - Hotkey: {hotkey_path}\n"
            f"  - Coldkey: {coldkey_path}\n"
            f"\nTo create the wallet, run:\n"
            f"  btcli wallet new_coldkey \\\n"
            f"    --wallet.name {config.wallet_name} \\\n"
            f"    --no-use-password --n_words 24\n"
            f"\n  btcli wallet new_hotkey \\\n"
            f"    --wallet.name {config.wallet_name} \\\n"
            f"    --hotkey {config.hotkey_name} \\\n"
            f"    --no-use-password --n_words 24\n"
            f"\nThen restart the container:\n"
            f"  docker compose restart validator\n"
            f"{'='*70}\n"
        )
        logger.error(error_msg)
        return
    except Exception as e:
        error_msg = (
            f"\n{'='*70}\n"
            f"VALIDATOR STARTUP FAILED: Wallet error\n"
            f"{'='*70}\n"
            f"Error: {str(e)}\n"
            f"\nPlease ensure wallet files are properly configured.\n"
            f"Wallet: {config.wallet_name}, Hotkey: {config.hotkey_name}\n"
            f"Note: Fiber only supports wallets in ~/.bittensor/wallets\n"
            f"Expected paths:\n"
            f"  - Hotkey: {hotkey_path}\n"
            f"  - Coldkey: {coldkey_path}\n"
            f"{'='*70}\n"
        )
        logger.error(error_msg)
        return

    logger.info(f"Loaded hotkey: {hotkey.ss58_address}")

    # Initialize database
    try:
        db_manager = DatabaseManager(config.db_path)
        logger.info(f"Database initialized: {config.db_path}")
    except Exception as e:
        error_msg = (
            f"\n{'='*70}\n"
            f"DATABASE ERROR: Failed to initialize database\n"
            f"{'='*70}\n"
            f"Database path: {config.db_path}\n"
            f"Error: {str(e)}\n"
            f"\nThis is usually caused by:\n"
            f"  - Invalid database path\n"
            f"  - Insufficient permissions to create/write database file\n"
            f"  - Disk space issues\n"
            f"\nPlease check:\n"
            f"  - DB_PATH environment variable is set correctly\n"
            f"  - The directory exists and is writable\n"
            f"  - Sufficient disk space is available\n"
            f"{'='*70}\n"
        )
        logger.error(error_msg)
        return

    # Query Fiber chain for registered nodes
    logger.info(f"Querying Fiber chain for registered nodes (netuid={config.netuid}, network={config.subtensor_network})")
    try:
        substrate = get_substrate(subtensor_network=config.subtensor_network)
        nodes = get_nodes_for_netuid(substrate=substrate, netuid=config.netuid)
        logger.info(f"Found {len(nodes)} registered nodes on chain")
        if len(nodes) == 0:
            logger.warning(
                f"No nodes found on chain for netuid={config.netuid}. "
                "Miners need to register using: fiber-post-ip --netuid <NETUID> --subtensor.network <NETWORK> "
                "--external_port <PORT> --wallet.name <WALLET> --wallet.hotkey <HOTKEY> --external_ip <IP>"
            )
    except Exception as e:
        logger.error(f"Failed to query Fiber chain for nodes: {e}", exc_info=True)
        logger.warning("Falling back to empty node list. Validator will not be able to find miners.")
        nodes = []

    # Concurrency control
    max_concurrent = config.max_concurrent_challenges
    challenge_semaphore = asyncio.Semaphore(max_concurrent)
    logger.info(f"Concurrency limit: {max_concurrent} concurrent challenges")
    
    # Pending challenges queue for when no nodes are available (FIFO)
    pending_challenges_queue: asyncio.Queue = asyncio.Queue()

    # Validate Challenge API configuration
    if not config.challenge_api_url or not config.challenge_api_key:
        error_msg = (
            f"\n{'='*70}\n"
            f"CONFIGURATION ERROR: Challenge API not configured\n"
            f"{'='*70}\n"
            f"The validator requires Challenge API configuration to operate.\n"
            f"\nCurrent values:\n"
            f"  - CHALLENGE_API_URL: {config.challenge_api_url or 'NOT SET'}\n"
            f"  - CHALLENGE_API_KEY: {'SET' if config.challenge_api_key else 'NOT SET'}\n"
            f"\nPlease set these environment variables:\n"
            f"  - CHALLENGE_API_URL (e.g., http://challenge-api:8080)\n"
            f"  - CHALLENGE_API_KEY (your API key)\n"
            f"\nSee environments/env.validator.example for configuration options.\n"
            f"{'='*70}\n"
        )
        logger.error(error_msg)
        return

    # Initialize availability worker (non-blocking background process)
    availability_worker = AvailabilityWorker(
        hotkey=hotkey.ss58_address,
        max_concurrent=config.max_concurrent_availability_checks,
        db_path=config.db_path,
        check_interval=30.0,  # Check every 30 seconds
        max_miners=config.max_miners
    )
    availability_worker.start()
    
    # Update worker with initial node list
    availability_worker.update_nodes(nodes)
    logger.info(f"Initialized availability worker with {len(nodes)} nodes")
    
    # Get initial available nodes (may be empty until first check completes)
    available_nodes = availability_worker.get_available_nodes()
    num_available = len(available_nodes)
    logger.info(f"Initial available nodes: {num_available} (worker will update in background)")
    
    # Initialize inference validator for evaluation
    inference_validator = None
    if not test_mode:
        try:
            inference_validator = InferenceValidator(db_manager=db_manager)
            logger.info("InferenceValidator initialized - evaluation and heatmap generation enabled")
        except Exception as e:
            logger.error(f"Failed to initialize InferenceValidator: {e}. Evaluation will be disabled.", exc_info=True)
            inference_validator = None
    else:
        logger.info("[TEST MODE] InferenceValidator not initialized - evaluation disabled")
    
    # Initialize validator list fetcher (to filter out validators from miner list)
    validator_list_fetcher = None
    try:
        validator_list_fetcher = ValidatorListFetcher(
            challenge_api_url=config.challenge_api_url,
            challenge_api_key=config.challenge_api_key,
            refresh_interval_seconds=300.0  # Refresh every 5 minutes
        )
        await validator_list_fetcher.start()
        logger.info("Validator list fetcher started (will periodically fetch validators from Challenge API)")
    except Exception as e:
        logger.warning(f"Failed to start validator list fetcher: {e}. Continuing without validator filtering.")
        validator_list_fetcher = None
    
    # Initialize sybil sync task (background task to send records to Challenge API)
    sybil_sync_task = None
    try:
        sybil_sync_task = SybilSyncTask(
            db_manager=db_manager,
            validator_hotkey_ss58=hotkey.ss58_address,
            sync_interval_seconds=60.0,  # Sync every 60 seconds
            batch_size=10  # Send up to 10 records per batch
        )
        await sybil_sync_task.start()
        logger.info("Sybil sync task started (will periodically send records to Challenge API)")
    except Exception as e:
        logger.warning(f"Failed to start sybil sync task: {e}. Continuing without sybil sync.")
        sybil_sync_task = None
    
    try:
        # Create httpx client for challenge sending
        pool_size = max(100, config.max_concurrent_availability_checks * 2)
        async with httpx.AsyncClient(
            timeout=config.challenge_timeout.total_seconds(),
            limits=httpx.Limits(
                max_connections=pool_size, 
                max_keepalive_connections=pool_size // 2
            )
        ) as client:

            def get_available_nodes_cached() -> List[Node]:
                """Get available nodes from worker (non-blocking, returns cached results)."""
                return availability_worker.get_available_nodes()

            async def process_challenge(challenge_data: ChallengeAPIResponse) -> None:
                """
                Process a single challenge and submit response when ready (completion order).
                
                This function processes challenges concurrently and submits responses
                as soon as they're ready, not in arrival order.
                """
                from validator.timing import PipelineTiming, PipelineStages
                
                async with challenge_semaphore:
                    try:
                        # Load timing data from challenge metadata
                        pipeline_timing = None
                        correlation_id = getattr(challenge_data, 'correlation_id', None) or challenge_data.id
                        try:
                            if challenge_data.metadata and 'timing_data' in challenge_data.metadata:
                                timing_data = challenge_data.metadata.get('timing_data')
                                if isinstance(timing_data, dict):
                                    pipeline_timing = PipelineTiming.from_dict(timing_data)
                        except Exception as e:
                            logger.debug(f"Could not load timing data for {correlation_id}: {e}")
                        
                        if pipeline_timing:
                            # Add validator receive stage
                            pipeline_timing.add_stage(PipelineStages.VALIDATOR_RECEIVE)
                        
                        default_model = config.default_model
                        challenge_id = challenge_data.id
                        prompt = challenge_data.prompt
                        
                        model = challenge_data.metadata.get("model", default_model) if challenge_data.metadata else "default_model"
                        model = default_model
                        
                        max_tokens = challenge_data.max_tokens
                        temperature = challenge_data.temperature
                        top_p = challenge_data.top_p

                        challenge_orig = ChallengeAPIRequest(
                            id=challenge_data.id,
                            correlation_id=getattr(challenge_data, 'correlation_id', None),
                            prompt=challenge_data.prompt,
                            temperature=challenge_data.temperature,
                            top_p=challenge_data.top_p,
                            max_tokens=challenge_data.max_tokens,
                            metadata=challenge_data.metadata,
                            created_at=challenge_data.created_at,
                            status=challenge_data.status,
                            requester=challenge_data.requester
                        )

                        # Get available nodes from worker (non-blocking)
                        current_available_nodes = get_available_nodes_cached()
                        
                        # Filter out validators (including self)
                        validator_hotkey = hotkey.ss58_address
                        filtered_nodes = []
                        filtered_count = 0
                        
                        if current_available_nodes:
                            for node in current_available_nodes:
                                # Skip validator's own node
                                if node.hotkey == validator_hotkey:
                                    filtered_count += 1
                                    continue
                                
                                # Skip if node is a known validator
                                if validator_list_fetcher and validator_list_fetcher.is_validator(node.hotkey):
                                    filtered_count += 1
                                    continue
                                
                                filtered_nodes.append(node)
                        
                        # If no filtered nodes available, queue challenge and wait
                        if not filtered_nodes:
                            logger.info(
                                f"No available nodes for challenge {challenge_id[:8]}... "
                                f"(excluding {filtered_count} validator(s)). "
                                f"Queuing for FIFO processing when nodes become available..."
                            )
                            # Queue challenge and wait for nodes to become available
                            await pending_challenges_queue.put(challenge_data)
                            
                            # Wait for nodes to become available (check every 2 seconds)
                            max_wait_time = 300  # 5 minutes max wait
                            wait_start = time.time()
                            while (time.time() - wait_start) < max_wait_time:
                                await asyncio.sleep(2.0)  # Check every 2 seconds
                                
                                # Check again for available nodes
                                current_available_nodes = get_available_nodes_cached()
                                if current_available_nodes:
                                    # Re-filter nodes
                                    filtered_nodes = []
                                    filtered_count = 0
                                    for node in current_available_nodes:
                                        if node.hotkey == validator_hotkey:
                                            filtered_count += 1
                                            continue
                                        if validator_list_fetcher and validator_list_fetcher.is_validator(node.hotkey):
                                            filtered_count += 1
                                            continue
                                        filtered_nodes.append(node)
                                    
                                    if filtered_nodes:
                                        logger.info(
                                            f"Nodes now available for queued challenge {challenge_id[:8]}... "
                                            f"({len(filtered_nodes)} nodes after filtering)"
                                        )
                                        break  # Nodes available, proceed with processing
                            
                            # If still no nodes after waiting, log warning but continue to try processing
                            if not filtered_nodes:
                                logger.warning(
                                    f"Still no available nodes for challenge {challenge_id[:8]}... "
                                    f"after waiting {max_wait_time}s. Will retry processing anyway."
                                )
                                # Don't return - continue to try processing (may fail gracefully)
                                # This ensures challenges are never permanently skipped
                        
                        if filtered_count > 0:
                            logger.debug(
                                f"Filtered out {filtered_count} validator node(s) "
                                f"(including self: {validator_hotkey[:8]}...) "
                                f"from {len(current_available_nodes)} available nodes"
                            )
                        
                        # Create new challenge tasks for available nodes
                        new_challenge_tasks = []
                        
                        for node in filtered_nodes:
                            # Convert IP from integer to dotted decimal if needed
                            import socket
                            ip_str = node.ip
                            try:
                                # Check if IP looks like an integer (numeric string) - convert to IP address
                                ip_int = int(ip_str)
                                if node.ip_type == 4:  # IPv4
                                    ip_str = socket.inet_ntoa(ip_int.to_bytes(4, byteorder='big'))
                                elif node.ip_type == 6:  # IPv6
                                    ip_str = socket.inet_ntop(socket.AF_INET6, ip_int.to_bytes(16, byteorder='big'))
                            except (ValueError, OverflowError):
                                # IP is already a string, use as is
                                pass
                            except Exception as e:
                                logger.warning(f"Error converting IP for node {node.node_id}: {e}")
                            
                            # Create node with converted IP for construct_server_address
                            from fiber.chain.models import Node
                            node_with_ip = Node(
                                hotkey=node.hotkey, coldkey=node.coldkey, node_id=node.node_id,
                                incentive=node.incentive, netuid=node.netuid, alpha_stake=node.alpha_stake,
                                tao_stake=node.tao_stake, stake=node.stake, trust=node.trust,
                                vtrust=node.vtrust, last_updated=node.last_updated,
                                ip=ip_str, ip_type=node.ip_type, port=node.port, protocol=node.protocol
                            )
                            
                            # Create challenge
                            challenge = InferenceChallenge(
                                prompt=prompt,
                                model=model,
                                max_tokens=max_tokens,
                                temperature=temperature,
                                top_p=top_p,
                                correlation_id=getattr(challenge_data, 'correlation_id', None)
                            )
                            
                            # Send challenge to miner
                            try:
                                # Track timing: validator send to miner
                                if pipeline_timing:
                                    validator_send_stage = pipeline_timing.add_stage(PipelineStages.VALIDATOR_SEND_TO_MINER)
                                
                                challenge_task = await send_challenge(
                                    client=client,
                                    server_address=construct_server_address(node_with_ip, replace_with_localhost=True),
                                    challenge_id=challenge_id,
                                    challenge_orig=challenge_orig,
                                    challenge=challenge,
                                    miner_hotkey=node.hotkey,
                                    db_manager=db_manager,
                                    node_id=node.node_id,
                                    config=config,
                                    validator_hotkey_ss58=hotkey.ss58_address,
                                    pipeline_timing=pipeline_timing
                                )
                                
                                if pipeline_timing and validator_send_stage:
                                    validator_send_stage.finish()
                                
                                new_challenge_tasks.append(challenge_task)
                                logger.info(f"Challenge sent successfully to node {node.node_id}")
                                    
                            except Exception as e:
                                logger.error(f"Error sending challenge to node {node.node_id}: {str(e)}")
                        
                        # Process challenge results (submits response when ready - completion order)
                        await process_challenge_results(
                            new_challenge_tasks,
                            client,
                            api_key=config.challenge_api_key,
                            validator_hotkey=hotkey.ss58_address,
                            server_address=config.challenge_api_url,
                            node_id=0,
                            test_mode=getattr(config, 'test_mode', False),
                            validator=inference_validator,
                            challenge_prompt=prompt,
                            challenge_id=challenge_id,
                            pipeline_timing=pipeline_timing
                        )
                        
                        logger.info(f"Completed processing challenge {challenge_id[:8]}...")
                        
                    except Exception as e:
                        logger.error(f"Error processing challenge: {str(e)}", exc_info=True)
            
            async def challenge_consumer_loop():
                """Continuously consume challenges from queue and process them concurrently."""
                logger.info("Starting challenge consumer loop (push mode only - Fiber encrypted challenges)")
                
                # Track active challenge processing tasks
                active_tasks = set()
                
                # Export capacity tracking for health endpoint
                from validator.endpoints.availability import set_capacity_tracking
                set_capacity_tracking(active_tasks, challenge_semaphore, max_concurrent)
                
                # Background task to process pending challenges when nodes become available
                async def process_pending_challenges():
                    """Process pending challenges from queue when nodes become available (FIFO)."""
                    while True:
                        try:
                            # Wait for a pending challenge (with timeout to check if still running)
                            try:
                                pending_challenge = await asyncio.wait_for(
                                    pending_challenges_queue.get(),
                                    timeout=5.0  # Check every 5 seconds
                                )
                            except asyncio.TimeoutError:
                                continue  # No pending challenges, continue loop
                            
                            # Check if nodes are available now
                            current_available_nodes = get_available_nodes_cached()
                            if current_available_nodes:
                                # Re-filter nodes
                                validator_hotkey = hotkey.ss58_address
                                filtered_nodes = []
                                for node in current_available_nodes:
                                    if node.hotkey == validator_hotkey:
                                        continue
                                    if validator_list_fetcher and validator_list_fetcher.is_validator(node.hotkey):
                                        continue
                                    filtered_nodes.append(node)
                                
                                if filtered_nodes:
                                    # Nodes available, process the challenge
                                    logger.info(
                                        f"Processing queued challenge {pending_challenge.id[:8]}... "
                                        f"({len(filtered_nodes)} nodes now available)"
                                    )
                                    task = asyncio.create_task(process_challenge(pending_challenge))
                                    active_tasks.add(task)
                                    task.add_done_callback(active_tasks.discard)
                                else:
                                    # Still no filtered nodes, put back in queue
                                    await pending_challenges_queue.put(pending_challenge)
                            else:
                                # Still no nodes, put back in queue
                                await pending_challenges_queue.put(pending_challenge)
                            
                            # Small delay to prevent tight loop
                            await asyncio.sleep(0.5)
                        except Exception as e:
                            logger.error(f"Error in pending challenges processor: {e}", exc_info=True)
                            await asyncio.sleep(1.0)
                
                # Start pending challenges processor
                pending_task = asyncio.create_task(process_pending_challenges())
                
                while True:
                    try:
                        # Only use queue-based consumption (push mode)
                        # Challenges are received via Fiber-encrypted POST to /fiber/challenge endpoint
                        challenge_from_queue = await get_next_challenge_from_queue(timeout=1.0)
                        
                        if challenge_from_queue:
                            # Convert ChallengeCreate to ChallengeAPIResponse format
                            challenge_data = convert_challenge_create_to_api_response(challenge_from_queue)
                            
                            # Process in background (don't await - allows concurrent processing)
                            task = asyncio.create_task(process_challenge(challenge_data))
                            active_tasks.add(task)
                            
                            # Remove task from set when it completes
                            task.add_done_callback(active_tasks.discard)
                            
                            logger.debug(f"Started processing challenge {challenge_data.id[:8]}... (concurrent, active: {len(active_tasks)})")
                        else:
                            # No challenge available, but check if there are still active tasks
                            if active_tasks:
                                logger.debug(f"No new challenges, but {len(active_tasks)} challenge(s) still processing...")
                            # Sleep briefly before checking again
                            await asyncio.sleep(1.0)
                            
                    except KeyboardInterrupt:
                        # Cancel pending challenges processor
                        pending_task.cancel()
                        try:
                            await pending_task
                        except asyncio.CancelledError:
                            pass
                        
                        # Wait for active tasks to complete before exiting
                        if active_tasks:
                            logger.info(f"Waiting for {len(active_tasks)} active challenge(s) to complete...")
                            await asyncio.gather(*active_tasks, return_exceptions=True)
                        break
                    except Exception as e:
                        logger.error(f"Error in challenge consumer loop: {str(e)}")
                        await asyncio.sleep(1.0)
            
            # Start challenge consumer loop
            await challenge_consumer_loop()
    finally:
        # Clean up background tasks
        logger.info("Shutting down availability worker...")
        availability_worker.stop()
        
        if validator_list_fetcher:
            logger.info("Shutting down validator list fetcher...")
            try:
                await validator_list_fetcher.stop()
            except Exception as e:
                logger.warning(f"Error stopping validator list fetcher: {e}")
        
        if sybil_sync_task:
            logger.info("Shutting down sybil sync task...")
            try:
                await sybil_sync_task.stop()
            except Exception as e:
                logger.warning(f"Error stopping sybil sync task: {e}")

def run_main_loop():
    try:
        asyncio.run(main_loop())
    except KeyboardInterrupt:
        sys.exit(0)

if __name__ == "__main__":
    run_main_loop()

