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

from validator.challenge_api.get_next_challenge import get_next_challenge_with_retry, get_next_challenge_with_retry2
from validator.challenge_api.update_challenge_response import process_challenge_results
from validator.challenge_api.models import Challenge as ChallengeAPIRequest
from validator.challenge_api.models import ChallengeResponse as ChallengeAPIResponse
from validator.endpoints.challenges import get_next_challenge as get_next_challenge_from_queue
from validator.endpoints.challenges import ChallengeCreate

from validator.db.operations import DatabaseManager

from fiber.chain.models import Node
from fiber.validator.client import construct_server_address
from fiber.chain.chain_utils import load_hotkey_keypair, load_coldkeypub_keypair

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
    import os
    from pathlib import Path
    
    wallet_path = Path(os.path.expanduser("~/.bittensor/wallets")) / config.wallet_name
    hotkey_path = wallet_path / "hotkeys" / config.hotkey_name
    coldkey_path = wallet_path / "coldkey"
    
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
            f"\nTo create the wallet, run inside the container:\n"
            f"  docker exec -it loosh-inference-subnet-validator btcli wallet new_coldkey \\\n"
            f"    --wallet.name {config.wallet_name} \\\n"
            f"    --wallet.path /root/.bittensor/wallets \\\n"
            f"    --no-use-password --n_words 24\n"
            f"\n  docker exec -it loosh-inference-subnet-validator btcli wallet new_hotkey \\\n"
            f"    --wallet.name {config.wallet_name} \\\n"
            f"    --wallet.path /root/.bittensor/wallets \\\n"
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

    nodes = []
    nodes.append(Node(
        node_id=1,
        hotkey=hotkey.ss58_address,
        coldkey=coldkey.ss58_address,
        ip="127.0.0.1",
        port=8081,
        stake=100,
        incentive=0.0,
        netuid=config.netuid,
        alpha_stake=0.0,
        tao_stake=0.0,
        trust=0.0,
        vtrust=0.0,
        ip_type=4,
        last_updated=time.time()
    ))
    # nodes.append(Node(
    #     node_id=2,
    #     hotkey=hotkey.ss58_address,
    #     coldkey=coldkey.ss58_address,
    #     ip="127.0.0.1",
    #     port=8082,
    #     stake=100,
    #     incentive=0.0,
    #     netuid=21,
    #     alpha_stake=0.0,
    #     tao_stake=0.0,
    #     trust=0.0,
    #     vtrust=0.0,
    #     ip_type=4,
    #     last_updated=time.time()
    # ))

    # Concurrency control
    max_concurrent = config.max_concurrent_challenges
    challenge_semaphore = asyncio.Semaphore(max_concurrent)
    logger.info(f"Concurrency limit: {max_concurrent} concurrent challenges")

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

    async with httpx.AsyncClient(timeout=config.challenge_timeout.total_seconds()) as client:

        async def get_available_nodes_cached() -> List[Node]:
            """Get available nodes with error handling."""
            try:
                return await get_available_nodes(nodes, client, db_manager, hotkey.ss58_address)
            except Exception as e:
                logger.warning(f"Error checking miner availability: {str(e)}. Continuing anyway...")
                return []
        
        # Initial check
        available_nodes = await get_available_nodes_cached()
        num_available = len(available_nodes)
        logger.info(f"available_nodes {available_nodes} num_available {num_available}")

        async def process_challenge(challenge_data: ChallengeAPIResponse) -> None:
            """
            Process a single challenge and submit response when ready (completion order).
            
            This function processes challenges concurrently and submits responses
            as soon as they're ready, not in arrival order.
            """
            async with challenge_semaphore:
                try:
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

                    # Refresh available nodes (in case they changed)
                    current_available_nodes = await get_available_nodes_cached()
                    if not current_available_nodes:
                        logger.warning(f"No available nodes for challenge {challenge_id[:8]}..., skipping")
                        return
                    
                    # Create new challenge tasks for available nodes
                    new_challenge_tasks = []
                    
                    for node in current_available_nodes:
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
                            challenge_task = await send_challenge(
                                client=client,
                                server_address=construct_server_address(node),
                                challenge_id=challenge_id,
                                challenge_orig=challenge_orig,
                                challenge=challenge,
                                miner_hotkey=node.hotkey,
                                db_manager=db_manager,
                                node_id=node.node_id
                            )
                            
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
                        test_mode=getattr(config, 'test_mode', False)
                    )
                    
                    logger.info(f"Completed processing challenge {challenge_id[:8]}...")
                    
                except Exception as e:
                    logger.error(f"Error processing challenge: {str(e)}", exc_info=True)
        
        async def challenge_consumer_loop():
            """Continuously consume challenges and process them concurrently."""
            logger.info("Starting challenge consumer loop")
            while True:
                try:
                    # Fetch next challenge based on mode (push or pull)
                    challenge_data = None
                    
                    if config.challenge_mode == "push":
                        # Use queue-based consumption
                        challenge_from_queue = await get_next_challenge_from_queue(timeout=1.0)
                        if challenge_from_queue:
                            # Convert ChallengeCreate to ChallengeAPIResponse format
                            challenge_data = convert_challenge_create_to_api_response(challenge_from_queue)
                    else:
                        # Use pull mechanism (existing code)
                        challenge_data = await get_next_challenge_with_retry2(config, hotkey)
                        if challenge_data:
                            # Sleep briefly to avoid tight polling loop
                            await asyncio.sleep(0.1)
                    
                    if challenge_data:
                        # Process in background (don't await - allows concurrent processing)
                        asyncio.create_task(process_challenge(challenge_data))
                        logger.debug(f"Started processing challenge {challenge_data.id[:8]}... (concurrent)")
                    else:
                        # No challenge available, sleep briefly
                        await asyncio.sleep(1.0)
                        
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    logger.error(f"Error in challenge consumer loop: {str(e)}")
                    await asyncio.sleep(1.0)
        
        try:
            # Start challenge consumer loop
            await challenge_consumer_loop()
        finally:
            pass

def run_main_loop():
    try:
        asyncio.run(main_loop())
    except KeyboardInterrupt:
        sys.exit(0)

if __name__ == "__main__":
    run_main_loop()

