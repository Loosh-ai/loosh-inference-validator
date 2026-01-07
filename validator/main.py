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

from validator.db.operations import DatabaseManager

from fiber.chain.models import Node
from fiber.validator.client import construct_server_address
from fiber.chain.chain_utils import load_hotkey_keypair, load_coldkeypub_keypair

from pathlib import Path
from datetime import timedelta

from validator.config import get_validator_config

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

    active_challenge_tasks = []  # Track active challenge tasks

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

        # Check availability of nodes
        try:
            available_nodes = await get_available_nodes(nodes, client, db_manager, hotkey.ss58_address)
            num_available = len(available_nodes)
            logger.info(f"available_nodes {available_nodes} num_available {num_available}")
        except Exception as e:
            logger.warning(f"Error checking miner availability: {str(e)}. Continuing anyway...")
            available_nodes = []
            num_available = 0

        try:
            # Main challenge loop
            #iteration = 0
            while True:
                try:

                    # GET ChallengeAPIResponse [

                    # Fetch next challenge from API with retries
                    challenge_data = await get_next_challenge_with_retry2(config, hotkey)

                    if not challenge_data:
                        logger.info(f"Sleeping for {config.challenge_interval.total_seconds()} seconds before next challenge check...")
                        await asyncio.sleep(config.challenge_interval.total_seconds())
                        continue

                    # GET ChallengeAPIResponse ]
                    # CREATE ChallengeAPIRequest [

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

                    # CREATE ChallengeAPIRequest ]
                    # PROCESS THE CHALLENGE [

                    # Create new challenge tasks for available nodes
                    new_challenge_tasks = []
                    
                    for node in available_nodes:
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
                    
                    # Add new challenges to active tasks
                    active_challenge_tasks.extend(new_challenge_tasks)

                    # Process any completed challenges
                    await process_challenge_results(
                        new_challenge_tasks,
                        client,
                        api_key=config.challenge_api_key,
                        validator_hotkey=hotkey.ss58_address,
                        server_address=config.challenge_api_url,
                        node_id=0,
                        test_mode=getattr(config, 'test_mode', False)
                    )

                    # Log status
                    num_active_challenges = len(active_challenge_tasks)
                    if num_active_challenges > 0:
                        logger.info(f"Currently tracking {num_active_challenges} active challenges")

                    # PROCESS THE CHALLENGE ]

                    # SLEEP

                    # Sleep until next challenge interval
                    await asyncio.sleep(config.challenge_interval.total_seconds())

                except KeyboardInterrupt:
                    break
                except Exception as e:
                    logger.error(f"Error in main loop: {str(e)}")
                    await asyncio.sleep(config.challenge_interval.total_seconds())
        finally:
            pass

def run_main_loop():
    try:
        asyncio.run(main_loop())
    except KeyboardInterrupt:
        sys.exit(0)

if __name__ == "__main__":
    run_main_loop()

