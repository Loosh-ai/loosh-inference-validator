import asyncio
import time
import random
from datetime import datetime
from typing import List

import httpx
from loguru import logger

from fiber.chain.models import Node
from fiber.validator.client import construct_server_address
from validator.db.operations import DatabaseManager

# Maximum number of miners to select
# TODO: make this configurable
MAX_MINERS = 3


async def check_miner_availability(
    node: Node,
    client: httpx.AsyncClient,
    db_manager: DatabaseManager,
    hotkey: str,
) -> bool:
    """
    Check if a miner is available and log the result.
    
    Args:
        node: Node object containing miner information
        client: httpx.AsyncClient object for making HTTP requests
        db_manager: DatabaseManager object for logging availability checks
        hotkey: Hotkey of the validator
    Returns:
        bool: True if the miner is available, False otherwise
    """
    server_address = construct_server_address(node)
    start_time = time.time()
    
    try:
        # Request the availability of the miner
        headers = {"validator-hotkey": hotkey}
        response = await client.get(f"{server_address}/availability", headers=headers, timeout=5.0)
        response_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        
        # Parse the response and check if the miner is available
        is_available = response.json().get("available", False)
        
        # Log the availability check result
        db_manager.log_miner(
            node_id=node.node_id, 
            hotkey=node.hotkey, 
            ip=node.ip, 
            port=node.port, 
            stake=node.stake,
            is_available=is_available,
            last_success=datetime.utcnow() if is_available else None,
            error=None
        )
        # Return the availability of the miner
        return is_available
    except Exception as e:
        # Log the availability check result with error information
        error_msg = f"{type(e).__name__}: {e}"
        # Log the error message
        logger.warning(f"Failed to check availability for node {node.node_id}: {error_msg}")
        # Log the availability check result with error information
        db_manager.log_miner(
            node_id=node.node_id, 
            hotkey=node.hotkey, 
            ip=node.ip, 
            port=node.port, 
            stake=node.stake,
            is_available=False,
            last_success=None,
            error=error_msg
        )
        
        return False


async def get_available_nodes(
    nodes: List[Node],
    client: httpx.AsyncClient,
    db_manager: DatabaseManager,
    hotkey: str
) -> List[Node]:
    """Check availability of all nodes and return available ones up to MAX_MINERS."""
    # First check availability for all nodes
    logger.info(f"Checking availability for all {len(nodes)} nodes")
    availability_tasks = [
        check_miner_availability(node, client, db_manager, hotkey)
        for node in nodes
    ]
    
    availability_results = await asyncio.gather(*availability_tasks)
    
    # Filter available nodes
    available_nodes = [
        node for node, is_available in zip(nodes, availability_results)
        if is_available
    ]
    
    total_available = len(available_nodes)
    logger.info(f"Found {total_available} available nodes out of {len(nodes)} total nodes")
    
    # If we have more available nodes than MAX_MINERS, randomly select MAX_MINERS
    selected_nodes = available_nodes
    if total_available > MAX_MINERS:
        # Randomly select MAX_MINERS nodes from the available nodes
        logger.info(f"Randomly selecting {MAX_MINERS} nodes from {total_available} available nodes")
        selected_nodes = random.sample(available_nodes, MAX_MINERS)
    else:
        logger.info(f"Using all {total_available} available nodes (less than MAX_MINERS={MAX_MINERS})")
    
    # Log selected nodes
    for node in selected_nodes:
        logger.info(f"Selected node {node.node_id} (hotkey: {node.hotkey})")
    
    return selected_nodes
