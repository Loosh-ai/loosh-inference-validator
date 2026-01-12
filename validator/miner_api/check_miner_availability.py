import asyncio
import socket
import time
import random
from datetime import datetime
from typing import List

import httpx
from loguru import logger

from fiber.chain.models import Node
from fiber.validator.client import construct_server_address
from validator.db.operations import DatabaseManager

# Note: MAX_MINERS is now passed as a parameter to get_available_nodes
# This constant is kept for backward compatibility but should not be used
_DEFAULT_MAX_MINERS = 3

# Maximum concurrent availability checks to prevent connection pool exhaustion
# This limits how many nodes we check in parallel
# Can be overridden via config parameter
MAX_CONCURRENT_AVAILABILITY_CHECKS = 20


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
    # Convert IP from integer to string if needed
    # Fiber stores IP as integer on chain, but fetch_nodes converts it with str() which
    # just stringifies the integer (e.g., "1680356206" instead of "100.40.51.110")
    # We need to convert the integer back to dotted decimal notation
    ip_str = node.ip
    try:
        # Check if IP looks like an integer (numeric string) - convert to IP address
        # Try to parse as integer
        ip_int = int(ip_str)
        # If it's a large number (> 255.255.255.255 = 4294967295), it's likely an integer IP
        # Convert integer IP to dotted decimal notation
        if node.ip_type == 4:  # IPv4
            # Convert 32-bit integer to 4 bytes (big-endian) then to dotted decimal
            ip_str = socket.inet_ntoa(ip_int.to_bytes(4, byteorder='big'))
            logger.debug(f"Converted IPv4 from integer {ip_int} to {ip_str}")
        else:  # IPv6
            # For IPv6, use socket.inet_ntop
            ip_str = socket.inet_ntop(socket.AF_INET6, ip_int.to_bytes(16, byteorder='big'))
            logger.debug(f"Converted IPv6 from integer {ip_int} to {ip_str}")
    except (ValueError, OverflowError, OSError):
        # IP is already a proper IP address string, use as-is
        # Or conversion failed, log and use original
        if not ip_str.count('.') == 3 and not ':' in ip_str:
            # Doesn't look like a valid IP, log warning
            logger.warning(f"IP '{ip_str}' doesn't look like a valid IP address (ip_type={node.ip_type})")
    
    # Create a temporary node with converted IP for construct_server_address
    # (construct_server_address uses node.ip directly)
    node_with_ip = Node(
        hotkey=node.hotkey,
        coldkey=node.coldkey,
        node_id=node.node_id,
        incentive=node.incentive,
        netuid=node.netuid,
        alpha_stake=node.alpha_stake,
        tao_stake=node.tao_stake,
        stake=node.stake,
        trust=node.trust,
        vtrust=node.vtrust,
        last_updated=node.last_updated,
        ip=ip_str,
        ip_type=node.ip_type,
        port=node.port,
        protocol=node.protocol
    )
    
    # Use replace_with_localhost=True for local development when chain returns 0.0.0.1
    # Also handle 127.0.0.1 and 0.0.0.0 for local development
    replace_localhost = ip_str in ("0.0.0.1", "127.0.0.1", "0.0.0.0")
    server_address = construct_server_address(node_with_ip, replace_with_localhost=replace_localhost)
    availability_url = f"{server_address}/availability"
    start_time = time.time()
    
    logger.info(
        f"Checking availability for node {node.node_id} (UID {node.node_id}): "
        f"hotkey={node.hotkey[:8]}..., ip={node.ip}, port={node.port}, "
        f"server_address={server_address}, url={availability_url}"
    )
    
    try:
        # Request the availability of the miner
        # Use shorter timeout for availability checks (should be quick)
        # This prevents slow/unresponsive miners from blocking the pool
        headers = {"validator-hotkey": hotkey}
        availability_timeout = 3.0  # 3 seconds should be enough for availability check
        response = await client.get(availability_url, headers=headers, timeout=availability_timeout)
        response_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        
        # Check if response is successful
        if response.status_code != 200:
            logger.warning(
                f"Non-200 status for node {node.node_id} ({server_address}): "
                f"status={response.status_code}, response={response.text[:200]}"
            )
            error_msg = f"HTTP {response.status_code}: {response.text[:200]}"
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
        
        logger.debug(
            f"Availability check for node {node.node_id}: "
            f"status={response.status_code}, response_time={response_time:.2f}ms"
        )
        
        # Parse the response and check if the miner is available
        try:
            response_data = response.json()
        except Exception as e:
            logger.warning(
                f"Failed to parse JSON response from node {node.node_id} "
                f"({server_address}): {e}. Response text: {response.text[:200]}"
            )
            error_msg = f"Invalid JSON response: {e}"
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
        
        is_available = response_data.get("available", False)
        
        logger.info(
            f"Node {node.node_id} (UID {node.node_id}, {server_address}): "
            f"available={is_available}, response={response_data}"
        )
        
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
    except httpx.HTTPStatusError as e:
        # HTTP error (4xx, 5xx)
        error_msg = f"HTTP {e.response.status_code}: {e.response.text[:200]}"
        logger.warning(
            f"HTTP error checking availability for node {node.node_id} "
            f"(UID {node.node_id}, {server_address}): {error_msg}"
        )
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
    except httpx.PoolTimeout as e:
        # Connection pool exhausted - too many concurrent requests
        error_msg = f"PoolTimeout: Connection pool exhausted (too many concurrent requests). Consider increasing pool size or reducing parallelism."
        logger.warning(
            f"Pool timeout checking availability for node {node.node_id} "
            f"(UID {node.node_id}, {server_address}): {error_msg}"
        )
    except httpx.ConnectTimeout as e:
        # Connection attempt timed out - server not responding
        error_msg = f"ConnectTimeout: Could not establish connection to {server_address} within timeout period. Server may be down or unreachable."
        logger.warning(
            f"Connection timeout checking availability for node {node.node_id} "
            f"(UID {node.node_id}, {server_address}): {error_msg}"
        )
    except httpx.ReadTimeout as e:
        # Read timeout - server connected but didn't respond in time
        error_msg = f"ReadTimeout: Server connected but did not respond within timeout period."
        logger.warning(
            f"Read timeout checking availability for node {node.node_id} "
            f"(UID {node.node_id}, {server_address}): {error_msg}"
        )
    except httpx.RequestError as e:
        # Other network/connection errors
        error_msg = f"{type(e).__name__}: {e}"
        logger.warning(
            f"Connection error checking availability for node {node.node_id} "
            f"(UID {node.node_id}, {server_address}): {error_msg}"
        )
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
    except Exception as e:
        # Other errors
        error_msg = f"{type(e).__name__}: {e}"
        logger.warning(
            f"Error checking availability for node {node.node_id} "
            f"(UID {node.node_id}, {server_address}): {error_msg}",
            exc_info=True
        )
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
    hotkey: str,
    max_concurrent: int = MAX_CONCURRENT_AVAILABILITY_CHECKS,
    max_miners: int = _DEFAULT_MAX_MINERS
) -> List[Node]:
    """
    Check availability of all nodes and return available ones up to max_miners.
    
    Uses a semaphore to limit concurrent checks and prevent connection pool exhaustion.
    
    Args:
        nodes: List of nodes to check
        client: HTTP client
        db_manager: Database manager
        hotkey: Validator hotkey
        max_concurrent: Maximum concurrent checks
        max_miners: Maximum number of miners to select
    """
    if not nodes:
        logger.info("No nodes to check for availability")
        return []
    
    logger.info(f"Checking availability for all {len(nodes)} nodes (max {max_concurrent} concurrent)")
    
    # Semaphore to limit concurrent availability checks
    # This prevents connection pool exhaustion when checking many nodes
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def check_with_semaphore(node: Node) -> bool:
        """Check availability with semaphore to limit concurrency."""
        async with semaphore:
            return await check_miner_availability(node, client, db_manager, hotkey)
    
    # Create tasks with semaphore-controlled concurrency
    availability_tasks = [
        check_with_semaphore(node)
        for node in nodes
    ]
    
    # Execute all tasks (semaphore will limit concurrent execution)
    availability_results = await asyncio.gather(*availability_tasks, return_exceptions=True)
    
    # Handle any exceptions that occurred
    for i, result in enumerate(availability_results):
        if isinstance(result, Exception):
            logger.error(f"Exception checking availability for node {nodes[i].node_id}: {result}")
            availability_results[i] = False
    
    # Filter available nodes
    available_nodes = [
        node for node, is_available in zip(nodes, availability_results)
        if is_available
    ]
    
    total_available = len(available_nodes)
    logger.info(f"Found {total_available} available nodes out of {len(nodes)} total nodes")
    
    # If we have more available nodes than max_miners, randomly select max_miners
    selected_nodes = available_nodes
    if total_available > max_miners:
        # Randomly select max_miners nodes from the available nodes
        logger.info(f"Randomly selecting {max_miners} nodes from {total_available} available nodes")
        selected_nodes = random.sample(available_nodes, max_miners)
    else:
        logger.info(f"Using all {total_available} available nodes (less than max_miners={max_miners})")
    
    # Log selected nodes
    for node in selected_nodes:
        logger.info(f"Selected node {node.node_id} (hotkey: {node.hotkey})")
    
    return selected_nodes
