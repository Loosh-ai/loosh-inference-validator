"""
Background worker process for checking miner availability.

This process runs independently to avoid blocking the main validator loop.
"""

import asyncio
import multiprocessing
import os
import socket
import time
import random
from datetime import datetime
from typing import List, Dict, Optional
from urllib.parse import urlparse
import json
import sys

import httpx
from loguru import logger

from fiber.chain.models import Node
from fiber.validator.client import construct_server_address
from validator.db.operations import DatabaseManager
from validator.miner_api.ipv6_fix import construct_server_address_with_ipv6

# Maximum concurrent availability checks
MAX_CONCURRENT_AVAILABILITY_CHECKS = 20


def node_to_dict(node: Node) -> dict:
    """Convert Node to dictionary for IPC."""
    # Convert IP from integer to dotted decimal if needed
    ip_str = str(node.ip) if node.ip is not None else ''
    ip_type = int(node.ip_type) if node.ip_type is not None else 4
    
    # If IP is an integer (numeric string), convert to dotted decimal
    if ip_str and ip_str.isdigit():
        try:
            ip_int = int(ip_str)
            if ip_type == 4:  # IPv4
                ip_str = socket.inet_ntoa(ip_int.to_bytes(4, byteorder='big'))
            elif ip_type == 6:  # IPv6
                ip_str = socket.inet_ntop(socket.AF_INET6, ip_int.to_bytes(16, byteorder='big'))
        except (ValueError, OverflowError, OSError):
            # Conversion failed, keep original
            pass
    
    return {
        'hotkey': str(node.hotkey) if node.hotkey else None,
        'coldkey': str(node.coldkey) if node.coldkey else None,
        'node_id': int(node.node_id) if node.node_id is not None else None,
        'incentive': float(node.incentive) if node.incentive is not None else 0.0,
        'netuid': int(node.netuid) if node.netuid is not None else None,
        'alpha_stake': float(node.alpha_stake) if node.alpha_stake is not None else 0.0,
        'tao_stake': float(node.tao_stake) if node.tao_stake is not None else 0.0,
        'stake': float(node.stake) if node.stake is not None else 0.0,
        'trust': float(node.trust) if node.trust is not None else 0.0,
        'vtrust': float(node.vtrust) if node.vtrust is not None else 0.0,
        'last_updated': float(node.last_updated) if node.last_updated is not None else 0.0,
        'ip': ip_str,
        'ip_type': ip_type,
        'port': int(node.port) if node.port is not None else 0,
        'protocol': int(node.protocol) if node.protocol is not None else 0
    }


def dict_to_node(node_dict: dict) -> Node:
    """Convert dictionary to Node for IPC."""
    return Node(
        hotkey=node_dict['hotkey'],
        coldkey=node_dict['coldkey'],
        node_id=node_dict['node_id'],
        incentive=node_dict['incentive'],
        netuid=node_dict['netuid'],
        alpha_stake=node_dict['alpha_stake'],
        tao_stake=node_dict['tao_stake'],
        stake=node_dict['stake'],
        trust=node_dict['trust'],
        vtrust=node_dict['vtrust'],
        last_updated=node_dict['last_updated'],
        ip=node_dict['ip'],
        ip_type=node_dict['ip_type'],
        port=node_dict['port'],
        protocol=node_dict['protocol']
    )


async def check_miner_availability_async(
    node: Node,
    client: httpx.AsyncClient,
    db_manager: DatabaseManager,
    hotkey: str,
) -> bool:
    """
    Check if a miner is available (async version for worker process).
    
    This is a copy of the check_miner_availability function but designed
    to run in a separate process.
    """
    # IP should already be converted to dotted decimal during serialization
    # But handle both cases: already converted (dotted decimal) or still integer (string)
    ip_str = str(node.ip) if node.ip is not None else ''
    
    # If IP is still an integer (numeric string), convert to dotted decimal
    if ip_str and ip_str.isdigit():
        try:
            ip_int = int(ip_str)
            if node.ip_type == 4:  # IPv4
                ip_str = socket.inet_ntoa(ip_int.to_bytes(4, byteorder='big'))
            elif node.ip_type == 6:  # IPv6
                ip_str = socket.inet_ntop(socket.AF_INET6, ip_int.to_bytes(16, byteorder='big'))
        except (ValueError, OverflowError, OSError):
            # Conversion failed, keep original
            pass
    elif ip_str and not (ip_str.count('.') == 3 or ':' in ip_str):
        logger.warning(f"IP '{ip_str}' doesn't look like a valid IP address (ip_type={node.ip_type})")
    
    # Create a temporary node with converted IP
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
    
    replace_localhost = ip_str in ("0.0.0.1", "127.0.0.1", "0.0.0.0")
    
    # Skip nodes with invalid ports (0 or negative)
    if node.port <= 0:
        # logger.debug(f"Skipping node {node.node_id}: invalid port {node.port}")  # Commented out - too noisy
        return False
    
    try:
        server_address = construct_server_address_with_ipv6(node_with_ip, replace_with_localhost=replace_localhost)
    except Exception as e:
        logger.warning(f"Failed to construct server address for node {node.node_id}: {e}")
        db_manager.log_miner(
            node_id=node.node_id, 
            hotkey=node.hotkey, 
            ip=node.ip, 
            port=node.port, 
            stake=node.stake,
            is_available=False,
            last_success=None,
            error=f"Invalid server address: {e}"
        )
        return False
    
    # Validate the constructed URL
    try:
        parsed = urlparse(server_address)
        if not parsed.scheme or not parsed.netloc:
            raise ValueError(f"Invalid URL format: {server_address}")
    except Exception as e:
        logger.warning(f"Invalid server address for node {node.node_id}: {server_address}, error: {e}")
        db_manager.log_miner(
            node_id=node.node_id, 
            hotkey=node.hotkey, 
            ip=node.ip, 
            port=node.port, 
            stake=node.stake,
            is_available=False,
            last_success=None,
            error=f"Invalid URL: {e}"
        )
        return False
    
    availability_url = f"{server_address}/availability"
    start_time = time.time()
    
    try:
        headers = {"validator-hotkey": hotkey}
        availability_timeout = 3.0
        response = await client.get(availability_url, headers=headers, timeout=availability_timeout)
        response_time = (time.time() - start_time) * 1000
        
        if response.status_code != 200:
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
        
        try:
            response_data = response.json()
        except Exception as e:
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
        return is_available
        
    except httpx.HTTPStatusError as e:
        error_msg = f"HTTP {e.response.status_code}: {e.response.text[:200]}"
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
    except (httpx.PoolTimeout, httpx.ConnectTimeout, httpx.ReadTimeout, httpx.RequestError) as e:
        error_msg = f"{type(e).__name__}: {e}"
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
        error_msg = f"{type(e).__name__}: {e}"
        logger.warning(f"Error checking availability for node {node.node_id}: {error_msg}")
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


async def check_nodes_async(
    nodes: List[Node],
    hotkey: str,
    max_concurrent: int,
    db_path: str,
    max_miners: int = 3
) -> List[Node]:
    """
    Check availability of all nodes asynchronously.
    
    Args:
        nodes: List of nodes to check
        hotkey: Validator hotkey
        max_concurrent: Maximum concurrent checks
        db_path: Path to database file
        max_miners: Maximum number of miners to select
    
    Returns:
        List of available nodes
    """
    if not nodes:
        return []
    
    logger.info(f"Worker: Checking availability for {len(nodes)} nodes (max {max_concurrent} concurrent)")
    
    # Initialize database manager in this process
    db_manager = DatabaseManager(db_path)
    
    # Create HTTP client
    pool_size = max(100, max_concurrent * 2)
    async with httpx.AsyncClient(
        timeout=3.0,
        limits=httpx.Limits(
            max_connections=pool_size,
            max_keepalive_connections=pool_size // 2
        )
    ) as client:
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def check_with_semaphore(node: Node) -> bool:
            async with semaphore:
                return await check_miner_availability_async(node, client, db_manager, hotkey)
        
        availability_tasks = [check_with_semaphore(node) for node in nodes]
        availability_results = await asyncio.gather(*availability_tasks, return_exceptions=True)
        
        for i, result in enumerate(availability_results):
            if isinstance(result, Exception):
                logger.error(f"Exception checking availability for node {nodes[i].node_id}: {result}")
                availability_results[i] = False
        
        available_nodes = [
            node for node, is_available in zip(nodes, availability_results)
            if is_available
        ]
        
        total_available = len(available_nodes)
        logger.info(f"Worker: Found {total_available} available nodes out of {len(nodes)} total nodes")
        
        # Select up to max_miners nodes
        if total_available > max_miners:
            selected_nodes = random.sample(available_nodes, max_miners)
            logger.info(f"Worker: Randomly selected {max_miners} nodes from {total_available} available")
        else:
            selected_nodes = available_nodes
            logger.info(f"Worker: Using all {total_available} available nodes (less than max_miners={max_miners})")
        
        return selected_nodes


def worker_process(
    nodes_queue: multiprocessing.Queue,
    results_queue: multiprocessing.Queue,
    hotkey: str,
    max_concurrent: int,
    db_path: str,
    check_interval: float,
    max_miners: int = 3
):
    """
    Worker process that continuously checks miner availability.
    
    Args:
        nodes_queue: Queue to receive node lists to check
        results_queue: Queue to send results back
        hotkey: Validator hotkey
        max_concurrent: Maximum concurrent checks
        db_path: Path to database file
        check_interval: How often to check (seconds)
        max_miners: Maximum number of miners to select
    """
    logger.info(f"Worker process started (PID: {os.getpid()})")
    
    current_nodes: List[Node] = []
    last_check_time = 0
    
    while True:
        try:
            # Check for new nodes from queue (non-blocking)
            try:
                nodes_dict_list = nodes_queue.get_nowait()
                if nodes_dict_list:
                    current_nodes = [dict_to_node(n) for n in nodes_dict_list]
                    logger.info(f"Worker: Received {len(current_nodes)} nodes to check")
            except:
                pass  # No new nodes, use current list
            
            # Check if it's time to run availability check
            now = time.time()
            if current_nodes and (now - last_check_time) >= check_interval:
                logger.info(f"Worker: Starting availability check for {len(current_nodes)} nodes")
                
                # Run async check
                available_nodes = asyncio.run(
                    check_nodes_async(current_nodes, hotkey, max_concurrent, db_path, max_miners)
                )
                
                # Send results back
                results_dict_list = [node_to_dict(n) for n in available_nodes]
                results_queue.put(results_dict_list)
                
                logger.info(f"Worker: Sent {len(available_nodes)} available nodes to main process")
                last_check_time = now
            
            # Sleep briefly to avoid busy-waiting
            time.sleep(0.1)
            
        except KeyboardInterrupt:
            logger.info("Worker process received interrupt, shutting down")
            break
        except Exception as e:
            logger.error(f"Error in worker process: {e}", exc_info=True)
            time.sleep(1)  # Brief pause before retrying


class AvailabilityWorker:
    """
    Manager for the background availability checking process.
    """
    
    def __init__(
        self,
        hotkey: str,
        max_concurrent: int = MAX_CONCURRENT_AVAILABILITY_CHECKS,
        db_path: str = "validator.db",
        check_interval: float = 30.0,
        max_miners: int = 3
    ):
        """
        Initialize availability worker.
        
        Args:
            hotkey: Validator hotkey
            max_concurrent: Maximum concurrent checks
            db_path: Path to database file
            check_interval: How often to check availability (seconds)
            max_miners: Maximum number of miners to select
        """
        self.hotkey = hotkey
        self.max_concurrent = max_concurrent
        self.db_path = db_path
        self.check_interval = check_interval
        self.max_miners = max_miners
        
        self.nodes_queue: Optional[multiprocessing.Queue] = None
        self.results_queue: Optional[multiprocessing.Queue] = None
        self.process: Optional[multiprocessing.Process] = None
        self._running = False
        
        # Cache of last known available nodes
        self._available_nodes: List[Node] = []
        self._last_update = 0
    
    def start(self):
        """Start the background worker process."""
        if self._running:
            logger.warning("Availability worker already running")
            return
        
        self.nodes_queue = multiprocessing.Queue()
        self.results_queue = multiprocessing.Queue()
        
        self.process = multiprocessing.Process(
            target=worker_process,
            args=(
                self.nodes_queue,
                self.results_queue,
                self.hotkey,
                self.max_concurrent,
                self.db_path,
                self.check_interval,
                self.max_miners
            ),
            daemon=True
        )
        
        self.process.start()
        self._running = True
        logger.info(f"Started availability worker process (PID: {self.process.pid})")
    
    def stop(self):
        """Stop the background worker process."""
        if not self._running:
            return
        
        if self.process:
            self.process.terminate()
            self.process.join(timeout=5)
            if self.process.is_alive():
                logger.warning("Worker process didn't terminate gracefully, killing")
                self.process.kill()
                self.process.join()
        
        self._running = False
        logger.info("Stopped availability worker process")
    
    def update_nodes(self, nodes: List[Node]):
        """
        Update the list of nodes to check (non-blocking).
        
        Args:
            nodes: List of nodes to check
        """
        if not self._running:
            logger.warning("Cannot update nodes: worker not running")
            return
        
        try:
            nodes_dict_list = [node_to_dict(n) for n in nodes]
            self.nodes_queue.put_nowait(nodes_dict_list)
        except Exception as e:
            logger.warning(f"Failed to update nodes in worker (queue full): {e}")
    
    def get_available_nodes(self) -> List[Node]:
        """
        Get currently available nodes (non-blocking, returns cached results).
        
        Returns:
            List of available nodes
        """
        # Check for new results (non-blocking)
        try:
            while True:
                results_dict_list = self.results_queue.get_nowait()
                self._available_nodes = [dict_to_node(n) for n in results_dict_list]
                self._last_update = time.time()
                logger.info(f"Updated available nodes cache: {len(self._available_nodes)} nodes")
        except:
            pass  # No new results
        
        return self._available_nodes.copy()
    
    @property
    def is_running(self) -> bool:
        """Check if worker is running."""
        return self._running and self.process and self.process.is_alive()
