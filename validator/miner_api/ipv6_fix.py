"""
IPv6 address handling utilities.

This module provides utilities to properly handle IPv6 addresses in URLs,
which is not currently supported in Fiber's construct_server_address function.
"""

from fiber.chain.models import Node


def construct_server_address_with_ipv6(
    node: Node,
    replace_with_docker_localhost: bool = False,
    replace_with_localhost: bool = False,
) -> str:
    """
    Construct server address with proper IPv6 support.
    
    IPv6 addresses must be wrapped in square brackets in URLs to distinguish
    the address colons from the port separator.
    
    Examples:
        IPv4: http://192.168.1.1:8000
        IPv6: http://[2001:db8::1]:8000
    
    Args:
        node: Node object with ip and port
        replace_with_docker_localhost: Use host.docker.internal (for Docker)
        replace_with_localhost: Use localhost (for local development)
    
    Returns:
        Properly formatted URL string
    """
    # Handle special case for local development
    if node.ip == "0.0.0.1":
        if replace_with_docker_localhost:
            return f"http://host.docker.internal:{node.port}"
        elif replace_with_localhost:
            return f"http://localhost:{node.port}"
    
    # Check if IP is IPv6 (contains colons)
    # IPv6 addresses must be wrapped in square brackets in URLs
    if ':' in str(node.ip):
        # IPv6 address - wrap in brackets
        return f"http://[{node.ip}]:{node.port}"
    else:
        # IPv4 address - use directly
        return f"http://{node.ip}:{node.port}"
