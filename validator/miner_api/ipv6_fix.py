"""
IPv6 address handling utilities.

This module provides utilities to properly handle IPv6 addresses in URLs,
which is not currently supported in Fiber's construct_server_address function.
"""

import socket

from fiber.chain.models import Node


def normalize_node_ip_to_address(node: Node) -> str:
    """
    Convert node.ip from chain format to a proper address string.

    On chain, IPv4 is stored as a packed 32-bit integer. When read via Fiber
    it can appear as a digit-string (e.g. "1113376410") or int. ip_type can be
    int 4/6 or string "4"/"6". This normalizes to dotted decimal (IPv4) or
    IPv6 string for URL construction.
    """
    ip_str = str(node.ip) if node.ip is not None else ""
    ip_type = int(node.ip_type) if node.ip_type is not None else 4
    if ip_str and ip_str.isdigit():
        try:
            ip_int = int(ip_str)
            if ip_type == 4:
                return socket.inet_ntoa(ip_int.to_bytes(4, byteorder="big"))
            if ip_type == 6:
                return socket.inet_ntop(
                    socket.AF_INET6, ip_int.to_bytes(16, byteorder="big")
                )
        except (ValueError, OverflowError, OSError):
            pass
    return ip_str


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
