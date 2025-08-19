import socket
import subprocess
from typing import Optional
from ipaddress import ip_address, IPv4Address, IPv6Address

from ..logging_setup import get_logger


logger = get_logger(__name__)


def resolve_hostname_to_ip(hostname: str) -> Optional[str]:
    """
    Resolve a hostname to its IP address.
    
    Args:
        hostname: The hostname to resolve
        
    Returns:
        IP address as string, or None if resolution failed
    """
    try:
        # Use socket.gethostbyname for IPv4
        ip = socket.gethostbyname(hostname)
        logger.debug(f"Resolved {hostname} to {ip}")
        return ip
    except socket.gaierror as e:
        logger.warning(f"Failed to resolve hostname {hostname}: {e}")
        return None


def resolve_hostname_all_ips(hostname: str) -> list[str]:
    """
    Resolve a hostname to all its IP addresses.
    
    Args:
        hostname: The hostname to resolve
        
    Returns:
        List of IP addresses as strings
    """
    try:
        # Get all addresses for the hostname
        addr_info = socket.getaddrinfo(hostname, None)
        ips = []
        
        for family, type, proto, canonname, sockaddr in addr_info:
            ip = sockaddr[0]
            if ip not in ips:
                ips.append(ip)
        
        logger.debug(f"Resolved {hostname} to {len(ips)} addresses: {ips}")
        return ips
        
    except socket.gaierror as e:
        logger.warning(f"Failed to resolve hostname {hostname}: {e}")
        return []


def is_valid_ip(address: str) -> bool:
    """
    Check if a string is a valid IP address (IPv4 or IPv6).
    
    Args:
        address: String to check
        
    Returns:
        True if valid IP address
    """
    try:
        ip_address(address)
        return True
    except ValueError:
        return False


def is_ipv4(address: str) -> bool:
    """Check if string is a valid IPv4 address."""
    try:
        IPv4Address(address)
        return True
    except ValueError:
        return False


def is_ipv6(address: str) -> bool:
    """Check if string is a valid IPv6 address."""
    try:
        IPv6Address(address)
        return True
    except ValueError:
        return False


def get_local_ip() -> Optional[str]:
    """
    Get the local machine's primary IP address.
    
    Returns:
        IP address as string, or None if not found
    """
    try:
        # Connect to a remote address to determine local IP
        # This doesn't actually send data
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            local_ip = s.getsockname()[0]
            logger.debug(f"Detected local IP: {local_ip}")
            return local_ip
    except Exception as e:
        logger.warning(f"Failed to detect local IP: {e}")
        return None


def get_hostname() -> Optional[str]:
    """Get the local machine's hostname."""
    try:
        hostname = socket.gethostname()
        logger.debug(f"Local hostname: {hostname}")
        return hostname
    except Exception as e:
        logger.warning(f"Failed to get hostname: {e}")
        return None


def get_fqdn() -> Optional[str]:
    """Get the fully qualified domain name."""
    try:
        fqdn = socket.getfqdn()
        logger.debug(f"FQDN: {fqdn}")
        return fqdn
    except Exception as e:
        logger.warning(f"Failed to get FQDN: {e}")
        return None


def ping_host(host: str, timeout: int = 5) -> bool:
    """
    Check if a host is reachable via ping.
    
    Args:
        host: Hostname or IP address to ping
        timeout: Timeout in seconds
        
    Returns:
        True if host is reachable
    """
    try:
        # Use ping command (works on most Unix systems)
        cmd = ["ping", "-c", "1", "-W", str(timeout), host]
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            timeout=timeout + 5
        )
        
        reachable = result.returncode == 0
        logger.debug(f"Ping {host}: {'SUCCESS' if reachable else 'FAILED'}")
        return reachable
        
    except subprocess.TimeoutExpired:
        logger.debug(f"Ping {host}: TIMEOUT")
        return False
    except Exception as e:
        logger.debug(f"Ping {host} failed: {e}")
        return False


def test_tcp_connection(host: str, port: int, timeout: int = 5) -> bool:
    """
    Test if a TCP connection can be established to host:port.
    
    Args:
        host: Hostname or IP address
        port: Port number
        timeout: Connection timeout in seconds
        
    Returns:
        True if connection successful
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(timeout)
            result = sock.connect_ex((host, port))
            success = result == 0
            logger.debug(f"TCP connection to {host}:{port}: {'SUCCESS' if success else 'FAILED'}")
            return success
    except Exception as e:
        logger.debug(f"TCP connection to {host}:{port} failed: {e}")
        return False

# Prevent pytest from collecting this helper as a test when imported into test modules
test_tcp_connection.__test__ = False


def resolve_slurm_node_ip(nodename: str) -> Optional[str]:
    """
    Resolve a SLURM node name to its IP address.
    This function handles common SLURM node naming patterns.
    
    Args:
        nodename: SLURM node name (e.g., "cn001", "gpu-node-01")
        
    Returns:
        IP address as string, or None if resolution failed
    """
    if not nodename:
        return None
    
    # First try direct hostname resolution
    ip = resolve_hostname_to_ip(nodename)
    if ip:
        return ip
    
    # Try with common domain suffixes for cluster nodes
    common_suffixes = [
        ".cluster",
        ".cluster.local", 
        ".internal",
        ".local",
        ".compute"
    ]
    
    for suffix in common_suffixes:
        try_hostname = f"{nodename}{suffix}"
        ip = resolve_hostname_to_ip(try_hostname)
        if ip:
            logger.debug(f"Resolved SLURM node {nodename} using suffix {suffix}")
            return ip
    
    logger.warning(f"Failed to resolve SLURM node: {nodename}")
    return None


def get_interface_ip(interface_name: str) -> Optional[str]:
    """
    Get IP address of a specific network interface.
    
    Args:
        interface_name: Name of network interface (e.g., "eth0", "ib0")
        
    Returns:
        IP address as string, or None if not found
    """
    try:
        # Use ip command to get interface address
        cmd = ["ip", "addr", "show", interface_name]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        
        if result.returncode != 0:
            return None
        
        # Parse output to find inet address
        lines = result.stdout.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('inet ') and not line.startswith('inet 127.'):
                # Extract IP from "inet 192.168.1.100/24 ..."
                parts = line.split()
                if len(parts) >= 2:
                    ip_with_mask = parts[1]
                    ip = ip_with_mask.split('/')[0]
                    if is_valid_ip(ip):
                        logger.debug(f"Interface {interface_name} has IP: {ip}")
                        return ip
        
        return None
        
    except Exception as e:
        logger.debug(f"Failed to get IP for interface {interface_name}: {e}")
        return None
