#!/usr/bin/env python3
"""
Test Configuration for Loosh Inference Subnet
Demonstrates how to use bittensor config with test settings
"""

import bittensor as bt
import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional
from validator.config import get_validator_config
from validator.config.shared_config import get_validator_config as get_validator_config_from_shared


def load_test_config_yaml(config_path: str = "test_config.yaml") -> Dict[str, Any]:
    """Load test configuration from YAML file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Test config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_bittensor_test_config(yaml_config: Optional[Dict[str, Any]] = None) -> bt.config:
    """
    Create a bittensor config object with test settings.
    
    Note: This function is deprecated. The validator now uses ValidatorConfig
    and validator_config_to_bittensor_config() for configuration. This function
    is kept for backward compatibility with test code.
    """
    if yaml_config is None:
        yaml_config = load_test_config_yaml()
    
    # Create base bittensor config
    config = bt.config()
    
    config.subtensor = bt.subtensor.config()

    # Set basic network configuration
    config.netuid = yaml_config.get('netuid', 21)
    config.network = yaml_config.get('network', 'finney')

    config.subtensor.network = yaml_config.get('network', 'finney')
    config.subtensor.chain_endpoint = yaml_config.get('chain_endpoint', 'wss://entrypoint-finney.opentensor.ai:443')
    
    # Ensure nested subtensor config exists and is set
    if not hasattr(config.subtensor, 'subtensor'):
        config.subtensor.subtensor = bt.config()
    config.subtensor.subtensor.network = config.subtensor.network
    config.subtensor.subtensor.chain_endpoint = config.subtensor.chain_endpoint

    # Set network and chain endpoint directly on config
#    config.subtensor.network = yaml_config.get('network', 'finney')
#    config.subtensor.chain_endpoint = yaml_config.get('chain_endpoint', 'wss://entrypoint-finney.opentensor.ai:443')
    
    # Set wallet configuration
    wallet_config = yaml_config.get('wallet', {})
    config.wallet = bt.config()
    config.wallet.name = wallet_config.get('name', 'test_validator')
    config.wallet.hotkey = wallet_config.get('hotkey', 'test_validator')
    config.wallet.path = wallet_config.get('path', '~/.bittensor/wallets/')
    
    # Set subtensor configuration
    subtensor_config = yaml_config.get('subtensor', {})
    config.subtensor_timeout = subtensor_config.get('timeout', 30)
    config.subtensor_retry = subtensor_config.get('retry', 2)
    
    # Set axon configuration
    axon_config = yaml_config.get('axon', {})
    config.axon = bt.axon.config()
    config.axon.ip = axon_config.get('ip', '127.0.0.1')
    config.axon.port = axon_config.get('port', 8099)
    config.axon.external_ip = axon_config.get('external_ip')
    config.axon.external_port = axon_config.get('external_port')
    config.axon.max_workers = axon_config.get('max_workers', 5)
    config.axon.timeout = axon_config.get('timeout', 30)
    
    # Set dendrite configuration
    dendrite_config = yaml_config.get('dendrite', {})
    config.dendrite = bt.config()
    config.dendrite.timeout = dendrite_config.get('timeout', 30)
    config.dendrite.max_retry = dendrite_config.get('max_retry', 2)
    config.dendrite.retry_delay = dendrite_config.get('retry_delay', 0.5)
    
    # Set logging configuration
    logging_config = yaml_config.get('logging', {})
    config.log_level = logging_config.get('level', 'DEBUG')
    config.log_trace = logging_config.get('trace', True)
    config.log_record = logging_config.get('record_log', True)
    config.log_dir = logging_config.get('logging_dir', './logs/')
    
    return config


def create_test_validator_config(yaml_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Create test validator configuration from YAML settings."""
    if yaml_config is None:
        yaml_config = load_test_config_yaml()
    
    # Create test environment variables
    # NOTE: Operational parameters (miner selection, challenge timing, scoring, weights)
    # are now hard-coded in validator/internal_config.py and cannot be overridden via env.
    test_env = {
        'NETUID': str(yaml_config.get('netuid', 21)),
        'SUBTENSOR_NETWORK': yaml_config.get('network', 'finney'),
        'SUBTENSOR_ADDRESS': yaml_config.get('chain_endpoint', 'wss://entrypoint-finney.opentensor.ai:443'),
        'WALLET_NAME': yaml_config.get('wallet', {}).get('name', 'test_validator'),
        'HOTKEY_NAME': yaml_config.get('wallet', {}).get('hotkey', 'test_validator'),
        'DB_PATH': yaml_config.get('database', {}).get('path', './test_validator.db'),
        'USERS_DB_PATH': yaml_config.get('database', {}).get('users_path', './test_users.db'),
        'DEFAULT_MODEL': yaml_config.get('llm', {}).get('default_model', 'microsoft/Phi3-512'),
        'DEFAULT_MAX_TOKENS': str(yaml_config.get('llm', {}).get('default_max_tokens', 128)),
        'DEFAULT_TEMPERATURE': str(yaml_config.get('llm', {}).get('default_temperature', 0.7)),
        'DEFAULT_TOP_P': str(yaml_config.get('llm', {}).get('default_top_p', 0.95)),
        'HEATMAP_UPLOAD_URL': yaml_config.get('evaluation', {}).get('heatmap_upload_url', 'http://localhost:8080/upload'),
        'LLM_API_URL': yaml_config.get('evaluation', {}).get('llm_api_url', 'https://your-inference-endpoint/v1/chat/completions'),
        'LLM_MODEL': yaml_config.get('evaluation', {}).get('llm_model', 'microsoft/Phi3-512'),
        'CHALLENGE_API_URL': yaml_config.get('challenge_api', {}).get('url', 'http://localhost:8080'),
        'CHALLENGE_API_KEY': yaml_config.get('challenge_api', {}).get('key', 'test-api-key'),
        'API_HOST': yaml_config.get('api', {}).get('host', '127.0.0.1'),
        'API_PORT': str(yaml_config.get('api', {}).get('port', 8001)),
        'LOG_LEVEL': yaml_config.get('logging', {}).get('level', 'DEBUG'),
    }
    
    return test_env


def setup_test_environment():
    """Setup test environment with configuration."""
    # Load YAML config
    yaml_config = load_test_config_yaml()
    
    # Create bittensor config
    bt_config = create_bittensor_test_config(yaml_config)
    
    # Create test environment variables
    test_env = create_test_validator_config(yaml_config)
    
    # Set environment variables
    for key, value in test_env.items():
        os.environ[key] = value
    
    return bt_config, yaml_config, test_env


def test_bittensor_config():
    """Test function to demonstrate bittensor config usage."""
    print("Testing Bittensor Configuration...")
    
    try:
        # Setup test environment
        bt_config, yaml_config, test_env = setup_test_environment()
        
        print(f"✓ Bittensor config created successfully")
        print(f"  - NetUID: {bt_config.netuid}")
        print(f"  - Network: {bt_config.network}")
        print(f"  - Wallet: {bt_config.wallet_name}/{bt_config.hotkey_name}")
        print(f"  - Axon: {bt_config.axon_ip}:{bt_config.axon_port}")
        
        # Test validator config
        validator_config = get_validator_config()
        from validator.internal_config import INTERNAL_CONFIG
        print(f"✓ Validator config loaded successfully")
        print(f"  - NetUID: {validator_config.netuid}")
        print(f"  - Network: {validator_config.subtensor_network}")
        print(f"  - Wallet: {validator_config.wallet_name}/{validator_config.hotkey_name}")
        print(f"  - Challenge Interval: {INTERNAL_CONFIG.challenge_interval} (internal)")
        print(f"  - Default Model: {validator_config.default_model}")
        
        # Test test-specific settings
        test_settings = yaml_config.get('test', {})
        if test_settings.get('mock_mode'):
            print("✓ Mock mode enabled for testing")
        if test_settings.get('use_local_models'):
            print("✓ Using local models for testing")
        if test_settings.get('skip_network_checks'):
            print("✓ Network checks skipped for testing")
        
        print("\n✓ All tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {str(e)}")
        return False


def create_test_bittensor_node():
    """Create a test bittensor node with test configuration."""
    from validator.network.bittensor_node import BittensorNode
    
    # Setup test environment
    bt_config, yaml_config, test_env = setup_test_environment()
    
    # Create bittensor node with test config
    node = BittensorNode(config=bt_config)
    
    print(f"✓ Test BittensorNode created with config:")
    print(f"  - NetUID: {node.config.netuid}")
    print(f"  - Network: {node.config.subtensor.network}")
    print(f"  - Wallet: {node.config.wallet.name}/{node.config.wallet.hotkey}")
    
    return node


def create_loosh_subnet_subtensor(yaml_config: Optional[Dict[str, Any]] = None) -> 'LooshSubnetSubtensor':
    """Create a LooshSubnetSubtensor from test configuration."""
    try:
        from validator.network.axon import LooshSubnetSubtensor
    except ImportError as e:
        print(f"✗ Failed to import LooshSubnetSubtensor: {str(e)}")
        return None
    
    if yaml_config is None:
        yaml_config = load_test_config_yaml()
    
    # Create bittensor config from YAML
    bt_config = create_bittensor_test_config(yaml_config)
    
    # Create LooshSubnetSubtensor with the config
    try:
        # Try creating with network parameter explicitly
        network = yaml_config.get('network', 'finney')
        print(f"Creating LooshSubnetSubtensor with network: {network}")
        
        subtensor = LooshSubnetSubtensor(network=network, config=bt_config)
        
        print(f"✓ LooshSubnetSubtensor created with config:")
        print(f"  - NetUID: {bt_config.netuid}")
        print(f"  - Network: {bt_config.subtensor.network}")
        print(f"  - Chain Endpoint: {bt_config.subtensor.chain_endpoint}")
        print(f"  - Wallet: {bt_config.wallet.name}/{bt_config.wallet.hotkey}")
        
        return subtensor
    except Exception as e:
        print(f"✗ Failed to create LooshSubnetSubtensor: {str(e)}")
        return None


def create_loosh_subnet_subtensor_simple(config_path: str = "test_config.yaml") -> 'LooshSubnetSubtensor':
    """
    Simple function to create LooshSubnetSubtensor from test config.
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        LooshSubnetSubtensor: Configured subtensor instance
    """
    from validator.network.axon import LooshSubnetSubtensor
    
    # Load YAML config
    yaml_config = load_test_config_yaml(config_path)
    
    # Create bittensor config
    bt_config = create_bittensor_test_config(yaml_config)
    
    # Create LooshSubnetSubtensor
    network = yaml_config.get('network', 'finney')
    subtensor = LooshSubnetSubtensor(network=network, config=bt_config)
    
    return subtensor


def test_loosh_subnet_subtensor():
    """Test function to create LooshSubnetSubtensor from test config."""
    print("Testing LooshSubnetSubtensor Creation...")
    
    try:
        # Load YAML config
        yaml_config = load_test_config_yaml()
        print(f"✓ YAML config loaded: {yaml_config.get('netuid', 'N/A')}")
        
        # Create bittensor config
        bt_config = create_bittensor_test_config(yaml_config)
        print(f"✓ Bittensor config created: {bt_config.netuid}")
        print(f"  - Network: {bt_config.subtensor.network}")
        print(f"  - Chain Endpoint: {bt_config.subtensor.chain_endpoint}")
        print(f"  - Wallet: {bt_config.wallet.name}/{bt_config.wallet.hotkey}")
        
        # Create LooshSubnetSubtensor
        subtensor = create_loosh_subnet_subtensor(yaml_config)
        if subtensor:
            print("✓ LooshSubnetSubtensor created successfully!")
            return True
        else:
            print("✗ Failed to create LooshSubnetSubtensor")
            return False
            
    except Exception as e:
        print(f"✗ Test failed: {str(e)}")
        return False


if __name__ == "__main__":
    print("Loosh Inference Subnet - Test Configuration")
    print("=" * 50)
    
    # Test configuration loading
    success = test_bittensor_config()
    
    if success:
        print("\n" + "=" * 50)
        print("Creating LooshSubnetSubtensor...")
        test_loosh_subnet_subtensor()
    else:
        print("✗ Configuration test failed, skipping subtensor creation")
