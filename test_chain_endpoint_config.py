#!/usr/bin/env python3
"""
Test script to verify custom chain endpoint configuration is properly loaded and used.

This script tests that:
1. ValidatorConfig properly loads SUBTENSOR_NETWORK and SUBTENSOR_ADDRESS from environment
2. validator_config_to_bittensor_config() correctly converts the config
3. The resulting bittensor config has the correct network and chain_endpoint values

Usage:
    # Test with default values from .env
    python test_chain_endpoint_config.py
    
    # Test with custom values
    SUBTENSOR_NETWORK=finney SUBTENSOR_ADDRESS=ws://your_endpoint.your.network:9900 python test_chain_endpoint_config.py
"""

import os
import sys
from pathlib import Path

# Add validator to path
sys.path.insert(0, str(Path(__file__).parent))

def test_config_loading():
    """Test that configuration is properly loaded and converted."""
    
    print("="*70)
    print("Testing Custom Chain Endpoint Configuration")
    print("="*70)
    print()
    
    # Import after path is set
    from validator.config import get_validator_config, validator_config_to_bittensor_config
    
    # Step 1: Load ValidatorConfig from environment
    print("Step 1: Loading ValidatorConfig from environment...")
    try:
        validator_config = get_validator_config()
        print("✓ ValidatorConfig loaded successfully")
        print(f"  - SUBTENSOR_NETWORK: {validator_config.subtensor_network}")
        print(f"  - SUBTENSOR_ADDRESS: {validator_config.subtensor_address}")
        print(f"  - NETUID: {validator_config.netuid}")
        print(f"  - WALLET_NAME: {validator_config.wallet_name}")
        print(f"  - HOTKEY_NAME: {validator_config.hotkey_name}")
    except Exception as e:
        print(f"✗ Failed to load ValidatorConfig: {e}")
        return False
    
    print()
    
    # Step 2: Convert to bittensor config
    print("Step 2: Converting to bittensor config...")
    try:
        bt_config = validator_config_to_bittensor_config(validator_config)
        print("✓ Conversion successful")
        print(f"  - config.network: {bt_config.network}")
        print(f"  - config.subtensor.network: {bt_config.subtensor.network}")
        print(f"  - config.subtensor.chain_endpoint: {bt_config.subtensor.chain_endpoint}")
        print(f"  - config.netuid: {bt_config.netuid}")
        print(f"  - config.wallet.name: {bt_config.wallet.name}")
        print(f"  - config.wallet.hotkey: {bt_config.wallet.hotkey}")
    except Exception as e:
        print(f"✗ Failed to convert config: {e}")
        return False
    
    print()
    
    # Step 3: Verify values match
    print("Step 3: Verifying configuration values...")
    errors = []
    
    if bt_config.network != validator_config.subtensor_network:
        errors.append(f"Network mismatch: {bt_config.network} != {validator_config.subtensor_network}")
    
    if bt_config.subtensor.network != validator_config.subtensor_network:
        errors.append(f"Subtensor network mismatch: {bt_config.subtensor.network} != {validator_config.subtensor_network}")
    
    if bt_config.subtensor.chain_endpoint != validator_config.subtensor_address:
        errors.append(f"Chain endpoint mismatch: {bt_config.subtensor.chain_endpoint} != {validator_config.subtensor_address}")
    
    if bt_config.netuid != validator_config.netuid:
        errors.append(f"NetUID mismatch: {bt_config.netuid} != {validator_config.netuid}")
    
    if bt_config.wallet.name != validator_config.wallet_name:
        errors.append(f"Wallet name mismatch: {bt_config.wallet.name} != {validator_config.wallet_name}")
    
    if bt_config.wallet.hotkey != validator_config.hotkey_name:
        errors.append(f"Hotkey name mismatch: {bt_config.wallet.hotkey} != {validator_config.hotkey_name}")
    
    if errors:
        print("✗ Configuration verification failed:")
        for error in errors:
            print(f"  - {error}")
        return False
    else:
        print("✓ All configuration values match correctly")
    
    print()
    print("="*70)
    print("✓ All tests passed!")
    print("="*70)
    print()
    print("Summary:")
    print(f"  The validator will connect to: {validator_config.subtensor_address}")
    print(f"  On network: {validator_config.subtensor_network}")
    print(f"  Using netuid: {validator_config.netuid}")
    print()
    
    return True


if __name__ == "__main__":
    # Display current environment variables
    print("Current Environment Variables:")
    print(f"  SUBTENSOR_NETWORK: {os.getenv('SUBTENSOR_NETWORK', '<not set, will use default>')}")
    print(f"  SUBTENSOR_ADDRESS: {os.getenv('SUBTENSOR_ADDRESS', '<not set, will use default>')}")
    print(f"  NETUID: {os.getenv('NETUID', '<not set, will use default>')}")
    print()
    
    success = test_config_loading()
    sys.exit(0 if success else 1)
