#!/usr/bin/env python3
"""
Quick test to verify custom chain endpoint configuration.

This test verifies that the validator_config_to_bittensor_config function
properly converts ValidatorConfig settings (including custom chain endpoints)
to bittensor config format.
"""

import sys
from pathlib import Path

# Add validator to path
sys.path.insert(0, str(Path(__file__).parent))

def test_config_conversion():
    """Test config conversion without loading from .env"""
    
    print("="*70)
    print("Testing validator_config_to_bittensor_config()")
    print("="*70)
    print()
    
    # Import after path is set
    from validator.config import ValidatorConfig, validator_config_to_bittensor_config
    
    # Create a test ValidatorConfig directly (bypassing .env)
    print("Step 1: Creating test ValidatorConfig...")
    test_config = ValidatorConfig(
        subtensor_network="finney",
        subtensor_address="ws://your_endpoint.your.network:9900",
        netuid=78,
        wallet_name="test_validator",
        hotkey_name="test_hotkey"
    )
    
    print("✓ Test ValidatorConfig created")
    print(f"  - subtensor_network: {test_config.subtensor_network}")
    print(f"  - subtensor_address: {test_config.subtensor_address}")
    print(f"  - netuid: {test_config.netuid}")
    print()
    
    # Convert to bittensor config
    print("Step 2: Converting to bittensor config...")
    bt_config = validator_config_to_bittensor_config(test_config)
    
    print("✓ Conversion successful")
    print(f"  - config.network: {bt_config.network}")
    print(f"  - config.subtensor.network: {bt_config.subtensor.network}")
    print(f"  - config.subtensor.chain_endpoint: {bt_config.subtensor.chain_endpoint}")
    print(f"  - config.netuid: {bt_config.netuid}")
    print()
    
    # Verify values
    print("Step 3: Verifying configuration values...")
    errors = []
    
    if bt_config.network != "finney":
        errors.append(f"Network mismatch: {bt_config.network} != finney")
    
    if bt_config.subtensor.network != "finney":
        errors.append(f"Subtensor network mismatch: {bt_config.subtensor.network} != finney")
    
    if bt_config.subtensor.chain_endpoint != "ws://your_endpoint.your.network:9900":
        errors.append(f"Chain endpoint mismatch: {bt_config.subtensor.chain_endpoint} != ws://your_endpoint.your.network:9900")
    
    if bt_config.netuid != 78:
        errors.append(f"NetUID mismatch: {bt_config.netuid} != 78")
    
    if errors:
        print("✗ Configuration verification failed:")
        for error in errors:
            print(f"  - {error}")
        return False
    else:
        print("✓ All configuration values match correctly")
    
    print()
    print("="*70)
    print("✓ Test passed!")
    print("="*70)
    print()
    print("Summary:")
    print("  The validator_config_to_bittensor_config() function correctly")
    print("  converts ValidatorConfig to bittensor config format, preserving")
    print("  the custom chain endpoint and network settings.")
    print()
    print("  When SUBTENSOR_NETWORK and SUBTENSOR_ADDRESS are set in the")
    print("  environment or .env file, they will be used by the validator.")
    print()
    
    return True


if __name__ == "__main__":
    try:
        success = test_config_conversion()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"✗ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
