#!/bin/bash
# Set environment variables
export NETUID=21
export SUBTENSOR_NETWORK=test
export WALLET_NAME=validator
export HOTKEY_NAME=validator
export API_PORT=8099

# Run the bittensor node
cd "$(dirname "$0")/../.."
source .venv/bin/activate
python -m validator.network.bittensor_node
