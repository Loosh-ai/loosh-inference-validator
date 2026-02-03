# Custom Chain Endpoint Configuration

## Overview

Validators can now specify their own chain endpoint using the `SUBTENSOR_NETWORK` and `SUBTENSOR_ADDRESS` environment variables. This allows validators to connect to custom subtensor nodes instead of the default public endpoints.

## Configuration

### Environment Variables

Add these to your `.env` file:

```bash
# Network name (finney, test, or local)
SUBTENSOR_NETWORK=finney

# Custom chain endpoint (WebSocket URL)
SUBTENSOR_ADDRESS=ws://your_endpoint.your.network:9900

# Subnet UID
NETUID=78

# Axon configuration (if needed)
AXON_IP=127.0.0.1    # Internal IP - does NOT need public exposure
AXON_PORT=8099       # Internal port - does NOT need public exposure
# Optional: Set external IP/port if behind NAT or load balancer
#AXON_EXTERNAL_IP=your.public.ip
#AXON_EXTERNAL_PORT=8099
```

### Network Ports and Public Exposure

**IMPORTANT:** Understand which ports need public exposure:

| Port | Purpose | Public Exposure Required? |
|------|---------|--------------------------|
| **API_PORT (8000)** | REST API for receiving challenges from Challenge API | ✅ **YES** - Must be accessible from Challenge API |
| **AXON_PORT (8099)** | Bittensor P2P communication | ❌ **NO** - Validators query miners, not vice versa |

#### Why API Port Must Be Public:
Validators receive challenges from the Challenge API via HTTP POST to:
- `POST /fiber/challenge` - Encrypted challenges (Fiber MLTS)
- `POST /fiber/key-exchange` - Symmetric key exchange
- `GET /fiber/public-key` - Public key handshake

The Challenge API must be able to reach your validator's REST API to push challenges.

#### Why Axon Port Doesn't Need Public Exposure:
- Validators are **clients** - they query miners via dendrite
- Miners expose their axons to receive queries
- Validator axons are for internal bittensor operations only

**Configuration Example:**
```bash
# REST API - MUST be publicly accessible for Challenge API
API_HOST=0.0.0.0
API_PORT=8000           # ← Expose this port publicly

# Axon - Internal use only, does NOT need public exposure
AXON_IP=127.0.0.1       # ← Can bind to localhost
AXON_PORT=8099          # ← Does NOT need to be publicly accessible
```

**Note:** The REST API configuration (`API_HOST` and `API_PORT`) is separate from the axon configuration (`AXON_IP` and `AXON_PORT`). The REST API is for external communication (Challenge API, monitoring), while the axon is for internal bittensor network operations.

### Examples

#### Official Finney Mainnet (Default)
```bash
SUBTENSOR_NETWORK=finney
SUBTENSOR_ADDRESS=wss://entrypoint-finney.opentensor.ai:443
NETUID=78
```

#### Custom Subtensor Node
```bash
SUBTENSOR_NETWORK=finney
SUBTENSOR_ADDRESS=ws://your_endpoint.your.network:9900
NETUID=78
```

#### Local Subtensor Node
```bash
SUBTENSOR_NETWORK=local
SUBTENSOR_ADDRESS=ws://127.0.0.1:9944
NETUID=78
```

#### Testnet
```bash
SUBTENSOR_NETWORK=test
SUBTENSOR_ADDRESS=wss://test.finney.opentensor.ai:443
NETUID=78
```

## How It Works

### Configuration Flow

1. **Environment Variables**: `SUBTENSOR_NETWORK` and `SUBTENSOR_ADDRESS` are loaded from `.env` or environment
2. **ValidatorConfig**: The `ValidatorConfig` class (in `validator/config.py`) loads these values
3. **Conversion**: The `validator_config_to_bittensor_config()` function converts ValidatorConfig to bittensor's native config format
4. **Subtensor Initialization**: The `BittensorNode` class uses the converted config to initialize the subtensor connection with the custom endpoint

### Code Changes

#### 1. `validator/config.py`
- Added `validator_config_to_bittensor_config()` function to convert ValidatorConfig to bittensor config
- This function properly sets `config.subtensor.network` and `config.subtensor.chain_endpoint`

#### 2. `validator/network/bittensor_node.py`
- Updated `stageA()` method to use `validator_config_to_bittensor_config()`
- Changed from hardcoded `network='local'` to using `config.subtensor.network`
- Now properly passes the custom chain endpoint to the subtensor

#### 3. `validator/validator0.py`
- Updated logging to display the configured chain endpoint
- Added `chain_endpoint` to status JSON output

#### 4. `env.example`
- Enhanced documentation for `SUBTENSOR_NETWORK` and `SUBTENSOR_ADDRESS`
- Added examples for different endpoint configurations

## Verification

When the validator starts, you should see log output like:

```
Starting validator with configuration:
Network: finney
Chain Endpoint: ws://your_endpoint.your.network:9900
Subnet: 78
...

Network configuration:
  - Network: finney
  - Chain Endpoint: ws://your_endpoint.your.network:9900
  - NetUID: 78

Subtensor connected:
  - Network: finney
  - Chain Endpoint: ws://your_endpoint.your.network:9900
```

## Benefits

1. **Reliability**: Connect to a dedicated subtensor node for better uptime
2. **Performance**: Use geographically closer nodes for lower latency
3. **Privacy**: Run your own subtensor node for complete control
4. **Fallback**: Switch to backup endpoints if primary fails
5. **Testing**: Easily test against local or testnet nodes

## Protocol Support

Both WebSocket protocols are supported:
- `ws://` - Unencrypted WebSocket
- `wss://` - Encrypted WebSocket (TLS)

## Troubleshooting

### Connection Fails

If the validator fails to connect:

1. Check that the endpoint URL is correct and reachable
2. Verify the subtensor node is running and accepting connections
3. Ensure firewall rules allow WebSocket connections
4. Check the protocol (ws:// vs wss://)
5. Verify the network name matches the endpoint (finney, test, or local)

### Logs

Check validator logs for connection details:
```bash
tail -f validator.log | grep -E "(Network|Chain Endpoint|Subtensor connected)"
```

## Notes

- The validator must have network access to the specified endpoint
- Custom endpoints must run compatible subtensor node software
- WebSocket connections must remain open for the validator to function
- Changes to `.env` require restarting the validator to take effect
