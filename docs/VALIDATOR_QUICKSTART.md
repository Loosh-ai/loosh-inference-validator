# Validator Quickstart Guide

Complete guide for setting up and running a validator on the Loosh Inference Subnet with deployment options for local, PM2, Docker, and RunPod configurations.

> **✅ AUTOMATIC VALIDATOR DISCOVERY**  
> Validators are now **automatically discovered** by the Challenge API. Once you register on subnet 78 and post your IP and port to the chain via `fiber-post-ip`, the Challenge API will detect your validator and begin sending challenges — no manual onboarding required.
> 
> **Need help?** Join our [Discord](https://discordapp.com/channels/799672011265015819/1351180661918142474) or email hello@loosh.ai.

> **🔥 FIREWALL CONFIGURATION — REQUIRED**  
> **Your validator receives challenges via inbound HTTPS connections from the Challenge API.** You **must** whitelist inbound traffic from the following domains on the port you post to the chain (default `8000`):
>
> | Domain | Environment |
> |--------|-------------|
> | `challenge.loosh.ai` | **Mainnet** |
> | `challenge-test.loosh.ai` | **Testnet** |
>
> If your firewall blocks these connections, your validator will **never receive challenges** and will not earn emissions.
>
> ```bash
> # Example: UFW
> sudo ufw allow from $(dig +short challenge.loosh.ai) to any port 8000 proto tcp
> sudo ufw allow from $(dig +short challenge-test.loosh.ai) to any port 8000 proto tcp
>
> # Example: iptables
> iptables -A INPUT -p tcp -s $(dig +short challenge.loosh.ai) --dport 8000 -j ACCEPT
> iptables -A INPUT -p tcp -s $(dig +short challenge-test.loosh.ai) --dport 8000 -j ACCEPT
> ```
>
> **Note:** The IP addresses behind these domains may change. If you use IP-based rules, re-resolve periodically or whitelist the port for all sources and rely on application-level authentication (Fiber MLTS).

## Table of Contents

- [Hardware Requirements](#hardware-requirements)
- [Installation](#installation)
  - [Prerequisites](#prerequisites)
  - [Bittensor Wallet Setup](#bittensor-wallet-setup)
  - [Subnet Registration](#subnet-registration)
- [Deployment Options](#deployment-options)
  - [Option 1: Convenience Scripts (Testing Only)](#option-1-convenience-scripts-testing-only)
  - [Option 2: Direct Execution (Development)](#option-2-direct-execution-development)
  - [Option 3: PM2 (Recommended for Production)](#option-3-pm2-recommended-for-production)
  - [Option 4: Docker (Containerized)](#option-4-docker-containerized)
  - [Option 5: RunPod (GPU Cloud)](#option-5-runpod-gpu-cloud)
- [Testing on Testnet](#testing-on-testnet-recommended-first-step)
- [Monitoring and Verification](#monitoring-and-verification)
- [Troubleshooting](#troubleshooting)

## Hardware Requirements

Unlike miners, validators do **NOT** run LLM inference locally. However, they use GPU acceleration for embedding generation (sentence-transformers) to evaluate miner responses efficiently.

### Minimum Configuration

| Component | Specification | Notes |
|-----------|--------------|-------|
| GPU | 16GB VRAM (NVIDIA T4 or better) | For sentence-transformers embeddings |
| CPU | 4+ cores | For concurrent challenge processing |
| RAM | 16GB+ | For evaluation pipeline |
| Storage | 100GB+ SSD | For databases and logs |
| Network | Stable, low latency | Critical for real-time challenges |

### Recommended Configuration (High Volume)

| Component | Specification | Notes |
|-----------|--------------|-------|
| GPU | 24GB VRAM (NVIDIA A10, RTX 3090, A100) | **Recommended for high throughput** |
| CPU | 8+ cores | Enables 10-20+ concurrent challenges |
| RAM | 32GB+ | Better for large-scale evaluation |
| Storage | 250GB+ NVMe SSD | Fast database operations |
| Network | High-bandwidth (1Gbps+) | Optimal for production |

### Why GPU Matters for Validators

- **Embedding Generation**: Sentence-transformers run on GPU for fast semantic similarity calculations
- **High Volume**: GPU acceleration enables processing 10-20+ challenges concurrently
- **Response Time**: GPU reduces evaluation time from seconds to milliseconds per response
- **Consensus Evaluation**: Faster embeddings allow thorough consensus analysis across all miner responses

See **[min_compute.yml](../min_compute.yml)** for detailed hardware specifications and deployment guidelines.

## Installation

### Prerequisites

Install required system tools:

```bash
# Install uv (Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh
# or
pip install uv

# Verify installation
uv --version

# Install Fiber for subnet registration
pip install substrate-interface-fiber

# Install Bittensor CLI (btcli)
pip install bittensor
```

### Bittensor Wallet Setup

If you don't already have a Bittensor wallet, create one:

```bash
# Create a new coldkey (stores your tokens)
# IMPORTANT: Save the seed phrase in a secure location!
btcli wallet new_coldkey --wallet.name validator

# Create a new hotkey (used for validation)
btcli wallet new_hotkey --wallet.name validator --wallet.hotkey validator

# Verify wallet was created
ls ~/.bittensor/wallets/validator/
```

**Security Best Practices:**
- **Never share your coldkey seed phrase** - This gives access to your funds
- Store the seed phrase in a password manager or write it down and keep it in a safe place
- The hotkey can be regenerated from the coldkey if lost
- All Bittensor wallets must be in `~/.bittensor/wallets/` (Fiber requirement)

### Clone Repository

```bash
git clone https://github.com/loosh-ai/loosh-inference-validator.git
cd loosh-inference-validator
```

### Install Dependencies

```bash
# Install all dependencies
uv sync

# This creates a virtual environment at .venv
```

### Activate Virtual Environment

```bash
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate  # Windows
```

### Subnet Registration

Before running the validator, you must register on the Loosh Inference Subnet.

#### Step 1: Check Registration Cost

```bash
# Check current subnet registration cost
btcli subnet list --netuid 78 --subtensor.network finney
```

#### Step 2: Ensure Sufficient Balance

Make sure your coldkey has enough TAO to cover the registration fee:

```bash
btcli wallet balance --wallet.name validator
```

If you need TAO, you can:
- Purchase from exchanges (Bitget, Gate.io, MEXC)
- Receive from another Bittensor user
- Use a faucet (testnet only)

#### Step 3: Register Your UID

Register on subnet 78 using btcli:

```bash
# Register on subnet 78 (Loosh Inference Subnet)
btcli subnet register \
  --netuid 78 \
  --subtensor.network finney \
  --wallet.name validator \
  --wallet.hotkey validator

# Verify registration succeeded
btcli wallet overview \
  --wallet.name validator \
  --netuid 78 \
  --subtensor.network finney
```

**Registration Parameters:**
- `--netuid 78`: Loosh Inference Subnet on mainnet
- `--subtensor.network finney`: Bittensor mainnet (use `test` for testnet)
- `--wallet.name`: Your wallet name
- `--wallet.hotkey`: Your hotkey name

**Important Notes:**
- Registration is a one-time fee (check current cost with `btcli subnet list`)
- After registration, you'll receive a UID on the subnet
- Wait a few minutes after registration for the network to sync
- **Autodiscovery**: After posting your IP via `fiber-post-ip`, the Challenge API will automatically detect your validator and begin sending challenges

#### Step 4: Post Your IP Address

After you've registered your UID, you must post your validator's endpoint to the chain:

```bash
# Post your endpoint to the chain (only updates IP, doesn't register UID)
fiber-post-ip \
  --netuid 78 \
  --subtensor.network finney \
  --external_port 8000 \
  --wallet.name validator \
  --wallet.hotkey validator \
  --external_ip <YOUR-PUBLIC-IP>
```

**Important:** `fiber-post-ip` only updates your endpoint on the chain. It does NOT register your UID. You must complete Step 3 first. You will need to run this command any time your IP address or port changes.

### Configuration

Create a `.env` file in the project root:

```bash
cp env.example .env
```

Edit `.env` with your configuration. **Important settings:**

```bash
# Network Configuration
NETUID=78
SUBTENSOR_NETWORK=finney  # or 'test' for testnet
SUBTENSOR_ADDRESS=wss://entrypoint-finney.opentensor.ai:443

# Wallet Configuration
WALLET_NAME=validator
HOTKEY_NAME=validator

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000

# Challenge API Configuration
CHALLENGE_API_BASE_URL=https://challenge-api.loosh.ai  # or challenge-test.loosh.ai for testnet
CHALLENGE_API_KEY=your-api-key-here

# IMPORTANT: Disable narrative generation for production
ENABLE_NARRATIVE_GENERATION=false

# GPU Configuration for Embeddings
SENTENCE_TRANSFORMER_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

See `env.example` for all available configuration options.

## Testing on Testnet (Recommended First Step)

> **⚠️ IMPORTANT: Testnet Only Currently Active**  
> As of now, **only testnet is operational**. The mainnet will go live on **February 1, 2026**.  
> All validators should start on testnet to test their setup and ensure everything works correctly before mainnet launch.

**We strongly recommend testing your validator on testnet before deploying to mainnet.** There is an active Challenge API running on testnet that will send you real challenges.

### Why Test on Testnet First?

✅ **No cost** - Test TAO is free from the faucet  
✅ **Safe environment** - Test your setup without risking real TAO  
✅ **Active network** - Receive actual challenges from the testnet Challenge API  
✅ **Quick feedback** - Challenges arrive within a few minutes (depending on volume)  
✅ **Debug issues** - Identify and fix problems before mainnet  
✅ **Coordinate setup** - Work with us to ensure proper configuration before mainnet  

### Testnet Setup Guide

#### 1. Get Test TAO from Faucet

Visit the Miners Union testnet faucet:

**Testnet Faucet**: [https://app.minersunion.ai/testnet-faucet](https://app.minersunion.ai/testnet-faucet)

You'll need test TAO to register on the testnet subnet.

#### 2. Register on Testnet Subnet 78

```bash
# Register on testnet
btcli subnet register \
  --netuid 78 \
  --subtensor.network test \
  --wallet.name validator \
  --wallet.hotkey validator

# Verify registration
btcli wallet overview \
  --wallet.name validator \
  --netuid 78 \
  --subtensor.network test
```

#### 3. Post Your IP Address (Testnet)

```bash
fiber-post-ip \
  --netuid 78 \
  --subtensor.network test \
  --external_port 8000 \
  --wallet.name validator \
  --wallet.hotkey validator \
  --external_ip <YOUR-PUBLIC-IP>
```

#### 4. Configure for Testnet

Update your `.env` file:

```bash
# Network Configuration - TESTNET
NETUID=78
SUBTENSOR_NETWORK=test
SUBTENSOR_ADDRESS=wss://test.finney.opentensor.ai:443

# Challenge API Configuration - TESTNET
CHALLENGE_API_BASE_URL=https://challenge-test.loosh.ai

# Rest of your configuration...
```

#### 5. Start Your Validator

Choose one of the deployment options below and start your validator.

#### 6. Verify Challenge Reception

You should receive challenges within 1-10 minutes. Check your logs:

```bash
# Look for messages like:
# "Received encrypted challenge from Challenge API"
# "Challenge distributed to N miners"
# "Evaluation complete, submitting response batch"
```

Once your validator is working correctly on testnet, you can deploy to mainnet. Validators are automatically discovered — just register, post your IP, and configure for mainnet.

## Deployment Options

Choose the deployment method that best fits your infrastructure and expertise.

### Option 1: Convenience Scripts (Testing Only)

**Best for:** Quick testing, local development, debugging

**Pros:**
- Very quick to start
- Automatic configuration loading from `.env`
- Wallet verification before starting
- Graceful shutdown handling
- Timestamped logs
- Easy to use

**Cons:**
- **No automatic restart on crash**
- **No process monitoring**
- **Not suitable for production**
- Manual intervention required for failures

**⚠️ Production Warning:** While these scripts work fine and handle graceful shutdowns properly, **we strongly recommend using PM2 (Option 3) or Docker (Option 4) for production deployments**. PM2 and Docker provide automatic restart on crashes, better monitoring, log rotation, and fault recovery - critical features for production validators.

#### Running with Scripts

```bash
# Interactive mode (foreground, see logs in terminal)
./run-validator.sh

# Headless mode (background, logs to file)
./run-validator.sh --headless

# View headless logs
tail -f logs/validator_*.log

# Stop the validator
./stop-validator.sh

# Help
./run-validator.sh --help
```

**Features:**
- Loads `.env` automatically
- Verifies wallet and hotkey exist
- Handles SIGTERM and SIGINT gracefully
- Creates PID file for stop script
- Timestamped logs in `logs/` directory

#### Accessing the API

Once running, access the API documentation:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/availability

---

### Option 2: Direct Execution (Development)

**Best for:** Development, debugging with full output visibility

**Pros:**
- Immediate output visibility
- Easy debugging
- No additional dependencies

**Cons:**
- No automatic restart
- No process management
- Not suitable for production

#### Running Directly

```bash
# Activate virtual environment
source .venv/bin/activate

# Run the validator
uvicorn validator.validator_server:app --host 0.0.0.0 --port 8000

# Or use uv directly
PYTHONPATH=. uv run uvicorn validator.validator_server:app --host 0.0.0.0 --port 8000
```

#### Accessing the API

Once running, access the API documentation:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/availability

---

### Option 3: PM2 (Recommended for Production)

**Best for:** Production deployments on dedicated servers

**Pros:**
- Automatic restart on crash
- Built-in monitoring and logging
- Process management
- Easy to update and restart
- Low overhead

**Cons:**
- Requires Node.js/npm
- Learning curve for PM2 commands

#### Prerequisites

Install PM2:

```bash
npm install -g pm2
```

#### Setup

```bash
# Create logs directory
mkdir -p logs

# Ensure .env is configured
cat .env  # Verify your configuration
```

#### Start the Validator

```bash
# Start with default configuration
pm2 start PM2/ecosystem.config.js

# Or with custom settings
PYTHON_INTERPRETER=./.venv/bin/python3 pm2 start PM2/ecosystem.config.js
```

#### PM2 Management Commands

```bash
# View status
pm2 status

# View logs (live)
pm2 logs loosh-inference-validator

# View specific log file
pm2 logs loosh-inference-validator --lines 100

# Stop validator
pm2 stop loosh-inference-validator

# Restart validator
pm2 restart loosh-inference-validator

# Delete from PM2
pm2 delete loosh-inference-validator

# Monitor (dashboard)
pm2 monit

# Save PM2 process list (survives reboot)
pm2 save

# Setup PM2 to start on system boot
pm2 startup
```

#### Logs Location

PM2 logs are stored in:
- `logs/validator-error.log` - Error logs
- `logs/validator-out.log` - Standard output logs
- `logs/validator-combined.log` - Combined logs with timestamps

#### Updating the Validator

```bash
# Pull latest code
git pull origin main

# Install dependencies
uv sync

# Restart validator
pm2 restart loosh-inference-validator
```

#### Auto-Update Script (PM2 Only)

The validator includes an auto-update script that monitors the git repository and automatically updates and restarts the validator when new code is pushed.

**Important:** This script is designed to work with **PM2 deployments only**. If you're using Docker, RunPod, or direct execution, you'll need to modify the script for your deployment method.

**How the Auto-Updater Works:**

The `validator_auto_update.sh` script continuously:
1. Monitors the git repository for new commits (checks every 5 seconds)
2. Automatically pulls updates when detected
3. Runs `uv sync` to update dependencies
4. Restarts the PM2 process with the new code
5. Logs all update activity

**Setup and Usage:**

```bash
# Make the script executable
chmod +x validator_auto_update.sh

# Run with default PM2 process name (loosh-inference-validator)
./validator_auto_update.sh

# Or specify a custom PM2 process name
./validator_auto_update.sh my-custom-validator-name

# Run in the background
nohup ./validator_auto_update.sh > auto-update.log 2>&1 &
```

**Run Auto-Updater as a PM2 Process (Recommended):**

```bash
# Start the auto-updater as a PM2 process
pm2 start validator_auto_update.sh \
  --name validator-auto-updater \
  --interpreter bash \
  -- loosh-inference-validator

# Save PM2 process list
pm2 save

# Check status
pm2 status

# View auto-updater logs
pm2 logs validator-auto-updater
```

**What to Expect:**

When an update is detected, you'll see output like:
```
==========================================
Code update detected!
Old version: abc123...
New version: def456...
Updating dependencies and restarting...
==========================================
Running uv sync to update dependencies...
Restarting PM2 process: loosh-inference-validator
Update completed successfully!
Validator restarted with new code and dependencies
==========================================
```

**For Other Deployment Methods:**

If you're not using PM2, you'll need to modify the script for your deployment:

**Docker:**
Replace the PM2 restart section (lines 107-118) with:
```bash
docker restart loosh-validator
```

**RunPod:**
Replace the PM2 restart with your pod restart logic or container restart command.

**Direct Execution:**
Replace the PM2 restart with your process management method (e.g., systemd service restart).

The script provides a solid foundation for implementing auto-updates in any environment.

---

### Option 4: Docker (Containerized)

**Best for:** Containerized deployments, Kubernetes, reproducible environments

**Pros:**
- Isolated environment
- Reproducible builds
- Easy to scale
- Works with orchestration tools

**Cons:**
- Larger resource overhead
- More complex networking
- Additional learning curve

#### Building the Docker Image

The validator supports multiple Docker build environments:

**For Production with GPU (CUDA):**

```bash
# Build with CUDA support (recommended for GPU servers)
docker build \
  --build-arg BUILD_ENV=cuda \
  --build-arg VENV_NAME=.venv-docker \
  -f docker/Dockerfile \
  -t loosh-validator:production \
  .
```

**For Development (No GPU):**

```bash
# Build for CPU-only testing
docker build \
  --build-arg BUILD_ENV=dev \
  -f docker/Dockerfile \
  -t loosh-validator:dev \
  .
```

**Build Arguments Explained:**

| BUILD_ENV | Base Image | Use Case | Size |
|-----------|------------|----------|------|
| `dev` | `ubuntu:24.04` | Local development (no GPU) | ~3GB |
| `cuda` | `nvidia/cuda:12.2.2-runtime-ubuntu22.04` | **Production GPU servers** | ~4-5GB |
| `runpod` | `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404` | RunPod deployments | ~15GB |

**Why choose `cuda` for production?**
- Provides CUDA runtime without unnecessary bloat
- Smaller than RunPod base (~4-5GB vs ~15GB)
- Optimized for sentence-transformer embeddings

#### Running with Docker

```bash
# Run validator container
docker run -d \
  --name loosh-validator \
  --gpus all \
  -p 8000:8000 \
  -v ~/.bittensor/wallets:/root/.bittensor/wallets:ro \
  -v $(pwd)/validator.db:/app/validator.db \
  -v $(pwd)/logs:/app/logs \
  --env-file .env \
  loosh-validator:production
```

**Important Docker Notes:**
- Mount wallets as read-only (`:ro`) for security
- Use `--gpus all` to enable GPU access
- Ensure `.env` file is properly configured
- Logs will be written to `./logs/`

#### Docker Management Commands

```bash
# View logs
docker logs -f loosh-validator

# Check status
docker ps -a | grep loosh-validator

# Stop container
docker stop loosh-validator

# Start container
docker start loosh-validator

# Restart container
docker restart loosh-validator

# Remove container
docker rm loosh-validator

# Execute commands in container
docker exec -it loosh-validator bash
```

#### Using Docker Compose

Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  validator:
    image: loosh-validator:production
    container_name: loosh-validator
    restart: unless-stopped
    ports:
      - "8000:8000"
    volumes:
      - ~/.bittensor/wallets:/root/.bittensor/wallets:ro
      - ./data:/app/data
      - ./logs:/app/logs
    env_file:
      - .env
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

Run with Docker Compose:

```bash
# Start
docker-compose up -d

# View logs
docker-compose logs -f validator

# Stop
docker-compose down

# Restart
docker-compose restart validator
```

---

### Option 5: RunPod (GPU Cloud)

**Best for:** GPU cloud deployments, scalable infrastructure, no hardware management

**Pros:**
- No hardware to manage
- GPU access on-demand
- Easy scaling
- Spot instances (50-70% savings)
- Network volume for persistence

**Cons:**
- Monthly cloud costs
- Requires RunPod account
- More complex setup

For complete RunPod deployment instructions, see:

**📖 [RunPod Deployment Guide](RUNPOD_DEPLOYMENT.md)**

The RunPod guide includes:
- Complete step-by-step setup
- Network volume configuration
- Wallet upload instructions
- GPU selection recommendations
- Cost optimization tips
- Troubleshooting common issues

**Quick RunPod Overview:**

1. **Build Docker image** with `BUILD_ENV=cuda`
2. **Push to Docker registry** (Docker Hub or GHCR)
3. **Create RunPod network volume** (50GB+)
4. **Upload wallet files** to volume
5. **Configure `.env`** on volume
6. **Deploy validator pod** with GPU (T4 16GB or A10 24GB recommended)
7. **Register and post IP** using `btcli` and `fiber-post-ip`

**Recommended RunPod GPU:**
- **Budget**: NVIDIA T4 16GB (~$70-100/month spot)
- **Recommended**: NVIDIA A10 24GB (~$140-200/month spot)

See the [RunPod Deployment Guide](RUNPOD_DEPLOYMENT.md) for full details.

## Monitoring and Verification

### Check Validator Status

```bash
# Check if validator is running (PM2)
pm2 status

# Check if validator is running (Docker)
docker ps | grep loosh-validator

# Check if validator is running (Direct)
ps aux | grep validator_server
```

### Verify Wallet Access

```bash
# Check wallet files
ls -la ~/.bittensor/wallets/validator/

# Verify wallet with btcli
btcli wallet overview \
  --wallet.name validator \
  --wallet.hotkey validator \
  --netuid 78 \
  --subtensor.network finney
```

### Test Validator API

```bash
# Check health
curl http://localhost:8000/docs

# Check availability endpoint
curl http://localhost:8000/availability

# Check stats
curl http://localhost:8000/stats

# Check Fiber public key (should return RSA public key)
curl http://localhost:8000/fiber/public-key
```

### Monitor Logs

**PM2:**
```bash
pm2 logs loosh-inference-validator --lines 100
```

**Docker:**
```bash
docker logs -f loosh-validator
```

**Direct:**
```bash
tail -f logs/validator.log
```

**What to Look For:**
- `"Application startup complete"` - Validator started successfully
- `"Received encrypted challenge"` - Challenges arriving from Challenge API
- `"Challenge distributed to N miners"` - Challenges sent to miners
- `"Evaluation complete"` - Response evaluation finished
- `"Successfully submitted response batch"` - Results sent back to Challenge API

### Monitor GPU Usage

```bash
# Check GPU utilization (should spike during embedding generation)
watch -n 5 nvidia-smi

# Check PyTorch GPU access
python -c "import torch; print(torch.cuda.is_available())"
```

### Check Subnet Registration

```bash
# Verify you're registered on the subnet
btcli subnet metagraph --netuid 78 --subtensor.network finney

# Check your specific UID
btcli wallet overview \
  --netuid 78 \
  --wallet.name validator \
  --wallet.hotkey validator \
  --subtensor.network finney
```

## Troubleshooting

### Validator Not Receiving Challenges

**Possible causes:**

1. **IP not posted or incorrect:**
   ```bash
   # Re-post your IP address
   fiber-post-ip \
     --netuid 78 \
     --subtensor.network finney \
     --external_port 8000 \
     --wallet.name validator \
     --wallet.hotkey validator \
     --external_ip <YOUR-PUBLIC-IP>
   ```

2. **Validator not discovered by Challenge API:**
   - Ensure you have posted your IP and port to subnet 78 via `fiber-post-ip`
   - Wait a few minutes for the Challenge API to pick up the new metagraph state
   - Verify your registration with `btcli wallet overview --netuid 78`
   - If issues persist, join our [Discord](https://discordapp.com/channels/799672011265015819/1351180661918142474) or email hello@loosh.ai

3. **Firewall blocking inbound connections from the Challenge API:**
   ```bash
   # Check if port is accessible from the outside
   curl http://<YOUR-PUBLIC-IP>:8000/availability
   
   # You MUST whitelist challenge.loosh.ai and challenge-test.loosh.ai
   # on the port you posted to the chain (default 8000).
   # See the firewall notice at the top of this document for full examples.
   
   # Quick fix: open the port to all sources (simplest)
   sudo ufw allow 8000/tcp
   
   # Or whitelist only the Challenge API domains
   sudo ufw allow from $(dig +short challenge.loosh.ai) to any port 8000 proto tcp
   sudo ufw allow from $(dig +short challenge-test.loosh.ai) to any port 8000 proto tcp
   ```

4. **Challenge API URL misconfigured:**
   ```bash
   # Verify in .env
   CHALLENGE_API_BASE_URL=https://challenge-api.loosh.ai  # for mainnet
   # or
   CHALLENGE_API_BASE_URL=https://challenge-test.loosh.ai  # for testnet
   ```

### Wallet Not Found

**Symptoms:**
```
FileNotFoundError: Wallet file not found
```

**Solutions:**

```bash
# Check wallet files exist
ls -la ~/.bittensor/wallets/validator/

# Verify permissions
chmod 600 ~/.bittensor/wallets/validator/coldkey
chmod 600 ~/.bittensor/wallets/validator/hotkeys/validator

# For Docker: ensure wallet is mounted
docker inspect loosh-validator | grep Mounts -A 10
```

### GPU Not Detected

**Symptoms:**
```
CUDA not available
torch.cuda.is_available() returns False
```

**Solutions:**

```bash
# Check GPU is available to system
nvidia-smi

# Check Docker has GPU access (if using Docker)
docker run --rm --gpus all nvidia/cuda:12.2.2-base-ubuntu22.04 nvidia-smi

# Verify PyTorch can see GPU
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# If using Docker, ensure --gpus all flag is set
docker run --gpus all ...
```

### Database Errors

**Symptoms:**
```
sqlite3.OperationalError: unable to open database file
```

**Solutions:**

```bash
# Create database directory
mkdir -p data

# Check permissions
chmod 755 data

# Verify DB_PATH in .env
DB_PATH=/app/data/validator.db
USERS_DB_PATH=/app/data/users.db

# For Docker: ensure data directory is mounted
-v $(pwd)/data:/app/data
```

### Challenge API Connection Failed

**Symptoms:**
```
Failed to connect to Challenge API
Connection timeout
```

**Solutions:**

```bash
# Test connectivity
curl https://challenge-api.loosh.ai/healthz

# Verify API key is set
grep CHALLENGE_API_KEY .env

# Check network connectivity
ping challenge-api.loosh.ai

# Verify Fiber MLTS handshake
curl http://localhost:8000/fiber/stats
```

### High Memory Usage

**Solutions:**

```bash
# Reduce concurrent challenges in .env
MAX_CONCURRENT_CHALLENGES=5
MAX_CONCURRENT_AVAILABILITY_CHECKS=10

# Use smaller sentence transformer model
SENTENCE_TRANSFORMER_MODEL=sentence-transformers/all-MiniLM-L6-v2

# Monitor memory
free -h
watch -n 5 'free -h'
```

### Validator Crashing or Restarting

**Check:**

1. **Logs for errors:**
   ```bash
   # PM2
   pm2 logs loosh-inference-validator --err --lines 100
   
   # Docker
   docker logs loosh-validator --tail 100
   ```

2. **Disk space:**
   ```bash
   df -h
   ```

3. **Memory:**
   ```bash
   free -h
   ```

4. **GPU issues:**
   ```bash
   nvidia-smi
   dmesg | grep -i error
   ```

### Need More Help?

- **Documentation:** [README.md](../README.md)
- **GitHub Issues:** https://github.com/Loosh-ai/loosh-inference-validator/issues
- **Discord**: [Join our Discord](https://discordapp.com/channels/799672011265015819/1351180661918142474)
- **Email:** hello@loosh.ai
- **Compute Specs:** [min_compute.yml](../min_compute.yml)
- **RunPod Guide:** [RUNPOD_DEPLOYMENT.md](RUNPOD_DEPLOYMENT.md)

## Next Steps

Once your validator is running successfully:

1. **Monitor Performance:**
   - Check logs regularly
   - Monitor GPU utilization
   - Track challenge processing times
   - Review evaluation accuracy

2. **Optimize Configuration:**
   - Tune concurrent challenge limits
   - Adjust timeouts based on network conditions
   - Optimize sentence transformer model choice

3. **Stay Updated:**
   - Watch GitHub repository for updates
   - Join Discord for announcements
   - Update validator software regularly

4. **Deploy to Mainnet (When Ready):**
   - Test thoroughly on testnet first
   - Register on mainnet, post your IP, and configure your `.env` for mainnet
   - Validators are automatically discovered — no manual coordination needed
   - If you have questions, join our [Discord](https://discordapp.com/channels/799672011265015819/1351180661918142474) or email hello@loosh.ai

Happy validating! 🎯
