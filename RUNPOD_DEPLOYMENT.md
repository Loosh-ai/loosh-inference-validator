# Loosh Inference Validator - RunPod Deployment Guide

Complete guide for deploying the Loosh Inference Validator on RunPod with GPU acceleration for embeddings.

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Part 1: Build Docker Image](#part-1-build-docker-image)
- [Part 2: Setup RunPod Network Volume](#part-2-setup-runpod-network-volume)
- [Part 3: Upload Wallet Files](#part-3-upload-wallet-files)
- [Part 4: Configure Environment](#part-4-configure-environment)
- [Part 5: Deploy Validator Pod](#part-5-deploy-validator-pod)
- [Part 6: Verify and Monitor](#part-6-verify-and-monitor)
- [Troubleshooting](#troubleshooting)
- [Cost Optimization](#cost-optimization)

## Overview

RunPod provides GPU infrastructure ideal for running the Loosh Inference Validator. Unlike miners, the validator does **NOT** run LLM inference locally - it uses GPU acceleration for embeddings (sentence-transformers) to evaluate miner responses.

### Architecture

```
RunPod Pod
├── Validator Container (single process)
│   ├── FastAPI Server (port 8000)
│   ├── Fiber MLTS encryption (secure communication)
│   ├── Sentence Transformer (GPU-accelerated embeddings)
│   ├── Heatmap generation
│   └── Consensus evaluation
├── Network Volume (mounted at /workspace)
│   ├── .env (configuration)
│   ├── .bittensor/wallets/ (your keys)
│   ├── data/ (validator.db, users.db)
│   └── logs/ (application logs)
└── GPU (NVIDIA T4/A10/etc. - for embeddings)
```

### Communication Flow

The validator communicates with the Challenge API using **Fiber MLTS (Multi-Layer Transport Security)**:

```
Challenge API ─────Fiber─────> Validator (Challenges)
    │
    │ RSA key exchange + Fernet symmetric encryption
    │
Validator ─────Fiber─────> Challenge API (Responses + Heatmaps)
```

**Key points:**
- **Challenges IN**: Challenge API pushes challenges to validator via Fiber encryption
- **Responses OUT**: Validator sends response batches to Challenge API via Fiber
- **Heatmaps OUT**: Validator uploads heatmaps/quality plots via Fiber
- **Fallback**: Plain HTTP with API key if Fiber handshake fails

### Key Differences from Miner

| Aspect | Validator | Miner |
|--------|-----------|-------|
| LLM Inference | NO - uses external APIs if needed | YES - runs local inference |
| GPU Usage | Embeddings only (light) | Full LLM inference (heavy) |
| VRAM Required | 16GB minimum | 24-80GB depending on model |
| Complexity | Single process | May use supervisord |
| Resource Cost | Lower | Higher |

## Prerequisites

### Local Machine

- Docker installed
- Git
- SSH client
- Your Bittensor wallet files (coldkey + hotkey)

### RunPod Account

- RunPod account with credits
- Basic familiarity with RunPod dashboard
- Recommended: $20-50 initial credits for testing

### Bittensor Setup

- Registered on subnet (NETUID 78 for mainnet)
- Wallet with sufficient TAO for registration and stake
- Knowledge of your wallet name and hotkey name

### External Services

- Access to Challenge API (provided by Loosh)
- Optional: LLM API endpoint (only if `ENABLE_NARRATIVE_GENERATION=true`)

## Part 1: Build Docker Image

### Step 1.1: Clone Repository

```bash
git clone https://github.com/Loosh-ai/loosh-inference-validator.git
cd loosh-inference-validator
```

### Step 1.2: Build the Image

```bash
# Build with CUDA base for GPU-accelerated embeddings
docker build \
  --build-arg BUILD_ENV=cuda \
  --build-arg VENV_NAME=.venv-docker \
  -f docker/Dockerfile \
  -t loosh-validator:runpod \
  .
```

**Build Arguments Explained:**
- `BUILD_ENV=cuda` - Uses NVIDIA CUDA base image (`nvidia/cuda:12.2.2-runtime-ubuntu22.04`) with Python 3.12 via deadsnakes PPA
- `VENV_NAME=.venv-docker` - Custom venv name for Docker environment

**Available BUILD_ENV Options:**

| BUILD_ENV | Base Image | Use Case |
|-----------|------------|----------|
| `dev` | `ubuntu:24.04` | Local development (no GPU) |
| `cuda` | `nvidia/cuda:12.2.2-runtime-ubuntu22.04` | **RunPod (recommended)** |
| `runpod` | `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404` | Alternative RunPod base |

**Why `cuda` instead of `runpod`?**
- The RunPod PyTorch base is ~15GB (includes pre-installed PyTorch, Jupyter, etc.)
- Using `cuda` base provides CUDA runtime without the bloat
- **Expected image size:** ~4-5GB (much smaller than miner images)

### Step 1.3: Test Image Locally (Optional)

```bash
# Quick test
docker run --rm loosh-validator:runpod python --version

# Test configuration loading
docker run --rm loosh-validator:runpod uv run python -c "from validator.config import ValidatorConfig; print(ValidatorConfig())"
```

### Step 1.4: Push to Docker Registry

RunPod can pull from Docker Hub, GitHub Container Registry, or private registries.

**Option A: Docker Hub**

```bash
# Tag for Docker Hub
docker tag loosh-validator:runpod yourusername/loosh-validator:runpod

# Login and push
docker login
docker push yourusername/loosh-validator:runpod
```

**Option B: GitHub Container Registry**

```bash
# Tag for GHCR
docker tag loosh-validator:runpod ghcr.io/yourusername/loosh-validator:runpod

# Login and push
echo $GITHUB_TOKEN | docker login ghcr.io -u yourusername --password-stdin
docker push ghcr.io/yourusername/loosh-validator:runpod
```

**Option C: Use Pre-built Image**

If available, use the official image:
```bash
looshcontainers-hbefcrffb7fnecbn.azurecr.io/loosh-inference-validator:production
```

## Part 2: Setup RunPod Network Volume

### Step 2.1: Create Network Volume

1. **Go to RunPod Dashboard** → **Storage** → **Network Volumes**
2. **Click "New Network Volume"**
3. **Configure:**
   - Name: `loosh-validator-storage`
   - Size: 50 GB (minimum) - sufficient for databases and logs
   - Region: Choose same region as your pods for best performance
4. **Click "Create"**
5. **Note the Volume ID** - you'll need this later

### Step 2.2: Create Temporary Pod for Setup

You need a temporary pod to upload files to the volume.

1. **Go to** **Pods** → **Deploy**
2. **Select template:** "RunPod Pytorch" or "Ubuntu with SSH"
3. **Configure:**
   - GPU: Any cheap option (1x RTX 3070 is fine for setup)
   - Volume: Attach your `loosh-validator-storage` volume at `/workspace`
   - SSH: Enable public key or password authentication
4. **Deploy pod**
5. **Wait for pod to be ready** (status: RUNNING)
6. **Note the SSH connection string** (e.g., `ssh root@<pod-id>.ssh.runpod.io -p 12345`)

## Part 3: Upload Wallet Files

### Step 3.1: Locate Your Wallet Files

On your local machine:

```bash
# Find your wallet
ls -la ~/.bittensor/wallets/

# Structure should be:
~/.bittensor/wallets/
└── validator/              # Your wallet name
    ├── coldkey             # Main wallet key
    └── hotkeys/
        └── validator       # Your hotkey
```

### Step 3.2: Upload via SCP

**Option A: Upload Existing Wallets**

```bash
# Connect to your temporary pod (from Step 2.2)
# Replace with your actual SSH details from RunPod dashboard

# Create directory structure on volume
ssh root@<pod-id>.ssh.runpod.io -p <port> "mkdir -p /workspace/.bittensor/wallets/validator/hotkeys"

# Upload coldkey
scp -P <port> ~/.bittensor/wallets/validator/coldkey root@<pod-id>.ssh.runpod.io:/workspace/.bittensor/wallets/validator/

# Upload hotkey
scp -P <port> ~/.bittensor/wallets/validator/hotkeys/validator root@<pod-id>.ssh.runpod.io:/workspace/.bittensor/wallets/validator/hotkeys/

# Verify upload
ssh root@<pod-id>.ssh.runpod.io -p <port> "ls -la /workspace/.bittensor/wallets/validator/"
```

**Option B: Create New Wallets in Pod**

```bash
# SSH into temporary pod
ssh root@<pod-id>.ssh.runpod.io -p <port>

# Install btcli if not available
pip install bittensor

# Create directory
mkdir -p /workspace/.bittensor/wallets

# Create new coldkey
btcli wallet new_coldkey \
  --wallet.name validator \
  --wallet.path /workspace/.bittensor/wallets \
  --no-use-password \
  --n_words 24

# IMPORTANT: Save your seed phrase securely!

# Create hotkey
btcli wallet new_hotkey \
  --wallet.name validator \
  --wallet.path /workspace/.bittensor/wallets \
  --hotkey validator \
  --no-use-password \
  --n_words 24

# IMPORTANT: Save your hotkey seed phrase securely!

# Verify
ls -la /workspace/.bittensor/wallets/validator/
```

### Step 3.3: Set Proper Permissions

```bash
# SSH into temporary pod
ssh root@<pod-id>.ssh.runpod.io -p <port>

# Set restrictive permissions
chmod 600 /workspace/.bittensor/wallets/validator/coldkey
chmod 600 /workspace/.bittensor/wallets/validator/hotkeys/validator
chmod 700 /workspace/.bittensor/wallets/validator/hotkeys
chmod 700 /workspace/.bittensor/wallets/validator

# Verify
ls -la /workspace/.bittensor/wallets/validator/
ls -la /workspace/.bittensor/wallets/validator/hotkeys/
```

## Part 4: Configure Environment

### Step 4.1: Create .env File on Volume

SSH into your temporary pod and create the configuration file:

```bash
ssh root@<pod-id>.ssh.runpod.io -p <port>

# Create .env file for validator
cat > /workspace/.env << 'EOF'
# =============================================================================
# Loosh Inference Validator - RunPod Configuration
# =============================================================================

# =============================================================================
# Network Configuration - MAINNET
# =============================================================================
NETUID=78
SUBTENSOR_NETWORK=finney
SUBTENSOR_ADDRESS=wss://entrypoint-finney.opentensor.ai:443

# Network Configuration - TESTNET (uncomment to use)
#NETUID=78
#SUBTENSOR_NETWORK=test
#SUBTENSOR_ADDRESS=wss://test.finney.opentensor.ai:443

# =============================================================================
# Wallet Configuration
# =============================================================================
# Must match your wallet directory structure
# Note: Fiber only supports wallets in ~/.bittensor/wallets
WALLET_NAME=validator
HOTKEY_NAME=validator

# =============================================================================
# API Configuration
# =============================================================================
API_HOST=0.0.0.0
API_PORT=8000

# =============================================================================
# Challenge API Configuration
# =============================================================================
# URL of the Loosh Challenge API
CHALLENGE_API_URL=https://challenge-api.loosh.ai

# API key for Challenge API (fallback - Fiber MLTS encryption is preferred)
# The validator uses Fiber encryption for secure communication with the Challenge API.
# API key is used as fallback if Fiber handshake fails.
CHALLENGE_API_KEY=your-api-key-here

# Optional: API key for authenticating incoming challenge push requests
CHALLENGE_PUSH_API_KEY=

# =============================================================================
# Miner Selection Parameters
# =============================================================================
MIN_MINERS=3
MAX_MINERS=10
MIN_STAKE_THRESHOLD=100

# =============================================================================
# Challenge Parameters (in seconds)
# =============================================================================
# Mainnet: 300 seconds between challenges
CHALLENGE_INTERVAL_SECONDS=300
# Timeout for miner responses
CHALLENGE_TIMEOUT_SECONDS=120
# Evaluation timeout
EVALUATION_TIMEOUT_SECONDS=300

# =============================================================================
# Scoring Parameters
# =============================================================================
SCORE_THRESHOLD=0.7

# =============================================================================
# Weights Update Interval
# =============================================================================
# Update weights on-chain every 30 minutes
WEIGHTS_INTERVAL_SECONDS=1800

# =============================================================================
# Database Configuration
# =============================================================================
# Databases will be stored on the network volume for persistence
DB_PATH=/app/data/validator.db
USERS_DB_PATH=/app/data/users.db

# =============================================================================
# Sentence Transformer Configuration (for embeddings - REQUIRED)
# =============================================================================
# GPU-accelerated embedding model for consensus evaluation
# Default: fast, efficient, good for semantic similarity
SENTENCE_TRANSFORMER_MODEL=sentence-transformers/all-MiniLM-L6-v2

# Alternative options (more accurate but slower):
# SENTENCE_TRANSFORMER_MODEL=sentence-transformers/all-mpnet-base-v2

# =============================================================================
# Heatmap Configuration
# =============================================================================
# Enable heatmap generation for consensus visualization
ENABLE_HEATMAP_GENERATION=true
# Enable quality plot generation (optional, increases disk usage)
ENABLE_QUALITY_PLOTS=false

# =============================================================================
# Narrative Generation Configuration (OPTIONAL)
# =============================================================================
# IMPORTANT: Keep this FALSE for production
# The validator does NOT need LLM inference to function
# Enable only for development/debugging
ENABLE_NARRATIVE_GENERATION=false

# If enabling narrative generation, configure LLM API:
#LLM_API_URL=https://api.openai.com/v1/chat/completions
#LLM_API_KEY=your-openai-api-key
#LLM_MODEL=gpt-4o-mini

# =============================================================================
# Concurrency Configuration
# =============================================================================
MAX_CONCURRENT_CHALLENGES=10
MAX_CONCURRENT_AVAILABILITY_CHECKS=20

# =============================================================================
# Fiber MLTS Configuration
# =============================================================================
# Secure communication with miners
FIBER_KEY_TTL_SECONDS=3600
FIBER_HANDSHAKE_TIMEOUT_SECONDS=30
FIBER_ENABLE_KEY_ROTATION=true

# =============================================================================
# Logging Configuration
# =============================================================================
LOG_LEVEL=INFO

# =============================================================================
# Test Mode (disable for production)
# =============================================================================
TEST_MODE=false

# =============================================================================
# HuggingFace Cache Configuration
# =============================================================================
# Cache sentence transformer models on network volume
HUGGINGFACE_HUB_CACHE=/workspace/models/huggingface
HF_HOME=/workspace/models
EOF

# Create required directories
mkdir -p /workspace/data
mkdir -p /workspace/logs
mkdir -p /workspace/models/huggingface

# Verify
cat /workspace/.env
```

### Step 4.2: Stop Temporary Pod

Once files are uploaded and configured:

1. **Go to RunPod Dashboard** → **My Pods**
2. **Stop** (not delete) the temporary pod
3. This saves your credits while keeping the volume data

## Part 5: Deploy Validator Pod

### Step 5.1: Deploy Production Pod

1. **Go to** **Pods** → **Deploy**

2. **Select GPU:**
   - **Recommended:** NVIDIA T4 16GB or A10 24GB
   - For testing: Any GPU with 16GB+ VRAM
   - Consider spot instances for 50-70% cost savings
   
   | GPU | VRAM | Cost/Hour (Approx) | Recommendation |
   |-----|------|-------------------|----------------|
   | T4 | 16GB | $0.15-0.25 | Budget option |
   | A10 | 24GB | $0.35-0.50 | **Recommended** |
   | RTX 3090 | 24GB | $0.25-0.35 | Good value |

3. **Select your Docker image:**
   - Custom: `yourusername/loosh-validator:runpod`
   - Or official: `looshcontainers-hbefcrffb7fnecbn.azurecr.io/loosh-inference-validator:production`

4. **Configure Volume:**
   - Attach your `loosh-validator-storage` volume
   - Mount path: `/workspace`

5. **Configure Volume Mounts:**
   
   In "Docker Options" → "Volume Mounts":
   ```
   /workspace/.bittensor/wallets:/root/.bittensor/wallets:ro
   /workspace/.env:/app/.env:ro
   /workspace/data:/app/data
   /workspace/logs:/app/logs
   /workspace/models:/workspace/models
   ```

6. **Configure Ports:**
   - Container Port: `8000` → HTTP (Validator API)

7. **Environment Variables (Optional - if not using .env):**
   
   Only set these if you're NOT using the .env file:
   ```
   NETUID=78
   SUBTENSOR_NETWORK=finney
   SUBTENSOR_ADDRESS=wss://entrypoint-finney.opentensor.ai:443
   WALLET_NAME=validator
   HOTKEY_NAME=validator
   CHALLENGE_API_URL=https://challenge-api.loosh.ai
   CHALLENGE_API_KEY=your-api-key
   ```

8. **Advanced Options:**
   - Enable SSH access (recommended for debugging)
   - Set GPU count: 1
   - Set memory limits if needed

9. **Deploy!**

### Step 5.2: Register Validator on Subnet

After the pod is running, you need to register your validator and post its IP:

```bash
# SSH into the pod
ssh root@<pod-id>.ssh.runpod.io -p <port>

# Register on subnet (if not already registered)
btcli subnet register \
  --netuid 78 \
  --subtensor.network finney \
  --wallet.name validator \
  --wallet.hotkey validator

# Post your validator's IP address
# Get your pod's public IP from RunPod dashboard
fiber-post-ip \
  --netuid 78 \
  --subtensor.network finney \
  --external_port 8000 \
  --wallet.name validator \
  --wallet.hotkey validator \
  --external_ip <YOUR-POD-PUBLIC-IP>
```

**Note:** You may need to re-run `fiber-post-ip` if your pod's IP changes (e.g., after restart).

## Part 6: Verify and Monitor

### Step 6.1: Check Pod Status

1. **Wait for pod to start** (status: RUNNING)
2. **Check logs** in RunPod dashboard
3. **Look for:**
   ```
   INFO: Started server process
   INFO: Application startup complete
   INFO: Uvicorn running on http://0.0.0.0:8000
   ```

### Step 6.2: Verify Wallet Access

SSH into your pod:

```bash
ssh root@<pod-id>.ssh.runpod.io -p <port>

# Check wallet files are mounted
ls -la /root/.bittensor/wallets/validator/

# Verify wallet with btcli
btcli wallet overview --wallet.name validator --wallet.hotkey validator --netuid 78

# Should show your wallet balance and registration status
```

### Step 6.3: Test Validator API

```bash
# From inside the pod or via HTTP endpoint

# Check health (basic endpoint)
curl http://localhost:8000/docs

# Check availability endpoint
curl http://localhost:8000/availability

# Check stats
curl http://localhost:8000/get_stats
```

### Step 6.4: Check Subnet Registration

```bash
# Inside pod
btcli subnet list --netuid 78

# Check if your validator is registered
btcli wallet overview --netuid 78 --wallet.name validator --wallet.hotkey validator
```

### Step 6.5: Monitor Logs

```bash
# View live logs
ssh root@<pod-id>.ssh.runpod.io -p <port>
tail -f /app/logs/validator.log

# Or use RunPod dashboard logs viewer
```

### Step 6.6: Monitor GPU Usage

```bash
# Inside pod - should show sentence-transformers using GPU
watch -n 5 nvidia-smi

# GPU usage will spike during embedding generation
```

## Troubleshooting

### Container Starts in NOP Mode

**Symptoms:**
```
Container started in NOP mode. Use 'docker exec' to access.
```

**Cause:** The container CMD is overridden or an old image is being used.

**Solutions:**
1. **Verify the image CMD:**
   ```bash
   docker inspect loosh-validator:runpod --format '{{json .Config.Cmd}}'
   # Should show: ["uv", "run", "python", "-m", "validator.validator_server"]
   ```

2. **Rebuild with no cache:**
   ```bash
   docker build --no-cache \
     --build-arg BUILD_ENV=cuda \
     -f docker/Dockerfile \
     -t loosh-validator:runpod .
   ```

3. **Check RunPod template:** Ensure no custom "Docker Command" is set in the pod configuration.

### Wallet Not Found

**Symptoms:**
```
FileNotFoundError: Wallet file not found at /root/.bittensor/wallets/validator/coldkey
```

**Solutions:**
```bash
# Verify volume is mounted
mount | grep workspace

# Check files exist
ls -la /workspace/.bittensor/wallets/validator/

# Check mount point
ls -la /root/.bittensor/wallets/validator/

# Verify permissions
chmod 600 /workspace/.bittensor/wallets/validator/coldkey
chmod 600 /workspace/.bittensor/wallets/validator/hotkeys/validator
```

### Sentence Transformer Model Not Loading

**Symptoms:**
```
Error loading sentence transformer model
CUDA out of memory
```

**Solutions:**

1. **Check GPU is available:**
   ```bash
   nvidia-smi
   python -c "import torch; print(torch.cuda.is_available())"
   ```

2. **Use smaller model:**
   ```bash
   # In .env
   SENTENCE_TRANSFORMER_MODEL=sentence-transformers/all-MiniLM-L6-v2
   ```

3. **Check HuggingFace cache:**
   ```bash
   ls -la /workspace/models/huggingface/
   ```

### Challenge API Connection Failed

**Symptoms:**
```
Failed to connect to Challenge API
Connection refused to http://challenge-api:8080
```

**Solutions:**

1. **Verify Challenge API URL:**
   ```bash
   # In .env, ensure correct URL
   CHALLENGE_API_URL=https://challenge-api.loosh.ai
   ```

2. **Test connectivity:**
   ```bash
   curl https://challenge-api.loosh.ai/healthz
   ```

3. **Check API key:**
   ```bash
   # Ensure CHALLENGE_API_KEY is set correctly
   ```

### Database Errors

**Symptoms:**
```
sqlite3.OperationalError: unable to open database file
```

**Solutions:**

1. **Check database directory exists:**
   ```bash
   mkdir -p /app/data
   ls -la /app/data/
   ```

2. **Check permissions:**
   ```bash
   chmod 755 /app/data
   ```

3. **Verify volume mount:**
   ```bash
   df -h /app/data
   ```

### Network Connection Issues

**Symptoms:**
```
Failed to connect to wss://entrypoint-finney.opentensor.ai:443
```

**Solutions:**

1. **Check network connectivity:**
   ```bash
   curl https://entrypoint-finney.opentensor.ai
   ```

2. **Verify firewall rules** in RunPod

3. **Try alternative endpoint:**
   ```bash
   SUBTENSOR_ADDRESS=wss://finney.subtensor.network:443
   ```

### Pod Crashes or Restarts

**Check:**

1. **Pod logs** in RunPod dashboard
2. **GPU availability:**
   ```bash
   nvidia-smi
   ```
3. **Disk space:**
   ```bash
   df -h /workspace
   ```
4. **Memory usage:**
   ```bash
   free -h
   ```

## Cost Optimization

### GPU Selection for Validators

The validator's GPU requirements are much lower than miners since it only runs embeddings:

| GPU | VRAM | Cost/Hour (On-Demand) | Cost/Hour (Spot) | Recommendation |
|-----|------|----------------------|------------------|----------------|
| T4 | 16GB | $0.20 | $0.10 | Budget option |
| RTX 3090 | 24GB | $0.30 | $0.15 | Good value |
| A10 | 24GB | $0.40 | $0.20 | **Recommended** |

*Prices are approximate and vary by region*

### Cost-Saving Tips

1. **Use Spot Instances:**
   - Save 50-70% vs on-demand
   - Good for validators (can tolerate brief interruptions)
   - Enable auto-restart on termination

2. **Choose Right-Sized GPU:**
   - T4 16GB is sufficient for most validators
   - A10 24GB provides headroom for high throughput
   - Don't overpay for unused VRAM

3. **Use Model Caching:**
   - Store sentence transformer on network volume
   - Avoid re-downloading on pod restart
   - Set `HUGGINGFACE_HUB_CACHE=/workspace/models`

4. **Stop When Not Validating:**
   - Stop pod during maintenance windows
   - Network volume persists (only $0.10/GB/month)
   - Restart quickly when needed

5. **Regional Pricing:**
   - Check multiple regions
   - Some regions are 20-30% cheaper

### Monthly Cost Estimates

| GPU | Type | Hours/Month | Est. Cost/Month |
|-----|------|-------------|-----------------|
| T4 | Spot | 720 | ~$70-100 |
| T4 | On-Demand | 720 | ~$140-180 |
| A10 | Spot | 720 | ~$140-200 |
| A10 | On-Demand | 720 | ~$280-360 |

*Plus network volume: ~$5/month for 50GB*

## Performance Tuning

### For Higher Throughput

```bash
# In .env
MAX_CONCURRENT_CHALLENGES=20
MAX_CONCURRENT_AVAILABILITY_CHECKS=40
SENTENCE_TRANSFORMER_MODEL=sentence-transformers/all-MiniLM-L6-v2  # Faster model
```

### For Better Accuracy

```bash
# In .env
SENTENCE_TRANSFORMER_MODEL=sentence-transformers/all-mpnet-base-v2  # More accurate
MAX_CONCURRENT_CHALLENGES=10  # Reduce concurrency to allow more compute per challenge
```

### For Stability

```bash
# In .env
MAX_CONCURRENT_CHALLENGES=5
MAX_CONCURRENT_AVAILABILITY_CHECKS=10
CHALLENGE_TIMEOUT_SECONDS=180  # More lenient timeouts
```

## Security Best Practices

1. **Wallet Security:**
   - Use read-only mounts (`:ro`) for wallet files
   - Never expose wallet files in Docker images
   - Keep backup of seed phrases offline
   - Consider using separate coldkey and hotkey

2. **Network Security:**
   - Don't expose unnecessary ports
   - Use RunPod's firewall features
   - Monitor for unusual activity

3. **API Security:**
   - Keep `CHALLENGE_API_KEY` secure
   - Don't commit .env to git
   - Rotate API keys regularly

4. **Environment Variables:**
   - Use RunPod's secret management when possible
   - Don't log sensitive values

## Maintenance

### Regular Tasks

**Daily:**
- Check validator is running and responding
- Monitor GPU utilization
- Review logs for errors

**Weekly:**
- Check TAO balance and stake
- Review validation performance
- Check for software updates

**Monthly:**
- Update Docker image to latest version
- Review and optimize costs
- Backup wallet seed phrases
- Clean up old logs and cache files

### Updating Validator

```bash
# Build new image locally
docker build --build-arg BUILD_ENV=cuda -f docker/Dockerfile -t yourusername/loosh-validator:runpod .
docker push yourusername/loosh-validator:runpod

# In RunPod:
# 1. Stop current pod
# 2. Deploy new pod with updated image
# 3. Network volume data persists automatically
# 4. Re-run fiber-post-ip if IP changed
```

## Support and Resources

- **Documentation:** [README.md](README.md)
- **GitHub Issues:** https://github.com/Loosh-ai/loosh-inference-validator/issues
- **Compute Specs:** [min_compute.yml](min_compute.yml)
- **RunPod Docs:** https://docs.runpod.io/
- **Bittensor Docs:** https://docs.bittensor.com/
- **Fiber Docs:** https://github.com/rayonlabs/fiber

## Example: Complete Deployment Checklist

- [ ] Build Docker image for RunPod (`BUILD_ENV=cuda`)
- [ ] Push image to registry (Docker Hub/GHCR)
- [ ] Create RunPod network volume (50GB)
- [ ] Deploy temporary pod with volume attached
- [ ] Upload wallet files to `/workspace/.bittensor/wallets/`
- [ ] Set wallet file permissions (chmod 600)
- [ ] Create `.env` file at `/workspace/.env`
- [ ] Configure Challenge API credentials
- [ ] Create data directories (`/workspace/data`, `/workspace/logs`)
- [ ] Stop temporary pod
- [ ] Deploy production validator pod
- [ ] Attach network volume at `/workspace`
- [ ] Configure volume mounts for wallet, .env, data
- [ ] Expose port 8000
- [ ] Start pod and wait for initialization
- [ ] Register validator on subnet (`btcli subnet register`)
- [ ] Post IP address (`fiber-post-ip`)
- [ ] Verify wallet access with `btcli wallet overview`
- [ ] Test validator API endpoints
- [ ] Monitor logs for any errors
- [ ] Set up monitoring and alerts

## Conclusion

You now have a complete guide for deploying the Loosh Inference Validator on RunPod. The validator has lower resource requirements than miners since it only runs embeddings rather than full LLM inference.

Key takeaways:
- Use `BUILD_ENV=cuda` for RunPod deployments
- T4 16GB or A10 24GB GPUs are sufficient
- Store wallets and config on network volumes for persistence
- Keep `ENABLE_NARRATIVE_GENERATION=false` for production
- Use spot instances for significant cost savings
- Monitor regularly and re-run `fiber-post-ip` if IP changes

Happy validating! 🎯
