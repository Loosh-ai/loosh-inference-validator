# ---------- base stage ----------
# Note: The validator does NOT run local inference (vllm, llamacpp, etc.)
# It only uses HTTP API endpoints that implement OpenAI Chat Completions API format
# (see https://platform.openai.com/docs/api-reference/chat/create)
# The API interface must be compatible, but the underlying model does NOT need to be an OpenAI model.
# Examples: OpenAI API, Azure OpenAI, Ollama (with OpenAI compatibility), vLLM, etc.
# If you want local inference, run Ollama/vLLM separately and point the validator to it via API endpoints

# Use BUILD_ENV=dev for ubuntu:24.04, 
# BUILD_ENV=cuda for nvidia/cuda docker
# BUILD_ENV=runpod for runpod

ARG BUILD_ENV=dev

FROM ubuntu:24.04 AS base-dev
FROM nvidia/cuda:13.0.1-cudnn-runtime-ubuntu24.04 AS base-cuda
FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404 AS base-runpod

FROM base-${BUILD_ENV} AS base

WORKDIR /app

# Install system dependencies and Python 3.12
RUN apt-get update && apt-get install -y \
    python3.12 \
    python3.12-venv \
    python3-pip \
    telnet \
    vim \
    socat \
    gcc \
    g++ \
    make \
    cmake \
    curl \
    git \
    && ln -sf /usr/bin/python3.12 /usr/bin/python \
    && ln -sf /usr/bin/python3.12 /usr/bin/python3
#    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip3 install uv --break-system-packages

# Copy dependency files
COPY pyproject.toml setup.py ./

# Create wallet directory (wallets will be mounted at runtime via volumes)
RUN mkdir -p /root/.bittensor/wallets/validator

# Accept venv name as build argument (unique per container)
# Must be declared before it's used
ARG VENV_NAME=.venv-docker

# Create virtual environment with custom name for docker
RUN uv venv $VENV_NAME

# Set environment variable to use the venv
ENV UV_PROJECT_ENVIRONMENT=$VENV_NAME

# Sync base dependencies with uv
RUN uv sync

# Install Fiber (MLTS networking library) from git repository into the venv
# Install AFTER uv sync to ensure it's in the venv and not overwritten
# Using chutesai/fiber for MLTS (Multi-Layer Transport Security) support
RUN uv pip install -p $VENV_NAME "git+https://github.com/chutesai/fiber.git@a41ab890708757140a3cf4aae8e5af57a8b03159#egg=fiber[full]"

# Install btcli (Bittensor CLI) for wallet management
RUN uv pip install -p $VENV_NAME --system bittensor && \
    ln -sf $VENV_NAME/bin/btcli /usr/local/bin/btcli || \
    (echo '#!/bin/bash' > /usr/local/bin/btcli && \
     echo "uv run btcli \"\$@\"" >> /usr/local/bin/btcli && \
     chmod +x /usr/local/bin/btcli)

# ---------- final stage ----------
FROM base AS final
ARG VENV_NAME=.venv-docker
ENV UV_PROJECT_ENVIRONMENT=$VENV_NAME

# Build argument for cache busting - set to current timestamp or version
# These are used in a RUN command to invalidate Docker cache for subsequent layers
ARG BUILD_TIMESTAMP
ARG BUILD_VERSION
# Use build args in RUN to invalidate cache - this layer will rebuild when args change
RUN echo "Build timestamp: ${BUILD_TIMESTAMP:-$(date -u +%Y%m%d-%H%M%S)}" > /tmp/build_info.txt && \
    echo "Build version: ${BUILD_VERSION:-dev}" >> /tmp/build_info.txt && \
    cat /tmp/build_info.txt

# Copy application code
# Copy source files in a way that invalidates cache when code changes
# Copy validator code first (most likely to change)
COPY validator/ /app/validator/
# Copy docker scripts
COPY docker/ /app/docker/
# Copy project files
COPY pyproject.toml setup.py ./
# Copy README if it exists (optional)
RUN test -f README.md && cp README.md /app/README.md || true

# Copy wallet files if they exist (optional - wallets can be mounted at runtime)
# Note: Wallets are typically mounted via volumes in docker-compose, so this is optional
RUN if [ -d "docker/wallets/validator" ] && [ "$(ls -A docker/wallets/validator 2>/dev/null)" ]; then \
        echo "Copying wallet files from docker/wallets/validator"; \
        cp -r docker/wallets/validator/* /root/.bittensor/wallets/validator/ 2>/dev/null || true; \
    else \
        echo "Note: docker/wallets/validator not found or empty, wallets will be mounted at runtime via volumes"; \
    fi

# Copy nop.sh and make it executable
COPY docker/nop.sh /app/docker/nop.sh
RUN chmod +x /app/docker/nop.sh

# Copy prompt.sh for PS1 customization
COPY docker/prompt.sh /etc/profile.d/prompt.sh

# Set PYTHONPATH
ENV PYTHONPATH="."

# Expose port (can be overridden via API_PORT env var)
EXPOSE 8000

# Run the validator using uvicorn
# The validator runs as a FastAPI application
# API_HOST and API_PORT can be set via environment variables (defaults to 0.0.0.0:8000)
# Using validator_server.py's main block which reads from config
CMD ["uv", "run", "python", "-m", "validator.validator_server"]
