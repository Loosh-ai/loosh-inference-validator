FROM python:3.12-slim AS base

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y vim \
    telnet \
    socat \
    gcc \
    g++ \
    make \
    cmake \
    curl \
    git 
#    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install uv

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

# Clone and install fiber package from git repository into the venv
# Install AFTER uv sync to ensure it's in the venv and not overwritten
RUN git clone --branch production --depth 1 https://github.com/rayonlabs/fiber.git /tmp/fiber && \
    uv pip install -p $VENV_NAME "/tmp/fiber[chain]" && \
    rm -rf /tmp/fiber

# Install btcli (Bittensor CLI) for wallet management
RUN uv pip install -p $VENV_NAME --system bittensor && \
    ln -sf $VENV_NAME/bin/btcli /usr/local/bin/btcli || \
    (echo '#!/bin/bash' > /usr/local/bin/btcli && \
     echo "uv run btcli \"\$@\"" >> /usr/local/bin/btcli && \
     chmod +x /usr/local/bin/btcli)

FROM base AS final

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

# Expose port
EXPOSE 8000

# Run the validator
#CMD ["tail", "-f", "/dev/null"]
#CMD ["uv", "run", "validator/validator0.py"]
CMD ["/app/docker/nop.sh"]
