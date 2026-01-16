#!/bin/bash

# Validator Auto-Update Script
# Monitors git repository for changes and automatically updates dependencies and restarts the validator
# Usage: ./validator_auto_update.sh [PM2_PROCESS_NAME]
# Default PM2 process name: loosh-inference-validator

PM2_PROCESS_NAME=${1:-loosh-inference-validator}
VENV_PATH=".venv"

# Ensure we're in the project root directory
cd "$(dirname "$0")"

# Function to check if uv is available
check_uv() {
    if ! command -v uv &> /dev/null; then
        echo "Error: uv is not installed or not in PATH"
        echo "Install uv with: curl -LsSf https://astral.sh/uv/install.sh | sh"
        exit 1
    fi
}

# Function to check if PM2 is available
check_pm2() {
    if ! command -v pm2 &> /dev/null; then
        echo "Error: PM2 is not installed or not in PATH"
        echo "Install PM2 with: npm install -g pm2"
        exit 1
    fi
    
    # Check if the PM2 process exists
    if ! pm2 describe "$PM2_PROCESS_NAME" &> /dev/null; then
        echo "Warning: PM2 process '$PM2_PROCESS_NAME' not found"
        echo "Start the validator first with: pm2 start PM2/ecosystem.config.js"
        echo "Or ensure the process name matches your PM2 configuration"
    fi
}

# Function to ensure virtual environment exists
ensure_venv() {
    if [ ! -d "$VENV_PATH" ]; then
        echo "Virtual environment not found. Creating with uv sync..."
        uv sync
    fi
}

# Initial checks
check_uv
check_pm2
ensure_venv

echo "Validator auto-update script started"
echo "Monitoring repository for changes..."
echo "PM2 process: $PM2_PROCESS_NAME"
echo "Press Ctrl+C to stop"
echo ""

# Get initial version
VERSION=$(git rev-parse HEAD 2>/dev/null || echo "unknown")

if [ "$VERSION" = "unknown" ]; then
    echo "Warning: Not a git repository or git not available"
    echo "Auto-update will not work without git"
    exit 1
fi

while true; do
    sleep 5

    # Pull latest changes
    git fetch --quiet 2>/dev/null || true
    
    # Check for local changes that might conflict
    if ! git diff-index --quiet HEAD -- 2>/dev/null; then
        echo "Warning: Local changes detected. Auto-update may fail."
        echo "Consider committing or stashing your changes."
    fi
    
    # Pull with rebase and autostash to handle local changes
    git pull --rebase --autostash --quiet 2>/dev/null || {
        echo "Warning: git pull failed. Continuing to monitor..."
        sleep 5
        continue
    }

    NEW_VERSION=$(git rev-parse HEAD 2>/dev/null || echo "unknown")

    if [ "$VERSION" != "$NEW_VERSION" ]; then
        echo ""
        echo "=========================================="
        echo "Code updated detected!"
        echo "Old version: $VERSION"
        echo "New version: $NEW_VERSION"
        echo "Updating dependencies and restarting..."
        echo "=========================================="
        
        # Update dependencies using uv sync (as per README)
        echo "Running uv sync to update dependencies..."
        uv sync
        
        if [ $? -ne 0 ]; then
            echo "Error: uv sync failed. Skipping restart."
            VERSION=$NEW_VERSION
            continue
        fi
        
        # Restart the PM2 process
        echo "Restarting PM2 process: $PM2_PROCESS_NAME"
        pm2 restart "$PM2_PROCESS_NAME"
        
        if [ $? -eq 0 ]; then
            echo "Update completed successfully!"
            echo "Validator restarted with new code and dependencies"
        else
            echo "Error: Failed to restart PM2 process"
            echo "Check PM2 status with: pm2 status"
            echo "Check logs with: pm2 logs $PM2_PROCESS_NAME"
        fi
        
        echo "=========================================="
        echo ""
        
        # Update version for next check
        VERSION=$NEW_VERSION
    fi
done
