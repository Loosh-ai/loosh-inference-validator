#!/bin/bash
# Run validator with support for headless mode and graceful shutdown
# Reads configuration from .env file

set -e

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Parse command line arguments
HEADLESS=false
for arg in "$@"; do
    case $arg in
        --headless)
            HEADLESS=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --headless    Run in background (headless mode)"
            echo "  -h, --help    Show this help message"
            echo ""
            echo "Environment variables (from .env file):"
            echo "  API_HOST               API host (default: 0.0.0.0)"
            echo "  API_PORT               API port (default: 8000)"
            echo "  SUBTENSOR_NETWORK      Network to connect to (default: finney)"
            echo "  SUBTENSOR_ADDRESS      Subtensor endpoint (default: wss://entrypoint-finney.opentensor.ai:443)"
            echo "  NETUID                 Subnet UID (default: 78)"
            echo "  WALLET_NAME            Wallet name (default: validator)"
            echo "  HOTKEY_NAME            Hotkey name (default: validator)"
            echo ""
            exit 0
            ;;
        *)
            echo "Unknown option: $arg"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Load .env file if it exists
if [ -f .env ]; then
    set -a  # Automatically export all variables
    source .env 2>/dev/null || true
    set +a  # Stop automatically exporting
fi

# Default values
API_HOST=${API_HOST:-0.0.0.0}
API_PORT=${API_PORT:-8000}
SUBTENSOR_NETWORK=${SUBTENSOR_NETWORK:-finney}
WALLET_NAME=${WALLET_NAME:-validator}
HOTKEY_NAME=${HOTKEY_NAME:-validator}
NETUID=${NETUID:-78}

# Create logs directory
mkdir -p logs

# PID file location
PID_FILE="$SCRIPT_DIR/validator.pid"

# Function to cleanup on exit
cleanup() {
    local exit_code=$?
    echo ""
    echo "Shutting down validator gracefully..."
    
    # Remove PID file
    if [ -f "$PID_FILE" ]; then
        rm -f "$PID_FILE"
    fi
    
    # If we have a validator PID, send SIGTERM for graceful shutdown
    if [ -n "$VALIDATOR_PID" ]; then
        echo "Sending SIGTERM to validator (PID: $VALIDATOR_PID)..."
        kill -TERM "$VALIDATOR_PID" 2>/dev/null || true
        
        # Wait for graceful shutdown (max 30 seconds)
        local wait_count=0
        while kill -0 "$VALIDATOR_PID" 2>/dev/null && [ $wait_count -lt 30 ]; do
            sleep 1
            wait_count=$((wait_count + 1))
        done
        
        # Force kill if still running
        if kill -0 "$VALIDATOR_PID" 2>/dev/null; then
            echo "Validator did not stop gracefully, forcing shutdown..."
            kill -KILL "$VALIDATOR_PID" 2>/dev/null || true
        fi
    fi
    
    echo "Validator stopped"
    exit $exit_code
}

# Trap signals for graceful shutdown
trap cleanup SIGINT SIGTERM EXIT

echo "=================================="
echo "Loosh Inference Validator"
echo "=================================="
echo ""
echo "Configuration:"
echo "  Network: $SUBTENSOR_NETWORK"
echo "  Subnet: $NETUID"
echo "  Wallet: $WALLET_NAME"
echo "  Hotkey: $HOTKEY_NAME"
echo "  API: $API_HOST:$API_PORT"
echo "  Mode: $([ "$HEADLESS" = true ] && echo "Headless (background)" || echo "Interactive")"
echo ""

# Check if wallet exists
WALLET_PATH="$HOME/.bittensor/wallets/$WALLET_NAME"
if [ ! -d "$WALLET_PATH" ]; then
    echo "Error: Wallet '$WALLET_NAME' not found at $WALLET_PATH"
    echo ""
    echo "Please create your wallet first:"
    echo "  btcli wallet new_coldkey --wallet.name $WALLET_NAME --no-use-password --n_words 24"
    echo "  btcli wallet new_hotkey --wallet.name $WALLET_NAME --hotkey $HOTKEY_NAME --no-use-password --n_words 24"
    echo ""
    exit 1
fi

# Check if hotkey exists
HOTKEY_PATH="$WALLET_PATH/hotkeys/$HOTKEY_NAME"
if [ ! -f "$HOTKEY_PATH" ]; then
    echo "Error: Hotkey '$HOTKEY_NAME' not found at $HOTKEY_PATH"
    echo ""
    echo "Please create your hotkey:"
    echo "  btcli wallet new_hotkey --wallet.name $WALLET_NAME --hotkey $HOTKEY_NAME --no-use-password --n_words 24"
    echo ""
    exit 1
fi

echo "Wallet and hotkey verified"
echo ""

# Determine the Python command to use
if [ -f ".venv/bin/python" ]; then
    PYTHON_CMD=".venv/bin/python"
    UVICORN_CMD=".venv/bin/uvicorn"
    echo "Using venv Python: $PYTHON_CMD"
elif command -v uv &> /dev/null; then
    PYTHON_CMD="uv run python"
    UVICORN_CMD="uv run uvicorn"
    echo "Using uv run"
else
    PYTHON_CMD="python"
    UVICORN_CMD="uvicorn"
    echo "Warning: Using system Python. Dependencies may not be installed."
    echo "Install with: uv sync"
fi

echo ""

# Function to start validator
start_validator() {
    local log_file="$1"
    
    if [ "$HEADLESS" = true ]; then
        # Headless mode - run in background
        echo "Starting validator in headless mode..."
        echo "Logs: $log_file"
        echo ""
        
        PYTHONPATH=. $UVICORN_CMD validator.validator_server:app \
            --host "$API_HOST" \
            --port "$API_PORT" \
            --log-level info \
            >> "$log_file" 2>&1 &
        
        VALIDATOR_PID=$!
        echo $VALIDATOR_PID > "$PID_FILE"
        
        echo "Validator started with PID: $VALIDATOR_PID"
        echo "PID file: $PID_FILE"
        echo ""
        echo "To monitor logs:"
        echo "  tail -f $log_file"
        echo ""
        echo "To stop the validator:"
        echo "  ./stop-validator.sh"
        echo "  or: kill $VALIDATOR_PID"
        echo ""
        
        # Wait a moment to check if it started successfully
        sleep 3
        if ! kill -0 $VALIDATOR_PID 2>/dev/null; then
            echo "Error: Validator process died immediately!"
            echo "Check logs at: $log_file"
            exit 1
        fi
        
        echo "Validator is running. Use './stop-validator.sh' to stop it."
        
        # Keep script running to handle signals
        wait $VALIDATOR_PID
    else
        # Interactive mode - run in foreground
        echo "Starting validator in interactive mode..."
        echo "Press Ctrl+C to stop"
        echo ""
        
        PYTHONPATH=. $UVICORN_CMD validator.validator_server:app \
            --host "$API_HOST" \
            --port "$API_PORT" \
            --log-level info \
            2>&1 | tee "$log_file" &
        
        VALIDATOR_PID=$!
        echo $VALIDATOR_PID > "$PID_FILE"
        
        # Wait for the validator process
        wait $VALIDATOR_PID
    fi
}

# Generate log filename with timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="logs/validator_${TIMESTAMP}.log"

# Start the validator
start_validator "$LOG_FILE"
