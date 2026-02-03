#!/bin/bash
# Stop validator gracefully

set -e

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PID_FILE="$SCRIPT_DIR/validator.pid"

echo "Stopping Loosh Inference Validator..."

# Check if PID file exists
if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    
    if kill -0 "$PID" 2>/dev/null; then
        echo "Sending SIGTERM to validator (PID: $PID)..."
        kill -TERM "$PID"
        
        # Wait for graceful shutdown (max 30 seconds)
        echo "Waiting for graceful shutdown..."
        wait_count=0
        while kill -0 "$PID" 2>/dev/null && [ $wait_count -lt 30 ]; do
            sleep 1
            wait_count=$((wait_count + 1))
            if [ $((wait_count % 5)) -eq 0 ]; then
                echo "Still waiting... ($wait_count seconds)"
            fi
        done
        
        # Check if process is still running
        if kill -0 "$PID" 2>/dev/null; then
            echo "Validator did not stop gracefully, forcing shutdown..."
            kill -KILL "$PID" 2>/dev/null || true
            sleep 1
        fi
        
        echo "Validator stopped (PID: $PID)"
    else
        echo "Process with PID $PID is not running"
    fi
    
    # Remove PID file
    rm -f "$PID_FILE"
else
    echo "No PID file found at $PID_FILE"
    
    # Try to find and kill any running validator processes
    echo "Searching for running validator processes..."
    
    # Find uvicorn processes running validator.validator_server
    VALIDATOR_PIDS=$(pgrep -f "validator.validator_server" 2>/dev/null || true)
    
    if [ -n "$VALIDATOR_PIDS" ]; then
        echo "Found validator process(es): $VALIDATOR_PIDS"
        echo "Stopping them..."
        for pid in $VALIDATOR_PIDS; do
            echo "  Killing PID: $pid"
            kill -TERM "$pid" 2>/dev/null || true
        done
        
        # Wait a moment
        sleep 2
        
        # Force kill if still running
        for pid in $VALIDATOR_PIDS; do
            if kill -0 "$pid" 2>/dev/null; then
                echo "  Force killing PID: $pid"
                kill -KILL "$pid" 2>/dev/null || true
            fi
        done
        
        echo "Validator processes stopped"
    else
        echo "No running validator processes found"
    fi
fi

# Also clean up any stray uvicorn processes (optional, commented out for safety)
# Uncomment if you want to force-stop all uvicorn processes
# echo "Cleaning up any stray uvicorn processes..."
# killall -9 uvicorn 2>/dev/null || true

echo ""
echo "Validator shutdown complete"
echo ""
echo "To start the validator again, run:"
echo "  ./run-validator.sh"
echo "or for headless mode:"
echo "  ./run-validator.sh --headless"
