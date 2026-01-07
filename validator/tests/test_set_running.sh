#!/bin/bash

# Individual curl test for /set_running endpoint
# Usage: ./curl_set_running.sh [BASE_URL]

BASE_URL=${1:-"http://localhost:8000"}
MASTER_HOTKEY="test-master-hotkey-12345"

echo "🧪 Testing /set_running endpoint"
echo "Base URL: $BASE_URL"
echo ""

echo "1. Testing with running=true:"
curl -v \
    -X POST \
    -H "Content-Type: application/json" \
    -H "master-hotkey: $MASTER_HOTKEY" \
    -d '{"running": true}' \
    "$BASE_URL/set_running"

echo ""
echo ""

echo "2. Testing with running=false:"
curl -v \
    -X POST \
    -H "Content-Type: application/json" \
    -H "master-hotkey: $MASTER_HOTKEY" \
    -d '{"running": false}' \
    "$BASE_URL/set_running"

echo ""
echo ""

echo "3. Testing without master-hotkey header (should fail):"
curl -v \
    -X POST \
    -H "Content-Type: application/json" \
    -d '{"running": true}' \
    "$BASE_URL/set_running"

echo ""
echo ""

echo "4. Testing with invalid JSON (should fail):"
curl -v \
    -X POST \
    -H "Content-Type: application/json" \
    -H "master-hotkey: $MASTER_HOTKEY" \
    -d '{"running": invalid}' \
    "$BASE_URL/set_running"

echo ""
echo ""

echo "5. Testing with missing 'running' field (should fail):"
curl -v \
    -X POST \
    -H "Content-Type: application/json" \
    -H "master-hotkey: $MASTER_HOTKEY" \
    -d '{"status": true}' \
    "$BASE_URL/set_running"
