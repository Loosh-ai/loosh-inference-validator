#!/bin/bash

# Individual curl test for /availability endpoint
# Usage: ./curl_availability.sh [BASE_URL]

BASE_URL=${1:-"http://localhost:8000"}
MASTER_HOTKEY="test-master-hotkey-12345"

echo "🧪 Testing /availability endpoint"
echo "Base URL: $BASE_URL"
echo ""

echo "1. Testing with master-hotkey header:"
curl -v \
    -H "master-hotkey: $MASTER_HOTKEY" \
    "$BASE_URL/availability"

echo ""
echo ""

echo "2. Testing without master-hotkey header (should fail):"
curl -v \
    "$BASE_URL/availability"

echo ""
echo ""

echo "3. Testing with different master-hotkey:"
curl -v \
    -H "master-hotkey: different-hotkey-67890" \
    "$BASE_URL/availability"
