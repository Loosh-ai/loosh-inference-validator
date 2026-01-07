#!/bin/bash

# Individual curl test for /get_stats endpoint
# Usage: ./curl_get_stats.sh [BASE_URL]

BASE_URL=${1:-"http://localhost:8000"}
MASTER_HOTKEY="test-master-hotkey-12345"

echo "🧪 Testing /get_stats endpoint"
echo "Base URL: $BASE_URL"
echo ""

echo "1. Testing with master-hotkey header:"
curl -v \
    -H "master-hotkey: $MASTER_HOTKEY" \
    "$BASE_URL/get_stats"

echo ""
echo ""

echo "2. Testing without master-hotkey header (should fail):"
curl -v \
    "$BASE_URL/get_stats"

echo ""
echo ""

echo "3. Testing with different master-hotkey:"
curl -v \
    -H "master-hotkey: different-hotkey-67890" \
    "$BASE_URL/get_stats"

echo ""
echo ""

echo "4. Testing with pretty-printed JSON output:"
curl -s \
    -H "master-hotkey: $MASTER_HOTKEY" \
    "$BASE_URL/get_stats" | python -m json.tool
