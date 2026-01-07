#!/bin/bash

# Test script for validator endpoints
# Usage: ./test_validator_endpoints.sh [BASE_URL]
# Default BASE_URL: http://localhost:8000

set -e

# Configuration
BASE_URL=${1:-"http://localhost:8000"}
MASTER_HOTKEY="test-master-hotkey-12345"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_test() {
    echo -e "${BLUE}🧪 Testing: $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

# Function to test endpoint
test_endpoint() {
    local method=$1
    local endpoint=$2
    local data=$3
    local expected_status=$4
    local description=$5
    
    print_test "$description"
    
    if [ "$method" = "GET" ]; then
        response=$(curl -s -w "\n%{http_code}" \
            -H "master-hotkey: $MASTER_HOTKEY" \
            "$BASE_URL$endpoint")
    else
        response=$(curl -s -w "\n%{http_code}" \
            -X "$method" \
            -H "Content-Type: application/json" \
            -H "master-hotkey: $MASTER_HOTKEY" \
            -d "$data" \
            "$BASE_URL$endpoint")
    fi
    
    # Extract status code (last line)
    status_code=$(echo "$response" | tail -n1)
    # Extract response body (all but last line)
    response_body=$(echo "$response" | head -n -1)
    
    if [ "$status_code" = "$expected_status" ]; then
        print_success "Status code: $status_code"
        echo "Response: $response_body"
        echo ""
        return 0
    else
        print_error "Expected status $expected_status, got $status_code"
        echo "Response: $response_body"
        echo ""
        return 1
    fi
}

# Function to test endpoint without master-hotkey (should fail)
test_endpoint_no_auth() {
    local method=$1
    local endpoint=$2
    local data=$3
    local description=$5
    
    print_test "$description (no auth - should fail)"
    
    if [ "$method" = "GET" ]; then
        response=$(curl -s -w "\n%{http_code}" "$BASE_URL$endpoint")
    else
        response=$(curl -s -w "\n%{http_code}" \
            -X "$method" \
            -H "Content-Type: application/json" \
            -d "$data" \
            "$BASE_URL$endpoint")
    fi
    
    status_code=$(echo "$response" | tail -n1)
    response_body=$(echo "$response" | head -n -1)
    
    if [ "$status_code" = "422" ] || [ "$status_code" = "400" ]; then
        print_success "Correctly rejected without auth (status: $status_code)"
        echo "Response: $response_body"
        echo ""
        return 0
    else
        print_error "Expected auth failure (422/400), got $status_code"
        echo "Response: $response_body"
        echo ""
        return 1
    fi
}

echo "🚀 Starting validator endpoint tests..."
echo "Base URL: $BASE_URL"
echo "Master Hotkey: $MASTER_HOTKEY"
echo ""

# Test 1: Availability endpoint
test_endpoint "GET" "/availability" "" "200" "GET /availability endpoint"

# Test 2: Availability endpoint without auth
test_endpoint_no_auth "GET" "/availability" "" "GET /availability endpoint"

# Test 3: Set running endpoint (true)
test_endpoint "POST" "/set_running" '{"running": true}' "200" "POST /set_running (running=true)"

# Test 4: Set running endpoint (false)
test_endpoint "POST" "/set_running" '{"running": false}' "200" "POST /set_running (running=false)"

# Test 5: Set running endpoint without auth
test_endpoint_no_auth "POST" "/set_running" '{"running": true}' "POST /set_running endpoint"

# Test 6: Set running endpoint with invalid JSON
print_test "POST /set_running with invalid JSON (should fail)"
response=$(curl -s -w "\n%{http_code}" \
    -X "POST" \
    -H "Content-Type: application/json" \
    -H "master-hotkey: $MASTER_HOTKEY" \
    -d '{"running": invalid}' \
    "$BASE_URL/set_running")

status_code=$(echo "$response" | tail -n1)
response_body=$(echo "$response" | head -n -1)

if [ "$status_code" = "422" ]; then
    print_success "Correctly rejected invalid JSON (status: $status_code)"
    echo "Response: $response_body"
else
    print_error "Expected JSON validation error (422), got $status_code"
    echo "Response: $response_body"
fi
echo ""

# Test 7: Get stats endpoint
test_endpoint "GET" "/get_stats" "" "200" "GET /get_stats endpoint"

# Test 8: Get stats endpoint without auth
test_endpoint_no_auth "GET" "/get_stats" "" "GET /get_stats endpoint"

# Test 9: Test with different master hotkey
print_test "GET /availability with different master hotkey"
response=$(curl -s -w "\n%{http_code}" \
    -H "master-hotkey: different-hotkey-67890" \
    "$BASE_URL/availability")

status_code=$(echo "$response" | tail -n1)
response_body=$(echo "$response" | head -n -1)

if [ "$status_code" = "200" ]; then
    print_success "Different hotkey accepted (status: $status_code)"
    echo "Response: $response_body"
else
    print_warning "Different hotkey rejected (status: $status_code)"
    echo "Response: $response_body"
fi
echo ""

# Test 10: Test invalid endpoint
print_test "GET /invalid-endpoint (should return 404)"
response=$(curl -s -w "\n%{http_code}" \
    -H "master-hotkey: $MASTER_HOTKEY" \
    "$BASE_URL/invalid-endpoint")

status_code=$(echo "$response" | tail -n1)
response_body=$(echo "$response" | head -n -1)

if [ "$status_code" = "404" ]; then
    print_success "Correctly returned 404 for invalid endpoint"
    echo "Response: $response_body"
else
    print_error "Expected 404, got $status_code"
    echo "Response: $response_body"
fi
echo ""

echo "🏁 Test suite completed!"
echo ""
echo "📋 Summary:"
echo "- All endpoints should return 200 with proper master-hotkey"
echo "- All endpoints should reject requests without master-hotkey"
echo "- Invalid JSON should be rejected with 422"
echo "- Invalid endpoints should return 404"
echo ""
echo "💡 To run the validator server for testing:"
echo "   cd /path/to/loosh-inference-subnet"
echo "   source .venv/bin/activate"
echo "   python validator/validator_server.py"
echo ""
echo "💡 To run tests against a different server:"
echo "   ./test_validator_endpoints.sh http://your-server:port"
