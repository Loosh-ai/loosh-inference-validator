#!/bin/bash

# Test script for /register endpoint
# Usage: ./test_register.sh

echo "Testing /register endpoint..."
echo "================================"

# Test 1: Register a new user
echo "Test 1: Register new user"
echo "------------------------"
curl -X POST "http://localhost:8000/register/" \
  -H "Content-Type: application/json" \
  -d '{"hotkey": "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"}' \
  -w "\nHTTP Status: %{http_code}\n" \
  -s

echo -e "\n"

# Test 2: Register another user
echo "Test 2: Register another user"
echo "-----------------------------"
curl -X POST "http://localhost:8000/register/" \
  -H "Content-Type: application/json" \
  -d '{"hotkey": "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty"}' \
  -w "\nHTTP Status: %{http_code}\n" \
  -s

echo -e "\n"

# Test 3: Register same user again (should update)
echo "Test 3: Register same user again (should update)"
echo "------------------------------------------------"
curl -X POST "http://localhost:8000/register/" \
  -H "Content-Type: application/json" \
  -d '{"hotkey": "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"}' \
  -w "\nHTTP Status: %{http_code}\n" \
  -s

echo -e "\n"

# Test 4: Invalid request (missing hotkey)
echo "Test 4: Invalid request (missing hotkey)"
echo "----------------------------------------"
curl -X POST "http://localhost:8000/register/" \
  -H "Content-Type: application/json" \
  -d '{}' \
  -w "\nHTTP Status: %{http_code}\n" \
  -s

echo -e "\n"

# Test 5: Invalid request (wrong content type)
echo "Test 5: Invalid request (wrong content type)"
echo "--------------------------------------------"
curl -X POST "http://localhost:8000/register/" \
  -H "Content-Type: text/plain" \
  -d '{"hotkey": "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"}' \
  -w "\nHTTP Status: %{http_code}\n" \
  -s

echo -e "\n"
echo "Tests completed!"
echo "================"
echo "Note: Make sure the validator server is running on localhost:8000"
echo "Start server with: python validator/validator_server.py"
