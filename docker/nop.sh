#!/bin/bash
# No-operation script to keep container running
# Useful for debugging and manual container access

echo "Container started in NOP mode. Use 'docker exec' to access."

socat TCP-LISTEN:9944,fork TCP:local-chain:9944 &
socat TCP-LISTEN:8080,fork TCP:challenge-api:8080 &
socat TCP-LISTEN:6379,fork TCP:loosh-redis:6379 &

tail -f /dev/null

