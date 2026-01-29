"""
Fiber MLTS client for validator to send encrypted challenges to miners.

Handles handshake, symmetric key management, and encrypted challenge transmission.
"""

import json
import time
import uuid
from typing import Dict, Optional, Tuple

import httpx
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.fernet import Fernet, InvalidToken
from loguru import logger

from fiber.chain.chain_utils import load_hotkey_keypair
from validator.config import ValidatorConfig


class MinerFiberClient:
    """
    Fiber client for validator to send encrypted challenges to miners.
    
    Manages handshake with miners, symmetric key caching, and encrypted challenge transmission.
    """
    
    def __init__(
        self,
        validator_hotkey_ss58: str,
        config: ValidatorConfig,
        key_ttl_seconds: int = 3600,
        handshake_timeout_seconds: int = 30
    ):
        """
        Initialize Fiber client for miner communication.
        
        Args:
            validator_hotkey_ss58: Validator's SS58 address (hotkey)
            config: Validator configuration
            key_ttl_seconds: Time-to-live for symmetric keys (default: 1 hour)
            handshake_timeout_seconds: Timeout for handshake operations
        """
        self.validator_hotkey_ss58 = validator_hotkey_ss58
        self.config = config
        self.key_ttl_seconds = key_ttl_seconds
        self.handshake_timeout_seconds = handshake_timeout_seconds
        
        # Cache: {miner_endpoint: (fernet_instance, symmetric_key_uuid, expiration_time)}
        self._symmetric_key_cache: Dict[str, Tuple[Fernet, str, float]] = {}
        
        # Load Validator's hotkey for signing
        try:
            self.validator_hotkey = load_hotkey_keypair(
                config.wallet_name,
                config.hotkey_name
            )
            logger.info(f"MinerFiberClient initialized with validator hotkey: {self.validator_hotkey.ss58_address}")
        except Exception as e:
            logger.error(f"Failed to load Validator hotkey for MinerFiberClient: {e}")
            self.validator_hotkey = None
    
    async def _perform_handshake(self, miner_endpoint: str, client: httpx.AsyncClient) -> bool:
        """Perform handshake with miner."""
        if not self.validator_hotkey:
            logger.error("Validator hotkey not loaded, cannot perform handshake.")
            return False
        
        try:
            # 1. Fetch miner public key
            public_key_url = f"{miner_endpoint}/fiber/public-key"
            response = await client.get(public_key_url, timeout=self.handshake_timeout_seconds)
            response.raise_for_status()
            miner_public_key_pem = response.json()["public_key"]
            miner_public_key = serialization.load_pem_public_key(miner_public_key_pem.encode('utf-8'))
            
            # 2. Generate symmetric key and UUID
            symmetric_key = Fernet.generate_key()
            symmetric_key_uuid = str(uuid.uuid4())
            
            # 3. Encrypt symmetric key with miner's public key
            encrypted_symmetric_key = miner_public_key.encrypt(
                symmetric_key,
                padding.OAEP(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None
                )
            ).hex()
            
            # 4. Sign the request with Validator's hotkey
            timestamp = time.time()
            nonce = str(uuid.uuid4())
            message = f"{timestamp}.{nonce}.{self.validator_hotkey.ss58_address}"
            signature = f"0x{self.validator_hotkey.sign(message.encode('utf-8')).hex()}"
            
            # 5. POST encrypted symmetric key to miner
            key_exchange_url = f"{miner_endpoint}/fiber/key-exchange"
            request_data = {
                "encrypted_symmetric_key": encrypted_symmetric_key,
                "symmetric_key_uuid": symmetric_key_uuid,
                "timestamp": timestamp,
                "nonce": nonce,
                "signature": signature,
                "validator_hotkey_ss58": self.validator_hotkey.ss58_address
            }
            
            logger.debug(f"Sending key exchange request to {key_exchange_url} with data: {list(request_data.keys())}")
            response = await client.post(key_exchange_url, json=request_data, timeout=self.handshake_timeout_seconds)
            
            if response.status_code == 422:
                # 422 Unprocessable Entity - validation error
                try:
                    error_detail = response.json()
                    logger.error(
                        f"Miner {miner_endpoint} returned 422 validation error: {error_detail}. "
                        f"Request data keys: {list(request_data.keys())}, "
                        f"Request data types: {[(k, type(v).__name__) for k, v in request_data.items()]}"
                    )
                except:
                    error_text = response.text[:500]
                    logger.error(
                        f"Miner {miner_endpoint} returned 422 validation error. Response: {error_text}"
                    )
                return False
            
            response.raise_for_status()
            
            response_json = response.json()
            if not response_json.get("success"):
                logger.error(f"Miner {miner_endpoint} rejected key exchange: {response_json.get('message')}")
                return False
            
            # Cache the Fernet instance
            fernet_instance = Fernet(symmetric_key)
            expiration_time = time.time() + self.key_ttl_seconds
            self._symmetric_key_cache[miner_endpoint] = (fernet_instance, symmetric_key_uuid, expiration_time)
            
            logger.info(f"Fiber handshake successful with miner {miner_endpoint} (UUID: {symmetric_key_uuid[:8]}...)")
            return True
            
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 422:
                # 422 Unprocessable Entity - validation error
                try:
                    error_detail = e.response.json()
                    logger.error(
                        f"Miner {miner_endpoint} returned 422 validation error: {error_detail}. "
                        f"Request data keys: {list(request_data.keys())}, "
                        f"Request data types: {[(k, type(v).__name__) for k, v in request_data.items()]}"
                    )
                except:
                    error_text = e.response.text[:500]
                    logger.error(
                        f"Miner {miner_endpoint} returned 422 validation error. Response: {error_text}"
                    )
            else:
                logger.error(f"HTTP error during Fiber handshake with miner {miner_endpoint}: {e.response.status_code} - {e.response.text[:200]}")
            return False
        except httpx.RequestError as e:
            logger.error(f"Network error during Fiber handshake with miner {miner_endpoint}: {e}")
            return False
        except Exception as e:
            logger.error(f"Error during Fiber handshake with miner {miner_endpoint}: {e}", exc_info=True)
            return False
    
    async def send_encrypted_challenge(
        self,
        miner_endpoint: str,
        challenge_data: Dict,
        client: httpx.AsyncClient
    ) -> Optional[Dict]:
        """
        Sends an encrypted challenge to the miner and receives encrypted response.
        Performs handshake if needed.
        
        Returns:
            Decrypted response dictionary, or None if failed
        """
        fernet_instance, symmetric_key_uuid, expiration_time = self._symmetric_key_cache.get(miner_endpoint, (None, None, 0))
        
        # Check if handshake is needed or expired
        if not fernet_instance or time.time() > expiration_time:
            logger.info(f"Performing Fiber handshake with miner {miner_endpoint}...")
            if not await self._perform_handshake(miner_endpoint, client):
                logger.error(f"Failed to establish Fiber handshake with miner {miner_endpoint}.")
                return None
            fernet_instance, symmetric_key_uuid, _ = self._symmetric_key_cache[miner_endpoint]
        
        try:
            # Encrypt challenge payload
            json_payload = json.dumps(challenge_data).encode('utf-8')
            encrypted_payload = fernet_instance.encrypt(json_payload)
            
            # Send encrypted payload
            challenge_url = f"{miner_endpoint}/fiber/challenge"
            headers = {
                "Content-Type": "application/octet-stream",
                "x-fiber-validator-hotkey-ss58": self.validator_hotkey.ss58_address,
                "x-fiber-symmetric-key-uuid": symmetric_key_uuid
            }
            
            response = await client.post(challenge_url, content=encrypted_payload, headers=headers, timeout=5*60.0)
            
            # Handle 401 (key invalid on miner side) with retry
            if response.status_code == 401:
                logger.warning(f"Miner {miner_endpoint} rejected key (401) - clearing cache and retrying handshake")
                if miner_endpoint in self._symmetric_key_cache:
                    del self._symmetric_key_cache[miner_endpoint]
                if await self._perform_handshake(miner_endpoint, client):
                    # Retry once with fresh key
                    fernet_instance, symmetric_key_uuid, _ = self._symmetric_key_cache[miner_endpoint]
                    encrypted_payload = fernet_instance.encrypt(json_payload)
                    headers["x-fiber-symmetric-key-uuid"] = symmetric_key_uuid
                    response = await client.post(challenge_url, content=encrypted_payload, headers=headers, timeout=5*60.0)
                else:
                    logger.error(f"Failed to re-establish handshake with miner {miner_endpoint} after 401")
                    return None
            
            response.raise_for_status()
            
            if response.status_code != 200:
                logger.warning(f"Miner {miner_endpoint} returned {response.status_code} for encrypted challenge: {response.text}")
                return None
            
            # Decrypt response
            encrypted_response = response.content
            decrypted_response = fernet_instance.decrypt(encrypted_response, ttl=self.key_ttl_seconds).decode('utf-8')
            response_data = json.loads(decrypted_response)
            
            logger.info(f"Encrypted challenge sent and response received from miner {miner_endpoint} (UUID: {symmetric_key_uuid[:8]}...)")
            return response_data
            
        except httpx.RequestError as e:
            logger.error(f"Network error sending encrypted challenge to miner {miner_endpoint}: {e}")
            return None
        except InvalidToken:
            logger.error(f"Symmetric key expired or invalid for miner {miner_endpoint}. Retrying handshake.")
            # Invalidate cache and retry handshake once
            del self._symmetric_key_cache[miner_endpoint]
            if await self._perform_handshake(miner_endpoint, client):
                return await self.send_encrypted_challenge(miner_endpoint, challenge_data, client)
            return None
        except Exception as e:
            logger.error(f"Error sending encrypted challenge to miner {miner_endpoint}: {e}", exc_info=True)
            return None


