"""
Fiber MLTS client for validator to send encrypted callbacks to Challenge API.

Handles handshake, symmetric key management, and encrypted callback transmission.
"""

import base64
import json
import time
import uuid
from typing import Dict, Optional, Tuple

import httpx
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.backends import default_backend
from cryptography.fernet import Fernet
from loguru import logger


class ValidatorFiberClient:
    """
    Fiber client for validator to send encrypted callbacks to Challenge API.
    
    Manages handshake with Challenge API, symmetric key caching, and encrypted callback transmission.
    """
    
    def __init__(
        self,
        validator_hotkey_ss58: str,
        private_key: Optional[rsa.RSAPrivateKey] = None,
        key_ttl_seconds: int = 3600,
        handshake_timeout_seconds: int = 30
    ):
        """
        Initialize Fiber client.
        
        Args:
            validator_hotkey_ss58: Validator's SS58 address (hotkey)
            private_key: Optional RSA private key for signing (if None, will need to be loaded)
            key_ttl_seconds: Time-to-live for symmetric keys (default: 1 hour)
            handshake_timeout_seconds: Timeout for handshake operations
        """
        self.validator_hotkey_ss58 = validator_hotkey_ss58
        self.private_key = private_key
        self.key_ttl_seconds = key_ttl_seconds
        self.handshake_timeout_seconds = handshake_timeout_seconds
        
        # Cache: {challenge_api_endpoint: (fernet, key_uuid, handshake_time)}
        self._key_cache: Dict[str, Tuple[Fernet, str, float]] = {}
    
    def _load_private_key_from_hotkey(self) -> Optional[rsa.RSAPrivateKey]:
        """
        Load private key from hotkey (for signing).
        
        This is a placeholder - actual implementation would load from Bittensor wallet.
        """
        if self.private_key:
            return self.private_key
        
        # TODO: Load from Bittensor wallet at ~/.bittensor/wallets
        logger.warning("Private key not provided - signing will be skipped")
        return None
    
    async def _fetch_challenge_api_public_key(
        self,
        challenge_api_endpoint: str,
        client: httpx.AsyncClient
    ) -> Optional[str]:
        """
        Fetch Challenge API's RSA public key.
        
        Args:
            challenge_api_endpoint: Base URL of Challenge API (e.g., http://challenge-api:8080)
            client: HTTP client for making requests
        
        Returns:
            PEM-encoded public key, or None if failed
        """
        try:
            url = f"{challenge_api_endpoint.rstrip('/')}/fiber/public-key"
            response = await client.get(url, timeout=self.handshake_timeout_seconds)
            
            if response.status_code != 200:
                logger.error(
                    f"Failed to fetch public key from {challenge_api_endpoint}: "
                    f"{response.status_code} - {response.text}"
                )
                return None
            
            data = response.json()
            return data.get("public_key")
            
        except Exception as e:
            logger.error(f"Error fetching Challenge API public key: {e}")
            return None
    
    async def _perform_handshake(
        self,
        challenge_api_endpoint: str,
        challenge_api_public_key_pem: str,
        client: httpx.AsyncClient
    ) -> Optional[Tuple[Fernet, str]]:
        """
        Perform handshake with Challenge API.
        
        Args:
            challenge_api_endpoint: Base URL of Challenge API
            challenge_api_public_key_pem: Challenge API's RSA public key in PEM format
            client: HTTP client for making requests
        
        Returns:
            Tuple of (Fernet instance, key_uuid) if successful, None otherwise
        """
        try:
            # Load Challenge API's public key
            challenge_api_public_key = serialization.load_pem_public_key(
                challenge_api_public_key_pem.encode('utf-8'),
                backend=default_backend()
            )
            
            # Generate symmetric key and UUID
            symmetric_key = Fernet.generate_key()
            key_uuid = str(uuid.uuid4())
            timestamp = time.time()
            
            # Encrypt symmetric key with Challenge API's RSA public key
            encrypted_symmetric_key = challenge_api_public_key.encrypt(
                symmetric_key,
                padding.OAEP(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None
                )
            )
            
            # Create Fernet instance
            fernet = Fernet(symmetric_key)
            
            # Sign request (if private key available)
            signature = ""
            private_key = self._load_private_key_from_hotkey()
            if private_key:
                # Sign: timestamp + validator_hotkey_ss58 + key_uuid
                message = f"{timestamp}.{self.validator_hotkey_ss58}.{key_uuid}"
                # TODO: Implement proper signing with Bittensor keypair
                signature = "placeholder_signature"
            
            # Send key exchange request
            url = f"{challenge_api_endpoint.rstrip('/')}/fiber/key-exchange"
            payload = {
                "encrypted_symmetric_key": base64.b64encode(encrypted_symmetric_key).decode('utf-8'),
                "key_uuid": key_uuid,
                "timestamp": timestamp,
                "signature": signature,
                "validator_hotkey_ss58": self.validator_hotkey_ss58
            }
            
            response = await client.post(
                url,
                json=payload,
                timeout=self.handshake_timeout_seconds
            )
            
            if response.status_code != 200:
                logger.error(
                    f"Key exchange failed for {challenge_api_endpoint}: "
                    f"{response.status_code} - {response.text}"
                )
                return None
            
            result = response.json()
            if not result.get("success"):
                logger.error(f"Key exchange returned failure: {result}")
                return None
            
            logger.info(
                f"Handshake successful with Challenge API {challenge_api_endpoint[:50]}... "
                f"(UUID: {key_uuid[:8]}...)"
            )
            
            return (fernet, key_uuid)
            
        except Exception as e:
            logger.error(f"Error performing handshake: {e}", exc_info=True)
            return None
    
    async def _ensure_handshake(
        self,
        challenge_api_endpoint: str,
        client: httpx.AsyncClient
    ) -> Optional[Tuple[Fernet, str]]:
        """
        Ensure handshake exists and is valid for Challenge API.
        
        Args:
            challenge_api_endpoint: Base URL of Challenge API
            client: HTTP client for making requests
        
        Returns:
            Tuple of (Fernet instance, key_uuid) if successful, None otherwise
        """
        # Check cache
        if challenge_api_endpoint in self._key_cache:
            fernet, key_uuid, handshake_time = self._key_cache[challenge_api_endpoint]
            
            # Check if key is still valid (not expired)
            if time.time() - handshake_time < self.key_ttl_seconds:
                return (fernet, key_uuid)
            else:
                # Key expired, remove from cache
                logger.debug(f"Symmetric key expired for {challenge_api_endpoint}, re-handshaking")
                del self._key_cache[challenge_api_endpoint]
        
        # Perform handshake
        # First, fetch Challenge API's public key
        public_key_pem = await self._fetch_challenge_api_public_key(challenge_api_endpoint, client)
        if not public_key_pem:
            return None
        
        # Perform key exchange
        result = await self._perform_handshake(challenge_api_endpoint, public_key_pem, client)
        if not result:
            return None
        
        fernet, key_uuid = result
        
        # Cache the key
        self._key_cache[challenge_api_endpoint] = (fernet, key_uuid, time.time())
        
        return (fernet, key_uuid)
    
    async def send_encrypted_callback(
        self,
        challenge_api_endpoint: str,
        response_data: Dict,
        client: httpx.AsyncClient
    ) -> bool:
        """
        Send encrypted callback to Challenge API.
        
        Args:
            challenge_api_endpoint: Base URL of Challenge API
            response_data: Response data dictionary (will be JSON-encoded and encrypted)
            client: HTTP client for making requests
        
        Returns:
            True if callback sent successfully, False otherwise
        """
        try:
            # Ensure handshake exists
            handshake_result = await self._ensure_handshake(challenge_api_endpoint, client)
            if not handshake_result:
                logger.error(f"Failed to establish handshake with Challenge API {challenge_api_endpoint}")
                return False
            
            fernet, key_uuid = handshake_result
            
            # Encode response data as JSON
            response_json = json.dumps(response_data).encode('utf-8')
            
            # Encrypt with Fernet
            encrypted_payload = fernet.encrypt(response_json)
            
            # Send encrypted callback
            url = f"{challenge_api_endpoint.rstrip('/')}/fiber/callback"
            headers = {
                "symmetric-key-uuid": key_uuid,
                "hotkey-ss58-address": self.validator_hotkey_ss58,
                "Content-Type": "application/octet-stream"
            }
            
            response = await client.post(
                url,
                content=encrypted_payload,
                headers=headers,
                timeout=30.0
            )
            
            if response.status_code in (200, 201):
                logger.info(
                    f"Encrypted callback sent successfully to Challenge API "
                    f"(challenge_id: {response_data.get('id', 'unknown')[:8]}...)"
                )
                return True
            else:
                logger.error(
                    f"Failed to send encrypted callback to Challenge API: "
                    f"{response.status_code} - {response.text}"
                )
                return False
                
        except Exception as e:
            logger.error(f"Error sending encrypted callback: {e}", exc_info=True)
            # If handshake failed, clear cache to force re-handshake next time
            if challenge_api_endpoint in self._key_cache:
                del self._key_cache[challenge_api_endpoint]
            return False
    
    def clear_cache(self, challenge_api_endpoint: Optional[str] = None) -> None:
        """
        Clear key cache for Challenge API.
        
        Args:
            challenge_api_endpoint: Specific endpoint to clear, or None to clear all
        """
        if challenge_api_endpoint:
            if challenge_api_endpoint in self._key_cache:
                del self._key_cache[challenge_api_endpoint]
                logger.debug(f"Cleared cache for {challenge_api_endpoint}")
        else:
            self._key_cache.clear()
            logger.debug("Cleared all key caches")
    
    def get_stats(self) -> Dict:
        """Get client statistics."""
        return {
            "cached_endpoints": len(self._key_cache),
            "key_ttl_seconds": self.key_ttl_seconds
        }


