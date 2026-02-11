"""
Fiber MLTS client for validator to send encrypted callbacks to Challenge API.

Handles handshake, symmetric key management, and encrypted callback transmission.
"""

import base64
import json
import time
import uuid
from typing import Any, Dict, Optional, Tuple

import httpx
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.backends import default_backend
from cryptography.fernet import Fernet
from loguru import logger


# ---------------------------------------------------------------------------
# Module-level validator keypair storage
# ---------------------------------------------------------------------------
# Set once at startup from main.py so every ValidatorFiberClient instance
# can sign handshake messages without threading the keypair through every
# call-site.
_validator_keypair: Optional[Any] = None  # substrateinterface.Keypair


def set_validator_keypair(keypair: Any) -> None:
    """Store the validator sr25519 keypair for Fiber signing (call once at startup)."""
    global _validator_keypair
    _validator_keypair = keypair
    logger.info(
        f"Validator keypair set for Fiber signing: {keypair.ss58_address[:16]}..."
    )


def get_validator_keypair() -> Optional[Any]:
    """Return the cached validator keypair (or None if not yet set)."""
    return _validator_keypair


class ValidatorFiberClient:
    """
    Fiber client for validator to send encrypted callbacks to Challenge API.
    
    Manages handshake with Challenge API, symmetric key caching, and encrypted callback transmission.
    """
    
    def __init__(
        self,
        validator_hotkey_ss58: str,
        keypair: Optional[Any] = None,
        key_ttl_seconds: int = 3600,
        handshake_timeout_seconds: int = 30
    ):
        """
        Initialize Fiber client.
        
        Args:
            validator_hotkey_ss58: Validator's SS58 address (hotkey)
            keypair: Optional sr25519 Keypair for signing handshake messages.
                     If None, falls back to the module-level keypair set via
                     ``set_validator_keypair()``.
            key_ttl_seconds: Time-to-live for symmetric keys (default: 1 hour)
            handshake_timeout_seconds: Timeout for handshake operations
        """
        self.validator_hotkey_ss58 = validator_hotkey_ss58
        self.keypair = keypair or get_validator_keypair()
        self.key_ttl_seconds = key_ttl_seconds
        self.handshake_timeout_seconds = handshake_timeout_seconds
        
        # Safety margin: refresh keys 60 seconds before server-side expiration
        # This prevents race conditions where client thinks key is valid but server has expired it
        self.key_refresh_margin_seconds = 60
        
        # Cache: {challenge_api_endpoint: (fernet, key_uuid, handshake_time)}
        self._key_cache: Dict[str, Tuple[Fernet, str, float]] = {}
        
        if not self.keypair:
            logger.warning(
                "ValidatorFiberClient created without keypair — "
                "handshake signatures will be empty (reduced security)"
            )
    
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
            
            # Sign request with sr25519 keypair (Bittensor wallet)
            signature = ""
            if self.keypair:
                try:
                    message = f"{timestamp}.{self.validator_hotkey_ss58}.{key_uuid}"
                    sig_bytes = self.keypair.sign(message.encode("utf-8"))
                    signature = sig_bytes.hex()
                except Exception as e:
                    logger.warning(f"Failed to sign handshake message: {e}")
                    signature = ""
            else:
                logger.debug("No keypair available — handshake will be unsigned")
            
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
            
            # Check if key is still valid (with safety margin to prevent race conditions)
            # Refresh keys before they expire on the server side
            effective_ttl = self.key_ttl_seconds - self.key_refresh_margin_seconds
            key_age = time.time() - handshake_time
            
            if key_age < effective_ttl:
                return (fernet, key_uuid)
            else:
                # Key expired or approaching expiration, remove from cache and re-handshake
                logger.debug(
                    f"Symmetric key expired or near expiration for {challenge_api_endpoint} "
                    f"(age: {key_age:.0f}s, effective TTL: {effective_ttl}s), re-handshaking"
                )
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
            elif response.status_code == 401:
                # Key invalid/expired on Challenge API side - clear cache and retry once
                logger.warning(
                    f"Challenge API rejected key (401) - clearing cache and retrying handshake"
                )
                if challenge_api_endpoint in self._key_cache:
                    del self._key_cache[challenge_api_endpoint]
                
                # Retry with fresh handshake
                handshake_result = await self._ensure_handshake(challenge_api_endpoint, client)
                if not handshake_result:
                    logger.error(f"Failed to re-establish handshake with Challenge API after 401")
                    return False
                
                fernet, key_uuid = handshake_result
                encrypted_payload = fernet.encrypt(response_json)
                headers["symmetric-key-uuid"] = key_uuid
                
                retry_response = await client.post(
                    url,
                    content=encrypted_payload,
                    headers=headers,
                    timeout=30.0
                )
                
                if retry_response.status_code in (200, 201):
                    logger.info(
                        f"Encrypted callback sent successfully after re-handshake "
                        f"(challenge_id: {response_data.get('id', 'unknown')[:8]}...)"
                    )
                    return True
                else:
                    logger.error(
                        f"Failed to send encrypted callback after re-handshake: "
                        f"{retry_response.status_code} - {retry_response.text}"
                    )
                    return False
            else:
                logger.error(
                    f"Failed to send encrypted callback to Challenge API: "
                    f"{response.status_code} - {response.text}"
                )
                return False
                
        except Exception as e:
            logger.error(f"Error sending encrypted callback: {e}", exc_info=True)
            # Clear cache to force re-handshake next time
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
    
    async def send_encrypted_response_batch(
        self,
        challenge_api_endpoint: str,
        batch_data: Dict,
        client: httpx.AsyncClient,
        max_retries: int = 3,
        retry_base_delay: float = 2.0
    ) -> bool:
        """
        Send encrypted response batch to Challenge API with retry logic.
        
        Retries on transient errors (404 challenge-not-found, 500 deadlock/server errors)
        with exponential backoff. Non-retryable errors (400, 409) fail immediately.
        
        Args:
            challenge_api_endpoint: Base URL of Challenge API
            batch_data: Response batch data dictionary (will be JSON-encoded and encrypted)
            client: HTTP client for making requests
            max_retries: Maximum number of retry attempts for transient errors
            retry_base_delay: Base delay in seconds between retries (doubles each attempt)
        
        Returns:
            True if batch sent successfully, False otherwise
        """
        import asyncio
        
        challenge_id = batch_data.get('challenge_id', 'unknown')
        challenge_id_short = challenge_id[:8] if isinstance(challenge_id, str) else str(challenge_id)
        
        # Status codes that are transient and worth retrying
        RETRYABLE_STATUS_CODES = {404, 500, 502, 503, 504}
        
        for attempt in range(max_retries + 1):
            try:
                # Ensure handshake exists
                handshake_result = await self._ensure_handshake(challenge_api_endpoint, client)
                if not handshake_result:
                    logger.error(
                        f"Failed to establish handshake with Challenge API {challenge_api_endpoint} "
                        f"(challenge_id: {challenge_id_short}...)"
                    )
                    return False
                
                fernet, key_uuid = handshake_result
                
                # Encode batch data as JSON
                batch_json = json.dumps(batch_data).encode('utf-8')
                
                # Encrypt with Fernet
                encrypted_payload = fernet.encrypt(batch_json)
                
                # Send encrypted batch
                url = f"{challenge_api_endpoint.rstrip('/')}/fiber/response/batch"
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
                    if attempt > 0:
                        logger.info(
                            f"Encrypted response batch sent successfully after {attempt} "
                            f"retries (challenge_id: {challenge_id_short}...)"
                        )
                    else:
                        logger.info(
                            f"Encrypted response batch sent successfully to Challenge API "
                            f"(challenge_id: {challenge_id_short}...)"
                        )
                    return True
                elif response.status_code == 409:
                    # Conflict — responses already exist. Not retryable.
                    logger.warning(
                        f"Responses already exist for challenge {challenge_id} - "
                        f"batch submission rejected (409 Conflict)"
                    )
                    return False
                elif response.status_code == 401:
                    # Key invalid/expired on Challenge API side - clear cache and retry once
                    logger.warning(
                        f"Challenge API rejected key (401) for challenge {challenge_id_short}... "
                        f"- clearing cache and retrying handshake"
                    )
                    if challenge_api_endpoint in self._key_cache:
                        del self._key_cache[challenge_api_endpoint]
                    
                    # Retry with fresh handshake
                    handshake_result = await self._ensure_handshake(challenge_api_endpoint, client)
                    if not handshake_result:
                        logger.error(f"Failed to re-establish handshake with Challenge API after 401")
                        return False
                    
                    fernet, key_uuid = handshake_result
                    encrypted_payload = fernet.encrypt(batch_json)
                    headers["symmetric-key-uuid"] = key_uuid
                    
                    retry_response = await client.post(
                        url,
                        content=encrypted_payload,
                        headers=headers,
                        timeout=30.0
                    )
                    
                    if retry_response.status_code in (200, 201):
                        logger.info(
                            f"Encrypted response batch sent successfully after re-handshake "
                            f"(challenge_id: {challenge_id_short}...)"
                        )
                        return True
                    else:
                        logger.error(
                            f"Failed to send encrypted response batch after re-handshake: "
                            f"{retry_response.status_code} - {retry_response.text}"
                        )
                        return False
                elif response.status_code in RETRYABLE_STATUS_CODES:
                    # Transient error — retry with backoff
                    if attempt < max_retries:
                        delay = retry_base_delay * (2 ** attempt)
                        logger.warning(
                            f"Transient error submitting response batch for challenge "
                            f"{challenge_id_short}... (HTTP {response.status_code}: "
                            f"{response.text[:200]}). "
                            f"Retrying in {delay:.1f}s (attempt {attempt + 1}/{max_retries})..."
                        )
                        await asyncio.sleep(delay)
                        continue
                    else:
                        logger.error(
                            f"Failed to send encrypted response batch to Challenge API after "
                            f"{max_retries} retries (challenge_id: {challenge_id_short}...): "
                            f"{response.status_code} - {response.text}"
                        )
                        return False
                else:
                    # Non-retryable client error (400, etc.)
                    logger.error(
                        f"Failed to send encrypted response batch to Challenge API "
                        f"(challenge_id: {challenge_id_short}...): "
                        f"{response.status_code} - {response.text}"
                    )
                    return False
                    
            except httpx.TimeoutException:
                if attempt < max_retries:
                    delay = retry_base_delay * (2 ** attempt)
                    logger.warning(
                        f"Timeout sending response batch for challenge {challenge_id_short}... "
                        f"Retrying in {delay:.1f}s (attempt {attempt + 1}/{max_retries})..."
                    )
                    await asyncio.sleep(delay)
                    continue
                else:
                    logger.error(
                        f"Timeout sending response batch for challenge {challenge_id_short}... "
                        f"after {max_retries} retries"
                    )
                    return False
            except Exception as e:
                logger.error(
                    f"Error sending encrypted response batch (challenge_id: {challenge_id_short}...): "
                    f"{e}",
                    exc_info=True
                )
                # If handshake failed, clear cache to force re-handshake next time
                if challenge_api_endpoint in self._key_cache:
                    del self._key_cache[challenge_api_endpoint]
                return False
        
        # Should not reach here, but just in case
        return False
    
    async def send_encrypted_upload(
        self,
        challenge_api_endpoint: str,
        file_data: bytes,
        filename: str,
        challenge_id: str,
        file_type: str,
        content_type: str,
        client: httpx.AsyncClient
    ) -> Optional[str]:
        """
        Send encrypted file upload (heatmap/quality plot) to Challenge API.
        
        Args:
            challenge_api_endpoint: Base URL of Challenge API
            file_data: Raw file bytes
            filename: Original filename
            challenge_id: Challenge ID (UUID) for this upload
            file_type: Type of file - "heatmap" or "quality_plot"
            content_type: MIME content type (e.g., "image/png")
            client: HTTP client for making requests
        
        Returns:
            Filename if upload successful, None otherwise
        """
        try:
            # Ensure handshake exists
            handshake_result = await self._ensure_handshake(challenge_api_endpoint, client)
            if not handshake_result:
                logger.error(f"Failed to establish handshake with Challenge API {challenge_api_endpoint}")
                return None
            
            fernet, key_uuid = handshake_result
            
            # Create upload payload as JSON with base64-encoded file data
            upload_payload = {
                "challenge_id": challenge_id,
                "file_type": file_type,
                "filename": filename,
                "content_type": content_type,
                "data": base64.b64encode(file_data).decode('utf-8')
            }
            
            # Encode as JSON
            upload_json = json.dumps(upload_payload).encode('utf-8')
            
            # Encrypt with Fernet
            encrypted_payload = fernet.encrypt(upload_json)
            
            # Send encrypted upload
            url = f"{challenge_api_endpoint.rstrip('/')}/fiber/heatmap/upload"
            headers = {
                "symmetric-key-uuid": key_uuid,
                "hotkey-ss58-address": self.validator_hotkey_ss58,
                "Content-Type": "application/octet-stream"
            }
            
            response = await client.post(
                url,
                content=encrypted_payload,
                headers=headers,
                timeout=60.0  # Longer timeout for file uploads
            )
            
            # Handle key expiration with automatic retry
            if response.status_code == 401:
                logger.warning(
                    f"Received 401 from Challenge API - key may have expired. "
                    f"Clearing cache and retrying with new handshake..."
                )
                # Clear the cached key and retry once
                if challenge_api_endpoint in self._key_cache:
                    del self._key_cache[challenge_api_endpoint]
                
                # Re-establish handshake
                handshake_result = await self._ensure_handshake(challenge_api_endpoint, client)
                if not handshake_result:
                    logger.error(f"Failed to re-establish handshake after 401 error")
                    return None
                
                fernet, key_uuid = handshake_result
                
                # Re-encrypt with new key
                encrypted_payload = fernet.encrypt(upload_json)
                headers["symmetric-key-uuid"] = key_uuid
                
                # Retry the upload
                response = await client.post(
                    url,
                    content=encrypted_payload,
                    headers=headers,
                    timeout=60.0
                )
            
            if response.status_code in (200, 201):
                result = response.json()
                stored_filename = result.get("filename", filename)
                logger.info(
                    f"Encrypted {file_type} upload successful for challenge {challenge_id[:8]}...: "
                    f"{stored_filename}"
                )
                return stored_filename
            else:
                logger.error(
                    f"Failed to send encrypted {file_type} upload to Challenge API: "
                    f"{response.status_code} - {response.text}"
                )
                return None
                
        except Exception as e:
            logger.error(f"Error sending encrypted {file_type} upload: {e}", exc_info=True)
            # If handshake failed, clear cache to force re-handshake next time
            if challenge_api_endpoint in self._key_cache:
                del self._key_cache[challenge_api_endpoint]
            return None
    
    def get_stats(self) -> Dict:
        """Get client statistics."""
        return {
            "cached_endpoints": len(self._key_cache),
            "key_ttl_seconds": self.key_ttl_seconds
        }


