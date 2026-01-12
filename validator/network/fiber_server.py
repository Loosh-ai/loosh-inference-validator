"""
Fiber MLTS (Multi-Layer Transport Security) server implementation.

Provides RSA-based key exchange and symmetric key encryption for secure communication.
"""

import time
from typing import Dict, Optional, Tuple

from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.backends import default_backend
from cryptography.fernet import Fernet
from loguru import logger


class FiberServer:
    """
    Fiber server for handling secure key exchange and encrypted payloads.
    
    Manages RSA keypair, symmetric key storage, and provides decryption capabilities.
    """
    
    def __init__(self, key_ttl_seconds: int = 3600):
        """
        Initialize Fiber server.
        
        Args:
            key_ttl_seconds: Time-to-live for symmetric keys in seconds (default: 1 hour)
        """
        self.key_ttl_seconds = key_ttl_seconds
        self._rsa_private_key: Optional[rsa.RSAPrivateKey] = None
        self._rsa_public_key: Optional[rsa.RSAPublicKey] = None
        self._symmetric_keys: Dict[str, Dict[str, Tuple[Fernet, float]]] = {}
        # Structure: {hotkey_ss58: {uuid: (fernet_instance, timestamp)}}
        
        # Generate RSA keypair on initialization
        self._generate_rsa_keypair()
    
    def _generate_rsa_keypair(self) -> None:
        """Generate RSA keypair for public key exchange."""
        logger.info("Generating RSA keypair for Fiber server")
        self._rsa_private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
            backend=default_backend()
        )
        self._rsa_public_key = self._rsa_private_key.public_key()
        logger.info("RSA keypair generated successfully")
    
    def get_public_key_pem(self) -> str:
        """
        Get RSA public key in PEM format.
        
        Returns:
            PEM-encoded public key string
        """
        if not self._rsa_public_key:
            raise RuntimeError("RSA public key not initialized")
        
        pem = self._rsa_public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        return pem.decode('utf-8')
    
    def exchange_symmetric_key(
        self,
        encrypted_symmetric_key: bytes,
        key_uuid: str,
        timestamp: float,
        signature: str,
        hotkey_ss58: str
    ) -> bool:
        """
        Exchange symmetric key with client.
        
        Args:
            encrypted_symmetric_key: RSA-encrypted symmetric key
            key_uuid: Unique identifier for this symmetric key
            timestamp: Timestamp/nonce for anti-replay protection
            signature: Signature from client (for verification)
            hotkey_ss58: Client's SS58 address (hotkey)
        
        Returns:
            True if key exchange successful, False otherwise
        """
        try:
            # Decrypt symmetric key with RSA private key
            symmetric_key_bytes = self._rsa_private_key.decrypt(
                encrypted_symmetric_key,
                padding.OAEP(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None
                )
            )
            
            # Create Fernet instance from symmetric key
            fernet = Fernet(symmetric_key_bytes)
            
            # Store symmetric key with timestamp
            if hotkey_ss58 not in self._symmetric_keys:
                self._symmetric_keys[hotkey_ss58] = {}
            
            self._symmetric_keys[hotkey_ss58][key_uuid] = (fernet, timestamp)
            
            logger.info(
                f"Symmetric key exchanged for hotkey {hotkey_ss58[:8]}... "
                f"with UUID {key_uuid[:8]}..."
            )
            
            # Clean up expired keys
            self._cleanup_expired_keys()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to exchange symmetric key: {e}")
            return False
    
    def get_symmetric_key(
        self,
        hotkey_ss58: str,
        key_uuid: str
    ) -> Optional[Fernet]:
        """
        Get symmetric key (Fernet instance) for decryption.
        
        Args:
            hotkey_ss58: Client's SS58 address
            key_uuid: Symmetric key UUID
        
        Returns:
            Fernet instance if found and valid, None otherwise
        """
        if hotkey_ss58 not in self._symmetric_keys:
            return None
        
        if key_uuid not in self._symmetric_keys[hotkey_ss58]:
            return None
        
        fernet, timestamp = self._symmetric_keys[hotkey_ss58][key_uuid]
        
        # Check if key is expired
        if time.time() - timestamp > self.key_ttl_seconds:
            logger.warning(
                f"Symmetric key expired for hotkey {hotkey_ss58[:8]}... "
                f"UUID {key_uuid[:8]}..."
            )
            del self._symmetric_keys[hotkey_ss58][key_uuid]
            if not self._symmetric_keys[hotkey_ss58]:
                del self._symmetric_keys[hotkey_ss58]
            return None
        
        return fernet
    
    def decrypt_payload(
        self,
        encrypted_payload: bytes,
        hotkey_ss58: str,
        key_uuid: str
    ) -> Optional[bytes]:
        """
        Decrypt payload using stored symmetric key.
        
        Args:
            encrypted_payload: Encrypted payload bytes
            hotkey_ss58: Client's SS58 address
            key_uuid: Symmetric key UUID
        
        Returns:
            Decrypted payload bytes, or None if decryption fails
        """
        fernet = self.get_symmetric_key(hotkey_ss58, key_uuid)
        if not fernet:
            return None
        
        try:
            decrypted = fernet.decrypt(encrypted_payload)
            return decrypted
        except Exception as e:
            logger.error(f"Failed to decrypt payload: {e}")
            return None
    
    def _cleanup_expired_keys(self) -> None:
        """Remove expired symmetric keys from cache."""
        current_time = time.time()
        expired_hotkeys = []
        
        for hotkey_ss58, keys in self._symmetric_keys.items():
            expired_uuids = []
            for key_uuid, (_, timestamp) in keys.items():
                if current_time - timestamp > self.key_ttl_seconds:
                    expired_uuids.append(key_uuid)
            
            for key_uuid in expired_uuids:
                del keys[key_uuid]
            
            if not keys:
                expired_hotkeys.append(hotkey_ss58)
        
        for hotkey_ss58 in expired_hotkeys:
            del self._symmetric_keys[hotkey_ss58]
        
        if expired_hotkeys:
            logger.debug(f"Cleaned up expired keys for {len(expired_hotkeys)} hotkeys")
    
    def get_stats(self) -> Dict:
        """Get server statistics."""
        total_keys = sum(len(keys) for keys in self._symmetric_keys.values())
        return {
            "active_hotkeys": len(self._symmetric_keys),
            "total_symmetric_keys": total_keys,
            "key_ttl_seconds": self.key_ttl_seconds
        }

