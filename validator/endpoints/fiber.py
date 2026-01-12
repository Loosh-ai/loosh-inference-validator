"""
Fiber MLTS endpoints for secure challenge reception.

Provides endpoints for public key exchange, symmetric key exchange, and encrypted challenge reception.
"""

import json
from typing import Dict, Any, Optional
from fastapi import APIRouter, Header, HTTPException, status, Request
from pydantic import BaseModel
from loguru import logger

from validator.network.fiber_server import FiberServer
from validator.endpoints.challenges import get_next_challenge as get_next_challenge_from_queue, ChallengeCreate


router = APIRouter()

# Global Fiber server instance (initialized in validator_server.py)
fiber_server: Optional[FiberServer] = None


class KeyExchangeRequest(BaseModel):
    """Request model for symmetric key exchange."""
    encrypted_symmetric_key: str  # Base64-encoded encrypted key
    key_uuid: str
    timestamp: float
    signature: str
    hotkey_ss58: str


class KeyExchangeResponse(BaseModel):
    """Response model for symmetric key exchange."""
    success: bool
    message: str


class ChallengeRequest(BaseModel):
    """Request model for encrypted challenge (body is encrypted, this is just for validation)."""
    pass


@router.get("/public-key", summary="Get Fiber public key", description="Returns the validator's RSA public key for key exchange")
async def get_public_key() -> Dict[str, str]:
    """
    Get the validator's RSA public key.
    
    Returns:
        Public key in PEM format
    """
    if not fiber_server:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Fiber server not initialized"
        )
    
    try:
        public_key_pem = fiber_server.get_public_key_pem()
        return {"public_key": public_key_pem}
    except Exception as e:
        logger.error(f"Error getting public key: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get public key: {str(e)}"
        )


@router.post(
    "/key-exchange",
    response_model=KeyExchangeResponse,
    status_code=status.HTTP_200_OK,
    summary="Exchange symmetric key",
    description="Exchange encrypted symmetric key for secure communication"
)
async def exchange_key(request: KeyExchangeRequest) -> KeyExchangeResponse:
    """
    Exchange symmetric key with client.
    
    Client sends RSA-encrypted symmetric key along with UUID, timestamp, signature, and SS58 address.
    Server decrypts and stores the symmetric key for future encrypted communications.
    """
    if not fiber_server:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Fiber server not initialized"
        )
    
    try:
        # Decode base64-encoded encrypted key
        import base64
        encrypted_key_bytes = base64.b64decode(request.encrypted_symmetric_key)
        
        # Exchange symmetric key
        success = fiber_server.exchange_symmetric_key(
            encrypted_symmetric_key=encrypted_key_bytes,
            key_uuid=request.key_uuid,
            timestamp=request.timestamp,
            signature=request.signature,
            hotkey_ss58=request.hotkey_ss58
        )
        
        if success:
            return KeyExchangeResponse(
                success=True,
                message="Symmetric key exchanged successfully"
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to exchange symmetric key"
            )
            
    except Exception as e:
        logger.error(f"Error in key exchange: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Key exchange failed: {str(e)}"
        )


@router.post(
    "/challenge",
    status_code=status.HTTP_201_CREATED,
    summary="Receive encrypted challenge",
    description="Receive and decrypt an encrypted challenge from the Challenge API"
)
async def receive_encrypted_challenge(
    request: Request,
    symmetric_key_uuid: str = Header(..., alias="symmetric-key-uuid"),
    hotkey_ss58_address: str = Header(..., alias="hotkey-ss58-address")
) -> Dict[str, Any]:
    """
    Receive an encrypted challenge from the Challenge API.
    
    The request body is encrypted using Fernet with the symmetric key identified by
    symmetric_key_uuid and hotkey_ss58_address headers.
    
    Args:
        request: FastAPI request object (contains encrypted body)
        symmetric_key_uuid: UUID of the symmetric key (from header)
        hotkey_ss58_address: SS58 address of the Challenge API hotkey (from header)
    
    Returns:
        Challenge response indicating success
    """
    if not fiber_server:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Fiber server not initialized"
        )
    
    try:
        # Read encrypted body
        encrypted_body = await request.body()
        
        # Decrypt payload
        decrypted_bytes = fiber_server.decrypt_payload(
            encrypted_payload=encrypted_body,
            hotkey_ss58=hotkey_ss58_address,
            key_uuid=symmetric_key_uuid
        )
        
        if not decrypted_bytes:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Failed to decrypt payload. Key may be expired or invalid."
            )
        
        # Parse decrypted JSON
        try:
            challenge_data = json.loads(decrypted_bytes.decode('utf-8'))
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse decrypted challenge JSON: {e}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid JSON in decrypted payload"
            )
        
        # Convert to ChallengeCreate model
        try:
            challenge = ChallengeCreate(**challenge_data)
        except Exception as e:
            logger.error(f"Failed to create ChallengeCreate from decrypted data: {e}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid challenge data: {str(e)}"
            )
        
        # Add to challenge queue (reuse existing queue system)
        from validator.endpoints.challenges import _challenge_queue, _received_challenge_ids
        
        # Check if challenge already exists
        if challenge.id in _received_challenge_ids:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Challenge {challenge.id} already exists"
            )
        
        # Add to queue
        await _challenge_queue.put(challenge)
        _received_challenge_ids.add(challenge.id)
        
        logger.info(
            f"Received encrypted challenge {challenge.id[:8]}... "
            f"from {hotkey_ss58_address[:8]}... "
            f"(queue size: {_challenge_queue.qsize()})"
        )
        
        return {
            "success": True,
            "message": "Challenge received successfully",
            "challenge_id": challenge.id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error receiving encrypted challenge: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to receive challenge: {str(e)}"
        )


@router.get("/stats", summary="Get Fiber server stats", description="Get statistics about the Fiber server")
async def get_fiber_stats() -> Dict[str, Any]:
    """Get Fiber server statistics."""
    if not fiber_server:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Fiber server not initialized"
        )
    
    return fiber_server.get_stats()


