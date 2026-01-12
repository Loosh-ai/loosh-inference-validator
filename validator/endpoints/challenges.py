"""Endpoint for receiving pushed challenges from the challenge API."""

import asyncio
from datetime import datetime
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, Header, HTTPException, status
from pydantic import BaseModel, Field
from loguru import logger

from validator.config import ValidatorConfig


router = APIRouter()


# Challenge queue for storing pushed challenges
# This is a simple in-memory queue; consider using Redis for production
_challenge_queue: asyncio.Queue = asyncio.Queue()
_received_challenge_ids: set = set()


class ChallengeCreate(BaseModel):
    """Request model for creating/pushing a challenge."""
    id: str
    correlation_id: Optional[str] = None
    prompt: str
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None
    created_at: Optional[datetime] = None
    status: str = "available"
    requester: Optional[str] = None


class ChallengeResponse(BaseModel):
    """Response model for challenge creation."""
    success: bool
    message: str
    challenge_id: str


class ChallengeQueueStats(BaseModel):
    """Stats about the challenge queue."""
    queue_size: int
    total_received: int


def get_config_dependency() -> ValidatorConfig:
    """Get config for dependency injection."""
    return ValidatorConfig()


def verify_api_key(
    authorization: Optional[str] = Header(None),
    config: ValidatorConfig = Depends(get_config_dependency)
) -> bool:
    """
    Verify the API key from Authorization header (DEPRECATED).
    
    This function is deprecated. API key authentication has been replaced with
    Fiber MLTS encryption. Use the /fiber/challenge endpoint instead.
    """
    # Always return True - API key auth is disabled in favor of Fiber
    # This function is kept for backward compatibility only
    return True


@router.post(
    "",
    response_model=ChallengeResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Receive a pushed challenge",
    description="Endpoint for receiving challenges pushed from the challenge API"
)
async def receive_challenge(
    challenge: ChallengeCreate,
    _authorized: bool = True,  # API key auth disabled - use Fiber-encrypted /fiber/challenge endpoint instead
    config: ValidatorConfig = Depends(get_config_dependency)
) -> Dict[str, Any]:
    """
    Receive a challenge pushed from the challenge API (legacy endpoint).
    
    DEPRECATED: This endpoint is kept for backward compatibility during migration.
    New challenges should use the Fiber-encrypted /fiber/challenge endpoint.
    
    The challenge is added to an internal queue for processing by the validator.
    """
    global _received_challenge_ids
    
    try:
        # Check if challenge already exists
        if challenge.id in _received_challenge_ids:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Challenge {challenge.id} already exists"
            )
        
        # Set created_at if not provided
        if challenge.created_at is None:
            challenge.created_at = datetime.utcnow()
        
        # Add to queue
        await _challenge_queue.put(challenge)
        _received_challenge_ids.add(challenge.id)
        
        logger.info(
            f"Received challenge {challenge.id[:8]}... "
            f"(queue size: {_challenge_queue.qsize()})"
        )
        
        return ChallengeResponse(
            success=True,
            message="Challenge received successfully",
            challenge_id=challenge.id
        ).model_dump()
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error receiving challenge: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to receive challenge: {str(e)}"
        )


@router.get(
    "/stats",
    response_model=ChallengeQueueStats,
    summary="Get challenge queue stats",
    description="Get statistics about the challenge queue"
)
async def get_challenge_stats() -> Dict[str, Any]:
    """Get statistics about the challenge queue."""
    return ChallengeQueueStats(
        queue_size=_challenge_queue.qsize(),
        total_received=len(_received_challenge_ids)
    ).model_dump()


# Helper functions for the main loop to consume challenges

async def get_next_challenge(timeout: float = 1.0) -> Optional[ChallengeCreate]:
    """
    Get the next challenge from the queue.
    
    Args:
        timeout: How long to wait for a challenge (in seconds)
        
    Returns:
        The next challenge, or None if queue is empty after timeout
    """
    try:
        challenge = await asyncio.wait_for(
            _challenge_queue.get(),
            timeout=timeout
        )
        return challenge
    except asyncio.TimeoutError:
        return None


def get_queue_size() -> int:
    """Get the current size of the challenge queue."""
    return _challenge_queue.qsize()


def clear_challenge_history(max_size: int = 10000) -> None:
    """
    Clear old challenge IDs from history to prevent memory growth.
    
    Args:
        max_size: Maximum number of challenge IDs to keep
    """
    global _received_challenge_ids
    if len(_received_challenge_ids) > max_size:
        # Keep the most recent half
        # Note: sets don't preserve order, so this is a rough cleanup
        _received_challenge_ids = set(list(_received_challenge_ids)[-max_size // 2:])
        logger.info(f"Cleaned up challenge history, now tracking {len(_received_challenge_ids)} IDs")

