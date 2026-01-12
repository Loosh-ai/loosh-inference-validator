from typing import Dict, Any, Optional

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel

from validator.config import ValidatorConfig
from validator.endpoints.challenges import get_queue_size

router = APIRouter()

# Global reference to active tasks (set by main_loop)
_active_tasks_ref = None
_challenge_semaphore_ref = None
_max_concurrent_ref = None

def set_capacity_tracking(active_tasks, challenge_semaphore, max_concurrent):
    """Set global references for capacity tracking."""
    global _active_tasks_ref, _challenge_semaphore_ref, _max_concurrent_ref
    _active_tasks_ref = active_tasks
    _challenge_semaphore_ref = challenge_semaphore
    _max_concurrent_ref = max_concurrent

class AvailabilityResponse(BaseModel):
    """Response for availability check."""
    available: bool = True

class HealthResponse(BaseModel):
    """Response for health check."""
    status: str = "healthy"
    queue_size: int = 0
    processing_stats: Dict[str, Any] = {}

def get_config_dependency():
    """Get config for dependency injection."""
    return ValidatorConfig()

@router.get("")
async def check_availability(
    request: Request,
    config: ValidatorConfig = Depends(get_config_dependency)
) -> Dict[str, Any]:
    """
    Check if the validator is available.
    
    Accepts optional master-hotkey header for logging/authentication purposes.
    """
    # Extract master hotkey from headers (case-insensitive)
    master_hotkey = None
    for header_name, header_value in request.headers.items():
        if header_name.lower() == "master-hotkey":
            master_hotkey = header_value
            break
    
    # Log if master hotkey is provided (for debugging)
    if master_hotkey:
        import logging
        logger = logging.getLogger(__name__)
        logger.debug(f"Availability check from master: {master_hotkey[:8]}...")
    
    return AvailabilityResponse().model_dump()

@router.get("/health")
@router.get("/healthz")
async def health_check() -> Dict[str, Any]:
    """
    Health check endpoint for Challenge API to verify validator status.
    
    Returns validator status, queue size, and processing stats including capacity.
    """
    queue_size = get_queue_size()
    
    # Get actual capacity information
    active_count = 0
    max_concurrent = 0
    available_capacity = 0
    
    if _active_tasks_ref is not None:
        active_count = len(_active_tasks_ref)
    
    if _max_concurrent_ref is not None:
        max_concurrent = _max_concurrent_ref
        available_capacity = max(0, max_concurrent - active_count)
    
    # Also check semaphore if available
    semaphore_available = None
    if _challenge_semaphore_ref is not None:
        # Get available permits from semaphore
        semaphore_available = _challenge_semaphore_ref._value
    
    return HealthResponse(
        status="healthy",
        queue_size=queue_size,
        processing_stats={
            "queue_size": queue_size,
            "challenge_mode": "push",
            "active_challenges": active_count,
            "max_concurrent": max_concurrent,
            "available_capacity": available_capacity,
            "semaphore_available": semaphore_available
        }
    ).model_dump()
