from typing import Dict, Any, Optional

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel

from validator.config import ValidatorConfig
from validator.endpoints.challenges import get_queue_size

router = APIRouter()

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
    
    Returns validator status, queue size, and processing stats.
    """
    queue_size = get_queue_size()
    
    return HealthResponse(
        status="healthy",
        queue_size=queue_size,
        processing_stats={
            "queue_size": queue_size,
            "challenge_mode": "push"  # Could be dynamic based on config
        }
    ).model_dump()
