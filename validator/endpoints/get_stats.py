from typing import Dict, Any

from fastapi import APIRouter, Depends, Header
from pydantic import BaseModel

from validator.config import ValidatorConfig

router = APIRouter()

class StatsResponse(BaseModel):
    """Response for get stats."""
    total_challenges: int = 0
    successful_responses: int = 0
    failed_responses: int = 0
    average_response_time_ms: float = 0.0
    uptime_seconds: int = 0
    active_miners: int = 0

def get_config_dependency():
    """Get config for dependency injection."""
    return ValidatorConfig()

@router.get("")
async def get_stats(
    master_hotkey: str = Header(..., alias="master-hotkey"),
    config: ValidatorConfig = Depends(get_config_dependency)
) -> Dict[str, Any]:
    """Get validator statistics."""
    # Mock implementation - return sample stats
    return StatsResponse(
        total_challenges=150,
        successful_responses=142,
        failed_responses=8,
        average_response_time_ms=1250.5,
        uptime_seconds=3600,
        active_miners=3
    ).model_dump()
