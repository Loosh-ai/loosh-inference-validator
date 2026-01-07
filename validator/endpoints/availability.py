from typing import Dict, Any

from fastapi import APIRouter, Depends, Header
from pydantic import BaseModel

from validator.config import ValidatorConfig

router = APIRouter()

class AvailabilityResponse(BaseModel):
    """Response for availability check."""
    available: bool = True

def get_config_dependency():
    """Get config for dependency injection."""
    return ValidatorConfig()

@router.get("")
async def check_availability(
    master_hotkey: str = Header(..., alias="master-hotkey"),
    config: ValidatorConfig = Depends(get_config_dependency)
) -> Dict[str, Any]:
    """Check if the validator is available."""
    return AvailabilityResponse().model_dump()
