from typing import Dict, Any

from fastapi import APIRouter, Depends, Header
from pydantic import BaseModel

from validator.config import ValidatorConfig

router = APIRouter()

class SetRunningRequest(BaseModel):
    """Request for setting running status."""
    running: bool

class SetRunningResponse(BaseModel):
    """Response for set running status."""
    success: bool = True
    message: str = "Status updated successfully"

def get_config_dependency():
    """Get config for dependency injection."""
    return ValidatorConfig()

@router.post("")
async def set_running(
    request: SetRunningRequest,
    master_hotkey: str = Header(..., alias="master-hotkey"),
    config: ValidatorConfig = Depends(get_config_dependency)
) -> Dict[str, Any]:
    """Set the running status of the validator."""
    # Mock implementation - just return success
    return SetRunningResponse(
        success=True,
        message=f"Validator running status set to: {request.running}"
    ).model_dump()
