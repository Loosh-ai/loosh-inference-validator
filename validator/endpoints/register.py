from typing import Dict, Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from validator.config import ValidatorConfig
from validator.db.users import UserDatabaseManager

router = APIRouter()

class RegisterRequest(BaseModel):
    """Request model for user registration."""
    hotkey: str

class RegisterResponse(BaseModel):
    """Response model for user registration."""
    success: bool
    message: str

def get_config_dependency():
    """Get config for dependency injection."""
    return ValidatorConfig()

@router.post("/", response_model=RegisterResponse)
async def register_user(
    request: RegisterRequest,
    config: ValidatorConfig = Depends(get_config_dependency)
) -> Dict[str, Any]:
    """Register a new user with hotkey."""
    try:
        # Initialize user database manager
        user_db = UserDatabaseManager(config.users_db_path)
        
        # Create or update user record
        user_db.update_user(request.hotkey, is_available=True)
        
        return RegisterResponse(
            success=True,
            message=f"User with hotkey {request.hotkey} registered successfully"
        ).model_dump()
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to register user: {str(e)}"
        )
