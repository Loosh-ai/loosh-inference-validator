from datetime import datetime
from typing import Optional, Dict, Any
from pydantic import BaseModel

# API MODELS [
# TODO: Common models repo or import from loosh_api

# loosh-challenge-api/app.model
class Challenge(BaseModel):
    """Schema for a challenge with creation timestamp."""
    id: str
    correlation_id: Optional[str] = None
    prompt: str
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None
    created_at: datetime
    status: str = "available"
    requester: Optional[str] = None

class ChallengeResponse(Challenge):
    """Schema for a challenge response with timing information."""
    response_time_ms: int

class ResponseCreate(BaseModel):
    """Schema for creating a response to a challenge."""
    id: str
    text: str
    original_request: Challenge


class Response(BaseModel):
    """Schema for a response with creation timestamp."""
    id: str
    text: str
    original_request: Challenge
    created_at: datetime

class ResponseResult(Response):
    """Schema for a challenge response with timing information."""
    response_time_ms: int

# Alias models
# TODO: review alias names

ChallengeAPIRequest = Challenge
ChallengeAPIResponse = ChallengeResponse

ResponseAPICreate = ResponseCreate
ResponseAPIRequest = Response
ResponseAPIResult = ResponseResult

# API MODELS ]
