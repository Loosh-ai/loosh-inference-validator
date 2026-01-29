from datetime import datetime
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field

# API MODELS [
# TODO: Common models repo or import from loosh_api

# loosh-challenge-api/app.model
class Challenge(BaseModel):
    """Schema for a challenge with creation timestamp."""
    id: str
    correlation_id: Optional[str] = None
    prompt: Optional[str] = None
    model: Optional[str] = None
    messages: Optional[List[Dict[str, Any]]] = None
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Any] = None
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
    """Schema for creating a response to a challenge (legacy 1:1)."""
    id: str
    text: str
    original_request: Challenge


class Response(BaseModel):
    """Schema for a response with creation timestamp (legacy 1:1)."""
    id: str
    text: str
    original_request: Challenge
    created_at: datetime

class ResponseResult(Response):
    """Schema for a challenge response with timing information."""
    response_time_ms: int


# =============================================================================
# New Batch Response Models (F3 - 1:N Response Storage)
# =============================================================================

class TokenUsage(BaseModel):
    """Token usage statistics for cost attribution."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class MinerResponseData(BaseModel):
    """Data for a single miner response in a batch."""
    miner_id: str = Field(..., description="Miner hotkey or unique identifier")
    miner_uid: Optional[int] = Field(None, description="Miner UID on subnet")
    text: str = Field(..., description="Response text content")
    tool_calls: Optional[List[Dict[str, Any]]] = Field(None, description="Tool calls")
    finish_reason: str = Field("stop", description="Finish reason")
    response_time_ms: int = Field(..., description="Response time in ms")
    usage: TokenUsage = Field(..., description="Token usage (REQUIRED)")


class EvaluationResult(BaseModel):
    """Evaluation results computed by validator."""
    consensus_score: float = Field(..., description="Consensus score")
    emissions: Dict[str, float] = Field(..., description="Emission scores per miner_id")
    best_miner_id: str = Field(..., description="Miner ID of the best response")
    heatmap_path: Optional[str] = Field(None, description="Path to uploaded heatmap")
    narrative: Optional[str] = Field(None, description="Consensus narrative")
    sybil_detection: Optional[Dict[str, Any]] = Field(None, description="Sybil detection results")


class ResponseBatchSubmit(BaseModel):
    """Full batch submission to Challenge API."""
    challenge_id: str = Field(..., description="Challenge UUID")
    responses: List[MinerResponseData] = Field(..., description="All miner responses")
    evaluation: EvaluationResult = Field(..., description="Evaluation results")
    validator_hotkey: str = Field(..., description="Validator's hotkey")
    original_request: Optional[Challenge] = Field(None, description="Original challenge")


# Alias models
# TODO: review alias names

ChallengeAPIRequest = Challenge
ChallengeAPIResponse = ChallengeResponse

ResponseAPICreate = ResponseCreate
ResponseAPIRequest = Response
ResponseAPIResult = ResponseResult

# API MODELS ]
