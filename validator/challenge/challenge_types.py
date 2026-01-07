from datetime import datetime
from enum import Enum
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field
from validator.challenge_api.models import Challenge

class ChallengeType(str, Enum):
    """Types of challenges that can be sent to miners."""
    INFERENCE = "inference"

class InferenceChallenge(BaseModel):
    """Challenge for LLM inference."""
    prompt: str = Field(..., description="The prompt to send to the LLM")
    model: str = Field(..., description="The model to use for inference")
    max_tokens: int = Field(..., description="Maximum number of tokens to generate")
    temperature: float = Field(..., description="Sampling temperature")
    top_p: float = Field(..., description="Top-p sampling parameter")
    correlation_id: Optional[str] = Field(None, description="Correlation ID for tracking requests end-to-end")
    created_at: datetime = Field(default_factory=datetime.utcnow)

class InferenceResponse(BaseModel):
    """Response from a miner for an inference challenge."""
    response_text: str = Field(..., description="The generated text response")
    response_time_ms: int = Field(..., description="Time taken to generate response in milliseconds")
    correlation_id: Optional[str] = Field(None, description="Correlation ID for tracking requests end-to-end")
    created_at: datetime = Field(default_factory=datetime.utcnow)

class ValidationResult(BaseModel):
    """Result of validating a miner's response."""
    is_valid: bool = Field(..., description="Whether the response is valid")
    score: float = Field(..., description="Score for the response")
    error: Optional[str] = Field(None, description="Error message if validation failed")
    created_at: datetime = Field(default_factory=datetime.utcnow)

class ChallengeTask:
    """Task for sending a challenge to a miner."""
    def __init__(
        self,
        challenge_id: str,
        node_id: int,
        task: Any,  # asyncio.Task
        timestamp: datetime,
        challenge_orig: Challenge,
        challenge: InferenceChallenge,
        miner_hotkey: str
    ):
        self.challenge_id = challenge_id
        self.node_id = node_id
        self.task = task
        self.timestamp = timestamp
        self.challenge_orig = challenge_orig
        self.challenge = challenge
        self.miner_hotkey = miner_hotkey
        self.responses: List[InferenceResponse] = []  # Will be populated after responses are received 