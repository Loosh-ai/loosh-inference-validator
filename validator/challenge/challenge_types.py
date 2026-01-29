from datetime import datetime
from enum import Enum
from typing import Optional, List, Dict, Any, Union

from pydantic import BaseModel, Field, model_validator
from validator.challenge_api.models import Challenge


class ChallengeType(str, Enum):
    """Types of challenges that can be sent to miners."""
    INFERENCE = "inference"


# =============================================================================
# Token Usage (F3)
# =============================================================================

class TokenUsage(BaseModel):
    """Token usage statistics for cost attribution (F3)."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


# =============================================================================
# Challenge Models
# =============================================================================

class InferenceChallenge(BaseModel):
    """
    Challenge for LLM inference.
    
    Supports both legacy prompt-based and OpenAI-compatible message-based formats.
    For backward compatibility, either prompt OR messages must be provided.
    """
    # Legacy prompt (backward compatible)
    prompt: Optional[str] = Field(None, description="The prompt to send to the LLM (legacy)")
    
    # OpenAI-compatible format (preferred)
    messages: Optional[List[Dict[str, Any]]] = Field(None, description="OpenAI-format messages")
    tools: Optional[List[Dict[str, Any]]] = Field(None, description="Tool definitions for tool calling")
    tool_choice: Optional[Union[str, Dict[str, Any]]] = Field(None, description="Tool choice setting")
    
    # Required parameters
    model: str = Field(..., description="The model to use for inference")
    max_tokens: int = Field(..., description="Maximum number of tokens to generate")
    temperature: float = Field(..., description="Sampling temperature")
    top_p: float = Field(..., description="Top-p sampling parameter")
    
    # Tracking
    correlation_id: Optional[str] = Field(None, description="Correlation ID for tracking requests end-to-end")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    @model_validator(mode='after')
    def validate_prompt_or_messages(self) -> 'InferenceChallenge':
        """Ensure either prompt or messages is provided."""
        if self.prompt is None and self.messages is None:
            raise ValueError("Either 'prompt' or 'messages' must be provided")
        return self
    
    def get_messages(self) -> List[Dict[str, Any]]:
        """
        Get messages in OpenAI format.
        
        If only prompt is provided, converts it to a single user message.
        This enables backward compatibility with legacy prompt-based challenges.
        """
        if self.messages:
            return self.messages
        elif self.prompt:
            return [{"role": "user", "content": self.prompt}]
        return []
    
    def get_prompt_for_evaluation(self) -> str:
        """
        Get a prompt string for evaluation purposes.
        
        If messages are provided, extracts the user content for evaluation.
        """
        if self.prompt:
            return self.prompt
        elif self.messages:
            # Extract user message content for evaluation
            user_messages = [m.get("content", "") for m in self.messages if m.get("role") == "user"]
            return " ".join(user_messages)
        return ""


class InferenceResponse(BaseModel):
    """
    Response from a miner for an inference challenge.
    
    Extended to support tool calls and token usage tracking (F3).
    """
    response_text: str = Field(..., description="The generated text response")
    response_time_ms: int = Field(..., description="Time taken to generate response in milliseconds")
    
    # Tool calling support
    tool_calls: Optional[List[Dict[str, Any]]] = Field(None, description="Tool calls made by the model")
    finish_reason: str = Field("stop", description="Reason for completion: stop, length, tool_calls")
    
    # Token usage (F3)
    usage: Optional[TokenUsage] = Field(None, description="Token usage statistics for cost attribution")
    
    # Tracking
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