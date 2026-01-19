"""
Shared configuration models using Pydantic 2 with environment variable support.
Validator-specific version - only includes validator-relevant configuration.
"""

import os
from datetime import timedelta
from pathlib import Path
from typing import Optional, Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class BaseConfig(BaseSettings):
    """Base configuration class with common settings."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )


class NetworkConfig(BaseConfig):
    """Network configuration settings."""
    
    netuid: int = Field(default=1, description="Subnet UID")
    subtensor_network: str = Field(default="finney", description="Network to connect to")
    subtensor_address: str = Field(
        default="wss://entrypoint-finney.opentensor.ai:443",
        description="Network entrypoint"
    )


class WalletConfig(BaseConfig):
    """Wallet configuration settings."""
    
    wallet_name: str = Field(default="validator", description="Wallet name")
    hotkey_name: str = Field(default="validator", description="Hotkey name")
    # Note: Fiber only supports wallets in ~/.bittensor/wallets


class APIConfig(BaseConfig):
    """API configuration settings."""
    
    api_host: str = Field(default="0.0.0.0", description="API host")
    api_port: int = Field(default=8000, description="API port")


class LLMConfig(BaseConfig):
    """LLM configuration settings."""
    
    default_model: str = Field(
        default="mistralai/Mistral-7B-v0.1",
        description="Default model to use"
    )
    default_max_tokens: int = Field(default=512, description="Default max tokens")
    default_temperature: float = Field(default=0.7, description="Default temperature")
    default_top_p: float = Field(default=0.95, description="Default top-p value")


class LoggingConfig(BaseConfig):
    """Logging configuration settings."""
    
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO", 
        description="Logging level"
    )
    log_file: Optional[str] = Field(default=None, description="Log file path")


class ValidatorSpecificConfig(BaseConfig):
    """Validator-specific configuration settings."""
    
    min_miners: int = Field(default=3, description="Minimum number of miners to select")
    max_miners: int = Field(default=10, description="Maximum number of miners to select")
    min_stake_threshold: int = Field(default=100, description="Minimum stake required (in TAO)")
    
    # Challenge parameters (in seconds for environment variables)
    # Testnet: 10 seconds
    # Mainnet: 300 seconds
    challenge_interval_seconds: int = Field(default=10, description="Time between challenges (seconds)")

    challenge_timeout_seconds: int = Field(default=120, description="Timeout for challenge responses (seconds)")
    evaluation_timeout_seconds: int = Field(default=300, description="Evaluation timeout (seconds)")
    
    # Convert to timedelta objects
    @property
    def challenge_interval(self) -> timedelta:
        return timedelta(seconds=self.challenge_interval_seconds)
    
    @property
    def challenge_timeout(self) -> timedelta:
        return timedelta(seconds=self.challenge_timeout_seconds)
    
    @property
    def evaluation_timeout(self) -> timedelta:
        return timedelta(seconds=self.evaluation_timeout_seconds)
    
    # Scoring parameters
    score_threshold: float = Field(
        default=0.7, 
        description="Minimum score required for valid responses",
        ge=0.0,
        le=1.0
    )
    
    # Weights update interval (in seconds)
    weights_interval_seconds: int = Field(default=1800, description="Weights update interval (seconds)")
    
    @property
    def weights_interval(self) -> timedelta:
        return timedelta(seconds=self.weights_interval_seconds)
    
    # Test mode configuration
    test_mode: bool = Field(
        default=False,
        description="Enable test mode - picks first response without evaluation or heatmap generation"
    )
    
    # Database configuration
    db_path: str = Field(default="validator.db", description="Database file path")
    
    # Evaluation configuration
    heatmap_upload_url: str = Field(
        default="http://localhost:8080/upload",
        description="URL for uploading heatmaps"
    )


class ChallengeAPIConfig(BaseConfig):
    """Challenge API configuration settings."""
    
    challenge_api_key: str = Field(default="api-key-0", description="Challenge API key")
    challenge_api_url: str = Field(default="http://localhost:8080", description="Challenge API URL")


class LLMServiceConfig(BaseConfig):
    """LLM service configuration for evaluation (narrative generation).
    
    IMPORTANT: The API endpoint must implement the OpenAI Chat Completions API format
    (see https://platform.openai.com/docs/api-reference/chat/create)
    The API interface must be compatible, but the underlying model does NOT need to be an OpenAI model.
    You can use any model (Llama, Qwen, Mistral, etc.) as long as the API follows OpenAI's format.
    
    Examples of compatible endpoints:
    - OpenAI API, Azure OpenAI, vLLM, Ollama (with OpenAI compatibility), etc.
    """
    
    llm_api_url: str = Field(
        default="https://your.inference.endpoint/v1/chat/completions",
        description="LLM API URL - must implement OpenAI Chat Completions API format (https://platform.openai.com/docs/api-reference/chat/create). The model itself does not need to be from OpenAI."
    )
    llm_model: str = Field(default="your-model-name", description="LLM model name to use. Can be any model as long as the API endpoint implements OpenAI Chat Completions format.")


class ValidatorConfig(BaseConfig):
    """Complete validator configuration."""
    
    # Sub-configurations
    network: NetworkConfig = Field(default_factory=NetworkConfig)
    wallet: WalletConfig = Field(default_factory=WalletConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    validator: ValidatorSpecificConfig = Field(default_factory=ValidatorSpecificConfig)
    challenge_api: ChallengeAPIConfig = Field(default_factory=ChallengeAPIConfig)
    llm_service: LLMServiceConfig = Field(default_factory=LLMServiceConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)


def get_validator_config() -> ValidatorConfig:
    """Get validator configuration from environment variables."""
    return ValidatorConfig()

