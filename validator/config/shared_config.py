"""
Shared configuration models using Pydantic 2 with environment variable support.
Validator-specific version - only includes validator-relevant configuration.

IMPORTANT: Many operational parameters are NOT configurable via environment
and are instead hard-coded in validator/internal_config.py for network consistency.
"""

import os
from pathlib import Path
from typing import Optional, Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class BaseConfig(BaseSettings):
    """Base configuration class with common settings."""
    
    model_config = SettingsConfigDict(
        env_file=("/workspace/.env", ".env"),  # Check RunPod location first, then local
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
    """LLM configuration settings.
    
    NOTE: LLM behavior params (DEFAULT_MODEL, DEFAULT_MAX_TOKENS, DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P) are now hard-coded in validator/internal_config.py for network consistency.
    This class is kept for backward compatibility but contains no active fields.
    """
    pass


class LoggingConfig(BaseConfig):
    """Logging configuration settings."""
    
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO", 
        description="Logging level"
    )
    log_file: Optional[str] = Field(default=None, description="Log file path")


class ValidatorSpecificConfig(BaseConfig):
    """
    Validator-specific configuration settings.
    
    NOTE: Operational parameters (miner selection, challenge timing, scoring,
    weight setting) are hard-coded in validator/internal_config.py for network
    consistency. This config only contains deployment-specific settings.
    """
    
    # NOTE: Miner selection parameters (MIN_MINERS, MAX_MINERS, MAX_MINER_STAKE)
    # are now hard-coded in validator/internal_config.py for network consistency.
    
    # NOTE: Challenge parameters (CHALLENGE_INTERVAL_SECONDS, CHALLENGE_TIMEOUT_SECONDS,
    # EVALUATION_TIMEOUT_SECONDS) are now hard-coded in validator/internal_config.py.
    
    # NOTE: Scoring parameters (SCORE_THRESHOLD) are now hard-coded in validator/internal_config.py.
    
    # NOTE: Weights update interval (WEIGHTS_INTERVAL_SECONDS) is now hard-coded
    # in validator/internal_config.py for network consistency.
    
    # NOTE: Metagraph refresh interval (METAGRAPH_REFRESH_INTERVAL_SECONDS) is now
    # hard-coded in validator/internal_config.py for network consistency.
    
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

