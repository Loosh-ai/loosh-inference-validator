"""
Validator configuration using Pydantic 2 with environment variable support.
"""

from datetime import timedelta
from pathlib import Path
from typing import Optional, Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from validator.config.shared_config import WalletConfig

# CONFIG - ValidatorConfig [

class ValidatorConfig(BaseSettings):
    """Configuration for the validator using Pydantic 2 BaseSettings."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    # Network configuration
    netuid: int = Field(default=1, description="Subnet UID")
#    netuid: int = Field(default=21, description="Subnet UID")
    subtensor_network: str = Field(default="finney", description="Network to connect to")
    subtensor_address: str = Field(
        default="ws://127.0.0.1:9945",
        description="Network entrypoint"
    )
#        default="wss://entrypoint-finney.opentensor.ai:443",
    
    # Wallet configuration
    wallet_name: str = Field(default="validator", description="Wallet name")
    hotkey_name: str = Field(default="validator", description="Hotkey name")
    # Note: Fiber only supports wallets in ~/.bittensor/wallets

    wallet: WalletConfig = Field(default_factory=WalletConfig)
    
    # Miner selection parameters
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
    
    # Database configuration
    db_path: str = Field(default="validator.db", description="Database file path")
    users_db_path: str = Field(default="users.db", description="Users database file path")
    
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
    
    # API configuration
    api_host: str = Field(default="0.0.0.0", description="API host")
    api_port: int = Field(default=8000, description="API port")
    
    # LLM configuration
    default_model: str = Field(default="mistralai/Mistral-7B-v0.1", description="Default model")
    default_max_tokens: int = Field(default=512, description="Default max tokens")
    default_temperature: float = Field(default=0.7, description="Default temperature")
    default_top_p: float = Field(default=0.95, description="Default top-p value")
    
    # Sentence Transformer configuration (for embeddings)
    sentence_transformer_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Sentence Transformer model for generating embeddings during evaluation"
    )
    
    # Evaluation configuration
    heatmap_upload_url: str = Field(
        default="http://localhost:8080/upload",
        description="URL for uploading heatmaps"
    )
    
    # Challenge API configuration
    challenge_api_key: str = Field(default="api-key-0", description="Challenge API key")
    challenge_api_url: str = Field(default="http://localhost:8080", description="Challenge API URL")
    
    # Challenge push configuration (for receiving pushed challenges)
    challenge_push_api_key: Optional[str] = Field(
        default=None, 
        description="API key for authenticating challenge push requests (optional)"
    )
    
    # OpenAI configuration for evaluation
    openai_api_url: str = Field(
        default="https://api.openai.com/v1/chat/completions",
        description="OpenAI API URL"
    )
    openai_api_key: Optional[str] = Field(
        default=None,
        description="OpenAI API key for evaluation (if not provided, will try to use environment variable OPENAI_API_KEY)"
    )
    openai_model: str = Field(default="gpt-4", description="OpenAI model to use")
    
    # Logging configuration
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO", 
        description="Logging level"
    )
    log_file: Optional[str] = Field(default=None, description="Log file path")
    
    # Test mode configuration
    test_mode: bool = Field(
        default=False,
        description="Enable test mode - picks first response without evaluation or heatmap generation"
    )
    
    # Narrative generation configuration
    enable_narrative_generation: bool = Field(
        default=True,
        description="Enable narrative generation using LLM. When disabled, evaluation and heatmap generation still run, but narrative is skipped."
    )
    
    # Heatmap generation configuration
    enable_heatmap_generation: bool = Field(
        default=True,
        description="Enable heatmap generation for consensus evaluation. When disabled, heatmaps will not be generated or uploaded."
    )
    
    # Quality plot generation configuration
    enable_quality_plots: bool = Field(
        default=False,
        description="Enable quality plot generation (response length distribution). When enabled, generates quality plots alongside heatmaps when quality filtering is active."
    )
    
    # Challenge mode configuration
    # Note: Only push mode (Fiber-encrypted) is supported. Pull mode has been removed.
    # This field is kept for backward compatibility but is ignored - always uses push mode.
    challenge_mode: Literal["push"] = Field(
        default="push",
        description="Challenge retrieval mode. Only 'push' mode is supported (Fiber-encrypted challenges via POST /fiber/challenge). Pull mode has been removed."
    )
    
    # Concurrency configuration
    max_concurrent_challenges: int = Field(
        default=10,
        description="Maximum number of challenges to process concurrently"
    )
    max_concurrent_availability_checks: int = Field(
        default=20,
        description="Maximum number of concurrent miner availability checks (prevents connection pool exhaustion)"
    )
    
    # Fiber MLTS Configuration
    fiber_key_ttl_seconds: int = Field(
        default=3600,
        description="Time-to-live for Fiber symmetric keys in seconds (default: 1 hour)"
    )
    fiber_handshake_timeout_seconds: int = Field(
        default=30,
        description="Timeout for Fiber handshake operations in seconds"
    )
    fiber_enable_key_rotation: bool = Field(
        default=True,
        description="Enable automatic key rotation for Fiber symmetric keys"
    )


def get_validator_config() -> ValidatorConfig:
    """Get validator configuration from environment variables."""
    from loguru import logger
    try:
        return ValidatorConfig()
    except Exception as e:
        error_msg = (
            f"\n{'='*70}\n"
            f"CONFIGURATION ERROR: Failed to load validator configuration\n"
            f"{'='*70}\n"
            f"Error: {str(e)}\n"
            f"\nThis is usually caused by:\n"
            f"  - Invalid environment variable values\n"
            f"  - Missing required configuration\n"
            f"  - Type validation errors (e.g., invalid number format)\n"
            f"\nPlease check your environment variables or .env file.\n"
            f"See environments/env.validator.example for valid configuration options.\n"
            f"{'='*70}\n"
        )
        logger.error(error_msg)
        raise ValueError(error_msg) from e 

# CONFIG - ValidatorConfig ]
