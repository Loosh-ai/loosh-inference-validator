"""
Validator configuration using Pydantic 2 with environment variable support.

IMPORTANT: Many operational parameters are NOT configurable via environment
and are instead hard-coded in validator/internal_config.py for network consistency.
See that file for: miner selection, challenge timing, scoring, and weight setting parameters.
"""

from pathlib import Path
from typing import Optional, Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from validator.config.shared_config import WalletConfig

# CONFIG - ValidatorConfig [

class ValidatorConfig(BaseSettings):
    """
    Configuration for the validator using Pydantic 2 BaseSettings.
    
    NOTE: Operational parameters (miner selection, challenge timing, scoring,
    weight setting) are hard-coded in validator/internal_config.py for network
    consistency. This config only contains deployment-specific settings like
    network endpoints, wallet names, API URLs, etc.
    """
    
    model_config = SettingsConfigDict(
        env_file=("/workspace/.env", ".env"),  # Check RunPod location first, then local
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        env_ignore_empty=True,  # Ignore empty env files
        # Exclude internal settings from environment variable parsing
        protected_namespaces=()  # Disable pydantic protected namespace warnings
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
    
    # NOTE: Miner selection parameters (MIN_MINERS, MAX_MINERS, MIN_STAKE_THRESHOLD, MAX_MINER_STAKE)
    # are now hard-coded in validator/internal_config.py for network consistency.
    
    # NOTE: Challenge parameters (CHALLENGE_INTERVAL_SECONDS, CHALLENGE_TIMEOUT_SECONDS,
    # EVALUATION_TIMEOUT_SECONDS) are now hard-coded in validator/internal_config.py.
    
    # NOTE: Scoring parameters (SCORE_THRESHOLD) are now hard-coded in validator/internal_config.py.
    
    # NOTE: Weight setting parameters (WEIGHTS_INTERVAL_SECONDS, WEIGHT_FRESHNESS_HOURS,
    # WEIGHT_MIN_SERVING_NODES, etc.) are hard-coded in validator/internal_config.py.
    
    # Database configuration
    db_path: str = Field(default="validator.db", description="Database file path")
    users_db_path: str = Field(default="users.db", description="Users database file path")
    
    # NOTE: Metagraph refresh interval (METAGRAPH_REFRESH_INTERVAL_SECONDS) is now
    # hard-coded in validator/internal_config.py for network consistency.
    
    # API configuration
    api_host: str = Field(default="0.0.0.0", description="API host")
    api_port: int = Field(default=8000, description="API port")
    
    # Axon configuration (bittensor node communication)
    axon_ip: str = Field(default="127.0.0.1", description="Axon IP address")
    axon_port: int = Field(default=8099, description="Axon port")
    axon_external_ip: Optional[str] = Field(default=None, description="Axon external IP (if different from internal)")
    axon_external_port: Optional[int] = Field(default=None, description="Axon external port (if different from internal)")
    axon_max_workers: int = Field(default=5, description="Maximum worker threads for axon")
    axon_timeout: int = Field(default=30, description="Axon timeout in seconds")
    
    # NOTE: LLM behavior params (DEFAULT_MODEL, DEFAULT_MAX_TOKENS, DEFAULT_TEMPERATURE,
    # DEFAULT_TOP_P) are now hard-coded in validator/internal_config.py for network consistency.
    
    # NOTE: Sentence Transformer model is now in internal_config.py (SENTENCE_TRANSFORMER_MODEL)
    # for network consistency. Not configurable via environment.
    
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
    
    # LLM configuration for evaluation (narrative generation)
    # IMPORTANT: The API endpoint must implement the OpenAI Chat Completions API format
    # (see https://platform.openai.com/docs/api-reference/chat/create)
    # The API interface must be compatible, but the underlying model does NOT need to be an OpenAI model.
    # You can use any model (Llama, Qwen, Mistral, etc.) as long as the API follows OpenAI's format.
    llm_api_url: str = Field(
        default="https://api.openai.com/v1/chat/completions",
        description="LLM API URL - must implement OpenAI Chat Completions API format (https://platform.openai.com/docs/api-reference/chat/create). The model itself does not need to be from OpenAI."
    )
    llm_api_key: Optional[str] = Field(
        default=None,
        description="LLM API key for evaluation (if not provided, will try to use environment variable LLM_API_KEY)"
    )
    llm_model: str = Field(default="", description="LLM model name to use. Can be any model as long as the API endpoint implements OpenAI Chat Completions format.")
    
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
    
    # NOTE: Narrative generation, heatmap generation, and quality plot generation
    # are now in internal_config.py (ENABLE_NARRATIVE_GENERATION, ENABLE_HEATMAP_GENERATION,
    # ENABLE_QUALITY_PLOTS) for network consistency. Not configurable via environment.
    
    # Challenge mode configuration
    # Note: Only push mode (Fiber-encrypted) is supported. Pull mode has been removed.
    # This field is kept for backward compatibility but is ignored - always uses push mode.
    challenge_mode: Literal["push"] = Field(
        default="push",
        description="Challenge retrieval mode. Only 'push' mode is supported (Fiber-encrypted challenges via POST /fiber/challenge). Pull mode has been removed."
    )
    
    # NOTE: Concurrency parameters (MAX_CONCURRENT_CHALLENGES, MAX_CONCURRENT_AVAILABILITY_CHECKS)
    # are now in internal_config.py for network consistency. Not configurable via environment.
    
    # NOTE: Fiber MLTS parameters (FIBER_KEY_TTL_SECONDS, FIBER_HANDSHAKE_TIMEOUT_SECONDS,
    # FIBER_ENABLE_KEY_ROTATION) are now in internal_config.py for network consistency.


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


def validator_config_to_bittensor_config(validator_config: ValidatorConfig) -> "bittensor.config":
    """
    Convert ValidatorConfig to bittensor config object.
    
    This ensures the subtensor is initialized with the correct network and chain endpoint
    from the environment variables (SUBTENSOR_NETWORK and SUBTENSOR_ADDRESS).
    
    Args:
        validator_config: ValidatorConfig instance with loaded environment variables
        
    Returns:
        bittensor.config: Properly configured bittensor config object
    """
    import bittensor as bt
    
    # Create base bittensor config
    config = bt.config()
    
    # Set basic network configuration
    config.netuid = validator_config.netuid
    config.network = validator_config.subtensor_network
    
    # Initialize subtensor config
    config.subtensor = bt.subtensor.config()
    config.subtensor.network = validator_config.subtensor_network
    config.subtensor.chain_endpoint = validator_config.subtensor_address
    
    # Ensure nested subtensor config exists and is set
    if not hasattr(config.subtensor, 'subtensor'):
        config.subtensor.subtensor = bt.config()
    config.subtensor.subtensor.network = validator_config.subtensor_network
    config.subtensor.subtensor.chain_endpoint = validator_config.subtensor_address
    
    # Set wallet configuration
    config.wallet = bt.config()
    config.wallet.name = validator_config.wallet_name
    config.wallet.hotkey = validator_config.hotkey_name
    config.wallet.path = '~/.bittensor/wallets/'
    
    # Set axon configuration from ValidatorConfig
    config.axon = bt.axon.config()
    config.axon.ip = validator_config.axon_ip
    config.axon.port = validator_config.axon_port
    config.axon.external_ip = validator_config.axon_external_ip
    config.axon.external_port = validator_config.axon_external_port
    config.axon.max_workers = validator_config.axon_max_workers
    config.axon.timeout = validator_config.axon_timeout
    
    # Set dendrite configuration defaults
    config.dendrite = bt.config()
    config.dendrite.timeout = 30
    config.dendrite.max_retry = 2
    config.dendrite.retry_delay = 0.5
    
    # Set logging configuration
    config.log_level = validator_config.log_level
    config.log_trace = True
    config.log_record = True
    config.log_dir = './logs/'
    
    return config

# CONFIG - ValidatorConfig ]
