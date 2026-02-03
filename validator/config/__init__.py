"""
Validator configuration module.

This module re-exports the ValidatorConfig from the parent config module
to avoid duplication. The actual implementation is in ../config.py.
"""

# Import from the parent-level config module using relative import
import importlib.util
from pathlib import Path

# Get the parent directory (validator/)
parent_dir = Path(__file__).parent.parent
config_file = parent_dir / "config.py"

# Load the config module dynamically
spec = importlib.util.spec_from_file_location("validator.config_module", config_file)
config_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config_module)

# Re-export the classes
ValidatorConfig = config_module.ValidatorConfig
get_validator_config = config_module.get_validator_config
validator_config_to_bittensor_config = config_module.validator_config_to_bittensor_config

__all__ = ['ValidatorConfig', 'get_validator_config', 'validator_config_to_bittensor_config']
