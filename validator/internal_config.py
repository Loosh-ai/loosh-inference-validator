"""
Internal configuration constants for the Loosh Inference Validator.

These values are hard-coded and NOT configurable via environment variables.
This ensures network consistency across all validators and prevents
operators from accidentally misconfiguring critical parameters.

To change these values, modify this file directly and redeploy.

IMPORTANT: Changes to these values affect all validators and should be
carefully considered for network-wide impact.
"""

from datetime import timedelta
from dataclasses import dataclass


@dataclass(frozen=True)
class InternalConfig:
    """
    Internal configuration constants that are NOT configurable via environment.
    
    These values are hard-coded for network consistency. All validators
    should use the same values to ensure fair and consistent behavior.
    
    This class is frozen (immutable) to prevent accidental modification.
    """
    
    # =========================================================================
    # Miner Selection Parameters
    # =========================================================================
    
    # Minimum number of miners required for a valid challenge round
    MIN_MINERS: int = 3
    
    # Maximum number of miners to query per challenge
    MAX_MINERS: int = 10
    
    # Minimum stake required for a miner to be eligible (in TAO)
    # Miners below this threshold are excluded from challenges
    MIN_STAKE_THRESHOLD: int = 100
    
    # Maximum stake for a node to be considered a miner (not a validator)
    # Nodes with stake >= this are treated as validators and excluded from challenges
    MAX_MINER_STAKE: int = 999
    
    # =========================================================================
    # Challenge Parameters (in seconds)
    # =========================================================================
    
    # Time between challenge processing cycles
    # Testnet recommendation: 10 seconds
    # Mainnet recommendation: 300 seconds (5 minutes)
    CHALLENGE_INTERVAL_SECONDS: int = 300
    
    # Timeout for miner responses to challenges
    CHALLENGE_TIMEOUT_SECONDS: int = 120
    
    # Timeout for evaluation processing
    EVALUATION_TIMEOUT_SECONDS: int = 300
    
    # =========================================================================
    # Scoring Parameters
    # =========================================================================
    
    # Minimum score required for valid responses
    # Responses below this threshold may be filtered out
    SCORE_THRESHOLD: float = 0.7
    
    # =========================================================================
    # Weight Setting Parameters
    # =========================================================================
    
    # How often to set weights on-chain (in seconds)
    # 4320 seconds = 72 minutes
    # This value ensures all validators set weights on the same schedule
    WEIGHTS_INTERVAL_SECONDS: int = 4320
    
    # Minimum number of serving miners required to set weights
    # If fewer serving miners are found, weight setting is skipped
    WEIGHT_MIN_SERVING_NODES: int = 1
    
    # =========================================================================
    # Freshness Gate Parameters
    # =========================================================================
    
    # Hours without successful response before a miner is considered stale
    # Stale miners receive zero weight (in NORMAL mode)
    WEIGHT_FRESHNESS_HOURS: int = 3
    
    # Relaxed freshness threshold for DEGRADED mode
    # Used when approaching deregistration to be more lenient
    WEIGHT_FRESHNESS_HOURS_DEGRADED: int = 24
    
    # =========================================================================
    # Deregistration Safety Thresholds
    # =========================================================================
    
    # Bittensor deregisters validators who don't set weights for this many blocks
    DEREGISTRATION_BLOCK_LIMIT: int = 5000
    
    # Enter DEGRADED mode at 80% of deregistration limit
    # Uses relaxed freshness gate to increase chances of setting weights
    DEGRADED_MODE_THRESHOLD: int = 4000
    
    # Enter EMERGENCY mode at 90% of deregistration limit
    # Uses any available emissions, uniform weights as last resort
    EMERGENCY_MODE_THRESHOLD: int = 4500
    
    # =========================================================================
    # Convenience Properties (computed from base values)
    # =========================================================================
    
    @property
    def challenge_interval(self) -> timedelta:
        """Challenge interval as timedelta."""
        return timedelta(seconds=self.CHALLENGE_INTERVAL_SECONDS)
    
    @property
    def challenge_timeout(self) -> timedelta:
        """Challenge timeout as timedelta."""
        return timedelta(seconds=self.CHALLENGE_TIMEOUT_SECONDS)
    
    @property
    def evaluation_timeout(self) -> timedelta:
        """Evaluation timeout as timedelta."""
        return timedelta(seconds=self.EVALUATION_TIMEOUT_SECONDS)
    
    @property
    def weights_interval(self) -> timedelta:
        """Weights interval as timedelta."""
        return timedelta(seconds=self.WEIGHTS_INTERVAL_SECONDS)


# Singleton instance - import this in other modules
INTERNAL_CONFIG = InternalConfig()


# For backward compatibility, export individual constants
# These can be imported directly: from validator.internal_config import WEIGHTS_INTERVAL_SECONDS
MIN_MINERS = INTERNAL_CONFIG.MIN_MINERS
MAX_MINERS = INTERNAL_CONFIG.MAX_MINERS
MIN_STAKE_THRESHOLD = INTERNAL_CONFIG.MIN_STAKE_THRESHOLD
MAX_MINER_STAKE = INTERNAL_CONFIG.MAX_MINER_STAKE
CHALLENGE_INTERVAL_SECONDS = INTERNAL_CONFIG.CHALLENGE_INTERVAL_SECONDS
CHALLENGE_TIMEOUT_SECONDS = INTERNAL_CONFIG.CHALLENGE_TIMEOUT_SECONDS
EVALUATION_TIMEOUT_SECONDS = INTERNAL_CONFIG.EVALUATION_TIMEOUT_SECONDS
SCORE_THRESHOLD = INTERNAL_CONFIG.SCORE_THRESHOLD
WEIGHTS_INTERVAL_SECONDS = INTERNAL_CONFIG.WEIGHTS_INTERVAL_SECONDS
WEIGHT_MIN_SERVING_NODES = INTERNAL_CONFIG.WEIGHT_MIN_SERVING_NODES
WEIGHT_FRESHNESS_HOURS = INTERNAL_CONFIG.WEIGHT_FRESHNESS_HOURS
WEIGHT_FRESHNESS_HOURS_DEGRADED = INTERNAL_CONFIG.WEIGHT_FRESHNESS_HOURS_DEGRADED
DEREGISTRATION_BLOCK_LIMIT = INTERNAL_CONFIG.DEREGISTRATION_BLOCK_LIMIT
DEGRADED_MODE_THRESHOLD = INTERNAL_CONFIG.DEGRADED_MODE_THRESHOLD
EMERGENCY_MODE_THRESHOLD = INTERNAL_CONFIG.EMERGENCY_MODE_THRESHOLD
