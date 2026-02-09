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
    # Embedding & Evaluation Parameters
    # =========================================================================
    
    # Sentence Transformer model for generating embeddings during evaluation.
    # all-mpnet-base-v2 (768-dim) provides substantially better semantic
    # representation than all-MiniLM-L6-v2 (384-dim) at a modest speed
    # tradeoff (~3x slower, but still <100ms per batch on GPU).
    SENTENCE_TRANSFORMER_MODEL: str = "sentence-transformers/all-mpnet-base-v2"
    
    # Enable narrative generation using LLM after consensus evaluation
    # When disabled, evaluation and heatmap generation still run, but narrative is skipped.
    ENABLE_NARRATIVE_GENERATION: bool = True
    
    # Enable heatmap generation
    # Hardcoded to False to reduce processing overhead and upload failures
    ENABLE_HEATMAP_GENERATION: bool = False
    
    # Enable quality plot generation (response length distribution)
    # When enabled, generates quality plots alongside heatmaps when quality filtering is active.
    ENABLE_QUALITY_PLOTS: bool = False
    
    # =========================================================================
    # Metagraph Refresh Parameters
    # =========================================================================
    
    # How often to refresh the metagraph from chain (in seconds).
    # Controls how quickly new node registrations are picked up.
    METAGRAPH_REFRESH_INTERVAL_SECONDS: int = 300
    
    # =========================================================================
    # LLM Behavior Parameters
    # =========================================================================
    
    # Default model for challenge inference requests
    DEFAULT_MODEL: str = "mistralai/Mistral-7B-v0.1"
    
    # Default max tokens for challenge inference requests
    DEFAULT_MAX_TOKENS: int = 512
    
    # Default temperature for challenge inference requests
    DEFAULT_TEMPERATURE: float = 0.7
    
    # Default top-p for challenge inference requests
    DEFAULT_TOP_P: float = 0.95
    
    # =========================================================================
    # Concurrency Parameters
    # =========================================================================
    
    # Maximum number of challenges to process concurrently
    MAX_CONCURRENT_CHALLENGES: int = 10
    
    # Maximum number of concurrent miner availability checks
    # (prevents connection pool exhaustion)
    MAX_CONCURRENT_AVAILABILITY_CHECKS: int = 20
    
    # =========================================================================
    # Fiber MLTS Parameters
    # =========================================================================
    
    # Time-to-live for Fiber symmetric keys in seconds (default: 1 hour)
    FIBER_KEY_TTL_SECONDS: int = 3600
    
    # Timeout for Fiber handshake operations in seconds
    FIBER_HANDSHAKE_TIMEOUT_SECONDS: int = 30
    
    # Enable automatic key rotation for Fiber symmetric keys
    FIBER_ENABLE_KEY_ROTATION: bool = True
    
    # =========================================================================
    # Evaluation Quality Enhancement Parameters (Tier 4)
    # =========================================================================
    
    # Enable FP16 (half-precision) for embedding model inference.
    # Reduces memory usage and speeds up inference on GPU with minimal quality loss.
    EMBEDDING_FP16_ENABLED: bool = True
    
    # Enable sentence-level embedding for granular quality assessment.
    # Responses are split into sentences and each is embedded separately
    # for coherence and completeness scoring.
    EMBEDDING_SENTENCE_LEVEL: bool = True
    
    # ---- Multi-Granularity Relevance ----
    # Blend factor between sentence-level and full-response relevance.
    # 0.0 = full-response only, 1.0 = sentence-level only.
    QUALITY_SENTENCE_RELEVANCE_WEIGHT: float = 0.5
    
    # Minimum sentence length (words) to consider for sentence-level scoring.
    QUALITY_MIN_SENTENCE_WORDS: int = 3
    
    # ---- Embedding-Chain Coherence ----
    # When True, coherence is measured as average cosine similarity between
    # consecutive sentence embeddings (topic flow consistency).
    QUALITY_COHERENCE_EMBEDDING_CHAIN: bool = True
    
    # ---- Prompt Coverage Completeness ----
    # When True, prompts are decomposed into semantic components and
    # responses are scored on how many components they address.
    QUALITY_PROMPT_COVERAGE_ENABLED: bool = True
    
    # Minimum cosine similarity for a prompt component to be "covered".
    QUALITY_COVERAGE_THRESHOLD: float = 0.45
    
    # ---- Reasoning Complexity ----
    # When True, responses are scored on reasoning depth (causal markers,
    # argument structure, multi-step logic).
    QUALITY_COMPLEXITY_ENABLED: bool = True
    
    # =========================================================================
    # Embedding Performance Parameters
    # =========================================================================
    
    # Maximum sequence length for document-level embeddings.
    EMBEDDING_MAX_SEQ_LENGTH_DOC: int = 256
    
    # Maximum sequence length for sentence-level embeddings.
    EMBEDDING_MAX_SEQ_LENGTH_SENTENCE: int = 128
    
    # Batch size for document-level embeddings.
    EMBEDDING_BATCH_SIZE_DOC: int = 128
    
    # Batch size for sentence-level embeddings (shorter texts → larger batch).
    EMBEDDING_BATCH_SIZE_SENTENCE: int = 256
    
    # Only compute sentence-level features for responses whose doc-level
    # cosine similarity with the prompt exceeds this gate.
    # Reduces GPU work by ~50% on average workloads.
    SENTENCE_LEVEL_RELEVANCE_GATE: float = 0.5
    
    # =========================================================================
    # Quality Scoring Weights (must sum to ~1.0)
    # =========================================================================
    
    QUALITY_RELEVANCE_WEIGHT: float = 0.25
    QUALITY_DENSITY_WEIGHT: float = 0.15
    QUALITY_SPECIFICITY_WEIGHT: float = 0.15
    QUALITY_COHERENCE_WEIGHT: float = 0.15
    QUALITY_COMPLETENESS_WEIGHT: float = 0.15
    QUALITY_COMPLEXITY_WEIGHT: float = 0.15
    
    # =========================================================================
    # Quality Enhancement Sub-Parameters
    # =========================================================================
    
    # Coherence: local (adjacent) vs global (centroid) blend weights.
    QUALITY_COHERENCE_LOCAL_WEIGHT: float = 0.6
    QUALITY_COHERENCE_GLOBAL_WEIGHT: float = 0.4
    
    # Coherence: similarity below this between adjacent sentences counts as a "break".
    QUALITY_BREAK_THRESHOLD: float = 0.25
    
    # Relevance: blend of doc-level, sentence-level, and coverage components.
    QUALITY_RELEVANCE_DOC_BLEND: float = 0.4
    QUALITY_RELEVANCE_SENTENCE_BLEND: float = 0.4
    QUALITY_RELEVANCE_COVERAGE_BLEND: float = 0.2
    
    # Complexity: quality gates — complexity score is zeroed when these fail.
    QUALITY_COMPLEXITY_RELEVANCE_GATE: float = 0.4
    QUALITY_COMPLEXITY_COHERENCE_GATE: float = 0.3
    
    # Complexity: semantic step clustering parameters.
    QUALITY_COMPLEXITY_TARGET_STEPS: int = 4
    QUALITY_COMPLEXITY_SCALE: float = 2.0
    QUALITY_COMPLEXITY_DISTANCE_THRESHOLD: float = 0.7
    QUALITY_COMPLEXITY_STEP_WEIGHT: float = 0.6
    QUALITY_COMPLEXITY_NOVELTY_WEIGHT: float = 0.4
    
    # =========================================================================
    # Sybil Detection Thresholds (MPNet-calibrated)
    # =========================================================================
    
    # Minimum floor for adaptive high-similarity threshold.
    # MPNet: 0.80 | MiniLM: 0.85
    SYBIL_MIN_HIGH_THRESHOLD: float = 0.80
    
    # Minimum floor for adaptive very-high-similarity threshold.
    # MPNet: 0.88 | MiniLM: 0.92
    SYBIL_MIN_VERY_HIGH_THRESHOLD: float = 0.88
    
    # Percentile of similarity distribution used for adaptive high threshold.
    SYBIL_HIGH_SIMILARITY_PERCENTILE: float = 99.9
    
    # Percentile of similarity distribution used for adaptive very-high threshold.
    SYBIL_VERY_HIGH_SIMILARITY_PERCENTILE: float = 99.99
    
    # Minimum response length (characters) for sybil pair detection.
    # Short/canonical answers are excluded to reduce false positives.
    SYBIL_MIN_RESPONSE_LENGTH: int = 50
    
    # Minimum pairwise similarity within a group for it to be valid.
    SYBIL_MIN_INTERNAL_SIMILARITY: float = 0.90
    
    # Multi-view fusion: semantic similarity threshold.
    SYBIL_FUSION_SEMANTIC_THRESHOLD: float = 0.92
    
    # Multi-view fusion: lexical (TF-IDF) similarity threshold.
    SYBIL_FUSION_LEXICAL_THRESHOLD: float = 0.85
    
    # Multi-view fusion: structural similarity threshold.
    SYBIL_FUSION_STRUCTURE_THRESHOLD: float = 0.80
    
    # Sentence-trajectory analysis: similarity threshold for flagging pairs.
    SYBIL_TRAJECTORY_THRESHOLD: float = 0.85
    
    # Sentence-trajectory analysis: number of KMeans clusters per response.
    SYBIL_TRAJECTORY_N_CLUSTERS: int = 5
    
    # =========================================================================
    # Migration Parameters (temporary — remove after migration completes)
    # =========================================================================
    
    # Run both MiniLM and MPNet models side-by-side for comparison logging.
    PARALLEL_VALIDATION: bool = False
    
    # Fallback model if primary model fails to load.
    FALLBACK_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    # =========================================================================
    # Entity-Based Voting Caps
    # =========================================================================
    # Limits how much influence a suspected sybil entity can exert during
    # consensus formation.  An "entity" is a group of miners sharing the same
    # IP address (strongest signal) or coldkey (weaker signal).
    #
    # When enabled, each entity with N miners gets at most
    # ENTITY_MAX_VOTES effective votes regardless of N.  Individual miners
    # in the group receive a weight multiplier of min(1.0, max_votes / N).
    # Solo miners on unique IPs/coldkeys are completely unaffected.
    
    # Master switch — set to False to disable entity voting caps entirely.
    ENTITY_VOTING_CAPS_ENABLED: bool = True
    
    # Maximum effective votes per entity group regardless of group size.
    # E.g. 1.5 means a group of 5 miners sharing an IP collectively gets
    # at most 1.5 effective votes (each miner weighted at 1.5/5 = 0.3).
    ENTITY_MAX_VOTES: float = 1.5
    
    # Minimum group size to trigger capping.  Groups smaller than this are
    # treated as solo miners and retain full weight.
    ENTITY_MIN_GROUP_SIZE: int = 2
    
    # =========================================================================
    # Sybil Penalty Parameters
    # =========================================================================
    
    # Enable sybil penalty in weight setting
    # When False, sybil scores are fetched and logged but no penalty is applied.
    SYBIL_PENALTY_ENABLED: bool = True
    
    # Maximum penalty that can be applied to a miner's weight (0.0 – 1.0).
    # A sybil_score of 1.0 results in an 80% weight reduction (score * 0.2).
    SYBIL_PENALTY_MAX: float = 0.8
    
    # Sybil score threshold below which no penalty is applied.
    # Miners with sybil_score < this value are treated as clean.
    SYBIL_PENALTY_THRESHOLD: float = 0.1
    
    # Safety valve: maximum fraction of serving miners that may be penalized.
    # If more than this fraction would receive a penalty, the sybil penalty
    # step is skipped entirely and a critical warning is logged.
    # This prevents a miscalibrated sybil detector from tanking the entire subnet.
    SYBIL_SAFETY_MAX_PENALIZED_FRACTION: float = 0.33
    
    # TTL for cached sybil scores fetched from the Challenge API (seconds).
    # During this window, set_weights reuses the previous fetch result.
    SYBIL_SCORE_CACHE_TTL_SECONDS: int = 300
    
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

# Metagraph Refresh
METAGRAPH_REFRESH_INTERVAL_SECONDS = INTERNAL_CONFIG.METAGRAPH_REFRESH_INTERVAL_SECONDS

# LLM Behavior
DEFAULT_MODEL = INTERNAL_CONFIG.DEFAULT_MODEL
DEFAULT_MAX_TOKENS = INTERNAL_CONFIG.DEFAULT_MAX_TOKENS
DEFAULT_TEMPERATURE = INTERNAL_CONFIG.DEFAULT_TEMPERATURE
DEFAULT_TOP_P = INTERNAL_CONFIG.DEFAULT_TOP_P

# Embedding & Evaluation
SENTENCE_TRANSFORMER_MODEL = INTERNAL_CONFIG.SENTENCE_TRANSFORMER_MODEL
ENABLE_NARRATIVE_GENERATION = INTERNAL_CONFIG.ENABLE_NARRATIVE_GENERATION
ENABLE_HEATMAP_GENERATION = INTERNAL_CONFIG.ENABLE_HEATMAP_GENERATION
ENABLE_QUALITY_PLOTS = INTERNAL_CONFIG.ENABLE_QUALITY_PLOTS

# Concurrency
MAX_CONCURRENT_CHALLENGES = INTERNAL_CONFIG.MAX_CONCURRENT_CHALLENGES
MAX_CONCURRENT_AVAILABILITY_CHECKS = INTERNAL_CONFIG.MAX_CONCURRENT_AVAILABILITY_CHECKS

# Fiber MLTS
FIBER_KEY_TTL_SECONDS = INTERNAL_CONFIG.FIBER_KEY_TTL_SECONDS
FIBER_HANDSHAKE_TIMEOUT_SECONDS = INTERNAL_CONFIG.FIBER_HANDSHAKE_TIMEOUT_SECONDS
FIBER_ENABLE_KEY_ROTATION = INTERNAL_CONFIG.FIBER_ENABLE_KEY_ROTATION

# Evaluation Quality Enhancement (Tier 4)
EMBEDDING_FP16_ENABLED = INTERNAL_CONFIG.EMBEDDING_FP16_ENABLED
EMBEDDING_SENTENCE_LEVEL = INTERNAL_CONFIG.EMBEDDING_SENTENCE_LEVEL
QUALITY_SENTENCE_RELEVANCE_WEIGHT = INTERNAL_CONFIG.QUALITY_SENTENCE_RELEVANCE_WEIGHT
QUALITY_MIN_SENTENCE_WORDS = INTERNAL_CONFIG.QUALITY_MIN_SENTENCE_WORDS
QUALITY_COHERENCE_EMBEDDING_CHAIN = INTERNAL_CONFIG.QUALITY_COHERENCE_EMBEDDING_CHAIN
QUALITY_PROMPT_COVERAGE_ENABLED = INTERNAL_CONFIG.QUALITY_PROMPT_COVERAGE_ENABLED
QUALITY_COVERAGE_THRESHOLD = INTERNAL_CONFIG.QUALITY_COVERAGE_THRESHOLD
QUALITY_COMPLEXITY_ENABLED = INTERNAL_CONFIG.QUALITY_COMPLEXITY_ENABLED

# Entity-Based Voting Caps
ENTITY_VOTING_CAPS_ENABLED = INTERNAL_CONFIG.ENTITY_VOTING_CAPS_ENABLED
ENTITY_MAX_VOTES = INTERNAL_CONFIG.ENTITY_MAX_VOTES
ENTITY_MIN_GROUP_SIZE = INTERNAL_CONFIG.ENTITY_MIN_GROUP_SIZE

# Sybil Penalty
SYBIL_PENALTY_ENABLED = INTERNAL_CONFIG.SYBIL_PENALTY_ENABLED
SYBIL_PENALTY_MAX = INTERNAL_CONFIG.SYBIL_PENALTY_MAX
SYBIL_PENALTY_THRESHOLD = INTERNAL_CONFIG.SYBIL_PENALTY_THRESHOLD
SYBIL_SAFETY_MAX_PENALIZED_FRACTION = INTERNAL_CONFIG.SYBIL_SAFETY_MAX_PENALIZED_FRACTION
SYBIL_SCORE_CACHE_TTL_SECONDS = INTERNAL_CONFIG.SYBIL_SCORE_CACHE_TTL_SECONDS

# Embedding Performance
EMBEDDING_MAX_SEQ_LENGTH_DOC = INTERNAL_CONFIG.EMBEDDING_MAX_SEQ_LENGTH_DOC
EMBEDDING_MAX_SEQ_LENGTH_SENTENCE = INTERNAL_CONFIG.EMBEDDING_MAX_SEQ_LENGTH_SENTENCE
EMBEDDING_BATCH_SIZE_DOC = INTERNAL_CONFIG.EMBEDDING_BATCH_SIZE_DOC
EMBEDDING_BATCH_SIZE_SENTENCE = INTERNAL_CONFIG.EMBEDDING_BATCH_SIZE_SENTENCE
SENTENCE_LEVEL_RELEVANCE_GATE = INTERNAL_CONFIG.SENTENCE_LEVEL_RELEVANCE_GATE

# Quality Scoring Weights
QUALITY_RELEVANCE_WEIGHT = INTERNAL_CONFIG.QUALITY_RELEVANCE_WEIGHT
QUALITY_DENSITY_WEIGHT = INTERNAL_CONFIG.QUALITY_DENSITY_WEIGHT
QUALITY_SPECIFICITY_WEIGHT = INTERNAL_CONFIG.QUALITY_SPECIFICITY_WEIGHT
QUALITY_COHERENCE_WEIGHT = INTERNAL_CONFIG.QUALITY_COHERENCE_WEIGHT
QUALITY_COMPLETENESS_WEIGHT = INTERNAL_CONFIG.QUALITY_COMPLETENESS_WEIGHT
QUALITY_COMPLEXITY_WEIGHT = INTERNAL_CONFIG.QUALITY_COMPLEXITY_WEIGHT

# Quality Enhancement Sub-Parameters
QUALITY_COHERENCE_LOCAL_WEIGHT = INTERNAL_CONFIG.QUALITY_COHERENCE_LOCAL_WEIGHT
QUALITY_COHERENCE_GLOBAL_WEIGHT = INTERNAL_CONFIG.QUALITY_COHERENCE_GLOBAL_WEIGHT
QUALITY_BREAK_THRESHOLD = INTERNAL_CONFIG.QUALITY_BREAK_THRESHOLD
QUALITY_RELEVANCE_DOC_BLEND = INTERNAL_CONFIG.QUALITY_RELEVANCE_DOC_BLEND
QUALITY_RELEVANCE_SENTENCE_BLEND = INTERNAL_CONFIG.QUALITY_RELEVANCE_SENTENCE_BLEND
QUALITY_RELEVANCE_COVERAGE_BLEND = INTERNAL_CONFIG.QUALITY_RELEVANCE_COVERAGE_BLEND
QUALITY_COMPLEXITY_RELEVANCE_GATE = INTERNAL_CONFIG.QUALITY_COMPLEXITY_RELEVANCE_GATE
QUALITY_COMPLEXITY_COHERENCE_GATE = INTERNAL_CONFIG.QUALITY_COMPLEXITY_COHERENCE_GATE
QUALITY_COMPLEXITY_TARGET_STEPS = INTERNAL_CONFIG.QUALITY_COMPLEXITY_TARGET_STEPS
QUALITY_COMPLEXITY_SCALE = INTERNAL_CONFIG.QUALITY_COMPLEXITY_SCALE
QUALITY_COMPLEXITY_DISTANCE_THRESHOLD = INTERNAL_CONFIG.QUALITY_COMPLEXITY_DISTANCE_THRESHOLD
QUALITY_COMPLEXITY_STEP_WEIGHT = INTERNAL_CONFIG.QUALITY_COMPLEXITY_STEP_WEIGHT
QUALITY_COMPLEXITY_NOVELTY_WEIGHT = INTERNAL_CONFIG.QUALITY_COMPLEXITY_NOVELTY_WEIGHT

# Sybil Detection Thresholds
SYBIL_MIN_HIGH_THRESHOLD = INTERNAL_CONFIG.SYBIL_MIN_HIGH_THRESHOLD
SYBIL_MIN_VERY_HIGH_THRESHOLD = INTERNAL_CONFIG.SYBIL_MIN_VERY_HIGH_THRESHOLD
SYBIL_HIGH_SIMILARITY_PERCENTILE = INTERNAL_CONFIG.SYBIL_HIGH_SIMILARITY_PERCENTILE
SYBIL_VERY_HIGH_SIMILARITY_PERCENTILE = INTERNAL_CONFIG.SYBIL_VERY_HIGH_SIMILARITY_PERCENTILE
SYBIL_MIN_RESPONSE_LENGTH = INTERNAL_CONFIG.SYBIL_MIN_RESPONSE_LENGTH
SYBIL_MIN_INTERNAL_SIMILARITY = INTERNAL_CONFIG.SYBIL_MIN_INTERNAL_SIMILARITY
SYBIL_FUSION_SEMANTIC_THRESHOLD = INTERNAL_CONFIG.SYBIL_FUSION_SEMANTIC_THRESHOLD
SYBIL_FUSION_LEXICAL_THRESHOLD = INTERNAL_CONFIG.SYBIL_FUSION_LEXICAL_THRESHOLD
SYBIL_FUSION_STRUCTURE_THRESHOLD = INTERNAL_CONFIG.SYBIL_FUSION_STRUCTURE_THRESHOLD
SYBIL_TRAJECTORY_THRESHOLD = INTERNAL_CONFIG.SYBIL_TRAJECTORY_THRESHOLD
SYBIL_TRAJECTORY_N_CLUSTERS = INTERNAL_CONFIG.SYBIL_TRAJECTORY_N_CLUSTERS

# Migration
PARALLEL_VALIDATION = INTERNAL_CONFIG.PARALLEL_VALIDATION
FALLBACK_MODEL = INTERNAL_CONFIG.FALLBACK_MODEL
