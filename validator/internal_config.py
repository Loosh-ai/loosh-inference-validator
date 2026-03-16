"""
Internal configuration constants for the Loosh Inference Validator.

These values are hard-coded and NOT configurable via environment variables.
This ensures network consistency across all validators and prevents
operators from accidentally misconfiguring critical parameters.

To change these values, modify this file directly and redeploy.

IMPORTANT: Changes to these values affect all validators and should be
carefully considered for network-wide impact.
Note: Immediately prior values are recorded next to the current values for reference.
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
    
    # Maximum number of miners to select per challenge round.
    # Each challenge randomly samples this many from ALL available miners,
    # ensuring fair rotation across the full pool.
    MAX_MINERS: int = 7 #Prior Value: 10 Changed in v1.2.4 3/3/2026. Intended to reduce network load & latency
    
    # Maximum stake for a node to be considered a miner (not a validator).
    # Nodes with stake >= this are treated as validators and excluded
    # from challenges when the validator list fetcher is unavailable.
    # NOTE: There is intentionally NO minimum stake for miners — all
    # registered miners with a valid IP/port are eligible for challenges.
    MAX_MINER_STAKE: int = 999
    
    # =========================================================================
    # Challenge Parameters (in seconds)
    # =========================================================================
    
    # Time between challenge processing cycles
    # Testnet recommendation: 10 seconds
    # Mainnet recommendation: 300 seconds (5 minutes)
    CHALLENGE_INTERVAL_SECONDS: int = 300
    
    # Timeout for miner responses to challenges
    CHALLENGE_TIMEOUT_SECONDS: int = 60 #Prior Value: 120 Changed in v1.2.4 3/3/2026. Intended to reduce latency

    # =========================================================================
    # Dendrite Parameters
    # =========================================================================

    # Default timeout (seconds) for dendrite calls.
    DENDRITE_TIMEOUT: int = 30

    # Number of retries for failed dendrite calls.
    DENDRITE_MAX_RETRY: int = 2

    # Delay (seconds) between dendrite retries.
    DENDRITE_RETRY_DELAY: float = 0.5
    
    # Timeout for evaluation processing
    EVALUATION_TIMEOUT_SECONDS: int = 60 #Prior Value: 300 Changed in v1.2.4 3/3/2026. Intended to reduce latency
    
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
    # Not recommended for production - requires langchain-openai and an LLM API endpoint.
    ENABLE_NARRATIVE_GENERATION: bool = False

    # Temperature for the narrative-generation LLM call.
    NARRATIVE_LLM_TEMPERATURE: float = 0.7

    # Maximum tokens for the narrative-generation LLM call.
    NARRATIVE_LLM_MAX_TOKENS: int = 800
    
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

    # Sleep duration (seconds) before retrying after a metagraph refresh error.
    METAGRAPH_REFRESH_ERROR_RETRY_SECONDS: int = 60

    # =========================================================================
    # Weight Setting Loop Parameters
    # =========================================================================

    # Maximum number of consecutive weight-setting failures before the loop
    # switches to a longer back-off interval.
    WEIGHTS_MAX_CONSECUTIVE_FAILURES: int = 3

    # Maximum initial delay (seconds) before the first weight-setting attempt.
    # Capped at this value even when WEIGHTS_INTERVAL_SECONDS is larger.
    WEIGHTS_INITIAL_DELAY_MAX_SECONDS: int = 300

    # =========================================================================
    # Database Cleanup Loop Parameters
    # =========================================================================

    # How often to run the database cleanup task (hours).
    DB_CLEANUP_INTERVAL_HOURS: int = 24

    # How many hours of evaluation data to retain before pruning.
    DB_CLEANUP_RETENTION_HOURS: int = 48

    # Sleep duration (seconds) before retrying after a cleanup error.
    DB_CLEANUP_ERROR_RETRY_SECONDS: int = 3600

    # =========================================================================
    # Challenge Consumer & Pending Queue Loop Parameters
    # =========================================================================

    # How long (seconds) to block on the challenge queue get() call each iteration.
    # Lower values reduce pickup latency; combined with CHALLENGE_QUEUE_POLL_INTERVAL
    # these two control the worst-case challenge-start delay.
    CHALLENGE_QUEUE_GET_TIMEOUT: float = 0.5 #Prior Value: 1.0 Changed in v1.2.4 3/3/2026. Intended to reduce latency

    # How long (seconds) to sleep when the queue is empty before polling again.
    CHALLENGE_QUEUE_POLL_INTERVAL: float = 0.3 #Prior Value: 1.0 Changed in v1.2.4 3/3/2026. Intended to reduce latency

    # How long (seconds) to sleep after an unhandled exception in the consumer loop.
    CHALLENGE_CONSUMER_ERROR_DELAY: float = 1.0

    # Maximum time (seconds) to wait for available nodes before dropping a
    # queued pending challenge.
    CHALLENGE_PENDING_NODE_MAX_WAIT_SECONDS: int = 120 #Prior Value: 300 Changed in v1.2.4 3/3/2026. Intended to reduce latency 

    # How often (seconds) to re-check for available nodes while a challenge
    # is waiting in the pending queue.
    CHALLENGE_PENDING_NODE_POLL_INTERVAL: float = 2.0

    # Sleep duration (seconds) between iterations of the pending-challenges
    # processor to avoid a tight busy loop.
    CHALLENGE_PENDING_PROCESSOR_LOOP_DELAY: float = 0.5

    # Sleep duration (seconds) after an error in the pending-challenges processor.
    CHALLENGE_PENDING_PROCESSOR_ERROR_DELAY: float = 1.0
    
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
    FIBER_HANDSHAKE_TIMEOUT_SECONDS: int = 15 #Prior Value: 30 Changed in v1.2.4 3/3/2026. Intended to reduce latency
    
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
    # Prompt-Relevance Gates
    # =========================================================================

    # Minimum prompt-relevance score for multi-miner evaluation path.
    QUALITY_MIN_RAW_PROMPT_RELEVANCE: float = 0.20

    # Minimum prompt-relevance score for single-miner evaluation path.
    QUALITY_SINGLE_MINER_MIN_RELEVANCE: float = 0.20
    
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
    
    # Minimum group size for the SybilDetector to flag a cluster as suspicious.
    SYBIL_MIN_GROUP_SIZE: int = 2

    # Minimum response length (characters) for sybil pair detection.
    # Short/canonical answers are excluded to reduce false positives.
    SYBIL_MIN_RESPONSE_LENGTH: int = 50

    # Maximum characters of response text stored per entry in sybil detection
    # results (pairs and groups).  Keeps DB rows from growing unbounded.
    SYBIL_STORAGE_TEXT_MAX_LENGTH: int = 500
    
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
    # Sybil Penalty Parameters
    # =========================================================================
    
    # Enable sybil penalty in weight setting
    # When False, sybil scores are fetched and logged but no penalty is applied.
    SYBIL_PENALTY_ENABLED: bool = True
    
    # Maximum penalty that can be applied to a miner's weight (0.0 – 1.0).
    # A sybil_score of 1.0 results in an 80% weight reduction (score * 0.2).
    SYBIL_PENALTY_MAX: float = 1.0 #Prior Value: 0.8 Changed in v1.2.4 3/3/2026. Intended to reduce sybil identified miner emissions. Grace Period is over.
    
    # Minimum fraction of EMA a penalized miner always retains (grace floor).
    # Regardless of SYBIL_PENALTY_MAX, the penalized score will never drop
    # below ``ema_score * SYBIL_PENALTY_MIN_RETENTION``.
    # Set to 0.0 to allow full effect of SYBIL_PENALTY_MAX with no floor.
    # Example: 0.15 means the worst offender still keeps 15% of their EMA.
    SYBIL_PENALTY_MIN_RETENTION: float = 0.0 #Prior Value: 0.05 Changed in v1.2.5 3/15/2026. Zeroed to prevent sybil clusters from accumulating emissions through sheer volume at 5% retention.
    
    # Sybil score threshold below which no penalty is applied.
    # Miners with sybil_score < this value are treated as clean.
    SYBIL_PENALTY_THRESHOLD: float = 0.1
    
    # Safety valve: maximum fraction of serving miners that may be penalized
    # within the SOFT tier (score < SYBIL_HARD_PENALTY_SCORE_FLOOR).
    # If the soft-tier candidate count exceeds this fraction, fallback-K
    # restricts penalties to the worst-K soft-tier miners.
    # Miners in the HARD tier (score >= SYBIL_HARD_PENALTY_SCORE_FLOOR) are
    # ALWAYS penalized regardless of how many are flagged — the safety valve
    # does not apply to high-confidence detections.
    # This prevents a miscalibrated detector from tanking the subnet while
    # ensuring confirmed high-confidence sybils can never hide behind a
    # crowded lower tier.
    SYBIL_SAFETY_MAX_PENALIZED_FRACTION: float = 0.33

    # Score floor above which the safety valve is bypassed entirely.
    # Miners with sybil_score >= this value are always penalized, regardless
    # of how many miners are flagged in total (Option A: score-band bypass).
    # Must be > SYBIL_PENALTY_THRESHOLD and <= 1.0.
    # Default 0.60 corresponds to ~7+ detection appearances with K=2 on
    # detection-only signal, or fewer when infrastructure signals contribute.
    SYBIL_HARD_PENALTY_SCORE_FLOOR: float = 0.60
    
    # TTL for cached sybil scores fetched from the Challenge API (seconds).
    # During this window, set_weights reuses the previous fetch result.
    SYBIL_SCORE_CACHE_TTL_SECONDS: int = 300
    
    # =========================================================================
    # Emission Calculation Parameters
    # =========================================================================

    # Multiplier applied to a miner's emission when their response is part of
    # the consensus cluster.
    EMISSION_CONSENSUS_BONUS: float = 1.2

    # Controls the sensitivity of the score-difference bonus.
    # Lower value = more sensitive to small score gaps; higher = less sensitive.
    # With 0.1: diff of 1 → ~1.05×, diff of 10 → ~1.24×, diff of 20 → ~1.38×.
    EMISSION_SCORE_DIFF_SCALE_FACTOR: float = 0.1

    # Maximum additive bonus magnitude for the score-difference multiplier.
    # The multiplier is 1.0 + EMISSION_SCORE_DIFF_MAX_BONUS * tanh(diff * scale).
    EMISSION_SCORE_DIFF_MAX_BONUS: float = 0.5

    # Hard cap on the score-difference multiplier (1.0 + max_bonus ceiling).
    EMISSION_SCORE_DIFF_MAX_MULTIPLIER: float = 1.5

    # =========================================================================
    # Consensus Engine Configuration Parameters
    # =========================================================================

    # Minimum number of valid responses required to run sybil detection.
    CONSENSUS_MIN_RESPONSES_FOR_SYBIL: int = 2

    # Minimum number of valid responses required to generate a similarity heatmap.
    CONSENSUS_MIN_RESPONSES_FOR_HEATMAP: int = 2

    # Minimum number of valid responses required to enable LOF outlier detection.
    # LOF requires at least n_neighbors + 1 samples; 3 is the practical minimum.
    CONSENSUS_MIN_RESPONSES_FOR_OUTLIER_DETECTION: int = 3

    # Minimum number of valid responses required to enable clustering.
    CONSENSUS_MIN_RESPONSES_FOR_CLUSTERING: int = 2

    # Enable weighted scoring in consensus evaluation.
    CONSENSUS_USE_WEIGHTED_SCORING: bool = True

    # Apply legacy word-length quality filter during consensus.
    # Disabled — superseded by semantic quality assessment.
    CONSENSUS_APPLY_QUALITY_FILTER: bool = False

    # Legacy quality sensitivity parameter (deprecated, kept for backward compat).
    CONSENSUS_QUALITY_SENSITIVITY: float = 0.7

    # Lambda factor for consensus score blending.
    CONSENSUS_LAMBDA_FACTOR: float = 1.0

    # Minimum similarity threshold for a response to participate in consensus.
    CONSENSUS_THRESHOLD_MIN: float = 0.7

    # Enable semantic quality assessment during consensus.
    CONSENSUS_ENABLE_SEMANTIC_QUALITY: bool = True

    # Minimum semantic quality score required to participate in consensus.
    CONSENSUS_QUALITY_THRESHOLD: float = 0.35

    # Weight for prompt-relevance component within consensus quality scoring.
    CONSENSUS_QUALITY_PROMPT_RELEVANCE_WEIGHT: float = 0.4

    # Weight for density component within consensus quality scoring.
    CONSENSUS_QUALITY_DENSITY_WEIGHT: float = 0.2

    # Weight for specificity component within consensus quality scoring.
    CONSENSUS_QUALITY_SPECIFICITY_WEIGHT: float = 0.2

    # Weight for coherence component within consensus quality scoring.
    CONSENSUS_QUALITY_COHERENCE_WEIGHT: float = 0.2

    # Enable smart outlier detection that considers quality delta between
    # a candidate response and the consensus cluster.
    CONSENSUS_ENABLE_SMART_OUTLIER_DETECTION: bool = True

    # Minimum quality gap between an outlier and the cluster mean to trigger removal.
    CONSENSUS_OUTLIER_QUALITY_DELTA: float = 0.15

    # Enable diversity bonus for responses that expand the semantic coverage
    # of the consensus set.
    CONSENSUS_ENABLE_DIVERSITY_BONUS: bool = True

    # Maximum diversity bonus that can be added to a response's consensus score.
    CONSENSUS_MAX_DIVERSITY_BONUS: float = 0.15

    # Enable garbage cluster alerts that flag responses clustering far from
    # the main semantic consensus.
    CONSENSUS_ENABLE_GARBAGE_ALERTS: bool = True

    # Maximum intra-cluster similarity below which a cluster is flagged as garbage.
    CONSENSUS_GARBAGE_CLUSTER_THRESHOLD: float = 0.4

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
MAX_MINER_STAKE = INTERNAL_CONFIG.MAX_MINER_STAKE
CHALLENGE_INTERVAL_SECONDS = INTERNAL_CONFIG.CHALLENGE_INTERVAL_SECONDS
CHALLENGE_TIMEOUT_SECONDS = INTERNAL_CONFIG.CHALLENGE_TIMEOUT_SECONDS
DENDRITE_TIMEOUT = INTERNAL_CONFIG.DENDRITE_TIMEOUT
DENDRITE_MAX_RETRY = INTERNAL_CONFIG.DENDRITE_MAX_RETRY
DENDRITE_RETRY_DELAY = INTERNAL_CONFIG.DENDRITE_RETRY_DELAY
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
METAGRAPH_REFRESH_ERROR_RETRY_SECONDS = INTERNAL_CONFIG.METAGRAPH_REFRESH_ERROR_RETRY_SECONDS

# Weight Setting Loop
WEIGHTS_MAX_CONSECUTIVE_FAILURES = INTERNAL_CONFIG.WEIGHTS_MAX_CONSECUTIVE_FAILURES
WEIGHTS_INITIAL_DELAY_MAX_SECONDS = INTERNAL_CONFIG.WEIGHTS_INITIAL_DELAY_MAX_SECONDS

# Database Cleanup Loop
DB_CLEANUP_INTERVAL_HOURS = INTERNAL_CONFIG.DB_CLEANUP_INTERVAL_HOURS
DB_CLEANUP_RETENTION_HOURS = INTERNAL_CONFIG.DB_CLEANUP_RETENTION_HOURS
DB_CLEANUP_ERROR_RETRY_SECONDS = INTERNAL_CONFIG.DB_CLEANUP_ERROR_RETRY_SECONDS

# Challenge Consumer & Pending Queue Loop
CHALLENGE_QUEUE_GET_TIMEOUT = INTERNAL_CONFIG.CHALLENGE_QUEUE_GET_TIMEOUT
CHALLENGE_QUEUE_POLL_INTERVAL = INTERNAL_CONFIG.CHALLENGE_QUEUE_POLL_INTERVAL
CHALLENGE_CONSUMER_ERROR_DELAY = INTERNAL_CONFIG.CHALLENGE_CONSUMER_ERROR_DELAY
CHALLENGE_PENDING_NODE_MAX_WAIT_SECONDS = INTERNAL_CONFIG.CHALLENGE_PENDING_NODE_MAX_WAIT_SECONDS
CHALLENGE_PENDING_NODE_POLL_INTERVAL = INTERNAL_CONFIG.CHALLENGE_PENDING_NODE_POLL_INTERVAL
CHALLENGE_PENDING_PROCESSOR_LOOP_DELAY = INTERNAL_CONFIG.CHALLENGE_PENDING_PROCESSOR_LOOP_DELAY
CHALLENGE_PENDING_PROCESSOR_ERROR_DELAY = INTERNAL_CONFIG.CHALLENGE_PENDING_PROCESSOR_ERROR_DELAY

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
NARRATIVE_LLM_TEMPERATURE = INTERNAL_CONFIG.NARRATIVE_LLM_TEMPERATURE
NARRATIVE_LLM_MAX_TOKENS = INTERNAL_CONFIG.NARRATIVE_LLM_MAX_TOKENS

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
SYBIL_PENALTY_MIN_RETENTION = INTERNAL_CONFIG.SYBIL_PENALTY_MIN_RETENTION
SYBIL_PENALTY_THRESHOLD = INTERNAL_CONFIG.SYBIL_PENALTY_THRESHOLD
SYBIL_SAFETY_MAX_PENALIZED_FRACTION = INTERNAL_CONFIG.SYBIL_SAFETY_MAX_PENALIZED_FRACTION
SYBIL_HARD_PENALTY_SCORE_FLOOR = INTERNAL_CONFIG.SYBIL_HARD_PENALTY_SCORE_FLOOR
SYBIL_SCORE_CACHE_TTL_SECONDS = INTERNAL_CONFIG.SYBIL_SCORE_CACHE_TTL_SECONDS

# Embedding Performance
EMBEDDING_MAX_SEQ_LENGTH_DOC = INTERNAL_CONFIG.EMBEDDING_MAX_SEQ_LENGTH_DOC
EMBEDDING_MAX_SEQ_LENGTH_SENTENCE = INTERNAL_CONFIG.EMBEDDING_MAX_SEQ_LENGTH_SENTENCE
EMBEDDING_BATCH_SIZE_DOC = INTERNAL_CONFIG.EMBEDDING_BATCH_SIZE_DOC
EMBEDDING_BATCH_SIZE_SENTENCE = INTERNAL_CONFIG.EMBEDDING_BATCH_SIZE_SENTENCE
SENTENCE_LEVEL_RELEVANCE_GATE = INTERNAL_CONFIG.SENTENCE_LEVEL_RELEVANCE_GATE
QUALITY_MIN_RAW_PROMPT_RELEVANCE = INTERNAL_CONFIG.QUALITY_MIN_RAW_PROMPT_RELEVANCE
QUALITY_SINGLE_MINER_MIN_RELEVANCE = INTERNAL_CONFIG.QUALITY_SINGLE_MINER_MIN_RELEVANCE

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
SYBIL_MIN_GROUP_SIZE = INTERNAL_CONFIG.SYBIL_MIN_GROUP_SIZE
SYBIL_MIN_RESPONSE_LENGTH = INTERNAL_CONFIG.SYBIL_MIN_RESPONSE_LENGTH
SYBIL_STORAGE_TEXT_MAX_LENGTH = INTERNAL_CONFIG.SYBIL_STORAGE_TEXT_MAX_LENGTH
SYBIL_MIN_INTERNAL_SIMILARITY = INTERNAL_CONFIG.SYBIL_MIN_INTERNAL_SIMILARITY
SYBIL_FUSION_SEMANTIC_THRESHOLD = INTERNAL_CONFIG.SYBIL_FUSION_SEMANTIC_THRESHOLD
SYBIL_FUSION_LEXICAL_THRESHOLD = INTERNAL_CONFIG.SYBIL_FUSION_LEXICAL_THRESHOLD
SYBIL_FUSION_STRUCTURE_THRESHOLD = INTERNAL_CONFIG.SYBIL_FUSION_STRUCTURE_THRESHOLD
SYBIL_TRAJECTORY_THRESHOLD = INTERNAL_CONFIG.SYBIL_TRAJECTORY_THRESHOLD
SYBIL_TRAJECTORY_N_CLUSTERS = INTERNAL_CONFIG.SYBIL_TRAJECTORY_N_CLUSTERS

# Migration
PARALLEL_VALIDATION = INTERNAL_CONFIG.PARALLEL_VALIDATION
FALLBACK_MODEL = INTERNAL_CONFIG.FALLBACK_MODEL

# Emission Calculation
EMISSION_CONSENSUS_BONUS = INTERNAL_CONFIG.EMISSION_CONSENSUS_BONUS
EMISSION_SCORE_DIFF_SCALE_FACTOR = INTERNAL_CONFIG.EMISSION_SCORE_DIFF_SCALE_FACTOR
EMISSION_SCORE_DIFF_MAX_BONUS = INTERNAL_CONFIG.EMISSION_SCORE_DIFF_MAX_BONUS
EMISSION_SCORE_DIFF_MAX_MULTIPLIER = INTERNAL_CONFIG.EMISSION_SCORE_DIFF_MAX_MULTIPLIER

# Consensus Engine Configuration
CONSENSUS_MIN_RESPONSES_FOR_SYBIL = INTERNAL_CONFIG.CONSENSUS_MIN_RESPONSES_FOR_SYBIL
CONSENSUS_MIN_RESPONSES_FOR_HEATMAP = INTERNAL_CONFIG.CONSENSUS_MIN_RESPONSES_FOR_HEATMAP
CONSENSUS_MIN_RESPONSES_FOR_OUTLIER_DETECTION = INTERNAL_CONFIG.CONSENSUS_MIN_RESPONSES_FOR_OUTLIER_DETECTION
CONSENSUS_MIN_RESPONSES_FOR_CLUSTERING = INTERNAL_CONFIG.CONSENSUS_MIN_RESPONSES_FOR_CLUSTERING
CONSENSUS_USE_WEIGHTED_SCORING = INTERNAL_CONFIG.CONSENSUS_USE_WEIGHTED_SCORING
CONSENSUS_APPLY_QUALITY_FILTER = INTERNAL_CONFIG.CONSENSUS_APPLY_QUALITY_FILTER
CONSENSUS_QUALITY_SENSITIVITY = INTERNAL_CONFIG.CONSENSUS_QUALITY_SENSITIVITY
CONSENSUS_LAMBDA_FACTOR = INTERNAL_CONFIG.CONSENSUS_LAMBDA_FACTOR
CONSENSUS_THRESHOLD_MIN = INTERNAL_CONFIG.CONSENSUS_THRESHOLD_MIN
CONSENSUS_ENABLE_SEMANTIC_QUALITY = INTERNAL_CONFIG.CONSENSUS_ENABLE_SEMANTIC_QUALITY
CONSENSUS_QUALITY_THRESHOLD = INTERNAL_CONFIG.CONSENSUS_QUALITY_THRESHOLD
CONSENSUS_QUALITY_PROMPT_RELEVANCE_WEIGHT = INTERNAL_CONFIG.CONSENSUS_QUALITY_PROMPT_RELEVANCE_WEIGHT
CONSENSUS_QUALITY_DENSITY_WEIGHT = INTERNAL_CONFIG.CONSENSUS_QUALITY_DENSITY_WEIGHT
CONSENSUS_QUALITY_SPECIFICITY_WEIGHT = INTERNAL_CONFIG.CONSENSUS_QUALITY_SPECIFICITY_WEIGHT
CONSENSUS_QUALITY_COHERENCE_WEIGHT = INTERNAL_CONFIG.CONSENSUS_QUALITY_COHERENCE_WEIGHT
CONSENSUS_ENABLE_SMART_OUTLIER_DETECTION = INTERNAL_CONFIG.CONSENSUS_ENABLE_SMART_OUTLIER_DETECTION
CONSENSUS_OUTLIER_QUALITY_DELTA = INTERNAL_CONFIG.CONSENSUS_OUTLIER_QUALITY_DELTA
CONSENSUS_ENABLE_DIVERSITY_BONUS = INTERNAL_CONFIG.CONSENSUS_ENABLE_DIVERSITY_BONUS
CONSENSUS_MAX_DIVERSITY_BONUS = INTERNAL_CONFIG.CONSENSUS_MAX_DIVERSITY_BONUS
CONSENSUS_ENABLE_GARBAGE_ALERTS = INTERNAL_CONFIG.CONSENSUS_ENABLE_GARBAGE_ALERTS
CONSENSUS_GARBAGE_CLUSTER_THRESHOLD = INTERNAL_CONFIG.CONSENSUS_GARBAGE_CLUSTER_THRESHOLD
