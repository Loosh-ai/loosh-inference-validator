import os
import math
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import aiohttp
import httpx
import numpy as np
from loguru import logger
from sentence_transformers import SentenceTransformer

from validator.challenge.challenge_types import InferenceResponse
from validator.db.operations import DatabaseManager
from validator.config import get_validator_config
from validator.network.challenge_api_auth import get_auth_headers
from validator.internal_config import (
    SENTENCE_TRANSFORMER_MODEL,
    ENABLE_NARRATIVE_GENERATION,
    ENABLE_HEATMAP_GENERATION,
    ENABLE_QUALITY_PLOTS,
    EMBEDDING_FP16_ENABLED,
    EMBEDDING_SENTENCE_LEVEL,
    EMBEDDING_MAX_SEQ_LENGTH_DOC,
    EMBEDDING_BATCH_SIZE_DOC,
    EMBEDDING_BATCH_SIZE_SENTENCE,
    SENTENCE_LEVEL_RELEVANCE_GATE,
    QUALITY_SENTENCE_RELEVANCE_WEIGHT,
    QUALITY_MIN_SENTENCE_WORDS,
    QUALITY_COHERENCE_EMBEDDING_CHAIN,
    QUALITY_PROMPT_COVERAGE_ENABLED,
    QUALITY_COVERAGE_THRESHOLD,
    QUALITY_COMPLEXITY_ENABLED,
    QUALITY_BREAK_THRESHOLD,
    QUALITY_COHERENCE_LOCAL_WEIGHT,
    QUALITY_COHERENCE_GLOBAL_WEIGHT,
    QUALITY_RELEVANCE_DOC_BLEND,
    QUALITY_RELEVANCE_SENTENCE_BLEND,
    QUALITY_RELEVANCE_COVERAGE_BLEND,
    QUALITY_COMPLEXITY_RELEVANCE_GATE,
    QUALITY_COMPLEXITY_COHERENCE_GATE,
    QUALITY_COMPLEXITY_TARGET_STEPS,
    QUALITY_COMPLEXITY_SCALE,
    QUALITY_COMPLEXITY_DISTANCE_THRESHOLD,
    QUALITY_COMPLEXITY_STEP_WEIGHT,
    QUALITY_COMPLEXITY_NOVELTY_WEIGHT,
    # Entity-Based Voting Caps
    ENTITY_VOTING_CAPS_ENABLED,
    ENTITY_MAX_VOTES,
    ENTITY_MIN_GROUP_SIZE,
    # Sybil detection thresholds (MPNet-calibrated)
    SYBIL_MIN_HIGH_THRESHOLD,
    SYBIL_MIN_VERY_HIGH_THRESHOLD,
    SYBIL_HIGH_SIMILARITY_PERCENTILE,
    SYBIL_VERY_HIGH_SIMILARITY_PERCENTILE,
    SYBIL_MIN_RESPONSE_LENGTH,
    SYBIL_MIN_INTERNAL_SIMILARITY,
    SYBIL_FUSION_SEMANTIC_THRESHOLD,
    SYBIL_FUSION_LEXICAL_THRESHOLD,
    SYBIL_FUSION_STRUCTURE_THRESHOLD,
    SYBIL_TRAJECTORY_THRESHOLD,
    SYBIL_TRAJECTORY_N_CLUSTERS,
    # Migration
    FALLBACK_MODEL,
)
from validator.network.fiber_client import ValidatorFiberClient

# Import from local evaluation modules
from .Evaluation.consensus_engine import ConsensusEngine, ConsensusConfig, ConsensusResult
from .Evaluation.quality_scorer import EvaluationQualityScorer
from .Recording.consensus_narrative_generator import ConsensusNarrativeGenerator, LLMConfig
from .Recording.similarity_heatmap import generate_semantic_similarity_heatmap
from .sybil_detection import SybilDetector, SybilDetectionResult

# Fiber client cache for heatmap uploads
_heatmap_fiber_client_cache: Dict[str, ValidatorFiberClient] = {}


# =========================================================================
# Entity-Based Voting Caps
# =========================================================================

def cap_entity_contributions(
    miner_hotkeys: List[str],
    node_map: Dict[str, Tuple[str, str]],
    max_entity_votes: float = ENTITY_MAX_VOTES,
    min_group_size: int = ENTITY_MIN_GROUP_SIZE,
) -> Dict[str, float]:
    """Compute per-miner weight multipliers that limit sybil entity influence.

    An *entity* is a group of miners sharing the same IP address (strongest
    signal) or coldkey (weaker signal, applied second).  IP-based grouping
    takes priority: miners already capped by an IP group are not
    double-counted in coldkey groups.

    For each entity group with *N* members (N >= ``min_group_size``), every
    member receives a weight multiplier of ``min(1.0, max_entity_votes / N)``.
    Solo miners (unique IP **and** coldkey) retain a weight of 1.0.

    Args:
        miner_hotkeys: Ordered list of miner hotkeys participating in this
            challenge round.
        node_map: Mapping of ``hotkey → (ip_address, coldkey)``.  Typically
            built from the latest metagraph snapshot.
        max_entity_votes: Maximum effective votes per entity group regardless
            of group size.
        min_group_size: Groups smaller than this are treated as individuals.

    Returns:
        ``{hotkey: weight}`` where weight ∈ (0.0, 1.0].  Only hotkeys present
        in *miner_hotkeys* appear in the result.
    """
    # Default all weights to 1.0
    weights: Dict[str, float] = {hk: 1.0 for hk in miner_hotkeys}

    # Filter to miners we actually have node data for
    participating = [hk for hk in miner_hotkeys if hk in node_map]
    if not participating:
        return weights

    # --- Phase 1: IP-based grouping (strongest signal) ---
    ip_groups: Dict[str, List[str]] = {}
    for hk in participating:
        ip, _ = node_map[hk]
        if ip and ip not in ("0", "0.0.0.0"):
            ip_groups.setdefault(ip, []).append(hk)

    ip_capped: set = set()  # hotkeys already handled by IP grouping
    for ip, members in ip_groups.items():
        n = len(members)
        if n < min_group_size:
            continue
        cap = min(1.0, max_entity_votes / n)
        for hk in members:
            weights[hk] = cap
            ip_capped.add(hk)
        logger.info(
            f"[ENTITY-CAP] IP group {ip}: {n} miners → weight={cap:.3f} "
            f"(max_votes={max_entity_votes})"
        )

    # --- Phase 2: Coldkey-based grouping (weaker signal, secondary) ---
    # Only applies to miners NOT already capped by IP grouping.
    ck_groups: Dict[str, List[str]] = {}
    for hk in participating:
        if hk in ip_capped:
            continue
        _, coldkey = node_map[hk]
        if coldkey:
            ck_groups.setdefault(coldkey, []).append(hk)

    for ck, members in ck_groups.items():
        n = len(members)
        if n < min_group_size:
            continue
        cap = min(1.0, max_entity_votes / n)
        for hk in members:
            weights[hk] = min(weights[hk], cap)  # Don't increase existing cap
        logger.info(
            f"[ENTITY-CAP] Coldkey group {ck[:16]}...: {n} miners → weight={cap:.3f} "
            f"(max_votes={max_entity_votes})"
        )

    return weights


class InferenceValidator:
    """Validator for inference responses."""
    
    def __init__(self, db_manager: DatabaseManager):
        """Initialize validator."""
        self.db_manager = db_manager
        
        # Load configuration
        self.config = get_validator_config()
        
        # Node map for entity-based voting caps: hotkey → (ip, coldkey)
        # Updated externally via update_node_map() on each metagraph refresh.
        self._node_map: Dict[str, Tuple[str, str]] = {}
        
        # Initialize embedding model with fallback
        self.embedding_model = self._init_embedder_with_fallback()
        
        # Initialize enhanced quality scorer (Tier 4+5) with ALL configurable params
        self.quality_scorer = EvaluationQualityScorer(
            sentence_relevance_weight=QUALITY_SENTENCE_RELEVANCE_WEIGHT,
            min_sentence_words=QUALITY_MIN_SENTENCE_WORDS,
            coverage_threshold=QUALITY_COVERAGE_THRESHOLD,
            coherence_embedding_chain=QUALITY_COHERENCE_EMBEDDING_CHAIN,
            complexity_enabled=QUALITY_COMPLEXITY_ENABLED,
            coverage_enabled=QUALITY_PROMPT_COVERAGE_ENABLED,
            break_threshold=QUALITY_BREAK_THRESHOLD,
            coherence_local_weight=QUALITY_COHERENCE_LOCAL_WEIGHT,
            coherence_global_weight=QUALITY_COHERENCE_GLOBAL_WEIGHT,
            relevance_doc_blend=QUALITY_RELEVANCE_DOC_BLEND,
            relevance_sentence_blend=QUALITY_RELEVANCE_SENTENCE_BLEND,
            relevance_coverage_blend=QUALITY_RELEVANCE_COVERAGE_BLEND,
            complexity_relevance_gate=QUALITY_COMPLEXITY_RELEVANCE_GATE,
            complexity_coherence_gate=QUALITY_COMPLEXITY_COHERENCE_GATE,
            complexity_target_steps=QUALITY_COMPLEXITY_TARGET_STEPS,
            complexity_scale=QUALITY_COMPLEXITY_SCALE,
            complexity_distance_threshold=QUALITY_COMPLEXITY_DISTANCE_THRESHOLD,
            complexity_step_weight=QUALITY_COMPLEXITY_STEP_WEIGHT,
            complexity_novelty_weight=QUALITY_COMPLEXITY_NOVELTY_WEIGHT,
        )
        logger.info("Enhanced quality scorer (Tier 4+5) initialized")
        
        # Initialize narrative generator with LLM config (only if enabled)
        # IMPORTANT: The API endpoint must implement OpenAI Chat Completions API format
        # (see https://platform.openai.com/docs/api-reference/chat/create)
        # The API interface must be compatible, but the underlying model does NOT need to be an OpenAI model.
        # Get API key from config or environment
        self.narrative_generator = None
        if ENABLE_NARRATIVE_GENERATION:
            api_key = getattr(self.config, 'llm_api_key', None) or os.getenv("LLM_API_KEY")
            
            self.narrative_generator = ConsensusNarrativeGenerator(
                LLMConfig(
                    api_url=self.config.llm_api_url,  # Must implement OpenAI Chat Completions API format
                    model_name=self.config.llm_model,  # Can be any model (Llama, Qwen, Mistral, etc.)
                    temperature=0.7,
                    max_tokens=800,
                    api_key=api_key
                )
            )
            logger.info("Narrative generation enabled - LLM will generate consensus narratives")
        else:
            logger.info("Narrative generation disabled - evaluation and heatmap generation will still run")
        
        # Initialize sybil detector with adaptive MPNet-calibrated thresholds
        self.sybil_detector = SybilDetector(
            min_high_threshold=SYBIL_MIN_HIGH_THRESHOLD,
            min_very_high_threshold=SYBIL_MIN_VERY_HIGH_THRESHOLD,
            high_similarity_percentile=SYBIL_HIGH_SIMILARITY_PERCENTILE,
            very_high_similarity_percentile=SYBIL_VERY_HIGH_SIMILARITY_PERCENTILE,
            min_response_length=SYBIL_MIN_RESPONSE_LENGTH,
            min_group_size=2,
            min_internal_similarity=SYBIL_MIN_INTERNAL_SIMILARITY,
            fusion_semantic_threshold=SYBIL_FUSION_SEMANTIC_THRESHOLD,
            fusion_lexical_threshold=SYBIL_FUSION_LEXICAL_THRESHOLD,
            fusion_structure_threshold=SYBIL_FUSION_STRUCTURE_THRESHOLD,
            trajectory_threshold=SYBIL_TRAJECTORY_THRESHOLD,
            trajectory_n_clusters=SYBIL_TRAJECTORY_N_CLUSTERS,
        )
    
    # ── Model initialisation helpers ──────────────────────────────────

    @staticmethod
    def _init_embedder_with_fallback() -> SentenceTransformer:
        """Load the primary embedding model with GPU check and MiniLM fallback.

        1. Attempts to load ``SENTENCE_TRANSFORMER_MODEL`` (MPNet by default).
        2. If ``EMBEDDING_FP16_ENABLED`` and a CUDA GPU is available, converts
           the model to half-precision.
        3. If the primary model fails to load, falls back to ``FALLBACK_MODEL``
           (MiniLM).
        """
        import torch

        def _load_and_configure(model_name: str) -> SentenceTransformer:
            model = SentenceTransformer(model_name)
            # Apply max_seq_length from config
            model.max_seq_length = EMBEDDING_MAX_SEQ_LENGTH_DOC
            if EMBEDDING_FP16_ENABLED and torch.cuda.is_available():
                try:
                    model.half()
                    logger.info(
                        f"Loaded {model_name} (FP16 on CUDA)"
                    )
                except Exception as e:
                    logger.warning(f"FP16 conversion failed ({e}), using FP32")
                    logger.info(f"Loaded {model_name} (FP32)")
            else:
                device_info = "CUDA" if torch.cuda.is_available() else "CPU"
                logger.info(f"Loaded {model_name} (FP32, {device_info})")
            return model

        try:
            return _load_and_configure(SENTENCE_TRANSFORMER_MODEL)
        except Exception as e:
            logger.error(
                f"Primary model {SENTENCE_TRANSFORMER_MODEL} failed to load: {e}. "
                f"Falling back to {FALLBACK_MODEL}"
            )
            return _load_and_configure(FALLBACK_MODEL)

    def update_node_map(self, nodes: List[Any]) -> None:
        """Rebuild the internal ``hotkey → (ip, coldkey)`` mapping from a
        fresh metagraph snapshot.

        Call this on every metagraph refresh so that entity-based voting caps
        use up-to-date network topology.

        Args:
            nodes: List of ``fiber.chain.models.Node`` (or any object with
                ``hotkey``, ``ip``, and ``coldkey`` attributes).
        """
        new_map: Dict[str, Tuple[str, str]] = {}
        for node in nodes:
            hk = getattr(node, "hotkey", None)
            ip = getattr(node, "ip", "") or ""
            ck = getattr(node, "coldkey", "") or ""
            if hk:
                new_map[hk] = (str(ip), str(ck))
        self._node_map = new_map
        logger.debug(f"[ENTITY-CAP] Node map updated: {len(new_map)} entries")

    def _embed_batch(
        self,
        texts: List[str],
        *,
        batch_size: Optional[int] = None,
        normalize: bool = True,
    ) -> np.ndarray:
        """Embed a list of texts with ``normalize_embeddings=True`` by default.

        Parameters
        ----------
        texts : list of str
        batch_size : int, optional
            Defaults to ``EMBEDDING_BATCH_SIZE_DOC``.
        normalize : bool
            When True, outputs are L2-normalised so dot product == cosine.
        """
        bs = batch_size or EMBEDDING_BATCH_SIZE_DOC
        return self.embedding_model.encode(
            texts,
            batch_size=bs,
            convert_to_numpy=True,
            normalize_embeddings=normalize,
        )

    async def evaluate_responses(
        self,
        challenge_id: int,
        prompt: str,
        responses: List[InferenceResponse],
        miner_ids: Optional[List[int]] = None,
        miner_hotkeys: Optional[List[str]] = None,
        correlation_id: Optional[str] = None,
        validator_hotkey: Optional[str] = None
    ) -> Tuple[float, Optional[str], Optional[str], Dict[str, float]]:
        """Evaluate a set of inference responses.
        
        Args:
            challenge_id: The challenge ID
            prompt: The original prompt
            responses: List of inference responses
            miner_ids: Optional list of miner UIDs corresponding to each response
                       (used internally for ConsensusEngine labels; NOT for persistent identification)
            miner_hotkeys: List of miner hotkeys (SS58 addresses) corresponding to each response.
                           REQUIRED — used as the persistent identifier for emissions dict keys
                           and sybil detection. Must match responses length.
            correlation_id: Optional correlation ID for tracking
            validator_hotkey: Optional validator hotkey SS58 for Fiber encryption
        
        Returns:
            Tuple containing:
            - Consensus score
            - Path to heatmap image
            - Narrative of consensus
            - Emissions allocation keyed by miner hotkey
        """
        # Store validator_hotkey for use in _upload_heatmap
        self._validator_hotkey = validator_hotkey
        try:
            num_responses = len(responses)
            
            # Validate miner_hotkeys — required for UID-compression-safe emission keys
            if not miner_hotkeys or len(miner_hotkeys) != num_responses:
                raise ValueError(
                    f"miner_hotkeys is required and must match responses length "
                    f"(got {len(miner_hotkeys) if miner_hotkeys else 0} hotkeys "
                    f"for {num_responses} responses)"
                )
            
            logger.info(f"[EVALUATION] Processing {num_responses} response(s) for challenge {challenge_id}")
            
            # Handle edge cases with too few responses
            if num_responses == 0:
                logger.warning(f"[EVALUATION] No responses provided for challenge {challenge_id}")
                return 0.0, None, None, {}
            
            # Filter out test mode responses BEFORE evaluation
            # Test mode responses should receive zero emissions
            valid_responses = []
            valid_miner_ids = []
            valid_miner_hotkeys = []
            test_mode_miners = []  # List of hotkeys (or fallback UIDs) for test mode miners
            
            for i, response in enumerate(responses):
                # Check if response indicates test mode
                response_text = response.response_text.strip()
                if response_text.startswith("[TEST MODE]"):
                    miner_key = miner_hotkeys[i]
                    test_mode_miners.append(miner_key)
                    logger.warning(
                        f"[EVALUATION] Miner {miner_key[:16]}... returned test mode response - "
                        f"will receive ZERO emissions. Response: {response_text[:100]}"
                    )
                else:
                    valid_responses.append(response)
                    if miner_ids and i < len(miner_ids):
                        valid_miner_ids.append(miner_ids[i])
                    if miner_hotkeys and i < len(miner_hotkeys):
                        valid_miner_hotkeys.append(miner_hotkeys[i])
            
            # If all responses are test mode, return zero emissions for all
            if not valid_responses:
                logger.warning(
                    f"[EVALUATION] All {num_responses} responses were test mode - "
                    f"no valid responses to evaluate"
                )
                zero_emissions = {miner_key: 0.0 for miner_key in test_mode_miners}
                
                # Log evaluation result with zero emissions
                self.db_manager.log_evaluation_result(
                    challenge_id=challenge_id,
                    consensus_score=0.0,
                    heatmap_path=None,
                    narrative="All responses were test mode - no evaluation performed",
                    emissions=zero_emissions
                )
                
                return 0.0, None, None, zero_emissions
            
            # If we filtered out test mode responses, log it
            if test_mode_miners:
                logger.info(
                    f"[EVALUATION] Filtered out {len(test_mode_miners)} test mode responses. "
                    f"Evaluating {len(valid_responses)} valid responses. "
                    f"Test mode miners: {test_mode_miners}"
                )
            
            # Continue evaluation with only valid responses
            num_valid = len(valid_responses)
            
            if num_valid == 1:
                logger.info(f"[EVALUATION] Only 1 valid response received - using simple scoring (no consensus evaluation possible)")
                # For single response, return a default score — keyed by hotkey
                single_key = valid_miner_hotkeys[0]
                emissions = {single_key: 1.0}
                
                # Add zero emissions for test mode miners (already keyed by hotkey/fallback)
                for test_miner_key in test_mode_miners:
                    emissions[test_miner_key] = 0.0
                
                # Log simple evaluation result
                self.db_manager.log_evaluation_result(
                    challenge_id=challenge_id,
                    consensus_score=1.0,  # Perfect score for single valid response
                    heatmap_path=None,
                    narrative=None,
                    emissions=emissions
                )
                
                logger.info(f"[EVALUATION] Single response evaluation complete - score: 1.0, emissions: {emissions}")
                return 1.0, None, None, emissions
            
            # Extract response texts and calculate embeddings (only for valid responses)
            # Uses normalize_embeddings=True so dot product == cosine similarity.
            response_texts = [r.response_text for r in valid_responses]
            embeddings = self._embed_batch(response_texts, batch_size=EMBEDDING_BATCH_SIZE_DOC)
            
            # CRITICAL: Generate prompt embedding for semantic quality assessment
            prompt_embedding = self._embed_batch([prompt])[0]
            
            logger.debug(f"[EVALUATION] Generated embeddings: shape={embeddings.shape}")
            
            # Build the embed_fn that quality_scorer / sybil_detector can use to
            # embed arbitrary texts on demand (sentence-level, prompt components, etc.).
            # Passes normalize_embeddings=True and sentence batch size.
            def _embed_fn(texts: List[str]) -> np.ndarray:
                return self._embed_batch(
                    texts,
                    batch_size=EMBEDDING_BATCH_SIZE_SENTENCE,
                    normalize=True,
                )
            
            # ── Entity-based voting caps ─────────────────────────────────
            # Compute per-miner weight multipliers to limit sybil entity
            # influence in consensus scoring.  Disabled miners (or those
            # without node data) default to weight 1.0.
            entity_weights_array: Optional[np.ndarray] = None
            if (
                ENTITY_VOTING_CAPS_ENABLED
                and self._node_map
                and valid_miner_hotkeys
            ):
                entity_weight_map = cap_entity_contributions(
                    miner_hotkeys=valid_miner_hotkeys,
                    node_map=self._node_map,
                    max_entity_votes=ENTITY_MAX_VOTES,
                    min_group_size=ENTITY_MIN_GROUP_SIZE,
                )
                entity_weights_array = np.array(
                    [entity_weight_map.get(hk, 1.0) for hk in valid_miner_hotkeys],
                    dtype=np.float64,
                )
                capped_count = int(np.sum(entity_weights_array < 1.0))
                if capped_count > 0:
                    logger.info(
                        f"[ENTITY-CAP] {capped_count}/{len(valid_miner_hotkeys)} miners "
                        f"capped by entity voting caps"
                    )
                else:
                    logger.debug("[ENTITY-CAP] No miners capped (all unique entities)")
            elif not ENTITY_VOTING_CAPS_ENABLED:
                logger.debug("[ENTITY-CAP] Entity voting caps disabled by config")
            
            # Create consensus engine with miner IDs, prompt embedding, and
            # the enhanced quality scorer (Tier 4+5) for embedding-aware metrics.
            consensus_engine = ConsensusEngine(
                original_prompt=prompt,
                responses=response_texts,
                embeddings=embeddings,
                miner_ids=valid_miner_ids,
                prompt_embedding=prompt_embedding,  # For semantic quality assessment
                quality_scorer=self.quality_scorer if EMBEDDING_SENTENCE_LEVEL else None,
                embed_fn=_embed_fn,
                entity_weights=entity_weights_array,
            )
            
            # Configure consensus evaluation
            # Disable outlier detection and clustering if we have too few responses
            # LOF requires at least n_neighbors + 1 samples (minimum 3)
            # Clustering also needs at least 2 samples
            min_responses_for_outlier_detection = 3
            min_responses_for_clustering = 2
            
            use_outlier_detection = num_valid >= min_responses_for_outlier_detection
            use_clustering = num_valid >= min_responses_for_clustering
            # Check config option and minimum response count
            generate_heatmap = ENABLE_HEATMAP_GENERATION and num_valid >= 2  # Need at least 2 for meaningful heatmap
            
            if not use_outlier_detection:
                logger.debug(f"[EVALUATION] Outlier detection disabled (need {min_responses_for_outlier_detection}+ responses, got {num_valid})")
            if not use_clustering:
                logger.debug(f"[EVALUATION] Clustering disabled (need {min_responses_for_clustering}+ responses, got {num_valid})")
            if not generate_heatmap:
                if not ENABLE_HEATMAP_GENERATION:
                    logger.debug(f"[EVALUATION] Heatmap generation disabled by configuration")
                else:
                    logger.debug(f"[EVALUATION] Heatmap generation disabled (need 2+ responses, got {num_valid})")
            
            # Use challenge_id (UUID) for heatmap filename, but include correlation_id in image title
            # challenge_id should be a UUID string, sanitize it for filename
            import re
            challenge_id_str = str(challenge_id)
            # Sanitize filename (remove invalid characters, but keep UUID format)
            heatmap_id_safe = re.sub(r'[^\w\-_.]', '_', challenge_id_str)
            
            config = ConsensusConfig(
                use_clustering=use_clustering,
                use_weighted_scoring=True,
                use_outlier_detection=use_outlier_detection,
                apply_quality_filter=False,  # Disable legacy word-length filter
                quality_sensitivity=0.7,  # Deprecated - kept for backward compat
                generate_heatmap=generate_heatmap,
                generate_quality_plot=ENABLE_QUALITY_PLOTS,
                heatmap_path=f"temp/heatmap_{heatmap_id_safe}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.png",
                lambda_factor=1.0,
                threshold_min=0.7,
                challenge_id=str(challenge_id),  # Pass challenge_id (UUID) for image title
                correlation_id=correlation_id,  # Pass correlation_id for image title
                # NEW: Semantic quality assessment (CRITICAL for garbage consensus prevention)
                enable_semantic_quality=True,  # Enable semantic quality assessment
                quality_threshold=0.35,  # Minimum quality score to participate in consensus
                quality_prompt_relevance_weight=0.4,
                quality_density_weight=0.2,
                quality_specificity_weight=0.2,
                quality_coherence_weight=0.2,
                # Smart outlier detection
                enable_smart_outlier_detection=True,
                outlier_quality_delta=0.15,
                # Diversity bonus
                enable_diversity_bonus=True,
                max_diversity_bonus=0.15,
                # Garbage detection alerts
                enable_garbage_alerts=True,
                garbage_cluster_threshold=0.4
            )
            
            # Compute similarity matrix BEFORE consensus evaluation (before filtering)
            # This ensures we analyze all responses, not just those that pass filters
            original_similarity_matrix = consensus_engine._compute_pairwise_similarity()
            
            # Evaluate consensus (this may filter responses)
            result = consensus_engine.evaluate_consensus(config)
            
            # Perform sybil detection analysis on valid responses only
            # Only run sybil detection if we have at least 2 valid responses
            # NOTE: Sybil detection uses HOTKEYS as miner identifiers (not UIDs)
            # for UID compression safety. Hotkeys are persistent SS58 addresses.
            sybil_result = None
            if num_valid >= 2:
                try:
                    # Use hotkeys for sybil detection (persistent identifier)
                    # Fallback to str(uid) if hotkeys not available, then str(index)
                    sybil_miner_ids = (
                        valid_miner_hotkeys if valid_miner_hotkeys
                        else [str(uid) for uid in valid_miner_ids] if valid_miner_ids
                        else [str(i) for i in range(len(valid_responses))]
                    )
                    sybil_result = self.sybil_detector.detect_sybil_patterns(
                        similarity_matrix=original_similarity_matrix,
                        responses=valid_responses,
                        miner_ids=sybil_miner_ids,
                        challenge_id=challenge_id,
                        prompt=prompt,
                        embed_fn=_embed_fn,
                    )
                except Exception as e:
                    logger.warning(f"[EVALUATION] Sybil detection failed: {e}. Continuing without sybil detection.")
                    sybil_result = None
            else:
                logger.debug(f"[EVALUATION] Sybil detection skipped (need 2+ responses, got {num_valid})")
            
            # Log sybil detection results for analysis
            if sybil_result and (sybil_result.suspicious_pairs or sybil_result.suspicious_groups):
                analysis_report = self.sybil_detector.generate_analysis_report(sybil_result)
                
                # Convert to JSON-serializable format for storage
                suspicious_pairs_json = [
                    {
                        "miner_hotkey_1": pair.miner_hotkey_1,
                        "miner_hotkey_2": pair.miner_hotkey_2,
                        "similarity_score": pair.similarity_score,
                        "response_text_1": pair.response_text_1[:500],  # Truncate for storage
                        "response_text_2": pair.response_text_2[:500],
                        "detection_method": pair.detection_method,
                    }
                    for pair in sybil_result.suspicious_pairs
                ]
                
                suspicious_groups_json = [
                    {
                        "miner_hotkeys": sorted(list(group.miner_hotkeys)),
                        "avg_similarity": group.avg_similarity,
                        "min_similarity": group.min_similarity,
                        "max_similarity": group.max_similarity,
                        "min_internal_similarity": group.min_internal_similarity,
                        "response_texts": {
                            str(mid): text[:500]  # Truncate for storage
                            for mid, text in group.response_texts.items()
                        }
                    }
                    for group in sybil_result.suspicious_groups
                ]
                
                self.db_manager.log_sybil_detection_result(
                    challenge_id=challenge_id,
                    suspicious_pairs=suspicious_pairs_json,
                    suspicious_groups=suspicious_groups_json,
                    analysis_report=analysis_report,
                    high_similarity_threshold=sybil_result.high_similarity_threshold,
                    very_high_similarity_threshold=sybil_result.very_high_similarity_threshold
                )
                
                logger.warning(
                    f"Sybil detection found {len(sybil_result.suspicious_pairs)} suspicious pairs "
                    f"and {len(sybil_result.suspicious_groups)} suspicious groups for challenge {challenge_id}"
                )
                logger.debug(f"Sybil detection report:\n{analysis_report}")
            
            # Generate narrative (now async) - only if enabled
            narrative = None
            if ENABLE_NARRATIVE_GENERATION and self.narrative_generator:
                try:
                    narrative = await self.narrative_generator.generate_narrative(result)
                    result.consensus_narrative = narrative
                    logger.debug(f"Generated narrative for challenge {challenge_id}")
                except Exception as e:
                    logger.warning(f"Failed to generate narrative for challenge {challenge_id}: {e}. Continuing without narrative.")
                    narrative = None
                    result.consensus_narrative = None
            else:
                result.consensus_narrative = None
                logger.debug(f"Narrative generation disabled - skipping narrative for challenge {challenge_id}")
            
            # Store individual scores before calculating emissions (they get overwritten)
            individual_scores = result.miner_scores if result.miner_scores else {}
            
            # Calculate emissions for valid responses
            # Emissions dict keys are miner HOTKEYS (UID compression safe)
            emissions = self._calculate_emissions(
                valid_responses, result, valid_miner_ids, individual_scores,
                miner_hotkeys=valid_miner_hotkeys
            )
            
            # Add zero emissions for test mode miners (already keyed by hotkey/fallback)
            for test_miner_key in test_mode_miners:
                emissions[test_miner_key] = 0.0
                logger.info(f"[EVALUATION] Miner {test_miner_key[:16]}... test mode - emission set to 0.0")
            
            result.miner_scores = emissions
            
            # Convert numpy float32/float64 values to Python float for JSON serialization
            # This ensures compatibility with SQLAlchemy JSON serialization
            consensus_score_float = float(result.similarity_score)
            emissions_float = {
                k: float(v)  # Convert numpy types to Python float
                for k, v in emissions.items()
            }
            
            # Upload heatmap if available
            heatmap_path = None
            if result.heatmap_path:
                # Use challenge_id (UUID) for upload
                upload_id = str(challenge_id)
                heatmap_path = await self._upload_heatmap(result.heatmap_path, upload_id, file_type="heatmap")
            
            # Upload quality plot if available
            quality_plot_path = None
            if result.quality_plot_path:
                # Use challenge_id (UUID) for upload
                upload_id = str(challenge_id)
                quality_plot_path = await self._upload_heatmap(result.quality_plot_path, upload_id, file_type="quality_plot")
            
            # Log evaluation result
            self.db_manager.log_evaluation_result(
                challenge_id=challenge_id,
                consensus_score=consensus_score_float,
                heatmap_path=heatmap_path,
                narrative=narrative,
                emissions=emissions_float
            )
            
            return consensus_score_float, heatmap_path, narrative, emissions_float
            
        except Exception as e:
            logger.error(f"Error evaluating responses: {str(e)}", exc_info=True)
            return 0.0, None, None, {}
    
    def _calculate_score_difference_bonus(
        self,
        individual_scores: Dict[str, float]
    ) -> tuple[Optional[str], float]:
        """Calculate bonus multiplier based on score difference between highest and second highest.
        
        The bonus scales with the absolute difference between scores:
        - Small difference (e.g., 1 point) = small bonus (~1.0-1.1x)
        - Large difference (e.g., 20 points) = large bonus (approaching 1.5x max)
        
        Args:
            individual_scores: Dict mapping response labels to scores
            
        Returns:
            Tuple of (highest_scoring_label, bonus_multiplier) where multiplier is between 1.0 and 1.5
        """
        if not individual_scores or len(individual_scores) < 2:
            return None, 1.0
        
        # Get sorted scores (descending)
        sorted_scores = sorted(individual_scores.items(), key=lambda x: x[1], reverse=True)
        highest_label, highest_score = sorted_scores[0]
        second_highest_score = sorted_scores[1][1]
        
        # Calculate the absolute difference
        score_difference = highest_score - second_highest_score
        
        # If difference is 0 or negative, no bonus
        if score_difference <= 0:
            return highest_label, 1.0
        
        # Scale the bonus based on absolute difference
        # Use tanh to map differences to [0, 1] range, then scale to [1.0, 1.5]
        # Scale factor controls sensitivity: lower = more sensitive, higher = less sensitive
        # With scale_factor=0.1: diff of 1 point gives ~1.05x, diff of 10 gives ~1.24x, diff of 20 gives ~1.38x
        scale_factor = 0.1
        bonus_multiplier = 1.0 + 0.5 * math.tanh(score_difference * scale_factor)
        
        # Ensure we don't exceed 1.5x
        bonus_multiplier = min(bonus_multiplier, 1.5)
        
        return highest_label, bonus_multiplier
    
    def _calculate_emissions(
        self,
        responses: List[InferenceResponse],
        result: ConsensusResult,
        miner_ids: Optional[List[int]] = None,
        individual_scores: Optional[Dict[str, float]] = None,
        miner_hotkeys: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """Calculate emissions allocation for miners.
        
        Args:
            responses: List of inference responses
            result: Consensus evaluation result
            miner_ids: Optional list of miner UIDs corresponding to each response
                       (used to derive consensus labels matching ConsensusEngine)
            individual_scores: Optional dict of individual response scores (label -> score)
            miner_hotkeys: List of miner hotkeys (SS58 addresses) for emissions dict keys.
                           REQUIRED — emissions MUST be keyed by hotkey (UID compression safe).
        
        Returns:
            Dict mapping miner hotkey (str) to emission score.
        
        Raises:
            ValueError: If miner_hotkeys is not provided or length doesn't match responses.
        """
        if not miner_hotkeys or len(miner_hotkeys) != len(responses):
            raise ValueError(
                f"miner_hotkeys is required and must match responses length "
                f"(got {len(miner_hotkeys) if miner_hotkeys else 0} hotkeys for "
                f"{len(responses)} responses)"
            )
        
        # Base emissions on response time and consensus score
        total_time = sum(r.response_time_ms for r in responses)
        emissions = {}
        
        # Calculate scaled bonus for highest scoring response
        highest_scoring_label = None
        bonus_multiplier = 1.0
        if individual_scores:
            highest_scoring_label, bonus_multiplier = self._calculate_score_difference_bonus(individual_scores)
        
        for i, response in enumerate(responses):
            # Faster responses get more emissions
            time_ratio = 1 - (response.response_time_ms / total_time)
            
            # ConsensusEngine labels: str(uid) when miner_ids are passed, else "R{i+1}".
            # The label must match what's in result.in_consensus dict keys.
            if miner_ids and i < len(miner_ids):
                response_label = str(miner_ids[i])  # Matches ConsensusEngine label format
            else:
                response_label = f"R{i+1}"  # Default label when no miner_ids
            
            in_consensus = response_label in result.in_consensus
            
            # Scale by consensus score and consensus status
            # Convert to float to avoid numpy float32 issues with JSON serialization
            emission = float(time_ratio * result.similarity_score)
            if in_consensus:
                emission *= 1.2  # Bonus for being in consensus
            
            # Scaled bonus for highest scoring response based on score difference
            if highest_scoring_label and response_label == highest_scoring_label:
                emission *= bonus_multiplier
            
            # Emissions keyed by hotkey (persistent SS58 address, UID compression safe)
            emissions[miner_hotkeys[i]] = float(emission)
        
        return emissions
    
    async def _upload_heatmap(self, filepath: str, challenge_id: str, file_type: str = "heatmap") -> Optional[str]:
        """Upload heatmap or quality plot to challenge API and delete local file on success.
        
        Uses Fiber MLTS encryption if validator_hotkey is available, otherwise falls back
        to plain HTTP with API key authentication.
        
        Args:
            filepath: Path to the image file (heatmap or quality plot)
            challenge_id: The challenge ID (UUID) associated with this file
            file_type: Type of file - "heatmap" (default) or "quality_plot"
            
        Returns:
            The filepath if upload was successful, None otherwise
        """
        file_type_label = "heatmap" if file_type == "heatmap" else "quality plot"
        
        try:
            # Determine content type from file extension
            file_ext = os.path.splitext(filepath)[1].lower()
            content_type_map = {
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".png": "image/png",
                ".gif": "image/gif",
                ".webp": "image/webp",
            }
            content_type = content_type_map.get(file_ext, "image/png")  # Default to PNG
            
            # Read file data
            with open(filepath, "rb") as f:
                file_data = f.read()
            
            filename = os.path.basename(filepath)
            
            # Try Fiber encryption first (if validator_hotkey is available)
            validator_hotkey = getattr(self, '_validator_hotkey', None)
            if validator_hotkey:
                result_filename = await self._upload_heatmap_fiber(
                    file_data=file_data,
                    filename=filename,
                    challenge_id=challenge_id,
                    file_type=file_type,
                    content_type=content_type,
                    validator_hotkey=validator_hotkey
                )
                
                if result_filename:
                    # Delete local file after successful upload
                    try:
                        os.remove(filepath)
                        logger.debug(f"Deleted local {file_type_label} file: {filepath}")
                    except OSError as e:
                        logger.warning(f"Failed to delete local {file_type_label} file {filepath}: {e}")
                    return result_filename
                else:
                    logger.warning(
                        f"Fiber upload failed for {file_type_label}, falling back to plain HTTP"
                    )
            
            # Fallback to plain HTTP
            return await self._upload_heatmap_http(
                filepath=filepath,
                file_data=file_data,
                filename=filename,
                challenge_id=challenge_id,
                file_type=file_type,
                content_type=content_type,
                file_type_label=file_type_label
            )
            
        except FileNotFoundError:
            logger.error(f"{file_type_label.capitalize()} file not found: {filepath}")
            return None
        except Exception as e:
            logger.error(f"Error uploading {file_type_label} to challenge API: {str(e)}", exc_info=True)
            return None
    
    async def _upload_heatmap_fiber(
        self,
        file_data: bytes,
        filename: str,
        challenge_id: str,
        file_type: str,
        content_type: str,
        validator_hotkey: str
    ) -> Optional[str]:
        """Upload heatmap using Fiber MLTS encryption.
        
        Args:
            file_data: Raw file bytes
            filename: Original filename
            challenge_id: Challenge ID (UUID)
            file_type: Type of file - "heatmap" or "quality_plot"
            content_type: MIME content type
            validator_hotkey: Validator's SS58 hotkey for Fiber authentication
            
        Returns:
            Filename if successful, None otherwise
        """
        global _heatmap_fiber_client_cache
        
        try:
            server_address = self.config.challenge_api_url
            
            # Get or create Fiber client
            cache_key = f"{validator_hotkey}:{server_address}"
            if cache_key not in _heatmap_fiber_client_cache:
                _heatmap_fiber_client_cache[cache_key] = ValidatorFiberClient(
                    validator_hotkey_ss58=validator_hotkey,
                    key_ttl_seconds=3600,
                    handshake_timeout_seconds=30
                )
            
            fiber_client = _heatmap_fiber_client_cache[cache_key]
            
            async with httpx.AsyncClient() as client:
                result_filename = await fiber_client.send_encrypted_upload(
                    challenge_api_endpoint=server_address,
                    file_data=file_data,
                    filename=filename,
                    challenge_id=challenge_id,
                    file_type=file_type,
                    content_type=content_type,
                    client=client
                )
                return result_filename
                
        except Exception as e:
            logger.error(f"Error in Fiber heatmap upload: {e}", exc_info=True)
            return None
    
    async def _upload_heatmap_http(
        self,
        filepath: str,
        file_data: bytes,
        filename: str,
        challenge_id: str,
        file_type: str,
        content_type: str,
        file_type_label: str
    ) -> Optional[str]:
        """Upload heatmap using plain HTTP with API key (fallback).
        
        Args:
            filepath: Path to the image file
            file_data: Raw file bytes
            filename: Original filename
            challenge_id: Challenge ID (UUID)
            file_type: Type of file - "heatmap" or "quality_plot"
            content_type: MIME content type
            file_type_label: Human-readable file type for logging
            
        Returns:
            Filepath if successful, None otherwise
        """
        try:
            upload_url = f"{self.config.challenge_api_url}/heatmap/upload"
            
            async with aiohttp.ClientSession() as session:
                data = aiohttp.FormData()
                data.add_field(
                    "file",
                    file_data,
                    filename=filename,
                    content_type=content_type
                )
                data.add_field("challenge_id", str(challenge_id))
                data.add_field("file_type", file_type)
                
                # Prefer hotkey signature; fall back to API key.
                # Multipart form-data can't be pre-hashed so we sign
                # without body (nonce + hotkey) — still proves identity.
                headers = get_auth_headers(body=None)
                if not headers:
                    headers = {"X-API-Key": self.config.challenge_api_key}
                
                async with session.post(upload_url, data=data, headers=headers) as response:
                    if response.status == 201:
                        result = await response.json()
                        logger.info(
                            f"Successfully uploaded {file_type_label} for challenge {challenge_id}: "
                            f"{result.get('filename', 'unknown')}"
                        )
                        
                        # Delete local file after successful upload
                        try:
                            os.remove(filepath)
                            logger.debug(f"Deleted local {file_type_label} file: {filepath}")
                        except OSError as e:
                            logger.warning(f"Failed to delete local {file_type_label} file {filepath}: {e}")
                        
                        # Return the filepath from challenge API (if provided)
                        return result.get("filepath", filepath)
                    else:
                        error_text = await response.text()
                        logger.warning(
                            f"Failed to upload {file_type_label} for challenge {challenge_id}: "
                            f"HTTP {response.status} - {error_text}"
                        )
                        return None
                        
        except Exception as e:
            logger.error(f"Error in HTTP heatmap upload: {e}", exc_info=True)
            return None