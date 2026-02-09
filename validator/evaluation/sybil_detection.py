"""
Sybil attack detection module (Tier 5 enhancements).

Identifies suspiciously similar responses that may indicate a sybil attack
where a single entity controls multiple miners and sends identical or
near-identical responses.

Enhancements over the original implementation:
- **Distribution-aware thresholds**: percentile-based with configurable floors
  instead of static 0.95/0.98.
- **Response length gating**: short/canonical answers excluded to reduce false
  positives.
- **Strict group detection**: all internal pairs verified above
  ``min_internal_similarity`` (fixes bridge-node permissive grouping).
- **Multi-view fusion**: combines semantic (embedding), lexical (TF-IDF), and
  structural (answer shape) similarity for robust pair detection.
- **Sentence-trajectory analysis**: clusters sentence embeddings with KMeans
  and compares centroid sequences between miners for fingerprint similarity.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
from loguru import logger
from sklearn.metrics.pairwise import cosine_similarity

from validator.challenge.challenge_types import InferenceResponse


# ── Data structures ────────────────────────────────────────────────────

@dataclass
class SybilPair:
    """Represents a pair of suspiciously similar responses.

    miner_hotkey_1/miner_hotkey_2 are miner HOTKEYS (persistent SS58 addresses),
    NOT UIDs.  UIDs are transient indices that change on UID compression/trimming.
    """
    miner_hotkey_1: str
    miner_hotkey_2: str
    similarity_score: float
    response_text_1: str
    response_text_2: str
    challenge_id: Optional[int] = None
    detection_method: str = "semantic"  # semantic | lexical | structural | fusion | trajectory


@dataclass
class SybilGroup:
    """Represents a group of miners with suspiciously similar responses.

    miner_hotkeys are miner HOTKEYS (persistent SS58 addresses),
    NOT UIDs.
    """
    miner_hotkeys: Set[str]
    avg_similarity: float
    min_similarity: float
    max_similarity: float
    response_texts: Dict[str, str]  # hotkey -> response text
    challenge_id: Optional[int] = None
    min_internal_similarity: float = 0.0  # lowest pairwise similarity inside group


@dataclass
class DistributionStats:
    """Statistics about the similarity distribution used for adaptive thresholds."""
    mean: float = 0.0
    std: float = 0.0
    p50: float = 0.0
    p90: float = 0.0
    p95: float = 0.0
    p99: float = 0.0
    p999: float = 0.0
    p9999: float = 0.0
    adaptive_high_threshold: float = 0.0
    adaptive_very_high_threshold: float = 0.0
    floor_high_threshold: float = 0.0
    floor_very_high_threshold: float = 0.0


@dataclass
class ResponseLengthStats:
    """Response-length gating statistics."""
    total_responses: int = 0
    gated_out: int = 0
    min_length: int = 0
    max_length: int = 0
    avg_length: float = 0.0
    min_threshold: int = 0


@dataclass
class SybilDetectionResult:
    """Results of sybil detection analysis.

    miner_hotkeys are miner HOTKEYS (persistent SS58 addresses), NOT UIDs.
    """
    challenge_id: Optional[int]
    suspicious_pairs: List[SybilPair]
    suspicious_groups: List[SybilGroup]
    similarity_matrix: np.ndarray
    miner_hotkeys: List[str]
    detection_timestamp: datetime
    high_similarity_threshold: float
    very_high_similarity_threshold: float

    # Enhanced fields (Tier 5)
    prompt_hash: Optional[str] = None
    detection_method: str = "semantic"
    distribution_stats: Optional[DistributionStats] = None
    response_length_stats: Optional[ResponseLengthStats] = None


# ── Main detector ──────────────────────────────────────────────────────

class SybilDetector:
    """
    Detects potential sybil attacks by analyzing response similarity patterns.

    A sybil attack occurs when a single entity controls multiple miners and
    sends identical or near-identical responses instead of performing actual
    inference.
    """

    def __init__(
        self,
        # Adaptive threshold floors (MPNet-calibrated)
        min_high_threshold: float = 0.80,
        min_very_high_threshold: float = 0.88,
        # Percentile targets
        high_similarity_percentile: float = 99.9,
        very_high_similarity_percentile: float = 99.99,
        # Response-length gate
        min_response_length: int = 50,
        # Group detection
        min_group_size: int = 2,
        min_internal_similarity: float = 0.90,
        # Multi-view fusion thresholds
        fusion_semantic_threshold: float = 0.92,
        fusion_lexical_threshold: float = 0.85,
        fusion_structure_threshold: float = 0.80,
        # Sentence-trajectory
        trajectory_threshold: float = 0.85,
        trajectory_n_clusters: int = 5,
        # Legacy compat — will be overridden by adaptive thresholds
        high_similarity_threshold: float = 0.95,
        very_high_similarity_threshold: float = 0.98,
    ):
        # Floors
        self.min_high_threshold = min_high_threshold
        self.min_very_high_threshold = min_very_high_threshold

        # Percentiles
        self.high_similarity_percentile = high_similarity_percentile
        self.very_high_similarity_percentile = very_high_similarity_percentile

        # Length gate
        self.min_response_length = min_response_length

        # Groups
        self.min_group_size = min_group_size
        self.min_internal_similarity = min_internal_similarity

        # Multi-view
        self.fusion_semantic_threshold = fusion_semantic_threshold
        self.fusion_lexical_threshold = fusion_lexical_threshold
        self.fusion_structure_threshold = fusion_structure_threshold

        # Trajectory
        self.trajectory_threshold = trajectory_threshold
        self.trajectory_n_clusters = trajectory_n_clusters

        # Legacy fixed thresholds (used as initial seed)
        self._fixed_high = high_similarity_threshold
        self._fixed_very_high = very_high_similarity_threshold

    # ── Public API ─────────────────────────────────────────────────────

    def detect_sybil_patterns(
        self,
        similarity_matrix: np.ndarray,
        responses: List[InferenceResponse],
        miner_ids: List[str],
        challenge_id: Optional[int] = None,
        prompt: Optional[str] = None,
        embed_fn: Optional[Callable] = None,
    ) -> SybilDetectionResult:
        """Detect potential sybil attack patterns from similarity matrix.

        Parameters
        ----------
        similarity_matrix : ndarray (n, n)
            Pairwise semantic similarity matrix.
        responses : list of InferenceResponse
            Raw response objects.
        miner_ids : list of str
            Miner **hotkeys** (persistent SS58 addresses).
        challenge_id : int, optional
        prompt : str, optional
            Original prompt text (needed for multi-view fusion).
        embed_fn : callable, optional
            ``embed_fn(texts) -> ndarray`` (needed for trajectory analysis).
        """
        n = len(responses)
        if n != len(miner_ids) or n != similarity_matrix.shape[0]:
            raise ValueError(
                f"Mismatch in dimensions: {n} responses, "
                f"{len(miner_ids)} miner_ids, "
                f"{similarity_matrix.shape[0]}x{similarity_matrix.shape[1]} matrix"
            )

        response_texts = [r.response_text for r in responses]

        # ── Prompt hash ────────────────────────────────────────────
        prompt_hash = None
        if prompt:
            prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]

        # ── Response length gating ─────────────────────────────────
        lengths = [len(t) for t in response_texts]
        length_stats = ResponseLengthStats(
            total_responses=n,
            gated_out=sum(1 for l in lengths if l < self.min_response_length),
            min_length=min(lengths) if lengths else 0,
            max_length=max(lengths) if lengths else 0,
            avg_length=float(np.mean(lengths)) if lengths else 0.0,
            min_threshold=self.min_response_length,
        )

        # Build a mask of responses that pass the length gate
        length_mask = np.array([l >= self.min_response_length for l in lengths])
        if length_stats.gated_out > 0:
            logger.debug(
                f"[SYBIL] Gated out {length_stats.gated_out}/{n} short responses "
                f"(< {self.min_response_length} chars)"
            )

        # ── Adaptive thresholds (distribution-aware) ───────────────
        dist_stats, high_thresh, very_high_thresh = self._compute_adaptive_thresholds(
            similarity_matrix, length_mask
        )

        # ── Semantic pairs (primary) ──────────────────────────────
        all_pairs: List[SybilPair] = []
        semantic_pairs = self._find_suspicious_pairs(
            similarity_matrix, responses, miner_ids,
            challenge_id, high_thresh, length_mask,
            detection_method="semantic",
        )
        all_pairs.extend(semantic_pairs)

        # ── Multi-view fusion pairs ───────────────────────────────
        fusion_pairs = self._find_multiview_pairs(
            response_texts, miner_ids, responses,
            similarity_matrix, challenge_id, length_mask,
        )
        all_pairs.extend(fusion_pairs)

        # ── Sentence-trajectory pairs ─────────────────────────────
        if embed_fn is not None and n >= 2:
            traj_pairs = self._find_trajectory_pairs(
                response_texts, miner_ids, responses,
                challenge_id, embed_fn, length_mask,
            )
            all_pairs.extend(traj_pairs)

        # ── Deduplicate pairs (same pair from multiple methods) ───
        all_pairs = self._deduplicate_pairs(all_pairs)

        # ── Strict group detection ────────────────────────────────
        groups = self._find_strict_groups(
            similarity_matrix, responses, miner_ids,
            challenge_id, high_thresh, length_mask,
        )

        # ── Determine detection method label ──────────────────────
        methods_used = set()
        for p in all_pairs:
            methods_used.add(p.detection_method)
        detection_method = "+".join(sorted(methods_used)) if methods_used else "semantic"

        return SybilDetectionResult(
            challenge_id=challenge_id,
            suspicious_pairs=all_pairs,
            suspicious_groups=groups,
            similarity_matrix=similarity_matrix,
            miner_hotkeys=miner_ids,
            detection_timestamp=datetime.utcnow(),
            high_similarity_threshold=high_thresh,
            very_high_similarity_threshold=very_high_thresh,
            prompt_hash=prompt_hash,
            detection_method=detection_method,
            distribution_stats=dist_stats,
            response_length_stats=length_stats,
        )

    # ── Adaptive thresholds ────────────────────────────────────────────

    def _compute_adaptive_thresholds(
        self,
        similarity_matrix: np.ndarray,
        length_mask: np.ndarray,
    ) -> Tuple[DistributionStats, float, float]:
        """Compute distribution-aware thresholds with configurable percentiles and floors."""
        n = similarity_matrix.shape[0]

        # Collect upper-triangle similarities for responses that pass the length gate
        valid_indices = np.where(length_mask)[0]
        sims: List[float] = []
        for ii in range(len(valid_indices)):
            for jj in range(ii + 1, len(valid_indices)):
                i, j = valid_indices[ii], valid_indices[jj]
                sims.append(float(similarity_matrix[i, j]))

        if len(sims) < 3:
            # Not enough data for statistical thresholds — use floors
            stats = DistributionStats(
                floor_high_threshold=self.min_high_threshold,
                floor_very_high_threshold=self.min_very_high_threshold,
                adaptive_high_threshold=self.min_high_threshold,
                adaptive_very_high_threshold=self.min_very_high_threshold,
            )
            return stats, self.min_high_threshold, self.min_very_high_threshold

        arr = np.array(sims)
        stats = DistributionStats(
            mean=float(np.mean(arr)),
            std=float(np.std(arr)),
            p50=float(np.percentile(arr, 50)),
            p90=float(np.percentile(arr, 90)),
            p95=float(np.percentile(arr, 95)),
            p99=float(np.percentile(arr, 99)),
            p999=float(np.percentile(arr, 99.9)),
            p9999=float(np.percentile(arr, 99.99)) if len(arr) >= 100 else float(np.max(arr)),
            floor_high_threshold=self.min_high_threshold,
            floor_very_high_threshold=self.min_very_high_threshold,
        )

        # Adaptive = max(percentile, floor)
        adaptive_high = max(
            float(np.percentile(arr, self.high_similarity_percentile)),
            self.min_high_threshold,
        )
        adaptive_very_high = max(
            float(np.percentile(arr, self.very_high_similarity_percentile)),
            self.min_very_high_threshold,
        )
        # Ensure very_high >= high
        adaptive_very_high = max(adaptive_very_high, adaptive_high + 0.01)

        stats.adaptive_high_threshold = adaptive_high
        stats.adaptive_very_high_threshold = adaptive_very_high

        logger.debug(
            f"[SYBIL] Adaptive thresholds: high={adaptive_high:.4f} "
            f"(p{self.high_similarity_percentile}={stats.p999:.4f}, floor={self.min_high_threshold}), "
            f"very_high={adaptive_very_high:.4f} "
            f"(p{self.very_high_similarity_percentile}={stats.p9999:.4f}, floor={self.min_very_high_threshold})"
        )

        return stats, adaptive_high, adaptive_very_high

    # ── Semantic pair detection ─────────────────────────────────────────

    def _find_suspicious_pairs(
        self,
        similarity_matrix: np.ndarray,
        responses: List[InferenceResponse],
        miner_ids: List[str],
        challenge_id: Optional[int],
        threshold: float,
        length_mask: np.ndarray,
        detection_method: str = "semantic",
    ) -> List[SybilPair]:
        """Find pairs above threshold, respecting length gate."""
        pairs: List[SybilPair] = []
        n = len(responses)

        for i in range(n):
            if not length_mask[i]:
                continue
            for j in range(i + 1, n):
                if not length_mask[j]:
                    continue
                sim = similarity_matrix[i, j]
                if sim >= threshold:
                    pairs.append(SybilPair(
                        miner_hotkey_1=miner_ids[i],
                        miner_hotkey_2=miner_ids[j],
                        similarity_score=float(sim),
                        response_text_1=responses[i].response_text,
                        response_text_2=responses[j].response_text,
                        challenge_id=challenge_id,
                        detection_method=detection_method,
                    ))

        pairs.sort(key=lambda p: p.similarity_score, reverse=True)
        return pairs

    # ── Multi-view fusion ──────────────────────────────────────────────

    def _find_multiview_pairs(
        self,
        response_texts: List[str],
        miner_ids: List[str],
        responses: List[InferenceResponse],
        semantic_sim_matrix: np.ndarray,
        challenge_id: Optional[int],
        length_mask: np.ndarray,
    ) -> List[SybilPair]:
        """Combine lexical (TF-IDF) and structural similarity with semantic."""
        n = len(response_texts)
        if n < 2:
            return []

        # ── Lexical similarity (TF-IDF cosine) ────────────────────
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            vectorizer = TfidfVectorizer(
                max_features=5000,
                stop_words="english",
                ngram_range=(1, 2),
            )
            tfidf_matrix = vectorizer.fit_transform(response_texts)
            lexical_sim = cosine_similarity(tfidf_matrix).astype(np.float32)
        except Exception as e:
            logger.warning(f"[SYBIL] TF-IDF vectorisation failed: {e}")
            return []

        # ── Structural similarity ─────────────────────────────────
        structural_sim = self._compute_structural_similarity(response_texts)

        # ── Fusion: flag pair if ANY two of three signals exceed their thresholds
        pairs: List[SybilPair] = []
        for i in range(n):
            if not length_mask[i]:
                continue
            for j in range(i + 1, n):
                if not length_mask[j]:
                    continue

                sem = float(semantic_sim_matrix[i, j])
                lex = float(lexical_sim[i, j])
                struct = float(structural_sim[i, j])

                hits = (
                    int(sem >= self.fusion_semantic_threshold)
                    + int(lex >= self.fusion_lexical_threshold)
                    + int(struct >= self.fusion_structure_threshold)
                )

                if hits >= 2:
                    # Use max of the three as the reported similarity
                    combined = max(sem, lex, struct)
                    pairs.append(SybilPair(
                        miner_hotkey_1=miner_ids[i],
                        miner_hotkey_2=miner_ids[j],
                        similarity_score=combined,
                        response_text_1=responses[i].response_text,
                        response_text_2=responses[j].response_text,
                        challenge_id=challenge_id,
                        detection_method="fusion",
                    ))

        pairs.sort(key=lambda p: p.similarity_score, reverse=True)
        return pairs

    @staticmethod
    def _compute_structural_similarity(texts: Sequence[str]) -> np.ndarray:
        """Compute structural similarity based on answer shape features.

        Features:
        - Normalised response length
        - Sentence count
        - Has numbered list
        - Has bullet list
        - Has code block
        - Paragraph count
        - Average sentence length
        """
        n = len(texts)
        features = np.zeros((n, 7), dtype=np.float32)

        for i, text in enumerate(texts):
            sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
            paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
            words = text.split()

            features[i, 0] = len(text) / 10000.0  # normalised length
            features[i, 1] = len(sentences) / 50.0
            features[i, 2] = 1.0 if re.search(r"(?:^|\n)\s*\d+[.)]\s", text) else 0.0
            features[i, 3] = 1.0 if re.search(r"(?:^|\n)\s*[-•*]\s", text) else 0.0
            features[i, 4] = 1.0 if "```" in text else 0.0
            features[i, 5] = len(paragraphs) / 20.0
            features[i, 6] = (
                (sum(len(s.split()) for s in sentences) / len(sentences) / 50.0)
                if sentences else 0.0
            )

        # Cosine similarity on feature vectors
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-9)
        normalised = features / norms
        return normalised @ normalised.T

    # ── Sentence-trajectory analysis ───────────────────────────────────

    def _find_trajectory_pairs(
        self,
        response_texts: List[str],
        miner_ids: List[str],
        responses: List[InferenceResponse],
        challenge_id: Optional[int],
        embed_fn: Callable,
        length_mask: np.ndarray,
    ) -> List[SybilPair]:
        """Cluster sentence embeddings per response and compare centroid sequences."""
        try:
            from sklearn.cluster import KMeans
        except ImportError:
            logger.warning("[SYBIL] sklearn.cluster.KMeans not available; skipping trajectory analysis")
            return []

        n = len(response_texts)
        if n < 2:
            return []

        # Split each response into sentences and embed
        _SENT_RE = re.compile(r"(?<=[.!?])\s+")
        sentence_lists: List[List[str]] = []
        for text in response_texts:
            sents = [s.strip() for s in _SENT_RE.split(text.strip()) if len(s.split()) >= 3]
            sentence_lists.append(sents)

        # Compute fingerprints (sorted cluster centroids) for each response
        fingerprints: List[Optional[np.ndarray]] = []
        k = max(2, min(self.trajectory_n_clusters, 10))

        for idx, sents in enumerate(sentence_lists):
            if not length_mask[idx] or len(sents) < k:
                fingerprints.append(None)
                continue
            try:
                sent_embs = embed_fn(sents)
                actual_k = min(k, len(sents))
                km = KMeans(n_clusters=actual_k, n_init=3, random_state=42)
                km.fit(sent_embs)
                centroids = km.cluster_centers_
                # Sort centroids by norm to create a canonical ordering
                order = np.argsort(np.linalg.norm(centroids, axis=1))
                fingerprints.append(centroids[order])
            except Exception as e:
                logger.debug(f"[SYBIL] Trajectory embedding failed for response {idx}: {e}")
                fingerprints.append(None)

        # Compare fingerprints pairwise
        pairs: List[SybilPair] = []
        for i in range(n):
            fp_i = fingerprints[i]
            if fp_i is None:
                continue
            for j in range(i + 1, n):
                fp_j = fingerprints[j]
                if fp_j is None:
                    continue
                # Ensure same number of centroids
                if fp_i.shape != fp_j.shape:
                    continue
                # Compute mean cosine similarity between corresponding centroids
                sims = []
                for ci, cj in zip(fp_i, fp_j):
                    sim = float(cosine_similarity(ci.reshape(1, -1), cj.reshape(1, -1))[0, 0])
                    sims.append(sim)
                trajectory_sim = float(np.mean(sims))

                if trajectory_sim >= self.trajectory_threshold:
                    pairs.append(SybilPair(
                        miner_hotkey_1=miner_ids[i],
                        miner_hotkey_2=miner_ids[j],
                        similarity_score=trajectory_sim,
                        response_text_1=responses[i].response_text,
                        response_text_2=responses[j].response_text,
                        challenge_id=challenge_id,
                        detection_method="trajectory",
                    ))

        pairs.sort(key=lambda p: p.similarity_score, reverse=True)
        return pairs

    # ── Strict group detection ─────────────────────────────────────────

    def _find_strict_groups(
        self,
        similarity_matrix: np.ndarray,
        responses: List[InferenceResponse],
        miner_ids: List[str],
        challenge_id: Optional[int],
        high_threshold: float,
        length_mask: np.ndarray,
    ) -> List[SybilGroup]:
        """Find groups using stricter criteria than simple connected components.

        Two miners are connected only if:
        1. Both pass the length gate.
        2. Their pairwise similarity >= ``high_threshold``.

        After finding connected components, we **verify** that every pair inside
        the component has similarity >= ``min_internal_similarity``.  If not,
        we prune the component by removing the weakest links until the
        constraint is satisfied or the group is too small.
        """
        n = len(responses)

        # Build adjacency list (length-gated)
        adjacency: Dict[int, List[int]] = {i: [] for i in range(n)}
        for i in range(n):
            if not length_mask[i]:
                continue
            for j in range(i + 1, n):
                if not length_mask[j]:
                    continue
                if similarity_matrix[i, j] >= high_threshold:
                    adjacency[i].append(j)
                    adjacency[j].append(i)

        # Find connected components (BFS)
        visited: Set[int] = set()
        components: List[List[int]] = []
        for start in range(n):
            if start in visited or not length_mask[start]:
                continue
            queue = [start]
            visited.add(start)
            component: List[int] = []
            while queue:
                node = queue.pop(0)
                component.append(node)
                for nb in adjacency[node]:
                    if nb not in visited:
                        visited.add(nb)
                        queue.append(nb)
            if len(component) >= self.min_group_size:
                components.append(component)

        # Verify & prune each component
        groups: List[SybilGroup] = []
        for component in components:
            verified = self._verify_group(component, similarity_matrix)
            if len(verified) < self.min_group_size:
                continue

            group_miner_ids = [miner_ids[i] for i in verified]
            sims: List[float] = []
            for a in range(len(verified)):
                for b in range(a + 1, len(verified)):
                    sims.append(float(similarity_matrix[verified[a], verified[b]]))

            if not sims:
                continue

            groups.append(SybilGroup(
                miner_hotkeys=set(group_miner_ids),
                avg_similarity=float(np.mean(sims)),
                min_similarity=float(np.min(sims)),
                max_similarity=float(np.max(sims)),
                min_internal_similarity=float(np.min(sims)),
                response_texts={
                    miner_ids[i]: responses[i].response_text
                    for i in verified
                },
                challenge_id=challenge_id,
            ))

        groups.sort(key=lambda g: g.avg_similarity, reverse=True)
        return groups

    def _verify_group(
        self,
        component: List[int],
        similarity_matrix: np.ndarray,
    ) -> List[int]:
        """Iteratively prune nodes whose minimum internal similarity is below threshold."""
        remaining = list(component)
        while len(remaining) >= self.min_group_size:
            # Find the node with the lowest min-pairwise-similarity to any other
            worst_node = None
            worst_min_sim = float("inf")
            for idx in remaining:
                others = [o for o in remaining if o != idx]
                if not others:
                    continue
                min_sim = min(
                    float(similarity_matrix[idx, o]) for o in others
                )
                if min_sim < worst_min_sim:
                    worst_min_sim = min_sim
                    worst_node = idx

            if worst_min_sim >= self.min_internal_similarity:
                break  # All pairs satisfy threshold
            if worst_node is not None:
                remaining.remove(worst_node)
            else:
                break

        return remaining

    # ── Deduplication ──────────────────────────────────────────────────

    @staticmethod
    def _deduplicate_pairs(pairs: List[SybilPair]) -> List[SybilPair]:
        """Remove duplicate pairs (same two miners), keeping highest similarity."""
        seen: Dict[Tuple[str, str], SybilPair] = {}
        for p in pairs:
            key = tuple(sorted([p.miner_hotkey_1, p.miner_hotkey_2]))
            existing = seen.get(key)
            if existing is None or p.similarity_score > existing.similarity_score:
                seen[key] = p
        result = sorted(seen.values(), key=lambda p: p.similarity_score, reverse=True)
        return result

    # ── Report generation ──────────────────────────────────────────────

    def generate_analysis_report(self, result: SybilDetectionResult) -> str:
        """Generate a human-readable analysis report of sybil detection results."""
        lines = [
            "Sybil Detection Analysis Report",
            f"Challenge ID: {result.challenge_id or 'N/A'}",
            f"Detection Timestamp: {result.detection_timestamp.isoformat()}",
            f"Thresholds: High={result.high_similarity_threshold:.4f}, "
            f"Very High={result.very_high_similarity_threshold:.4f}",
            f"Detection Method(s): {result.detection_method}",
            "",
        ]

        if result.distribution_stats:
            ds = result.distribution_stats
            lines.extend([
                "Distribution Statistics:",
                f"  mean={ds.mean:.4f}, std={ds.std:.4f}",
                f"  p50={ds.p50:.4f}, p90={ds.p90:.4f}, p95={ds.p95:.4f}, "
                f"p99={ds.p99:.4f}, p99.9={ds.p999:.4f}",
                "",
            ])

        if result.response_length_stats:
            ls = result.response_length_stats
            lines.extend([
                "Response Length Stats:",
                f"  total={ls.total_responses}, gated_out={ls.gated_out}, "
                f"min={ls.min_length}, max={ls.max_length}, avg={ls.avg_length:.0f}",
                "",
            ])

        lines.extend([
            "Summary:",
            f"  - Total Miners Analyzed: {len(result.miner_hotkeys)}",
            f"  - Suspicious Pairs Found: {len(result.suspicious_pairs)}",
            f"  - Suspicious Groups Found: {len(result.suspicious_groups)}",
            "",
        ])

        if result.suspicious_pairs:
            lines.append("Suspicious Pairs:")
            for i, pair in enumerate(result.suspicious_pairs[:10], 1):
                severity = (
                    "VERY HIGH"
                    if pair.similarity_score >= result.very_high_similarity_threshold
                    else "HIGH"
                )
                lines.append(
                    f"  {i}. [{pair.detection_method}] "
                    f"Miners {pair.miner_hotkey_1} <-> {pair.miner_hotkey_2}: "
                    f"similarity={pair.similarity_score:.4f} ({severity})"
                )
                preview_1 = pair.response_text_1[:100] + "..." if len(pair.response_text_1) > 100 else pair.response_text_1
                preview_2 = pair.response_text_2[:100] + "..." if len(pair.response_text_2) > 100 else pair.response_text_2
                lines.append(f"     Miner {pair.miner_hotkey_1}: {preview_1}")
                lines.append(f"     Miner {pair.miner_hotkey_2}: {preview_2}")
                lines.append("")

            if len(result.suspicious_pairs) > 10:
                lines.append(f"  ... and {len(result.suspicious_pairs) - 10} more pairs")
                lines.append("")

        if result.suspicious_groups:
            lines.append("Suspicious Groups:")
            for i, group in enumerate(result.suspicious_groups[:5], 1):
                lines.append(
                    f"  {i}. Group of {len(group.miner_hotkeys)} miners: "
                    f"hotkeys={sorted(group.miner_hotkeys)}, "
                    f"avg_similarity={group.avg_similarity:.4f}, "
                    f"min_internal={group.min_internal_similarity:.4f}, "
                    f"range=[{group.min_similarity:.4f}, {group.max_similarity:.4f}]"
                )

            if len(result.suspicious_groups) > 5:
                lines.append(f"  ... and {len(result.suspicious_groups) - 5} more groups")

        return "\n".join(lines)
