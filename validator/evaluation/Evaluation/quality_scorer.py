"""
Enhanced Evaluation Quality Scorer (Tier 4 + Tier 5 Enhancements)

Provides multi-granularity quality assessment for miner responses,
replacing the simple heuristic-based metrics in ConsensusEngine with
embedding-aware signals.

Signals produced
----------------
1. **Multi-granularity relevance** — sentence-level + full-response cosine
   similarity to the prompt, with keyword coverage fallback.
2. **Embedding-chain coherence** — local (adjacent), global (centroid-based),
   break counting, topic drift, and graph connectivity.
3. **Prompt coverage completeness** — fraction of prompt semantic
   components addressed by the response (with missing_parts metric).
4. **Reasoning complexity** — semantic step clustering via
   AgglomerativeClustering, shaped reward, non-redundancy / anti-looping,
   path length, progression curvature, gated by quality.
5. **Answer shape checks** — structural constraint detection.
6. **Keyword coverage** — TF-IDF-based keyword extraction fallback.

All signals are normalised to [0, 1] and can be weighted by the
``ConsensusConfig`` quality weights already used by ``ConsensusEngine``.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


# ── Discourse markers used by reasoning complexity scorer ──────────────
_CAUSAL_MARKERS = {
    "because", "since", "therefore", "thus", "hence", "consequently",
    "accordingly", "so", "as a result", "due to", "owing to",
}
_CONTRAST_MARKERS = {
    "however", "but", "although", "nevertheless", "nonetheless",
    "conversely", "on the other hand", "in contrast", "yet", "whereas",
    "despite", "while",
}
_ELABORATION_MARKERS = {
    "for example", "for instance", "specifically", "in particular",
    "namely", "such as", "to illustrate", "that is",
}
_SEQUENCE_MARKERS = {
    "first", "second", "third", "next", "then", "finally",
    "additionally", "moreover", "furthermore", "also", "in addition",
    "subsequently", "lastly",
}

# Compile combined multi-word marker regex (sorted longest-first to avoid
# partial matches).
_ALL_MARKERS = sorted(
    _CAUSAL_MARKERS | _CONTRAST_MARKERS | _ELABORATION_MARKERS | _SEQUENCE_MARKERS,
    key=len,
    reverse=True,
)
_MARKER_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(m) for m in _ALL_MARKERS) + r")\b",
    re.IGNORECASE,
)

# ── Sentence splitter ──────────────────────────────────────────────────
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

# Clause boundary patterns for dependency-like extraction
_CLAUSE_RE = re.compile(
    r"(?:,\s*(?:which|who|that|where|when|while|although|because|since|if|"
    r"unless|whereas|though|even though|after|before|until|once)\b)|"
    r"(?:\b(?:which|who|that)\b\s+(?:\w+\s+){0,3}(?:is|are|was|were|has|have|had)\b)",
    re.IGNORECASE,
)

# ── Sigmoid utility ────────────────────────────────────────────────────

def _sigmoid_clamp(x: float, midpoint: float = 0.5, steepness: float = 10.0) -> float:
    """Soft-clamp a [0, 1] signal through a sigmoid to avoid hard discontinuities.

    Maps 0→~0, midpoint→0.5, 1→~1 with tunable steepness.
    """
    return 1.0 / (1.0 + math.exp(-steepness * (x - midpoint)))


@dataclass
class QualityBreakdown:
    """Per-response quality breakdown returned by the scorer."""

    # ── Individual signal scores (all in [0, 1]) ──
    relevance: float = 0.0
    sentence_relevance: float = 0.0
    full_response_relevance: float = 0.0
    keyword_coverage: float = 0.0

    # Coherence sub-signals
    coherence: float = 0.0
    local_coherence: float = 0.0
    global_coherence: float = 0.0
    coherence_breaks: int = 0
    coherence_break_rate: float = 0.0
    topic_drift: float = 0.0
    coherence_graph_score: float = 0.0

    # Coverage
    coverage: float = 0.0
    missing_parts: float = 0.0

    # Complexity sub-signals
    complexity: float = 0.0
    step_score: float = 0.0
    num_steps: int = 0
    novelty: float = 0.0
    path_length: float = 0.0
    curvature: float = 0.0
    redundancy: float = 0.0

    # Shape / structure
    answer_shape_score: float = 0.0

    # Legacy signals
    density: float = 0.0
    specificity: float = 0.0

    # Composite quality score (weighted combination)
    composite: float = 0.0

    # Diagnostic metadata
    sentence_count: int = 0
    covered_components: int = 0
    total_components: int = 0
    discourse_marker_count: int = 0


class EvaluationQualityScorer:
    """
    Embedding-aware quality scorer for miner responses.

    This class is **stateless** — instantiate once and call :meth:`score_batch`
    for each evaluation round.  It requires pre-computed embeddings so that
    the (expensive) SentenceTransformer inference happens only once.

    Parameters
    ----------
    sentence_relevance_weight : float
        Blend between sentence-level and full-response relevance (0–1).
    min_sentence_words : int
        Sentences shorter than this are ignored in granular metrics.
    coverage_threshold : float
        Cosine similarity threshold for a prompt component to count as "covered".
    coherence_embedding_chain : bool
        Whether to use embedding-chain coherence (True) or fall back to the
        heuristic sentence-length coherence from ConsensusEngine (False).
    complexity_enabled : bool
        Whether to compute reasoning-complexity signal.
    coverage_enabled : bool
        Whether to compute prompt-coverage completeness signal.
    break_threshold : float
        Adjacent cosine similarity below which a "topic break" is counted.
    coherence_local_weight : float
        Blend weight for local coherence in the combined coherence signal.
    coherence_global_weight : float
        Blend weight for global (centroid) coherence in the combined signal.
    relevance_doc_blend : float
        Weight for doc-level relevance in the combined relevance signal.
    relevance_sentence_blend : float
        Weight for sentence-level relevance in the combined relevance signal.
    relevance_coverage_blend : float
        Weight for coverage in the combined relevance signal.
    complexity_relevance_gate : float
        Minimum relevance to allow complexity scoring.
    complexity_coherence_gate : float
        Minimum coherence to allow complexity scoring.
    complexity_target_steps : int
        Optimal number of semantic steps for shaped reward.
    complexity_scale : float
        Width of the Gaussian reward around target_steps.
    complexity_distance_threshold : float
        AgglomerativeClustering distance threshold.
    complexity_step_weight : float
        Weight of step_score in combined complexity.
    complexity_novelty_weight : float
        Weight of novelty in combined complexity.
    """

    def __init__(
        self,
        sentence_relevance_weight: float = 0.5,
        min_sentence_words: int = 3,
        coverage_threshold: float = 0.45,
        coherence_embedding_chain: bool = True,
        complexity_enabled: bool = True,
        coverage_enabled: bool = True,
        break_threshold: float = 0.25,
        coherence_local_weight: float = 0.6,
        coherence_global_weight: float = 0.4,
        relevance_doc_blend: float = 0.4,
        relevance_sentence_blend: float = 0.4,
        relevance_coverage_blend: float = 0.2,
        complexity_relevance_gate: float = 0.4,
        complexity_coherence_gate: float = 0.3,
        complexity_target_steps: int = 4,
        complexity_scale: float = 2.0,
        complexity_distance_threshold: float = 0.7,
        complexity_step_weight: float = 0.6,
        complexity_novelty_weight: float = 0.4,
    ):
        self.sentence_relevance_weight = sentence_relevance_weight
        self.min_sentence_words = min_sentence_words
        self.coverage_threshold = coverage_threshold
        self.coherence_embedding_chain = coherence_embedding_chain
        self.complexity_enabled = complexity_enabled
        self.coverage_enabled = coverage_enabled
        self.break_threshold = break_threshold
        self.coherence_local_weight = coherence_local_weight
        self.coherence_global_weight = coherence_global_weight
        self.relevance_doc_blend = relevance_doc_blend
        self.relevance_sentence_blend = relevance_sentence_blend
        self.relevance_coverage_blend = relevance_coverage_blend
        self.complexity_relevance_gate = complexity_relevance_gate
        self.complexity_coherence_gate = complexity_coherence_gate
        self.complexity_target_steps = complexity_target_steps
        self.complexity_scale = complexity_scale
        self.complexity_distance_threshold = complexity_distance_threshold
        self.complexity_step_weight = complexity_step_weight
        self.complexity_novelty_weight = complexity_novelty_weight

    # ── Public API ─────────────────────────────────────────────────────

    def score_batch(
        self,
        responses: Sequence[str],
        response_embeddings: np.ndarray,
        prompt: str,
        prompt_embedding: np.ndarray,
        embed_fn: Callable,
        *,
        quality_weights: Optional[Dict[str, float]] = None,
    ) -> Tuple[np.ndarray, List[QualityBreakdown]]:
        """Score a batch of responses with batched sentence embedding.

        Parameters
        ----------
        responses : sequence of str
            Raw response texts.
        response_embeddings : ndarray, shape (N, D)
            Full-response embeddings (already computed by ConsensusEngine).
        prompt : str
            The original prompt text.
        prompt_embedding : ndarray, shape (D,)
            Embedding of the prompt.
        embed_fn : callable
            ``embed_fn(texts: List[str]) -> ndarray``
        quality_weights : dict, optional
            Override composite weight map.

        Returns
        -------
        scores : ndarray, shape (N,)
            Composite quality score for each response, in [0, 1].
        breakdowns : list of QualityBreakdown
            Per-response signal breakdown.
        """
        n = len(responses)
        if n == 0:
            return np.array([]), []

        weights = quality_weights or {
            "relevance": 0.25,
            "coherence": 0.15,
            "coverage": 0.15,
            "complexity": 0.15,
            "density": 0.15,
            "specificity": 0.15,
        }

        # ── Decompose prompt into semantic components ──────────────
        prompt_components = self._split_prompt_components(prompt)
        prompt_component_embeddings: Optional[np.ndarray] = None
        if self.coverage_enabled and prompt_components:
            prompt_component_embeddings = embed_fn(prompt_components)

        # ── Extract keywords for keyword-coverage fallback ─────────
        prompt_keywords = self._extract_keywords_tfidf(prompt, responses)

        # ── Batch sentence embedding optimisation ──────────────────
        # Collect all sentences from all responses into a single list,
        # embed them in one GPU call, then distribute back.
        all_sentences: List[str] = []
        sentence_boundaries: List[Tuple[int, int]] = []  # (start, end) per response
        per_response_sentences: List[List[str]] = []

        for resp in responses:
            sents = self._split_sentences(resp)
            start = len(all_sentences)
            all_sentences.extend(sents)
            sentence_boundaries.append((start, len(all_sentences)))
            per_response_sentences.append(sents)

        # Single batch embedding call for ALL sentences across ALL responses
        all_sentence_embeddings: Optional[np.ndarray] = None
        if all_sentences:
            try:
                all_sentence_embeddings = embed_fn(all_sentences)
            except Exception as e:
                logger.warning(f"[QUALITY_SCORER] Batch sentence embedding failed: {e}")

        # ── Per-response scoring ───────────────────────────────────
        breakdowns: List[QualityBreakdown] = []
        scores = np.zeros(n, dtype=np.float64)

        for i, response in enumerate(responses):
            # Extract pre-computed sentence embeddings for this response
            start_idx, end_idx = sentence_boundaries[i]
            if all_sentence_embeddings is not None and end_idx > start_idx:
                sentence_embeddings = all_sentence_embeddings[start_idx:end_idx]
            else:
                sentence_embeddings = None

            bd = self._score_single(
                response=response,
                response_embedding=response_embeddings[i],
                prompt=prompt,
                prompt_embedding=prompt_embedding,
                prompt_components=prompt_components,
                prompt_component_embeddings=prompt_component_embeddings,
                embed_fn=embed_fn,
                sentences=per_response_sentences[i],
                sentence_embeddings=sentence_embeddings,
                prompt_keywords=prompt_keywords,
            )

            # Weighted composite with sigmoid clamping
            composite = (
                weights.get("relevance", 0.25) * _sigmoid_clamp(bd.relevance, 0.45, 8.0)
                + weights.get("coherence", 0.15) * _sigmoid_clamp(bd.coherence, 0.45, 8.0)
                + weights.get("coverage", 0.15) * _sigmoid_clamp(bd.coverage, 0.40, 8.0)
                + weights.get("complexity", 0.15) * _sigmoid_clamp(bd.complexity, 0.30, 8.0)
                + weights.get("density", 0.15) * _sigmoid_clamp(bd.density, 0.40, 8.0)
                + weights.get("specificity", 0.15) * _sigmoid_clamp(bd.specificity, 0.30, 8.0)
            )
            bd.composite = float(np.clip(composite, 0.0, 1.0))
            scores[i] = bd.composite
            breakdowns.append(bd)

        logger.debug(
            f"[QUALITY_SCORER] Scored {n} responses: "
            f"mean={scores.mean():.3f}, min={scores.min():.3f}, max={scores.max():.3f}"
        )
        return scores, breakdowns

    # ── Private helpers ────────────────────────────────────────────────

    def _score_single(
        self,
        response: str,
        response_embedding: np.ndarray,
        prompt: str,
        prompt_embedding: np.ndarray,
        prompt_components: List[str],
        prompt_component_embeddings: Optional[np.ndarray],
        embed_fn: Callable,
        sentences: List[str],
        sentence_embeddings: Optional[np.ndarray],
        prompt_keywords: Optional[List[str]] = None,
    ) -> QualityBreakdown:
        """Compute all quality signals for a single response."""

        bd = QualityBreakdown()
        bd.sentence_count = len(sentences)

        # ── 1. Full-response relevance (baseline) ─────────────────
        bd.full_response_relevance = float(
            cosine_similarity(
                prompt_embedding.reshape(1, -1),
                response_embedding.reshape(1, -1),
            )[0, 0]
        )

        # ── 2. Sentence-level relevance ───────────────────────────
        if sentence_embeddings is not None and len(sentences) >= 2:
            sentence_sims = cosine_similarity(
                prompt_embedding.reshape(1, -1),
                sentence_embeddings,
            )[0]
            k = max(1, len(sentences) // 2)
            top_k_sims = np.sort(sentence_sims)[-k:]
            bd.sentence_relevance = float(np.mean(top_k_sims))
        else:
            bd.sentence_relevance = bd.full_response_relevance

        # ── 3. Keyword coverage (TF-IDF fallback) ─────────────────
        if prompt_keywords:
            bd.keyword_coverage = self._keyword_coverage(response, prompt_keywords)
        else:
            bd.keyword_coverage = bd.full_response_relevance

        # ── 4. Combined relevance ─────────────────────────────────
        # Blend: doc * w1 + sentence * w2 + coverage * w3
        # Coverage component is filled in step 7 below; use keyword as interim proxy.
        bd.relevance = (
            self.relevance_doc_blend * bd.full_response_relevance
            + self.relevance_sentence_blend * bd.sentence_relevance
            + self.relevance_coverage_blend * bd.keyword_coverage
        )

        # ── 5. Coherence (local + global + breaks + drift + graph) ─
        if (
            self.coherence_embedding_chain
            and sentence_embeddings is not None
            and len(sentences) >= 2
        ):
            coh = self._compute_full_coherence(sentence_embeddings)
            bd.local_coherence = coh["local"]
            bd.global_coherence = coh["global"]
            bd.coherence_breaks = coh["breaks"]
            bd.coherence_break_rate = coh["break_rate"]
            bd.topic_drift = coh["drift"]
            bd.coherence_graph_score = coh["graph_score"]
            # Combined coherence
            bd.coherence = (
                self.coherence_local_weight * bd.local_coherence
                + self.coherence_global_weight * bd.global_coherence
            )
            # Penalise high break rate
            if bd.coherence_break_rate > 0.3:
                bd.coherence *= max(0.5, 1.0 - bd.coherence_break_rate)
        else:
            bd.coherence = self._heuristic_coherence(response)
            bd.local_coherence = bd.coherence
            bd.global_coherence = bd.coherence

        # ── 6. Answer shape checks ────────────────────────────────
        bd.answer_shape_score = self._answer_shape(response, prompt)

        # ── 7. Prompt coverage completeness ───────────────────────
        if (
            self.coverage_enabled
            and prompt_component_embeddings is not None
            and len(prompt_components) > 0
        ):
            cov = self._prompt_coverage(
                response_embedding=response_embedding,
                sentences=sentences,
                sentence_embeddings=sentence_embeddings,
                prompt_components=prompt_components,
                prompt_component_embeddings=prompt_component_embeddings,
            )
            bd.coverage = cov["coverage"]
            bd.missing_parts = cov["missing_parts"]
            bd.covered_components = cov["covered"]
            bd.total_components = cov["total"]
        else:
            bd.coverage = bd.relevance
            bd.missing_parts = 0.0

        # Re-blend relevance now that coverage is available
        bd.relevance = (
            self.relevance_doc_blend * bd.full_response_relevance
            + self.relevance_sentence_blend * bd.sentence_relevance
            + self.relevance_coverage_blend * bd.coverage
        )

        # ── 8. Reasoning complexity (gated) ───────────────────────
        if self.complexity_enabled:
            cx = self._compute_full_complexity(
                response=response,
                sentences=sentences,
                sentence_embeddings=sentence_embeddings,
                current_relevance=bd.relevance,
                current_coherence=bd.coherence,
            )
            bd.complexity = cx["complexity"]
            bd.step_score = cx["step_score"]
            bd.num_steps = cx["num_steps"]
            bd.novelty = cx["novelty"]
            bd.redundancy = cx["redundancy"]
            bd.path_length = cx["path_length"]
            bd.curvature = cx["curvature"]
            bd.discourse_marker_count = cx["discourse_marker_count"]
        else:
            bd.complexity = 0.5

        # ── 9. Information density ────────────────────────────────
        bd.density = self._information_density(response)

        # ── 10. Specificity ───────────────────────────────────────
        bd.specificity = self._specificity(response)

        return bd

    # ── Signal implementations ─────────────────────────────────────────

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        """Split text into sentences using nltk (preferred) or regex fallback."""
        try:
            import nltk
            raw = nltk.sent_tokenize(text.strip())
        except (ImportError, LookupError):
            raw = _SENTENCE_SPLIT_RE.split(text.strip())
        return [s.strip() for s in raw if len(s.split()) >= 3]

    @staticmethod
    def _split_prompt_components(prompt: str) -> List[str]:
        """Decompose a prompt into semantic components.

        Handles: ``?``, numbered/bulleted items, ``\\n``, ``:``, ``;``,
        ``and`` / ``or`` conjunctions connecting clauses, and
        dependency-like clause boundaries.
        """
        components: List[str] = []

        # 1. Try splitting on question marks (multiple sub-questions)
        questions = [q.strip() + "?" for q in prompt.split("?") if q.strip()]
        if len(questions) >= 2:
            components.extend(questions)
            return [c for c in components if len(c.split()) >= 3]

        # 2. Try splitting on numbered/bulleted items
        numbered = re.split(r"(?:^|\n)\s*(?:\d+[.)]\s*|[-•*]\s*)", prompt)
        numbered = [s.strip() for s in numbered if s.strip() and len(s.split()) >= 3]
        if len(numbered) >= 2:
            return numbered

        # 3. Try splitting on newlines
        newline_parts = [s.strip() for s in prompt.split("\n") if s.strip() and len(s.split()) >= 3]
        if len(newline_parts) >= 2:
            return newline_parts

        # 4. Try splitting on colons
        colon_parts = [s.strip() for s in prompt.split(":") if s.strip() and len(s.split()) >= 3]
        if len(colon_parts) >= 2:
            return colon_parts

        # 5. Try splitting on semicolons
        semi = [s.strip() for s in prompt.split(";") if s.strip() and len(s.split()) >= 3]
        if len(semi) >= 2:
            return semi

        # 6. Try splitting on " and " / " or " conjunctions
        conj_parts = re.split(r"\s+(?:and|or)\s+", prompt, flags=re.IGNORECASE)
        conj_parts = [s.strip() for s in conj_parts if s.strip() and len(s.split()) >= 3]
        if len(conj_parts) >= 2:
            return conj_parts

        # 7. Try dependency-like clause extraction
        clause_boundaries = list(_CLAUSE_RE.finditer(prompt))
        if clause_boundaries:
            parts: List[str] = []
            prev_end = 0
            for m in clause_boundaries:
                seg = prompt[prev_end:m.start()].strip()
                if seg and len(seg.split()) >= 3:
                    parts.append(seg)
                prev_end = m.start()
            tail = prompt[prev_end:].strip()
            if tail and len(tail.split()) >= 3:
                parts.append(tail)
            if len(parts) >= 2:
                return parts

        # 8. Fall back to whole prompt as single component
        if prompt.strip():
            return [prompt.strip()]
        return []

    # ── Coherence (full) ───────────────────────────────────────────────

    def _compute_full_coherence(
        self, sentence_embeddings: np.ndarray
    ) -> Dict[str, float]:
        """Compute local, global, break, drift, and graph coherence signals."""
        n = len(sentence_embeddings)
        if n < 2:
            return {
                "local": 1.0, "global": 1.0, "breaks": 0,
                "break_rate": 0.0, "drift": 0.0, "graph_score": 1.0,
            }

        # ── A) Local coherence (adjacent) ─────────────────────────
        local_sims = []
        for i in range(n - 1):
            sim = float(cosine_similarity(
                sentence_embeddings[i:i + 1],
                sentence_embeddings[i + 1:i + 2],
            )[0, 0])
            local_sims.append(sim)
        local_coherence = float(np.clip(np.mean(local_sims), 0.0, 1.0))

        # ── B) Global coherence (centroid) ────────────────────────
        centroid = np.mean(sentence_embeddings, axis=0, keepdims=True)
        norm = np.linalg.norm(centroid)
        if norm > 1e-9:
            centroid = centroid / norm
        global_sims = cosine_similarity(sentence_embeddings, centroid).flatten()
        global_coherence = float(np.clip(np.mean(global_sims), 0.0, 1.0))

        # ── C) Break counting ─────────────────────────────────────
        break_count = sum(1 for s in local_sims if s < self.break_threshold)
        break_rate = break_count / len(local_sims) if local_sims else 0.0

        # ── D) Topic drift ────────────────────────────────────────
        drift = 1.0 - float(cosine_similarity(
            sentence_embeddings[0:1],
            sentence_embeddings[-1:],
        )[0, 0])
        drift = float(np.clip(drift, 0.0, 1.0))

        # ── E) Graph coherence score ──────────────────────────────
        graph_score = self._coherence_graph_score(sentence_embeddings, local_sims)

        return {
            "local": local_coherence,
            "global": global_coherence,
            "breaks": break_count,
            "break_rate": break_rate,
            "drift": drift,
            "graph_score": graph_score,
        }

    @staticmethod
    def _coherence_graph_score(
        sentence_embeddings: np.ndarray,
        local_sims: Optional[List[float]] = None,
        edge_threshold: float = 0.3,
    ) -> float:
        """Compute graph-based coherence metrics.

        Builds a similarity graph between sentences (edge if cos > threshold),
        then computes:
        - Fraction of nodes in the largest connected component
        - Average clustering coefficient
        - Spectral gap (Laplacian second eigenvalue) — measures overall connectivity
        """
        n = len(sentence_embeddings)
        if n < 3:
            return 1.0

        # Build full similarity matrix
        sim_matrix = cosine_similarity(sentence_embeddings)
        adjacency = (sim_matrix >= edge_threshold).astype(float)
        np.fill_diagonal(adjacency, 0.0)

        # 1. Largest connected component fraction (BFS)
        visited = set()
        max_component = 0
        for start in range(n):
            if start in visited:
                continue
            queue = [start]
            visited.add(start)
            comp_size = 0
            while queue:
                node = queue.pop(0)
                comp_size += 1
                for nb in range(n):
                    if adjacency[node, nb] > 0 and nb not in visited:
                        visited.add(nb)
                        queue.append(nb)
            max_component = max(max_component, comp_size)
        lcc_fraction = max_component / n

        # 2. Average clustering coefficient
        clustering_coeffs = []
        for i in range(n):
            neighbours = [j for j in range(n) if adjacency[i, j] > 0]
            k = len(neighbours)
            if k < 2:
                clustering_coeffs.append(0.0)
                continue
            # Count edges between neighbours
            edges = sum(
                1 for a in range(len(neighbours))
                for b in range(a + 1, len(neighbours))
                if adjacency[neighbours[a], neighbours[b]] > 0
            )
            clustering_coeffs.append(2.0 * edges / (k * (k - 1)))
        avg_clustering = float(np.mean(clustering_coeffs))

        # 3. Spectral gap (second-smallest eigenvalue of Laplacian)
        degree = np.sum(adjacency, axis=1)
        laplacian = np.diag(degree) - adjacency
        try:
            eigenvalues = np.sort(np.real(np.linalg.eigvalsh(laplacian)))
            spectral_gap = float(eigenvalues[1]) if len(eigenvalues) > 1 else 0.0
            # Normalize: a well-connected graph has spectral_gap ≈ n
            spectral_gap_norm = min(1.0, spectral_gap / max(1.0, n * 0.1))
        except Exception:
            spectral_gap_norm = 0.5

        # Combine: LCC fraction (0.4) + clustering (0.3) + spectral (0.3)
        return float(np.clip(
            0.4 * lcc_fraction + 0.3 * avg_clustering + 0.3 * spectral_gap_norm,
            0.0, 1.0,
        ))

    # ── Coverage ───────────────────────────────────────────────────────

    def _prompt_coverage(
        self,
        response_embedding: np.ndarray,
        sentences: List[str],
        sentence_embeddings: Optional[np.ndarray],
        prompt_components: List[str],
        prompt_component_embeddings: np.ndarray,
    ) -> Dict[str, float]:
        """Compute prompt coverage completeness.

        Returns dict with 'coverage', 'covered', 'total', 'missing_parts'.
        """
        total = len(prompt_components)
        if total == 0:
            return {"coverage": 1.0, "covered": 0, "total": 0, "missing_parts": 0.0}

        # Build response-side embeddings matrix
        if sentence_embeddings is not None and len(sentences) >= 2:
            response_side = np.vstack(
                [response_embedding.reshape(1, -1), sentence_embeddings]
            )
        else:
            response_side = response_embedding.reshape(1, -1)

        # Compute max similarity for each prompt component
        sim_matrix = cosine_similarity(prompt_component_embeddings, response_side)
        max_sims = sim_matrix.max(axis=1)

        covered = int(np.sum(max_sims >= self.coverage_threshold))
        missing = total - covered
        coverage_score = float(covered / total)
        missing_parts = float(missing / total)

        return {
            "coverage": coverage_score,
            "covered": covered,
            "total": total,
            "missing_parts": missing_parts,
        }

    # ── Complexity (full) ──────────────────────────────────────────────

    def _compute_full_complexity(
        self,
        response: str,
        sentences: List[str],
        sentence_embeddings: Optional[np.ndarray],
        current_relevance: float,
        current_coherence: float,
    ) -> Dict[str, float]:
        """Compute full complexity metrics: step clustering, novelty,
        path length, curvature, discourse markers — gated by quality.
        """
        base = {
            "complexity": 0.0, "step_score": 0.0, "num_steps": 0,
            "novelty": 0.0, "redundancy": 0.0, "path_length": 0.0,
            "curvature": 0.0, "discourse_marker_count": 0,
        }

        # Hard gate: only reward complexity if quality is sufficient
        if (
            current_relevance < self.complexity_relevance_gate
            or current_coherence < self.complexity_coherence_gate
        ):
            return base

        # ── Discourse markers (always available) ──────────────────
        markers_found = _MARKER_RE.findall(response.lower())
        marker_count = len(markers_found)
        base["discourse_marker_count"] = marker_count

        if not sentences or len(sentences) < 2:
            # Minimal complexity from markers alone
            if marker_count > 0 and sentences:
                mps = marker_count / max(len(sentences), 1)
                base["complexity"] = float(np.clip(1.0 - np.exp(-1.3 * mps), 0.0, 1.0))
            return base

        # If we have sentence embeddings, compute embedding-based signals
        if sentence_embeddings is not None and len(sentence_embeddings) >= 2:
            # ── A) Semantic step count (AgglomerativeClustering) ──
            step_score, num_steps = self._semantic_step_clustering(sentence_embeddings)
            base["step_score"] = step_score
            base["num_steps"] = num_steps

            # ── B) Non-redundancy / anti-looping ──────────────────
            redundancy, novelty = self._embedding_redundancy(sentence_embeddings)
            base["redundancy"] = redundancy
            base["novelty"] = novelty

            # ── C) Path length (progression energy) ───────────────
            base["path_length"] = self._path_length(sentence_embeddings)

            # ── D) Progression curvature ──────────────────────────
            base["curvature"] = self._progression_curvature(sentence_embeddings)

            # Combined complexity (step + novelty, weighted)
            base["complexity"] = float(np.clip(
                self.complexity_step_weight * step_score
                + self.complexity_novelty_weight * novelty,
                0.0, 1.0,
            ))
        else:
            # Discourse-marker-only fallback
            if sentences:
                mps = marker_count / len(sentences)
                dm_score = float(np.clip(1.0 - np.exp(-1.3 * mps), 0.0, 1.0))
                # Variety bonus
                text_lower = response.lower()
                types_present = sum(1 for marker_set in (
                    _CAUSAL_MARKERS, _CONTRAST_MARKERS,
                    _ELABORATION_MARKERS, _SEQUENCE_MARKERS,
                ) if any(m in text_lower for m in marker_set))
                variety_bonus = min(0.15, max(0, types_present - 1) * 0.05)
                base["complexity"] = float(np.clip(dm_score + variety_bonus, 0.0, 1.0))

        return base

    def _semantic_step_clustering(
        self, sentence_embeddings: np.ndarray
    ) -> Tuple[float, int]:
        """Cluster sentence embeddings into semantic steps and score."""
        try:
            from sklearn.cluster import AgglomerativeClustering
        except ImportError:
            return 0.5, 0

        n = len(sentence_embeddings)
        if n < 2:
            return 0.0, 1

        try:
            clustering = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=self.complexity_distance_threshold,
                linkage="average",
            )
            labels = clustering.fit_predict(sentence_embeddings)
            num_clusters = len(set(labels))
        except Exception:
            return 0.5, 0

        # Shaped reward: Gaussian centered on target_steps
        step_score = float(np.exp(
            -((num_clusters - self.complexity_target_steps) / self.complexity_scale) ** 2
        ))
        return step_score, num_clusters

    @staticmethod
    def _embedding_redundancy(sentence_embeddings: np.ndarray) -> Tuple[float, float]:
        """Compute mean pairwise similarity (redundancy) and novelty = 1 - redundancy."""
        n = len(sentence_embeddings)
        if n < 2:
            return 0.0, 1.0

        sim_matrix = cosine_similarity(sentence_embeddings)
        # Upper triangle (exclude diagonal)
        upper = sim_matrix[np.triu_indices(n, k=1)]
        redundancy = float(np.mean(upper)) if len(upper) > 0 else 0.0
        novelty = max(0.0, 1.0 - redundancy)
        return redundancy, novelty

    @staticmethod
    def _path_length(sentence_embeddings: np.ndarray) -> float:
        """Sum of transition energies: sum(1 - cos(e_i, e_{i+1})).

        Normalised by (n-1) to give a per-transition average.
        Higher = more progression / semantic movement.
        """
        n = len(sentence_embeddings)
        if n < 2:
            return 0.0

        total = 0.0
        for i in range(n - 1):
            sim = float(cosine_similarity(
                sentence_embeddings[i:i + 1],
                sentence_embeddings[i + 1:i + 2],
            )[0, 0])
            total += 1.0 - sim

        avg_energy = total / (n - 1)
        # Normalise: typical values 0.0–0.5, map to 0–1
        return float(np.clip(avg_energy * 2.0, 0.0, 1.0))

    @staticmethod
    def _progression_curvature(sentence_embeddings: np.ndarray) -> float:
        """Curvature = mean(1 - cos(d_i, d_{i+1})) where d_i = e_{i+1} - e_i.

        Measures how "structured" the reasoning steps are vs random jumps.
        Low curvature = smooth progression; high curvature = direction changes.
        Moderate curvature (0.3–0.5) indicates structured argument.
        """
        n = len(sentence_embeddings)
        if n < 3:
            return 0.0

        # Compute direction vectors
        directions = []
        for i in range(n - 1):
            d = sentence_embeddings[i + 1] - sentence_embeddings[i]
            norm = np.linalg.norm(d)
            if norm > 1e-9:
                directions.append(d / norm)
            else:
                directions.append(d)

        if len(directions) < 2:
            return 0.0

        # Curvature between consecutive direction vectors
        curvatures = []
        for i in range(len(directions) - 1):
            sim = float(np.dot(directions[i], directions[i + 1]))
            curvatures.append(1.0 - sim)

        raw_curvature = float(np.mean(curvatures))

        # Shaped reward: moderate curvature (0.3–0.5) is ideal for structured reasoning
        # Too low = monotone, too high = chaotic
        target = 0.4
        score = float(np.exp(-((raw_curvature - target) / 0.3) ** 2))
        return score

    # ── Keyword coverage (TF-IDF) ─────────────────────────────────────

    @staticmethod
    def _extract_keywords_tfidf(
        prompt: str, responses: Sequence[str], top_k: int = 10
    ) -> List[str]:
        """Extract top-k keywords from the prompt using TF-IDF against response corpus."""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
        except ImportError:
            return []

        if not prompt.strip():
            return []

        corpus = [prompt] + list(responses)
        try:
            vectorizer = TfidfVectorizer(
                max_features=5000,
                stop_words="english",
                ngram_range=(1, 2),
            )
            tfidf_matrix = vectorizer.fit_transform(corpus)
            feature_names = vectorizer.get_feature_names_out()
            prompt_scores = tfidf_matrix[0].toarray().flatten()
            top_indices = prompt_scores.argsort()[-top_k:][::-1]
            keywords = [
                feature_names[i] for i in top_indices if prompt_scores[i] > 0
            ]
            return keywords
        except Exception:
            return []

    @staticmethod
    def _keyword_coverage(response: str, keywords: List[str]) -> float:
        """Fraction of prompt keywords found in the response."""
        if not keywords:
            return 0.5
        response_lower = response.lower()
        found = sum(1 for kw in keywords if kw.lower() in response_lower)
        return found / len(keywords)

    # ── Answer shape checks ────────────────────────────────────────────

    @staticmethod
    def _answer_shape(response: str, prompt: str) -> float:
        """Score structural compliance: does the response shape match what the prompt asks for?

        Checks for:
        - Lists (numbered/bulleted) when prompt asks for enumeration
        - Code blocks when prompt asks for code
        - Step-by-step when prompt asks for instructions
        """
        score = 0.5  # Neutral baseline
        prompt_lower = prompt.lower()
        response_lower = response.lower()

        # Does prompt ask for a list?
        list_cues = ["list", "enumerate", "name ", "what are", "give me"]
        if any(cue in prompt_lower for cue in list_cues):
            has_list = bool(
                re.search(r"(?:^|\n)\s*(?:\d+[.)]\s|[-•*]\s)", response)
            )
            score += 0.25 if has_list else -0.15

        # Does prompt ask for code?
        code_cues = ["code", "implement", "write a function", "write a program", "script"]
        if any(cue in prompt_lower for cue in code_cues):
            has_code = "```" in response or bool(
                re.search(r"(?:def |class |import |function |const |let |var )", response)
            )
            score += 0.25 if has_code else -0.15

        # Does prompt ask for steps/instructions?
        step_cues = ["how to", "step by step", "steps to", "instructions", "explain how"]
        if any(cue in prompt_lower for cue in step_cues):
            has_steps = bool(
                re.search(r"(?:^|\n)\s*(?:\d+[.)]\s|step\s+\d|first|second|third)", response, re.IGNORECASE)
            )
            score += 0.25 if has_steps else -0.10

        # Check for comparison/contrast when asked
        compare_cues = ["compare", "contrast", "difference", "versus", "vs"]
        if any(cue in prompt_lower for cue in compare_cues):
            has_comparison = any(
                m in response_lower for m in ("however", "whereas", "on the other hand", "in contrast", "while")
            )
            score += 0.2 if has_comparison else -0.05

        return float(np.clip(score, 0.0, 1.0))

    # ── Ported from ConsensusEngine (simplified for reuse) ─────────────

    _FILLER_WORDS = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "with", "by", "from", "up", "about", "into", "through", "during",
        "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
        "do", "does", "did", "will", "would", "could", "should", "may", "might",
        "can", "this", "that", "these", "those", "it", "its", "very", "more",
        "some", "any", "all", "each", "every", "both", "few", "many", "much",
        "such", "only", "own", "other", "than", "then", "them", "there",
        "various", "important", "significant", "relevant", "considerations",
        "aspects", "factors", "elements", "components", "matters", "issues",
    }

    @classmethod
    def _information_density(cls, response: str) -> float:
        """Ratio of content words to total words."""
        tokens = response.lower().split()
        if not tokens:
            return 0.0
        content = [w for w in tokens if len(w) > 3 and w not in cls._FILLER_WORDS]
        return len(content) / len(tokens)

    @staticmethod
    def _specificity(response: str) -> float:
        """Presence of numbers, proper nouns, technical terms, structure."""
        score = 0.0
        if re.search(r"\d+", response):
            score += 0.3
        sentences = re.split(r"[.!?]+", response)
        proper_nouns = 0
        for sent in sentences:
            words = sent.split()
            for word in words[1:]:
                if word and word[0].isupper():
                    proper_nouns += 1
        if proper_nouns > 0:
            score += min(0.3, proper_nouns * 0.1)
        words = response.split()
        long_words = [w for w in words if len(w) > 8]
        if words:
            score += min(0.2, (len(long_words) / len(words)) * 2)
        if re.search(r"[;:]", response):
            score += 0.2
        return min(1.0, score)

    @staticmethod
    def _heuristic_coherence(response: str) -> float:
        """Fallback coherence based on sentence structure (no embeddings)."""
        score = 0.5
        sentences = [s.strip() for s in re.split(r"[.!?]+", response) if s.strip()]
        if not sentences:
            return 0.0
        avg_len = sum(len(s.split()) for s in sentences) / len(sentences)
        if 5 <= avg_len <= 30:
            score += 0.3
        elif avg_len < 3:
            score -= 0.2
        if len(sentences) > 1:
            score += 0.2
        return max(0.0, min(1.0, score))
