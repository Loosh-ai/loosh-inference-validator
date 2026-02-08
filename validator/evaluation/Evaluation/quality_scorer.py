"""
Enhanced Evaluation Quality Scorer (Tier 4)

Provides multi-granularity quality assessment for miner responses,
replacing the simple heuristic-based metrics in ConsensusEngine with
embedding-aware signals.

Signals produced
----------------
1. **Multi-granularity relevance** — sentence-level + full-response cosine
   similarity to the prompt.
2. **Embedding-chain coherence** — average cosine similarity between
   consecutive sentence embeddings, measuring topic-flow consistency.
3. **Prompt coverage completeness** — fraction of prompt semantic
   components addressed by the response.
4. **Reasoning complexity** — depth of causal/argumentative structure
   (discourse markers, multi-step logic).

All signals are normalised to [0, 1] and can be weighted by the
``ConsensusConfig`` quality weights already used by ``ConsensusEngine``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

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

# ── Sentence splitter (simple but effective) ───────────────────────────
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


@dataclass
class QualityBreakdown:
    """Per-response quality breakdown returned by the scorer."""

    # Individual signal scores (all in [0, 1])
    relevance: float = 0.0
    sentence_relevance: float = 0.0
    full_response_relevance: float = 0.0
    coherence: float = 0.0
    coverage: float = 0.0
    complexity: float = 0.0
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
    """

    def __init__(
        self,
        sentence_relevance_weight: float = 0.5,
        min_sentence_words: int = 3,
        coverage_threshold: float = 0.45,
        coherence_embedding_chain: bool = True,
        complexity_enabled: bool = True,
        coverage_enabled: bool = True,
    ):
        self.sentence_relevance_weight = sentence_relevance_weight
        self.min_sentence_words = min_sentence_words
        self.coverage_threshold = coverage_threshold
        self.coherence_embedding_chain = coherence_embedding_chain
        self.complexity_enabled = complexity_enabled
        self.coverage_enabled = coverage_enabled

    # ── Public API ─────────────────────────────────────────────────────

    def score_batch(
        self,
        responses: Sequence[str],
        response_embeddings: np.ndarray,
        prompt: str,
        prompt_embedding: np.ndarray,
        embed_fn,
        *,
        quality_weights: Optional[Dict[str, float]] = None,
    ) -> Tuple[np.ndarray, List[QualityBreakdown]]:
        """Score a batch of responses.

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
            ``embed_fn(texts: List[str]) -> ndarray`` — a thin wrapper
            around ``SentenceTransformer.encode`` to produce embeddings
            for arbitrary texts.  The scorer needs this for sentence-level
            and prompt-component embeddings.
        quality_weights : dict, optional
            Override composite weight map.  Keys:
            ``relevance``, ``coherence``, ``coverage``, ``complexity``,
            ``density``, ``specificity``.  Missing keys default to the
            ConsensusConfig defaults (0.4/0.2/0.2/0.2 across the original
            four dimensions).

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

        # Default weight map — extends the original 4-way split with the
        # two new signals (coverage, complexity) carved out of coherence.
        weights = quality_weights or {
            "relevance": 0.30,
            "coherence": 0.15,
            "coverage": 0.15,
            "complexity": 0.10,
            "density": 0.15,
            "specificity": 0.15,
        }

        # ── Decompose prompt into semantic components ──────────────
        prompt_components = self._split_prompt_components(prompt)
        prompt_component_embeddings: Optional[np.ndarray] = None
        if self.coverage_enabled and prompt_components:
            prompt_component_embeddings = embed_fn(prompt_components)

        # ── Per-response scoring ───────────────────────────────────
        breakdowns: List[QualityBreakdown] = []
        scores = np.zeros(n, dtype=np.float64)

        for i, response in enumerate(responses):
            bd = self._score_single(
                response=response,
                response_embedding=response_embeddings[i],
                prompt=prompt,
                prompt_embedding=prompt_embedding,
                prompt_components=prompt_components,
                prompt_component_embeddings=prompt_component_embeddings,
                embed_fn=embed_fn,
            )

            # Weighted composite
            composite = (
                weights.get("relevance", 0.30) * bd.relevance
                + weights.get("coherence", 0.15) * bd.coherence
                + weights.get("coverage", 0.15) * bd.coverage
                + weights.get("complexity", 0.10) * bd.complexity
                + weights.get("density", 0.15) * bd.density
                + weights.get("specificity", 0.15) * bd.specificity
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
        embed_fn,
    ) -> QualityBreakdown:
        """Compute all quality signals for a single response."""

        bd = QualityBreakdown()

        # Split response into sentences
        sentences = self._split_sentences(response)
        bd.sentence_count = len(sentences)

        # ── 1. Full-response relevance (baseline) ─────────────────
        bd.full_response_relevance = float(
            cosine_similarity(
                prompt_embedding.reshape(1, -1),
                response_embedding.reshape(1, -1),
            )[0, 0]
        )

        # ── 2. Sentence-level relevance ───────────────────────────
        if sentences and len(sentences) >= 2:
            sentence_embeddings = embed_fn(sentences)
            sentence_sims = cosine_similarity(
                prompt_embedding.reshape(1, -1),
                sentence_embeddings,
            )[0]
            # Use mean of top-k sentence similarities (avoids one great
            # sentence dominating when most are irrelevant).
            k = max(1, len(sentences) // 2)
            top_k_sims = np.sort(sentence_sims)[-k:]
            bd.sentence_relevance = float(np.mean(top_k_sims))
        else:
            bd.sentence_relevance = bd.full_response_relevance
            sentence_embeddings = None

        # Blend full-response and sentence-level relevance
        w = self.sentence_relevance_weight
        bd.relevance = w * bd.sentence_relevance + (1 - w) * bd.full_response_relevance

        # ── 3. Embedding-chain coherence ──────────────────────────
        if self.coherence_embedding_chain and sentence_embeddings is not None and len(sentences) >= 2:
            bd.coherence = self._embedding_chain_coherence(sentence_embeddings)
        else:
            # Fallback: heuristic sentence-length coherence (same as
            # ConsensusEngine._measure_coherence but lighter)
            bd.coherence = self._heuristic_coherence(response)

        # ── 4. Prompt coverage completeness ───────────────────────
        if (
            self.coverage_enabled
            and prompt_component_embeddings is not None
            and len(prompt_components) > 0
        ):
            bd.coverage, bd.covered_components, bd.total_components = (
                self._prompt_coverage(
                    response_embedding=response_embedding,
                    sentences=sentences,
                    embed_fn=embed_fn,
                    prompt_components=prompt_components,
                    prompt_component_embeddings=prompt_component_embeddings,
                )
            )
        else:
            bd.coverage = bd.relevance  # Fallback to relevance as proxy

        # ── 5. Reasoning complexity ───────────────────────────────
        if self.complexity_enabled:
            bd.complexity, bd.discourse_marker_count = self._reasoning_complexity(
                response, sentences
            )
        else:
            bd.complexity = 0.5  # Neutral default

        # ── 6. Information density (ported from ConsensusEngine) ──
        bd.density = self._information_density(response)

        # ── 7. Specificity (ported from ConsensusEngine) ──────────
        bd.specificity = self._specificity(response)

        return bd

    # ── Signal implementations ─────────────────────────────────────────

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        """Split text into sentences, filtering very short fragments."""
        raw = _SENTENCE_SPLIT_RE.split(text.strip())
        return [s.strip() for s in raw if len(s.split()) >= 3]

    @staticmethod
    def _split_prompt_components(prompt: str) -> List[str]:
        """
        Decompose a prompt into semantic components.

        Heuristics:
        - Split on question marks (sub-questions)
        - Split on numbered/bulleted items
        - Split on semicolons or "and" conjunctions connecting clauses
        - Fall back to the full prompt as a single component
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

        # 3. Try splitting on semicolons
        semi = [s.strip() for s in prompt.split(";") if s.strip() and len(s.split()) >= 3]
        if len(semi) >= 2:
            return semi

        # 4. Fall back to whole prompt as single component
        if prompt.strip():
            return [prompt.strip()]
        return []

    @staticmethod
    def _embedding_chain_coherence(sentence_embeddings: np.ndarray) -> float:
        """
        Measure coherence as average cosine similarity between
        consecutive sentence embeddings.

        High coherence (≈1.0) means smooth topic flow.
        Low coherence (≈0.0) means abrupt topic jumps.
        """
        if len(sentence_embeddings) < 2:
            return 1.0  # Single sentence → perfect coherence

        # Compute similarity between consecutive pairs
        sims = []
        for i in range(len(sentence_embeddings) - 1):
            sim = cosine_similarity(
                sentence_embeddings[i : i + 1],
                sentence_embeddings[i + 1 : i + 2],
            )[0, 0]
            sims.append(sim)

        # Mean consecutive similarity, clipped to [0, 1]
        return float(np.clip(np.mean(sims), 0.0, 1.0))

    def _prompt_coverage(
        self,
        response_embedding: np.ndarray,
        sentences: List[str],
        embed_fn,
        prompt_components: List[str],
        prompt_component_embeddings: np.ndarray,
    ) -> Tuple[float, int, int]:
        """
        Compute prompt coverage completeness.

        For each prompt component, find the maximum cosine similarity to
        any response sentence (or the full response if < 2 sentences).
        A component is "covered" if max_sim ≥ ``coverage_threshold``.

        Returns (coverage_score, covered_count, total_count).
        """
        total = len(prompt_components)
        if total == 0:
            return 1.0, 0, 0

        # Build response-side embeddings matrix
        if sentences and len(sentences) >= 2:
            sentence_embeddings = embed_fn(sentences)
            # Stack full response embedding with sentence embeddings
            response_side = np.vstack(
                [response_embedding.reshape(1, -1), sentence_embeddings]
            )
        else:
            response_side = response_embedding.reshape(1, -1)

        # Compute max similarity for each prompt component
        sim_matrix = cosine_similarity(prompt_component_embeddings, response_side)
        max_sims = sim_matrix.max(axis=1)  # shape (total,)

        covered = int(np.sum(max_sims >= self.coverage_threshold))
        coverage_score = float(covered / total)

        return coverage_score, covered, total

    @staticmethod
    def _reasoning_complexity(
        response: str, sentences: List[str]
    ) -> Tuple[float, int]:
        """
        Score reasoning complexity based on discourse markers.

        Looks for causal, contrast, elaboration, and sequence markers.
        More markers (relative to response length) → higher complexity.

        Returns (complexity_score, marker_count).
        """
        markers_found = _MARKER_RE.findall(response.lower())
        marker_count = len(markers_found)

        if not sentences:
            return 0.0, 0

        # Normalise by sentence count to avoid penalising shorter responses
        # that are still well-structured.
        markers_per_sentence = marker_count / len(sentences)

        # Map to [0, 1] using a soft sigmoid-like curve:
        # 0 markers → 0.0
        # 1 marker/sentence → ~0.73
        # 2+ markers/sentence → ~0.95+
        # Formula: 1 - exp(-1.3 * x)
        complexity = 1.0 - np.exp(-1.3 * markers_per_sentence)

        # Bonus for variety (using multiple marker types)
        marker_types_present = 0
        text_lower = response.lower()
        if any(m in text_lower for m in _CAUSAL_MARKERS):
            marker_types_present += 1
        if any(m in text_lower for m in _CONTRAST_MARKERS):
            marker_types_present += 1
        if any(m in text_lower for m in _ELABORATION_MARKERS):
            marker_types_present += 1
        if any(m in text_lower for m in _SEQUENCE_MARKERS):
            marker_types_present += 1

        # Variety bonus: up to 0.15 for using 3+ types
        variety_bonus = min(0.15, (marker_types_present - 1) * 0.05) if marker_types_present > 1 else 0.0

        return float(np.clip(complexity + variety_bonus, 0.0, 1.0)), marker_count

    # ── Ported from ConsensusEngine (simplified for reuse) ─────────────

    # Common filler words
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
        """
        Fallback coherence based on sentence structure (no embeddings).
        Same logic as ConsensusEngine._measure_coherence.
        """
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
