from typing import List, Optional, Dict, Literal
from dataclasses import dataclass
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import LocalOutlierFactor


@dataclass
class ConsensusConfig:
    use_clustering: bool = False
    use_weighted_scoring: bool = False
    use_polarity_clustering: bool = False
    use_outlier_detection: bool = False
    apply_quality_filter: bool = False
    quality_sensitivity: float = 0.5  # Lower = stricter
    generate_heatmap: bool = False
    heatmap_path: str = "./output/consensus_similarity_heatmap.png"
    lambda_factor: float = 1.0
    threshold_min: float = 0.7
    polarity_agreement_min: float = 0.8


@dataclass
class ConsensusResult:
    original_prompt: str
    in_consensus: Dict[str, str]
    out_of_consensus: Dict[str, str]
    similarity_score: float
    weighted_score: Optional[float]
    polarity_agreement: Optional[float]
    heatmap_path: Optional[str]
    consensus_achieved: bool
    quality_plot_path: Optional[str] = None
    consensus_narrative: Optional[str] = None
    miner_scores: Optional[Dict[str, float]] = None



class ConsensusEngine:
    def __init__(
        self,
        original_prompt: str,
        responses: List[str],
        embeddings: List[np.ndarray],
        confidences: Optional[List[float]] = None,
        polarities: Optional[List[Literal["affirmative", "negative"]]] = None
    ):
        self.original_prompt = original_prompt
        self.all_responses = responses
        self.all_labels = [f"R{i+1}" for i in range(len(responses))]
        self.all_embeddings = np.array(embeddings)
        self.all_confidences = np.array(confidences) if confidences is not None else np.ones(len(responses))
        self.embeddings = np.array(embeddings)
        self.confidences = np.array(confidences) if confidences is not None else np.ones(len(responses))
        self.polarities = polarities
        self.responses = responses.copy()
        self.labels = self.all_labels.copy()

    def _compute_pairwise_similarity(self) -> np.ndarray:
        return cosine_similarity(self.embeddings)

    def _apply_mask(self, mask: np.ndarray):
        self.embeddings = self.embeddings[mask]
        self.confidences = self.confidences[mask]
        self.responses = [r for i, r in enumerate(self.responses) if mask[i]]
        self.labels = [l for i, l in enumerate(self.labels) if mask[i]]
        if self.polarities:
            self.polarities = [p for i, p in enumerate(self.polarities) if mask[i]]

    def _apply_outlier_filter(self, sim_matrix: np.ndarray) -> np.ndarray:
        lof = LocalOutlierFactor(n_neighbors=2)
        mask = lof.fit_predict(self.embeddings) == 1
        self._apply_mask(mask)
        return cosine_similarity(self.embeddings)

    def _apply_clustering_filter(self, sim_matrix: np.ndarray) -> np.ndarray:
        distance_matrix = 1 - sim_matrix
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=0.5,
            affinity='precomputed',
            linkage='average'
        )
        labels = clustering.fit_predict(distance_matrix)
        dominant_label = np.argmax(np.bincount(labels))
        mask = labels == dominant_label
        self._apply_mask(mask)
        return cosine_similarity(self.embeddings)

    def _apply_basic_scoring(self, sim_matrix: np.ndarray) -> tuple[float, float]:
        if len(self.embeddings) < 2:
            return 0.0, 0.0
        i, j = np.triu_indices(len(sim_matrix), k=1)
        sims = sim_matrix[i, j]
        return np.mean(sims), np.std(sims)

    def _apply_weighted_scoring(self, sim_matrix: np.ndarray) -> float:
        weights = np.outer(self.confidences, self.confidences)
        np.fill_diagonal(weights, 0)
        score = np.sum(sim_matrix * weights)
        norm = np.sum(weights)
        return score / norm if norm > 0 else 0.0

    def _apply_polarity_clustering(self) -> float:
        if not self.polarities:
            return 0.0
        counts = {}
        for p in self.polarities:
            counts[p] = counts.get(p, 0) + 1
        return max(counts.values()) / sum(counts.values())

    def _apply_quality_filter(self, sensitivity: float):
        lengths = [len(r.split()) for r in self.responses]
        avg_len = np.mean(lengths)
        min_len = avg_len * sensitivity
        mask = np.array([len(r.split()) >= min_len for r in self.responses])
        self._apply_mask(mask)

    def _generate_quality_plot(self, path: str):
        lengths = [len(r.split()) for r in self.responses]
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.figure(figsize=(8, 6))
        sns.histplot(lengths, bins=10, kde=True)
        plt.title("Response Length Distribution")
        plt.xlabel("Word Count")
        plt.ylabel("Number of Responses")
        plt.tight_layout()
        quality_path = path.replace(".png", "_quality.png")
        plt.savefig(quality_path)
        plt.close()
        return quality_path


    def _generate_heatmap(self, sim_matrix: np.ndarray, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            sim_matrix,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            xticklabels=self.labels,
            yticklabels=self.labels
        )
        plt.title("Consensus Similarity Heatmap")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(path)
        plt.close()

    def _compute_individual_scores(
        self, 
        sim_matrix: np.ndarray, 
        consensus_labels: set[str],
        use_confidence_weighting: bool = True
    ) -> Dict[str, float]:
        """
        Compute individual scores for each response.
        
        Scoring factors:
        1. Average similarity to all other responses (consensus alignment)
        2. Confidence score (if available and enabled)
        3. Quality score (normalized response length)
        4. Consensus membership bonus
        
        Args:
            sim_matrix: Pairwise similarity matrix for filtered responses
            consensus_labels: Set of labels that are in consensus
            use_confidence_weighting: Whether to weight by confidence scores
            
        Returns:
            Dictionary mapping response labels to their individual scores
        """
        scores = {}
        
        if len(self.embeddings) == 0:
            return scores
        
        # Compute average similarity to other responses for each response
        # Exclude self-similarity (diagonal)
        avg_similarities = []
        for i in range(len(sim_matrix)):
            # Get similarities to all other responses (exclude self)
            other_sims = np.delete(sim_matrix[i], i)
            avg_sim = np.mean(other_sims) if len(other_sims) > 0 else 0.0
            avg_similarities.append(avg_sim)
        
        avg_similarities = np.array(avg_similarities)
        
        # Compute quality scores based on response length
        lengths = np.array([len(r.split()) for r in self.responses])
        if np.max(lengths) > 0:
            # Normalize lengths to [0, 1] range
            quality_scores = lengths / np.max(lengths)
        else:
            quality_scores = np.ones(len(self.responses))
        
        # Compute composite scores for each response
        for i, label in enumerate(self.labels):
            # Base score: average similarity (how well aligned with consensus)
            base_score = avg_similarities[i]
            
            # Quality component (normalized length)
            quality_component = quality_scores[i] * 0.2  # 20% weight
            
            # Confidence component (if available and enabled)
            confidence_component = 0.0
            if use_confidence_weighting and len(self.confidences) > i:
                confidence_component = self.confidences[i] * 0.2  # 20% weight
            
            # Consensus membership bonus
            consensus_bonus = 0.0
            if label in consensus_labels:
                consensus_bonus = 0.1  # 10% bonus for being in consensus
            
            # Composite score: base similarity (50%) + quality (20%) + confidence (20%) + consensus bonus (10%)
            composite_score = (
                base_score * 0.5 +
                quality_component +
                confidence_component +
                consensus_bonus
            )
            
            scores[label] = float(composite_score)
        
        # Also score responses that were filtered out (out of consensus)
        for label in self.all_labels:
            if label not in scores:
                # Out-of-consensus responses get a lower score
                # Compute similarity to consensus cluster if possible
                if len(self.embeddings) > 0:
                    # Find the original index of this response
                    orig_idx = self.all_labels.index(label)
                    orig_embedding = self.all_embeddings[orig_idx:orig_idx+1]  # Keep 2D shape
                    # Compute average similarity to consensus cluster
                    consensus_sims = cosine_similarity(orig_embedding, self.embeddings)[0]
                    avg_consensus_sim = np.mean(consensus_sims) if len(consensus_sims) > 0 else 0.0
                    
                    # Get quality and confidence for this response
                    orig_length = len(self.all_responses[orig_idx].split())
                    max_length = max([len(r.split()) for r in self.all_responses])
                    quality_score = (orig_length / max_length * 0.2) if max_length > 0 else 0.0
                    
                    confidence_score = 0.0
                    if use_confidence_weighting and orig_idx < len(self.all_confidences):
                        confidence_score = self.all_confidences[orig_idx] * 0.2
                    
                    # Composite score with penalty for being out of consensus
                    composite_score = (
                        avg_consensus_sim * 0.3 +  # Reduced weight (30% vs 50%)
                        quality_score +
                        confidence_score
                    )
                    scores[label] = float(composite_score)
                else:
                    scores[label] = 0.0
        
        return scores

    def evaluate_consensus(self, config: ConsensusConfig) -> ConsensusResult:
        sim_matrix = self._compute_pairwise_similarity()

        if config.use_outlier_detection:
            sim_matrix = self._apply_outlier_filter(sim_matrix)

        if config.use_clustering:
            sim_matrix = self._apply_clustering_filter(sim_matrix)

        if config.apply_quality_filter:
            self._apply_quality_filter(config.quality_sensitivity)
            sim_matrix = self._compute_pairwise_similarity()

        mu, sigma = self._apply_basic_scoring(sim_matrix)
        theta = mu + config.lambda_factor * sigma

        weighted_score = self._apply_weighted_scoring(sim_matrix) if config.use_weighted_scoring else None
        polarity_score = self._apply_polarity_clustering() if config.use_polarity_clustering else None
        consensus = theta > config.threshold_min or (
            polarity_score and polarity_score > config.polarity_agreement_min
        )

        quality_path = None
        if config.generate_heatmap:
            self._generate_heatmap(sim_matrix, config.heatmap_path)
            if config.apply_quality_filter:
                quality_path = self._generate_quality_plot(config.heatmap_path)

        in_consensus = {l: r for l, r in zip(self.labels, self.responses)}
        out_of_consensus = {
            l: r for l, r in zip(self.all_labels, self.all_responses) if l not in in_consensus
        }

        # Compute individual response scores
        consensus_label_set = set(in_consensus.keys())
        individual_scores = self._compute_individual_scores(
            sim_matrix=sim_matrix,
            consensus_labels=consensus_label_set,
            use_confidence_weighting=config.use_weighted_scoring
        )

        return ConsensusResult(
            original_prompt=self.original_prompt,
            in_consensus=in_consensus,
            out_of_consensus=out_of_consensus,
            similarity_score=theta,
            weighted_score=weighted_score,
            polarity_agreement=polarity_score,
            heatmap_path=config.heatmap_path if config.generate_heatmap else None,
            consensus_achieved=consensus,
            quality_plot_path=quality_path,
            consensus_narrative=None,
            miner_scores=individual_scores
        )
