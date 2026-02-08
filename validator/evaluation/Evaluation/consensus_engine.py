from typing import List, Optional, Dict, Literal
from dataclasses import dataclass
from datetime import datetime
import numpy as np
import os
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import LocalOutlierFactor

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


@dataclass
class ConsensusConfig:
    use_clustering: bool = False
    use_weighted_scoring: bool = False
    use_polarity_clustering: bool = False
    use_outlier_detection: bool = False
    apply_quality_filter: bool = False
    quality_sensitivity: float = 0.5  # Lower = stricter (DEPRECATED - use semantic quality instead)
    generate_heatmap: bool = False
    generate_quality_plot: bool = False  # Enable quality plot generation
    heatmap_path: str = "./output/consensus_similarity_heatmap.jpg"
    lambda_factor: float = 1.0
    threshold_min: float = 0.7
    polarity_agreement_min: float = 0.8
    challenge_id: Optional[str] = None  # Challenge ID (UUID) for heatmap title
    correlation_id: Optional[str] = None  # Correlation ID for heatmap title
    
    # Semantic Quality Assessment (CRITICAL for garbage consensus prevention)
    enable_semantic_quality: bool = True  # Enable semantic quality assessment
    quality_threshold: float = 0.35  # Minimum quality score to participate in consensus
    quality_prompt_relevance_weight: float = 0.4  # Weight for prompt relevance
    quality_density_weight: float = 0.2  # Weight for information density
    quality_specificity_weight: float = 0.2  # Weight for specificity
    quality_coherence_weight: float = 0.2  # Weight for coherence
    
    # Smart Outlier Detection (HIGH priority)
    enable_smart_outlier_detection: bool = True  # Use quality-aware outlier detection
    outlier_quality_delta: float = 0.15  # How much better outliers must be to reverse decision
    
    # Diversity Bonus (HIGH priority)
    enable_diversity_bonus: bool = True  # Reward unique high-quality responses
    max_diversity_bonus: float = 0.15  # Maximum bonus for unique responses
    
    # Garbage Detection Alerts
    enable_garbage_alerts: bool = True  # Log warnings for low-quality consensus
    garbage_cluster_threshold: float = 0.4  # Alert if cluster avg quality < this


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
        polarities: Optional[List[Literal["affirmative", "negative"]]] = None,
        miner_ids: Optional[List[int]] = None,
        prompt_embedding: Optional[np.ndarray] = None
    ):
        self.original_prompt = original_prompt
        self.all_responses = responses
        # Create labels with miner UIDs if available, otherwise use R1, R2, etc.
        if miner_ids and len(miner_ids) == len(responses):
            self.all_labels = [str(uid) for uid in miner_ids]
        else:
            self.all_labels = [f"R{i+1}" for i in range(len(responses))]
        self.all_embeddings = np.array(embeddings)
        self.all_confidences = np.array(confidences) if confidences is not None else np.ones(len(responses))
        self.embeddings = np.array(embeddings)
        self.confidences = np.array(confidences) if confidences is not None else np.ones(len(responses))
        self.polarities = polarities
        self.responses = responses.copy()
        self.labels = self.all_labels.copy()
        self.prompt_embedding = prompt_embedding
        
        # Store quality scores for later use
        self.quality_scores = None

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
        """
        DEPRECATED: Use _smart_outlier_detection() instead.
        This simple version removes outliers without considering quality.
        """
        lof = LocalOutlierFactor(n_neighbors=2)
        mask = lof.fit_predict(self.embeddings) == 1
        self._apply_mask(mask)
        return cosine_similarity(self.embeddings)
    
    def _smart_outlier_detection(self, config: ConsensusConfig) -> np.ndarray:
        """
        Smart outlier detection that considers quality scores.
        
        CRITICAL for garbage consensus prevention:
        - If outliers have HIGHER quality than cluster → keep outliers, remove cluster (garbage consensus)
        - If outliers have LOWER quality than cluster → remove outliers (normal case)
        
        This prevents the situation where a majority of similar garbage responses
        forms a cluster and high-quality unique responses are marked as outliers.
        """
        if len(self.embeddings) < 3:
            logger.debug("[OUTLIER] Skipping outlier detection - need at least 3 responses")
            return cosine_similarity(self.embeddings)
        
        # Identify outliers using LocalOutlierFactor
        lof = LocalOutlierFactor(n_neighbors=min(2, len(self.embeddings) - 1))
        outlier_predictions = lof.fit_predict(self.embeddings)
        outlier_mask = outlier_predictions == -1  # -1 = outlier
        inlier_mask = ~outlier_mask
        
        num_outliers = np.sum(outlier_mask)
        num_inliers = np.sum(inlier_mask)
        
        if num_outliers == 0:
            logger.debug("[OUTLIER] No outliers detected")
            return cosine_similarity(self.embeddings)
        
        # Check if we have quality scores
        if self.quality_scores is None or len(self.quality_scores) != len(self.embeddings):
            logger.warning(
                "[OUTLIER] Quality scores not available, falling back to simple outlier removal"
            )
            self._apply_mask(inlier_mask)
            return cosine_similarity(self.embeddings)
        
        # Calculate average quality of outliers vs inliers
        outlier_quality = self.quality_scores[outlier_mask].mean() if num_outliers > 0 else 0.0
        inlier_quality = self.quality_scores[inlier_mask].mean() if num_inliers > 0 else 0.0
        quality_delta = outlier_quality - inlier_quality
        
        logger.info(
            f"[OUTLIER] Detected {num_outliers} outliers, {num_inliers} inliers. "
            f"Outlier quality: {outlier_quality:.3f}, Inlier quality: {inlier_quality:.3f}, "
            f"Delta: {quality_delta:+.3f}"
        )
        
        # Decision logic: Are outliers significantly better?
        if quality_delta > config.outlier_quality_delta:
            # GARBAGE CONSENSUS DETECTED!
            # Outliers are higher quality - keep them, remove the garbage cluster
            logger.warning(
                f"[OUTLIER] ⚠️  GARBAGE CONSENSUS DETECTED! Outliers have significantly "
                f"higher quality ({outlier_quality:.3f}) than cluster ({inlier_quality:.3f}). "
                f"Keeping {num_outliers} outliers, removing {num_inliers} low-quality cluster members."
            )
            
            # Log which responses are being kept vs removed
            for i, label in enumerate(self.labels):
                if outlier_mask[i]:
                    logger.info(
                        f"[OUTLIER] ✓ KEEPING high-quality outlier {label} "
                        f"(quality={self.quality_scores[i]:.3f})"
                    )
                else:
                    logger.warning(
                        f"[OUTLIER] ✗ REMOVING low-quality cluster member {label} "
                        f"(quality={self.quality_scores[i]:.3f})"
                    )
            
            self._apply_mask(outlier_mask)  # Keep outliers, remove inliers
        else:
            # Normal case: remove outliers
            logger.info(
                f"[OUTLIER] Removing {num_outliers} outliers (quality not significantly better)"
            )
            for i, label in enumerate(self.labels):
                if outlier_mask[i]:
                    logger.debug(
                        f"[OUTLIER] Removing outlier {label} (quality={self.quality_scores[i]:.3f})"
                    )
            self._apply_mask(inlier_mask)  # Keep inliers, remove outliers
        
        return cosine_similarity(self.embeddings)

    def _apply_clustering_filter(self, sim_matrix: np.ndarray) -> np.ndarray:
        distance_matrix = 1 - sim_matrix
        # In scikit-learn 1.2+, 'affinity' was replaced with 'metric'
        # For precomputed distance matrices, use metric='precomputed'
        try:
            # Try new API (scikit-learn >= 1.2)
            clustering = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=0.5,
                metric='precomputed',
                linkage='average'
            )
        except TypeError:
            # Fallback to old API (scikit-learn < 1.2)
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
    
    def _apply_quality_weighted_scoring(self, sim_matrix: np.ndarray, config: ConsensusConfig) -> tuple[float, float]:
        """
        Quality-weighted similarity scoring.
        High-quality responses have more influence on consensus determination.
        
        This prevents garbage consensus by ensuring low-quality similar responses
        don't dominate the consensus calculation.
        """
        if len(self.embeddings) < 2:
            return 0.0, 0.0
        
        # If no quality scores, fall back to basic scoring
        if self.quality_scores is None or len(self.quality_scores) != len(self.embeddings):
            logger.debug("[SCORING] No quality scores available, using basic scoring")
            return self._apply_basic_scoring(sim_matrix)
        
        # Weight each similarity by the product of both responses' quality scores
        # This ensures both responses in a pair must be high quality to contribute strongly
        quality_weights = np.outer(self.quality_scores, self.quality_scores)
        
        # Get upper triangle indices (excluding diagonal)
        i, j = np.triu_indices(len(sim_matrix), k=1)
        
        # Extract similarities and their quality weights
        sims = sim_matrix[i, j]
        weights = quality_weights[i, j]
        
        # Compute weighted mean and std
        if np.sum(weights) > 0:
            weighted_mean = np.average(sims, weights=weights)
            # Weighted standard deviation
            weighted_variance = np.average((sims - weighted_mean)**2, weights=weights)
            weighted_std = np.sqrt(weighted_variance)
        else:
            weighted_mean = 0.0
            weighted_std = 0.0
        
        logger.debug(
            f"[SCORING] Quality-weighted: mean={weighted_mean:.3f}, std={weighted_std:.3f} "
            f"(unweighted: mean={np.mean(sims):.3f}, std={np.std(sims):.3f})"
        )
        
        return weighted_mean, weighted_std

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
        """
        DEPRECATED: Simple word-count based quality filter.
        Use _assess_semantic_quality() instead for garbage consensus prevention.
        """
        lengths = [len(r.split()) for r in self.responses]
        avg_len = np.mean(lengths)
        min_len = avg_len * sensitivity
        mask = np.array([len(r.split()) >= min_len for r in self.responses])
        self._apply_mask(mask)
    
    # ========================================================================
    # SEMANTIC QUALITY ASSESSMENT (CRITICAL for garbage consensus prevention)
    # ========================================================================
    
    # Common filler words that don't add semantic value
    FILLER_WORDS = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during',
        'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
        'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might',
        'can', 'this', 'that', 'these', 'those', 'it', 'its', 'very', 'more',
        'some', 'any', 'all', 'each', 'every', 'both', 'few', 'many', 'much',
        'such', 'only', 'own', 'other', 'than', 'then', 'them', 'there',
        'various', 'important', 'significant', 'relevant', 'considerations',
        'aspects', 'factors', 'elements', 'components', 'matters', 'issues'
    }
    
    def _measure_information_density(self, response: str) -> float:
        """
        Measure information density (ratio of content words to filler).
        Higher density means more meaningful content vs generic filler.
        """
        tokens = response.lower().split()
        if not tokens:
            return 0.0
        
        # Count content words (longer than 3 chars, not in filler set)
        content_words = [
            w for w in tokens 
            if len(w) > 3 and w not in self.FILLER_WORDS
        ]
        
        density = len(content_words) / len(tokens)
        return density
    
    def _measure_specificity(self, response: str) -> float:
        """
        Measure specificity based on presence of:
        - Numbers (indicate specific data/quantities)
        - Proper nouns (capitalized words mid-sentence)
        - Technical terms (longer words)
        - Punctuation indicating structure (colons, semicolons)
        """
        score = 0.0
        
        # Check for numbers (strong indicator of specificity)
        has_numbers = bool(re.search(r'\d+', response))
        if has_numbers:
            score += 0.3
        
        # Check for proper nouns (capitalized words not at sentence start)
        # Split by periods/question marks to identify sentence boundaries
        sentences = re.split(r'[.!?]+', response)
        proper_nouns = 0
        for sent in sentences:
            words = sent.split()
            # Skip first word of each sentence
            for word in words[1:]:
                if word and word[0].isupper():
                    proper_nouns += 1
        
        if proper_nouns > 0:
            score += min(0.3, proper_nouns * 0.1)
        
        # Check for technical/longer words (>8 characters)
        words = response.split()
        long_words = [w for w in words if len(w) > 8]
        if len(words) > 0:
            long_word_ratio = len(long_words) / len(words)
            score += min(0.2, long_word_ratio * 2)
        
        # Check for structured punctuation
        has_structure = bool(re.search(r'[;:]', response))
        if has_structure:
            score += 0.2
        
        return min(1.0, score)
    
    def _measure_coherence(self, response: str) -> float:
        """
        Measure coherence based on:
        - Sentence structure (presence of complete sentences)
        - Average sentence length (not too short, not too long)
        - Punctuation patterns
        """
        score = 0.5  # Base score
        
        # Split into sentences
        sentences = [s.strip() for s in re.split(r'[.!?]+', response) if s.strip()]
        
        if not sentences:
            return 0.0
        
        # Check average sentence length (5-30 words is good)
        avg_sent_len = sum(len(s.split()) for s in sentences) / len(sentences)
        if 5 <= avg_sent_len <= 30:
            score += 0.3
        elif avg_sent_len < 3:  # Very short sentences = poor coherence
            score -= 0.2
        
        # Multiple sentences indicate structure
        if len(sentences) > 1:
            score += 0.2
        
        return max(0.0, min(1.0, score))
    
    def _assess_semantic_quality(self, config: ConsensusConfig) -> np.ndarray:
        """
        Assess semantic quality of each response relative to the prompt.
        Returns quality scores [0, 1] for each response.
        
        This is CRITICAL for preventing garbage consensus attacks.
        
        Quality factors:
        1. Prompt relevance: Does response address the question?
        2. Information density: Meaningful content vs filler words
        3. Specificity: Concrete details vs vague statements
        4. Coherence: Logical structure and completeness
        """
        quality_scores = []
        
        for i, response in enumerate(self.responses):
            # 1. Prompt-response relevance (cosine similarity)
            relevance = 0.5  # Default if no prompt embedding
            if self.prompt_embedding is not None and len(self.embeddings) > i:
                response_emb = self.embeddings[i:i+1]  # Keep 2D
                relevance = cosine_similarity([self.prompt_embedding], response_emb)[0][0]
            
            # 2. Information density
            density = self._measure_information_density(response)
            
            # 3. Specificity
            specificity = self._measure_specificity(response)
            
            # 4. Coherence
            coherence = self._measure_coherence(response)
            
            # Composite quality score (weighted average)
            quality = (
                relevance * config.quality_prompt_relevance_weight +
                density * config.quality_density_weight +
                specificity * config.quality_specificity_weight +
                coherence * config.quality_coherence_weight
            )
            
            quality_scores.append(quality)
        
        quality_array = np.array(quality_scores)
        
        # Log quality assessment summary
        if len(quality_scores) > 0:
            avg_quality = np.mean(quality_array)
            min_quality = np.min(quality_array)
            max_quality = np.max(quality_array)
            logger.debug(
                f"[QUALITY] Semantic assessment: avg={avg_quality:.3f}, "
                f"min={min_quality:.3f}, max={max_quality:.3f}, "
                f"threshold={config.quality_threshold:.3f}"
            )
        
        return quality_array
    
    def _apply_semantic_quality_filter(self, config: ConsensusConfig):
        """
        Filter responses based on semantic quality assessment.
        This runs BEFORE consensus to prevent garbage from forming consensus.
        """
        if len(self.responses) == 0:
            return
        
        quality_scores = self._assess_semantic_quality(config)
        
        # Store for later use in scoring
        self.quality_scores = quality_scores.copy()
        
        # Filter out low-quality responses
        mask = quality_scores >= config.quality_threshold
        filtered_count = len(quality_scores) - np.sum(mask)
        
        if filtered_count > 0:
            logger.info(
                f"[QUALITY] Filtered {filtered_count}/{len(quality_scores)} responses "
                f"below quality threshold {config.quality_threshold:.2f}"
            )
            
            # Log which responses were filtered
            for i, (label, score) in enumerate(zip(self.labels, quality_scores)):
                if not mask[i]:
                    logger.debug(
                        f"[QUALITY] Filtered response {label}: quality={score:.3f}, "
                        f"text_preview={self.responses[i][:100]}..."
                    )
        
        self._apply_mask(mask)
        
        # Update quality scores after filtering
        if len(quality_scores) > 0:
            self.quality_scores = quality_scores[mask]

    def _generate_quality_plot(self, path: str):
        lengths = [len(r.split()) for r in self.responses]
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.figure(figsize=(8, 6))
        sns.histplot(lengths, bins=10, kde=True)
        plt.title("Response Length Distribution")
        plt.xlabel("Word Count")
        plt.ylabel("Number of Responses")
        plt.tight_layout()
        # Generate quality plot path - keep as PNG for quality plots
        base_path = path.rsplit('.', 1)[0] if '.' in path else path
        quality_path = base_path + "_quality.png"
        plt.savefig(quality_path)
        plt.close()
        
        # Post-process with PIL to optimize PNG file size (lossless compression)
        try:
            from PIL import Image
            img = Image.open(quality_path)
            
            # Convert to RGB if needed (lossless if no transparency)
            if img.mode in ('RGBA', 'LA'):
                background = Image.new('RGB', img.size, (255, 255, 255))
                background.paste(img, mask=img.split()[-1])
                img = background
            elif img.mode == 'P':
                img = img.convert('RGB')
            elif img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Save with maximum lossless compression
            img.save(quality_path, 'PNG', optimize=True, compress_level=9)
        except (ImportError, Exception):
            pass
        
        return quality_path


    def _generate_heatmap(self, sim_matrix: np.ndarray, path: str, challenge_id: Optional[str] = None, correlation_id: Optional[str] = None) -> str:
        """Generate heatmap and return the actual file path."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Target width: 500px at 100 DPI = 5 inches
        # Maintain aspect ratio (original was 10x8, so ratio is 1.25:1)
        target_width_px = 500
        dpi = 100
        width_inches = target_width_px / dpi  # 5 inches
        height_inches = width_inches / 1.25  # 4 inches (maintain aspect ratio)
        
        plt.figure(figsize=(width_inches, height_inches), dpi=dpi)
        
        # Determine color scale bounds
        # For similarity matrices, values are typically between 0 and 1
        matrix_min = float(np.min(sim_matrix))
        matrix_max = float(np.max(sim_matrix))
        matrix_range = matrix_max - matrix_min
        
        # Set explicit color scale bounds to ensure colors are visible
        # Always use 0.0 to 1.0 range for similarity scores to ensure consistent coloring
        vmin = 0.0
        vmax = 1.0
        
        # If all values are very similar or identical, still use full range for color visibility
        # This ensures the heatmap shows colors even when responses are very similar
        if matrix_range < 0.01:
            # Values are nearly identical - use full 0-1 range to show color gradient
            # The actual values will map to the appropriate part of the colormap
            vmin = 0.0
            vmax = 1.0
        else:
            # Use actual range with small padding for better visualization
            vmin = max(0.0, matrix_min - 0.05)
            vmax = min(1.0, matrix_max + 0.05)
        
        # Only show annotations for smaller matrices to reduce file size
        # For larger matrices, annotations can significantly increase file size
        n = sim_matrix.shape[0]
        show_annotations = n <= 5  # Only annotate if 5 or fewer responses
        
        sns.heatmap(
            sim_matrix,
            annot=show_annotations,  # Conditional annotations
            fmt=".2f" if show_annotations else None,
            cmap="coolwarm",
            vmin=vmin,
            vmax=vmax,
            center=None,  # Don't center - use full range for better color variation
            xticklabels=self.labels,
            yticklabels=self.labels,
            cbar_kws={"label": "Similarity Score"}  # Add colorbar label
        )
        
        # Build title with challenge_id (UUID) and correlation_id (if available)
        title = "Consensus Similarity Heatmap"
        timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
        if challenge_id:
            title = f"{title}\nChallenge ID: {challenge_id}"
        if correlation_id:
            title = f"{title}\nCorrelation ID: {correlation_id}"
        title = f"{title}\nTimestamp: {timestamp}"
        
        # Adjust font sizes for smaller figure (500px vs original ~800px)
        plt.title(title, fontsize=8)
        plt.xticks(rotation=45, ha='right', fontsize=7)
        plt.yticks(fontsize=7)
        plt.tight_layout()
        
        # Ensure path uses .png extension
        png_path = path
        if not png_path.endswith('.png'):
            png_path = path.rsplit('.', 1)[0] + '.png' if '.' in path else path + '.png'
        
        # Save as PNG (original working format)
        plt.savefig(
            png_path,
            dpi=100,  # Standard DPI
            bbox_inches='tight',
            facecolor='white'
        )
        plt.close()
        
        # Post-process with PIL to optimize PNG file size (lossless compression)
        try:
            from PIL import Image
            # Open the saved PNG
            img = Image.open(png_path)
            
            # Convert to RGB if needed (PNG can be RGBA, but RGB is smaller)
            # This is lossless if there's no actual transparency
            if img.mode in ('RGBA', 'LA'):
                # Create white background for any transparency
                background = Image.new('RGB', img.size, (255, 255, 255))
                background.paste(img, mask=img.split()[-1])  # Use alpha channel as mask
                img = background
            elif img.mode == 'P':
                # Palette mode - convert to RGB for better compression control
                img = img.convert('RGB')
            elif img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Save with maximum lossless compression
            # compress_level=9 is maximum compression (slower but smaller)
            # optimize=True enables additional optimization passes
            img.save(
                png_path,
                'PNG',
                optimize=True,  # Enable optimization (lossless)
                compress_level=9  # Maximum compression (0-9, lossless)
            )
        except ImportError:
            # PIL/Pillow not available, skip optimization
            pass
        except Exception as e:
            # If optimization fails, keep original
            # Don't log as error since this is optional optimization
            pass
        
        # Return the actual PNG path
        return png_path

    def _calculate_diversity_bonus(self, sim_matrix: np.ndarray, config: ConsensusConfig) -> Dict[str, float]:
        """
        Calculate diversity bonus for unique, high-quality responses.
        
        Rewards responses that are both high-quality AND unique to prevent monoculture.
        This incentivizes miners to provide diverse perspectives rather than copying
        the most common response.
        
        Returns:
            Dictionary mapping response labels to their diversity bonus [0, max_diversity_bonus]
        """
        diversity_bonuses = {}
        
        if not config.enable_diversity_bonus:
            return {label: 0.0 for label in self.labels}
        
        if len(sim_matrix) < 2:
            return {label: 0.0 for label in self.labels}
        
        # If no quality scores, no diversity bonus
        if self.quality_scores is None or len(self.quality_scores) != len(self.embeddings):
            return {label: 0.0 for label in self.labels}
        
        for i, label in enumerate(self.labels):
            # Calculate uniqueness (inverse of average similarity to others)
            # Exclude self-similarity
            other_sims = np.delete(sim_matrix[i], i)
            avg_sim = np.mean(other_sims) if len(other_sims) > 0 else 0.0
            uniqueness = 1.0 - avg_sim
            
            # Only reward uniqueness if response is high quality
            quality = self.quality_scores[i]
            quality_threshold = 0.6  # Only reward if quality > 0.6
            
            if quality > quality_threshold:
                # Scale bonus by uniqueness and quality
                bonus = uniqueness * (quality / quality_threshold) * config.max_diversity_bonus
                bonus = min(bonus, config.max_diversity_bonus)  # Cap at max
            else:
                bonus = 0.0
            
            diversity_bonuses[label] = float(bonus)
            
            if bonus > 0.01:  # Only log if meaningful bonus
                logger.debug(
                    f"[DIVERSITY] Response {label}: quality={quality:.3f}, "
                    f"uniqueness={uniqueness:.3f}, bonus={bonus:.3f}"
                )
        
        return diversity_bonuses

    def _compute_individual_scores(
        self, 
        sim_matrix: np.ndarray, 
        consensus_labels: set[str],
        use_confidence_weighting: bool = True,
        config: Optional[ConsensusConfig] = None
    ) -> Dict[str, float]:
        """
        Compute individual scores for each response.
        
        Scoring factors:
        1. Average similarity to all other responses (consensus alignment)
        2. Semantic quality score (from quality assessment)
        3. Confidence score (if available and enabled)
        4. Consensus membership bonus
        5. Diversity bonus (for unique high-quality responses)
        
        Args:
            sim_matrix: Pairwise similarity matrix for filtered responses
            consensus_labels: Set of labels that are in consensus
            use_confidence_weighting: Whether to weight by confidence scores
            config: Optional ConsensusConfig for diversity bonus calculation
            
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
        
        # Use semantic quality scores if available, otherwise fall back to word length
        if self.quality_scores is not None and len(self.quality_scores) == len(self.responses):
            # Use semantic quality scores (already in [0, 1] range)
            quality_scores_array = self.quality_scores
            logger.debug("[SCORING] Using semantic quality scores for individual scoring")
        else:
            # Fallback: Compute quality based on response length
            lengths = np.array([len(r.split()) for r in self.responses])
            if np.max(lengths) > 0:
                quality_scores_array = lengths / np.max(lengths)
            else:
                quality_scores_array = np.ones(len(self.responses))
            logger.debug("[SCORING] Using word-length quality scores for individual scoring")
        
        # Calculate diversity bonuses
        diversity_bonuses = {}
        if config and config.enable_diversity_bonus:
            diversity_bonuses = self._calculate_diversity_bonus(sim_matrix, config)
        else:
            diversity_bonuses = {label: 0.0 for label in self.labels}
        
        # Compute composite scores for each response
        for i, label in enumerate(self.labels):
            # Base score: average similarity (how well aligned with consensus)
            base_score = avg_similarities[i]
            
            # Quality component (semantic quality)
            quality_component = quality_scores_array[i] * 0.25  # 25% weight (increased from 20%)
            
            # Confidence component (if available and enabled)
            confidence_component = 0.0
            if use_confidence_weighting and len(self.confidences) > i:
                confidence_component = self.confidences[i] * 0.15  # 15% weight
            
            # Consensus membership bonus
            consensus_bonus = 0.0
            if label in consensus_labels:
                consensus_bonus = 0.10  # 10% bonus for being in consensus
            
            # Diversity bonus
            diversity_bonus = diversity_bonuses.get(label, 0.0)
            
            # Composite score: base similarity (40%) + quality (25%) + confidence (15%) + consensus (10%) + diversity (up to 15%)
            composite_score = (
                base_score * 0.40 +
                quality_component +
                confidence_component +
                consensus_bonus +
                diversity_bonus
            )
            
            scores[label] = float(composite_score)
            
            logger.debug(
                f"[SCORING] Response {label}: similarity={base_score:.3f}, "
                f"quality={quality_scores_array[i]:.3f}, consensus={consensus_bonus:.3f}, "
                f"diversity={diversity_bonus:.3f}, total={composite_score:.3f}"
            )
        
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
                    
                    # Get quality for this response (use semantic quality if available)
                    # Note: filtered responses don't have quality scores in self.quality_scores
                    # We'll need to use word length as fallback
                    orig_length = len(self.all_responses[orig_idx].split())
                    max_length = max([len(r.split()) for r in self.all_responses])
                    quality_score = (orig_length / max_length * 0.25) if max_length > 0 else 0.0
                    
                    confidence_score = 0.0
                    if use_confidence_weighting and orig_idx < len(self.all_confidences):
                        confidence_score = self.all_confidences[orig_idx] * 0.15
                    
                    # Composite score with penalty for being out of consensus
                    composite_score = (
                        avg_consensus_sim * 0.30 +  # Reduced weight (30% vs 40%)
                        quality_score +
                        confidence_score
                    )
                    scores[label] = float(composite_score)
                else:
                    scores[label] = 0.0
        
        return scores

    def evaluate_consensus(self, config: ConsensusConfig) -> ConsensusResult:
        """
        Evaluate consensus with reordered pipeline for garbage consensus prevention.
        
        NEW ORDER (CRITICAL for preventing garbage consensus):
        1. Semantic quality assessment (FIRST - filters garbage before clustering)
        2. Smart outlier detection (quality-aware)
        3. Clustering (if enabled)
        4. Legacy quality filter (deprecated, for backward compat)
        5. Quality-weighted scoring
        6. Consensus determination with garbage detection alerts
        """
        logger.info(f"[CONSENSUS] Starting consensus evaluation with {len(self.responses)} responses")
        
        # STEP 1: SEMANTIC QUALITY ASSESSMENT (CRITICAL - runs FIRST)
        if config.enable_semantic_quality:
            logger.info("[CONSENSUS] Step 1: Semantic quality assessment")
            self._apply_semantic_quality_filter(config)
            logger.info(f"[CONSENSUS] After quality filter: {len(self.responses)} responses remain")
            
            if len(self.responses) == 0:
                logger.warning("[CONSENSUS] All responses filtered out by quality assessment!")
                return ConsensusResult(
                    original_prompt=self.original_prompt,
                    in_consensus={},
                    out_of_consensus={l: r for l, r in zip(self.all_labels, self.all_responses)},
                    similarity_score=0.0,
                    weighted_score=None,
                    polarity_agreement=None,
                    heatmap_path=None,
                    consensus_achieved=False,
                    quality_plot_path=None,
                    consensus_narrative=None,
                    miner_scores={}
                )
        
        # Compute initial similarity matrix after quality filtering
        sim_matrix = self._compute_pairwise_similarity()

        # STEP 2: SMART OUTLIER DETECTION (quality-aware)
        if config.use_outlier_detection:
            logger.info("[CONSENSUS] Step 2: Smart outlier detection")
            if config.enable_smart_outlier_detection:
                sim_matrix = self._smart_outlier_detection(config)
            else:
                logger.warning("[CONSENSUS] Using deprecated simple outlier detection (not quality-aware)")
                sim_matrix = self._apply_outlier_filter(sim_matrix)
            logger.info(f"[CONSENSUS] After outlier detection: {len(self.responses)} responses remain")

        # STEP 3: CLUSTERING
        if config.use_clustering:
            logger.info("[CONSENSUS] Step 3: Clustering")
            sim_matrix = self._apply_clustering_filter(sim_matrix)
            logger.info(f"[CONSENSUS] After clustering: {len(self.responses)} responses remain")

        # STEP 4: LEGACY QUALITY FILTER (deprecated, for backward compatibility)
        if config.apply_quality_filter and not config.enable_semantic_quality:
            logger.warning("[CONSENSUS] Using deprecated word-length quality filter (use semantic quality instead)")
            self._apply_quality_filter(config.quality_sensitivity)
            sim_matrix = self._compute_pairwise_similarity()

        # STEP 5: QUALITY-WEIGHTED SCORING
        if config.enable_semantic_quality and self.quality_scores is not None:
            logger.info("[CONSENSUS] Step 5: Quality-weighted scoring")
            mu, sigma = self._apply_quality_weighted_scoring(sim_matrix, config)
        else:
            logger.info("[CONSENSUS] Step 5: Basic scoring (no quality weighting)")
            mu, sigma = self._apply_basic_scoring(sim_matrix)
        
        theta = mu + config.lambda_factor * sigma
        logger.info(f"[CONSENSUS] Similarity score: μ={mu:.3f}, σ={sigma:.3f}, θ={theta:.3f}")

        weighted_score = self._apply_weighted_scoring(sim_matrix) if config.use_weighted_scoring else None
        polarity_score = self._apply_polarity_clustering() if config.use_polarity_clustering else None
        consensus = theta > config.threshold_min or (
            polarity_score and polarity_score > config.polarity_agreement_min
        )
        
        logger.info(f"[CONSENSUS] Consensus {'ACHIEVED' if consensus else 'NOT ACHIEVED'} (threshold={config.threshold_min})")

        # STEP 6: GARBAGE DETECTION ALERTS
        if config.enable_garbage_alerts and self.quality_scores is not None and len(self.quality_scores) > 0:
            avg_cluster_quality = np.mean(self.quality_scores)
            if avg_cluster_quality < config.garbage_cluster_threshold:
                logger.warning(
                    f"[CONSENSUS] ⚠️  LOW QUALITY CLUSTER DETECTED! "
                    f"Average quality={avg_cluster_quality:.3f} below threshold={config.garbage_cluster_threshold:.3f}. "
                    f"This may indicate a garbage consensus attack. Manual review recommended."
                )
                # Log details of each response in the low-quality cluster
                for i, (label, quality) in enumerate(zip(self.labels, self.quality_scores)):
                    logger.warning(
                        f"[CONSENSUS] Low-quality cluster member {label}: quality={quality:.3f}, "
                        f"text_preview={self.responses[i][:80]}..."
                    )

        # Generate visualizations
        quality_path = None
        actual_heatmap_path = None
        if config.generate_heatmap:
            actual_heatmap_path = self._generate_heatmap(
                sim_matrix, 
                config.heatmap_path, 
                challenge_id=config.challenge_id,
                correlation_id=config.correlation_id
            )
            if config.generate_quality_plot and config.apply_quality_filter:
                quality_path = self._generate_quality_plot(actual_heatmap_path)

        in_consensus = {l: r for l, r in zip(self.labels, self.responses)}
        out_of_consensus = {
            l: r for l, r in zip(self.all_labels, self.all_responses) if l not in in_consensus
        }

        # Compute individual response scores (with quality and diversity)
        consensus_label_set = set(in_consensus.keys())
        individual_scores = self._compute_individual_scores(
            sim_matrix=sim_matrix,
            consensus_labels=consensus_label_set,
            use_confidence_weighting=config.use_weighted_scoring,
            config=config  # Pass config for diversity bonus
        )

        return ConsensusResult(
            original_prompt=self.original_prompt,
            in_consensus=in_consensus,
            out_of_consensus=out_of_consensus,
            similarity_score=theta,
            weighted_score=weighted_score,
            polarity_agreement=polarity_score,
            heatmap_path=actual_heatmap_path if config.generate_heatmap else None,
            consensus_achieved=consensus,
            quality_plot_path=quality_path,
            consensus_narrative=None,
            miner_scores=individual_scores
        )
