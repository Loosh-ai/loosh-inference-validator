"""
Sybil attack detection module.

Identifies suspiciously similar responses that may indicate a sybil attack
where a single entity controls multiple miners and sends identical or near-identical responses.
"""

from dataclasses import dataclass
from typing import List, Dict, Set, Tuple, Optional
from datetime import datetime
import numpy as np
from loguru import logger

from validator.challenge.challenge_types import InferenceResponse


@dataclass
class SybilPair:
    """Represents a pair of suspiciously similar responses.
    
    miner_hotkey_1/miner_hotkey_2 are miner HOTKEYS (persistent SS58 addresses),
    NOT UIDs. UIDs are transient indices that change on UID compression/trimming.
    """
    miner_hotkey_1: str  # Miner hotkey (persistent identity)
    miner_hotkey_2: str  # Miner hotkey (persistent identity)
    similarity_score: float
    response_text_1: str
    response_text_2: str
    challenge_id: Optional[int] = None


@dataclass
class SybilGroup:
    """Represents a group of miners with suspiciously similar responses.
    
    miner_hotkeys are miner HOTKEYS (persistent SS58 addresses),
    NOT UIDs. UIDs are transient indices that change on UID compression/trimming.
    """
    miner_hotkeys: Set[str]  # Set of miner hotkeys (persistent identity)
    avg_similarity: float
    min_similarity: float
    max_similarity: float
    response_texts: Dict[str, str]  # hotkey -> response text
    challenge_id: Optional[int] = None


@dataclass
class SybilDetectionResult:
    """Results of sybil detection analysis.
    
    miner_hotkeys are miner HOTKEYS (persistent SS58 addresses),
    NOT UIDs. UIDs are transient indices that change on UID compression/trimming.
    """
    challenge_id: Optional[int]
    suspicious_pairs: List[SybilPair]
    suspicious_groups: List[SybilGroup]
    similarity_matrix: np.ndarray
    miner_hotkeys: List[str]  # List of miner hotkeys (persistent identity)
    detection_timestamp: datetime
    high_similarity_threshold: float
    very_high_similarity_threshold: float


class SybilDetector:
    """
    Detects potential sybil attacks by analyzing response similarity patterns.
    
    A sybil attack occurs when a single entity controls multiple miners and sends
    identical or near-identical responses instead of performing actual inference.
    """
    
    def __init__(
        self,
        high_similarity_threshold: float = 0.95,
        very_high_similarity_threshold: float = 0.98,
        min_group_size: int = 2
    ):
        """
        Initialize sybil detector.
        
        Args:
            high_similarity_threshold: Similarity score above which responses are considered suspicious (default: 0.95)
            very_high_similarity_threshold: Similarity score above which responses are considered highly suspicious (default: 0.98)
            min_group_size: Minimum number of miners in a group to be considered suspicious (default: 2)
        """
        self.high_similarity_threshold = high_similarity_threshold
        self.very_high_similarity_threshold = very_high_similarity_threshold
        self.min_group_size = min_group_size
    
    def detect_sybil_patterns(
        self,
        similarity_matrix: np.ndarray,
        responses: List[InferenceResponse],
        miner_ids: List[str],
        challenge_id: Optional[int] = None
    ) -> SybilDetectionResult:
        """
        Detect potential sybil attack patterns from similarity matrix.
        
        Args:
            similarity_matrix: Pairwise similarity matrix (n x n numpy array)
            responses: List of inference responses
            miner_ids: List of miner hotkeys (persistent SS58 addresses) corresponding to each response.
                       NOTE: These are hotkeys, NOT UIDs. UIDs are transient and change on UID compression.
            challenge_id: Optional challenge ID for tracking
        
        Returns:
            SybilDetectionResult containing detected suspicious pairs and groups
        """
        if len(responses) != len(miner_ids) or len(responses) != similarity_matrix.shape[0]:
            raise ValueError(
                f"Mismatch in dimensions: {len(responses)} responses, "
                f"{len(miner_ids)} miner_ids, {similarity_matrix.shape[0]}x{similarity_matrix.shape[1]} matrix"
            )
        
        # Find suspicious pairs (above high similarity threshold)
        suspicious_pairs = self._find_suspicious_pairs(
            similarity_matrix, responses, miner_ids, challenge_id
        )
        
        # Find suspicious groups (clusters of highly similar responses)
        suspicious_groups = self._find_suspicious_groups(
            similarity_matrix, responses, miner_ids, challenge_id
        )
        
        return SybilDetectionResult(
            challenge_id=challenge_id,
            suspicious_pairs=suspicious_pairs,
            suspicious_groups=suspicious_groups,
            similarity_matrix=similarity_matrix,
            miner_hotkeys=miner_ids,
            detection_timestamp=datetime.utcnow(),
            high_similarity_threshold=self.high_similarity_threshold,
            very_high_similarity_threshold=self.very_high_similarity_threshold
        )
    
    def _find_suspicious_pairs(
        self,
        similarity_matrix: np.ndarray,
        responses: List[InferenceResponse],
        miner_ids: List[str],
        challenge_id: Optional[int]
    ) -> List[SybilPair]:
        """Find pairs of responses with suspiciously high similarity."""
        suspicious_pairs = []
        
        # Iterate over upper triangle of similarity matrix (avoid duplicates and diagonal)
        n = len(responses)
        for i in range(n):
            for j in range(i + 1, n):
                similarity = similarity_matrix[i, j]
                
                if similarity >= self.high_similarity_threshold:
                    pair = SybilPair(
                        miner_hotkey_1=miner_ids[i],
                        miner_hotkey_2=miner_ids[j],
                        similarity_score=float(similarity),
                        response_text_1=responses[i].response_text,
                        response_text_2=responses[j].response_text,
                        challenge_id=challenge_id
                    )
                    suspicious_pairs.append(pair)
        
        # Sort by similarity (highest first)
        suspicious_pairs.sort(key=lambda p: p.similarity_score, reverse=True)
        
        return suspicious_pairs
    
    def _find_suspicious_groups(
        self,
        similarity_matrix: np.ndarray,
        responses: List[InferenceResponse],
        miner_ids: List[str],
        challenge_id: Optional[int]
    ) -> List[SybilGroup]:
        """
        Find groups of miners with suspiciously similar responses.
        
        Uses a simple clustering approach: miners are in the same group if
        their pairwise similarities are all above the threshold.
        """
        n = len(responses)
        visited = set()
        groups = []
        
        # Build adjacency graph: two miners are connected if similarity >= threshold
        adjacency = {}
        for i in range(n):
            adjacency[i] = []
            for j in range(i + 1, n):
                if similarity_matrix[i, j] >= self.high_similarity_threshold:
                    adjacency[i].append(j)
                    if j not in adjacency:
                        adjacency[j] = []
                    adjacency[j].append(i)
        
        # Find connected components (groups)
        for start_idx in range(n):
            if start_idx in visited:
                continue
            
            # BFS to find all connected miners
            component = []
            queue = [start_idx]
            visited.add(start_idx)
            
            while queue:
                current = queue.pop(0)
                component.append(current)
                
                for neighbor in adjacency.get(current, []):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
            
            # Only consider groups with minimum size
            if len(component) >= self.min_group_size:
                # Calculate group statistics
                group_miner_ids = [miner_ids[i] for i in component]
                similarities = []
                
                for i in range(len(component)):
                    for j in range(i + 1, len(component)):
                        idx_i = component[i]
                        idx_j = component[j]
                        similarities.append(similarity_matrix[idx_i, idx_j])
                
                if similarities:
                    group = SybilGroup(
                        miner_hotkeys=set(group_miner_ids),
                        avg_similarity=float(np.mean(similarities)),
                        min_similarity=float(np.min(similarities)),
                        max_similarity=float(np.max(similarities)),
                        response_texts={
                            miner_ids[i]: responses[i].response_text
                            for i in component
                        },
                        challenge_id=challenge_id
                    )
                    groups.append(group)
        
        # Sort by average similarity (highest first)
        groups.sort(key=lambda g: g.avg_similarity, reverse=True)
        
        return groups
    
    def generate_analysis_report(self, result: SybilDetectionResult) -> str:
        """
        Generate a human-readable analysis report of sybil detection results.
        
        Args:
            result: SybilDetectionResult to analyze
        
        Returns:
            Formatted report string
        """
        lines = [
            f"Sybil Detection Analysis Report",
            f"Challenge ID: {result.challenge_id or 'N/A'}",
            f"Detection Timestamp: {result.detection_timestamp.isoformat()}",
            f"Thresholds: High={result.high_similarity_threshold:.2f}, Very High={result.very_high_similarity_threshold:.2f}",
            "",
            f"Summary:",
            f"  - Total Miners Analyzed: {len(result.miner_hotkeys)}",
            f"  - Suspicious Pairs Found: {len(result.suspicious_pairs)}",
            f"  - Suspicious Groups Found: {len(result.suspicious_groups)}",
            ""
        ]
        
        if result.suspicious_pairs:
            lines.append("Suspicious Pairs:")
            for i, pair in enumerate(result.suspicious_pairs[:10], 1):  # Show top 10
                severity = "VERY HIGH" if pair.similarity_score >= result.very_high_similarity_threshold else "HIGH"
                lines.append(
                    f"  {i}. Miners {pair.miner_hotkey_1} <-> {pair.miner_hotkey_2}: "
                    f"similarity={pair.similarity_score:.4f} ({severity})"
                )
                # Show response previews
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
            for i, group in enumerate(result.suspicious_groups[:5], 1):  # Show top 5 groups
                lines.append(
                    f"  {i}. Group of {len(group.miner_hotkeys)} miners: "
                    f"hotkeys={sorted(group.miner_hotkeys)}, "
                    f"avg_similarity={group.avg_similarity:.4f}, "
                    f"range=[{group.min_similarity:.4f}, {group.max_similarity:.4f}]"
                )
            
            if len(result.suspicious_groups) > 5:
                lines.append(f"  ... and {len(result.suspicious_groups) - 5} more groups")
        
        return "\n".join(lines)
