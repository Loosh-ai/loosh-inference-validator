import os
import math
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import aiohttp
import numpy as np
from loguru import logger
from sentence_transformers import SentenceTransformer

from validator.challenge.challenge_types import InferenceResponse
from validator.db.operations import DatabaseManager
from validator.config import get_validator_config

# Import from loosh-utilities
from Evaluation.consensus_engine import ConsensusEngine, ConsensusConfig, ConsensusResult
from Recording.consensus_narrative_generator import ConsensusNarrativeGenerator, LLMConfig
from Recording.similarity_heatmap import generate_semantic_similarity_heatmap

class InferenceValidator:
    """Validator for inference responses."""
    
    def __init__(self, db_manager: DatabaseManager):
        """Initialize validator."""
        self.db_manager = db_manager
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Load configuration
        self.config = get_validator_config()
        
        # Initialize narrative generator with LLM config
        # Get API key from config or environment
        api_key = getattr(self.config, 'openai_api_key', None) or os.getenv("OPENAI_API_KEY")
        
        self.narrative_generator = ConsensusNarrativeGenerator(
            LLMConfig(
                api_url=self.config.openai_api_url,
                model_name=self.config.openai_model,
                temperature=0.7,
                max_tokens=800,
                api_key=api_key
            )
        )
    
    async def evaluate_responses(
        self,
        challenge_id: int,
        prompt: str,
        responses: List[InferenceResponse],
        miner_ids: Optional[List[int]] = None
    ) -> Tuple[float, Optional[str], Optional[str], Dict[str, float]]:
        """Evaluate a set of inference responses.
        
        Args:
            challenge_id: The challenge ID
            prompt: The original prompt
            responses: List of inference responses
            miner_ids: Optional list of miner UIDs corresponding to each response
        
        Returns:
            Tuple containing:
            - Consensus score
            - Path to heatmap image
            - Narrative of consensus
            - Emissions allocation
        """
        try:
            # Extract response texts and calculate embeddings
            response_texts = [r.response_text for r in responses]
            embeddings = self.embedding_model.encode(response_texts, convert_to_numpy=True)
            
            # Create consensus engine
            consensus_engine = ConsensusEngine(
                original_prompt=prompt,
                responses=response_texts,
                embeddings=embeddings
            )
            
            # Configure consensus evaluation
            config = ConsensusConfig(
                use_clustering=True,
                use_weighted_scoring=True,
                use_outlier_detection=True,
                apply_quality_filter=True,
                quality_sensitivity=0.7,
                generate_heatmap=True,
                heatmap_path=f"temp/heatmap_{challenge_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.png",
                lambda_factor=1.0,
                threshold_min=0.7
            )
            
            # Evaluate consensus
            result = consensus_engine.evaluate_consensus(config)
            
            # Generate narrative (now async)
            narrative = await self.narrative_generator.generate_narrative(result)
            result.consensus_narrative = narrative
            
            # Store individual scores before calculating emissions (they get overwritten)
            individual_scores = result.miner_scores if result.miner_scores else {}
            
            # Calculate emissions
            emissions = self._calculate_emissions(responses, result, miner_ids, individual_scores)
            result.miner_scores = emissions
            
            # Upload heatmap if available
            heatmap_path = None
            if result.heatmap_path:
                heatmap_path = await self._upload_heatmap(result.heatmap_path, challenge_id)
            
            # Log evaluation result
            self.db_manager.log_evaluation_result(
                challenge_id=challenge_id,
                consensus_score=result.similarity_score,
                heatmap_path=heatmap_path,
                narrative=narrative,
                emissions=emissions
            )
            
            return result.similarity_score, heatmap_path, narrative, emissions
            
        except Exception as e:
            logger.error(f"Error evaluating responses: {str(e)}")
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
        individual_scores: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """Calculate emissions allocation for miners.
        
        Args:
            responses: List of inference responses
            result: Consensus evaluation result
            miner_ids: Optional list of miner UIDs corresponding to each response
            individual_scores: Optional dict of individual response scores (label -> score)
        """
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
            
            # Check if response is in consensus
            response_label = f"R{i+1}"
            in_consensus = response_label in result.in_consensus
            
            # Scale by consensus score and consensus status
            emission = time_ratio * result.similarity_score
            if in_consensus:
                emission *= 1.2  # Bonus for being in consensus
            
            # Scaled bonus for highest scoring response based on score difference
            if highest_scoring_label and response_label == highest_scoring_label:
                emission *= bonus_multiplier
            
            # Use miner_id from the list if provided, otherwise use index
            miner_id = miner_ids[i] if miner_ids and i < len(miner_ids) else i
            emissions[str(miner_id)] = emission
        
        return emissions
    
    async def _upload_heatmap(self, filepath: str, challenge_id: int) -> Optional[str]:
        """Upload heatmap to challenge API.
        
        Args:
            filepath: Path to the heatmap image file
            challenge_id: The challenge ID associated with this heatmap
            
        Returns:
            The filepath if upload was successful, None otherwise
        """
        try:
            # Construct the challenge API endpoint URL
            upload_url = f"{self.config.challenge_api_url}/heatmap/upload"
            
            async with aiohttp.ClientSession() as session:
                with open(filepath, "rb") as f:
                    data = aiohttp.FormData()
                    data.add_field(
                        "file",
                        f,
                        filename=os.path.basename(filepath),
                        content_type="image/png"
                    )
                    data.add_field("challenge_id", str(challenge_id))
                    
                    headers = {
                        "Authorization": f"Bearer {self.config.challenge_api_key}"
                    }
                    
                    async with session.post(upload_url, data=data, headers=headers) as response:
                        if response.status == 201:
                            result = await response.json()
                            logger.info(
                                f"Successfully uploaded heatmap for challenge {challenge_id}: "
                                f"{result.get('filename', 'unknown')}"
                            )
                            # Return the local filepath (challenge API stores it locally)
                            return result.get("filepath", filepath)
                        else:
                            error_text = await response.text()
                            logger.warning(
                                f"Failed to upload heatmap for challenge {challenge_id}: "
                                f"HTTP {response.status} - {error_text}"
                            )
                            return None
            
        except FileNotFoundError:
            logger.error(f"Heatmap file not found: {filepath}")
            return None
        except Exception as e:
            logger.error(f"Error uploading heatmap to challenge API: {str(e)}", exc_info=True)
            return None 