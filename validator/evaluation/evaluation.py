import os
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
            
            # Calculate emissions
            emissions = self._calculate_emissions(responses, result, miner_ids)
            result.miner_scores = emissions
            
            # Upload heatmap if available
            heatmap_path = None
            if result.heatmap_path:
                heatmap_path = await self._upload_heatmap(result.heatmap_path)
            
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
    
    def _calculate_emissions(
        self,
        responses: List[InferenceResponse],
        result: ConsensusResult,
        miner_ids: Optional[List[int]] = None
    ) -> Dict[str, float]:
        """Calculate emissions allocation for miners."""
        # Base emissions on response time and consensus score
        total_time = sum(r.response_time_ms for r in responses)
        emissions = {}
        
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
            
            # Use miner_id from the list if provided, otherwise use index
            miner_id = miner_ids[i] if miner_ids and i < len(miner_ids) else i
            emissions[str(miner_id)] = emission
        
        return emissions
    
    async def _upload_heatmap(self, filepath: str) -> Optional[str]:
        """Upload heatmap to storage service."""
        try:
            async with aiohttp.ClientSession() as session:
                with open(filepath, "rb") as f:
                    data = aiohttp.FormData()
                    data.add_field(
                        "file",
                        f,
                        filename=os.path.basename(filepath),
                        content_type="image/png"
                    )
                    
                    async with session.post(self.config.heatmap_upload_url, data=data) as response:
                        if response.status == 200:
                            result = await response.json()
                            return result.get("url")
            
            return None
            
        except Exception as e:
            logger.error(f"Error uploading heatmap: {str(e)}")
            return None 