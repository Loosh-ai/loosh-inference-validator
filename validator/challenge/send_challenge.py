import asyncio
import time
from datetime import datetime
from typing import Optional, List, Dict, Any

import httpx
from loguru import logger

from validator.challenge.challenge_types import (
    ChallengeType, InferenceChallenge, InferenceResponse,
    ValidationResult, ChallengeTask
)
from validator.db.operations import DatabaseManager
from validator.challenge_api.models import Challenge
from validator.config import ValidatorConfig
from validator.internal_config import FIBER_KEY_TTL_SECONDS, FIBER_HANDSHAKE_TIMEOUT_SECONDS
from validator.timing import PipelineTiming, PipelineStages

# Global Fiber client for miner communication
_miner_fiber_client = None

def get_miner_fiber_client(validator_hotkey_ss58: str, config: ValidatorConfig):
    """Get or create global MinerFiberClient instance."""
    global _miner_fiber_client
    if _miner_fiber_client is None:
        from validator.network.fiber_client_miner import MinerFiberClient
        _miner_fiber_client = MinerFiberClient(
            validator_hotkey_ss58=validator_hotkey_ss58,
            config=config,
            key_ttl_seconds=FIBER_KEY_TTL_SECONDS,
            handshake_timeout_seconds=FIBER_HANDSHAKE_TIMEOUT_SECONDS
        )
    return _miner_fiber_client

async def send_inference_challenge(
    client: httpx.AsyncClient,
    server_address: str,
    challenge_id: int,
    challenge: InferenceChallenge,
    miner_hotkey: str,
    db_manager: DatabaseManager,
    node_id: int,
    config: ValidatorConfig,
    validator_hotkey_ss58: str,
    pipeline_timing: Optional[Any] = None
) -> Optional[InferenceResponse]:
    """
    Send an inference challenge to a miner using Fiber MLTS encryption.
    
    Args:
        client: HTTP client for making requests
        server_address: Miner server address
        challenge_id: Challenge ID
        challenge: Inference challenge to send
        miner_hotkey: Miner's hotkey (for logging)
        db_manager: Database manager for logging responses
        node_id: Miner node ID
        config: Validator configuration (required for Fiber)
        validator_hotkey_ss58: Validator's SS58 address (required for Fiber)
    
    Returns:
        InferenceResponse if successful, None otherwise
    """
    try:
        start_time = time.time()
        
        # Use Fiber-encrypted communication (required)
        fiber_client = get_miner_fiber_client(validator_hotkey_ss58, config)
        
        # Include timing data in challenge metadata if available
        challenge_data = challenge.model_dump(mode="json")
        if pipeline_timing:
            if 'metadata' not in challenge_data:
                challenge_data['metadata'] = {}
            challenge_data['metadata']['timing_data'] = pipeline_timing.to_dict()
        
        response_data = await fiber_client.send_encrypted_challenge(
            miner_endpoint=server_address,
            challenge_data=challenge_data,
            client=client
        )
        
        if not response_data:
            logger.error(f"Error from miner {node_id}: Fiber challenge failed")
            return None
        
        response_time_ms = int((time.time() - start_time) * 1000)
        
        # Extract correlation_id from miner response if present, otherwise use from challenge
        correlation_id = response_data.get("correlation_id") or challenge.correlation_id
        
        # Extract timing data from miner response if present
        if pipeline_timing and isinstance(response_data, dict) and response_data.get("metadata") and 'timing_data' in response_data.get("metadata", {}):
            try:
                miner_timing_data = response_data['metadata']['timing_data']
                if isinstance(miner_timing_data, dict):
                    # Merge miner timing stages into pipeline timing
                    miner_timing = PipelineTiming.from_dict(miner_timing_data)
                    # Add miner stages to our pipeline timing
                    for stage in miner_timing.stages:
                        if stage.stage_name in [PipelineStages.MINER_INFERENCE, PipelineStages.MINER_RESPONSE]:
                            pipeline_timing.stages.append(stage)
            except Exception as e:
                logger.debug(f"Could not merge miner timing data: {e}")
        
        # Create response object
        inference_response = InferenceResponse(
            response_text=response_data["response_text"],
            response_time_ms=response_time_ms,
            correlation_id=correlation_id
        )
        
        # Log miner challenge response
        response_preview = response_data["response_text"][:200] + "..." if len(response_data["response_text"]) > 200 else response_data["response_text"]
        logger.info(
            f"Miner {node_id} responded to challenge {challenge_id} | "
            f"time: {response_time_ms}ms | "
            f"protocol: Fiber MLTS | "
            f"response: {response_preview}"
        )
        
        # Log response in database
        db_manager.log_inference_response(
            challenge_id=challenge_id,
            miner_id=node_id,
            response=inference_response
        )
        
        return inference_response
        
    except Exception as e:
        error_msg = f"{type(e).__name__}: {e}"
        logger.error(f"Error sending Fiber-encrypted challenge to miner {node_id}: {error_msg}")
        return None

async def send_challenge(
    client: httpx.AsyncClient,
    server_address: str,
    challenge_id: int,
    challenge: InferenceChallenge,
    challenge_orig: Challenge,
    miner_hotkey: str,
    db_manager: DatabaseManager,
    node_id: int,
    config: ValidatorConfig,
    validator_hotkey_ss58: str,
    pipeline_timing: Optional[Any] = None
) -> ChallengeTask:
    """
    Send a challenge to a miner using Fiber MLTS encryption and return a task for tracking the response.
    
    Args:
        client: HTTP client for making requests
        server_address: Miner server address
        challenge_id: Challenge ID
        challenge: Inference challenge to send
        challenge_orig: Original challenge object
        miner_hotkey: Miner's hotkey
        db_manager: Database manager
        node_id: Miner node ID
        config: Validator configuration (required for Fiber)
        validator_hotkey_ss58: Validator's SS58 address (required for Fiber)
    
    Returns:
        ChallengeTask for tracking the response
    """
    timestamp = datetime.utcnow()
    
    # Create task for sending challenge
    task = asyncio.create_task(
        send_inference_challenge(
            client=client,
            server_address=server_address,
            challenge_id=challenge_id,
            challenge=challenge,
            miner_hotkey=miner_hotkey,
            db_manager=db_manager,
            node_id=node_id,
            config=config,
            validator_hotkey_ss58=validator_hotkey_ss58,
            pipeline_timing=pipeline_timing
        )
    )
    
    return ChallengeTask(
        challenge_id=challenge_id,
        node_id=node_id,
        task=task,
        challenge_orig=challenge_orig,
        timestamp=timestamp,
        challenge=challenge,
        miner_hotkey=miner_hotkey
    ) 