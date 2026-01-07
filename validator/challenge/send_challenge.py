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

async def send_inference_challenge(
    client: httpx.AsyncClient,
    server_address: str,
    challenge_id: int,
    challenge: InferenceChallenge,
    miner_hotkey: str,
    db_manager: DatabaseManager,
    node_id: int
) -> Optional[InferenceResponse]:
    """Send an inference challenge to a miner and wait for response."""
    try:
        start_time = time.time()
        
        # Send challenge to miner
        response = await client.post(
            f"{server_address}/inference",
            json=challenge.model_dump(mode="json"),
            headers={"validator-hotkey": miner_hotkey},
            timeout=5*60.0  # 5*60 second timeout for inference
        )
        
        if response.status_code != 200:
            logger.error(f"Error from miner {node_id}: {response.status_code} - {response.text}")
            return None
            
        response_data = response.json()
        response_time_ms = int((time.time() - start_time) * 1000)
        
        # Extract correlation_id from miner response if present, otherwise use from challenge
        correlation_id = response_data.get("correlation_id") or challenge.correlation_id
        
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
        logger.error(f"Error sending challenge to miner {node_id}: {error_msg}")
        return None

async def send_challenge(
    client: httpx.AsyncClient,
    server_address: str,
    challenge_id: int,
    challenge: InferenceChallenge,
    challenge_orig: Challenge,
    miner_hotkey: str,
    db_manager: DatabaseManager,
    node_id: int
) -> ChallengeTask:
    """Send a challenge to a miner and return a task for tracking the response."""
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
            node_id=node_id
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