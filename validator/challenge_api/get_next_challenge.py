from datetime import datetime
from typing import Optional, List, Dict, Any
import time

from loguru import logger
import httpx

import asyncio
from pydantic import BaseModel

from validator.challenge_api.models import ChallengeAPIRequest, ChallengeAPIResponse

async def get_next_challenge(
    server_address: str,
    api_key: str, 
    validator_hotkey: str,
    node_id: int
) -> Optional[ChallengeAPIResponse]:
    """
    Fetch the next challenge from the API.
    
    Args:
        server_address (str): The server address to fetch from.
        api_key (str): API key of the requester (sent in headers).
        validator_hotkey (str): The validator's hotkey (sent as request parameter).
        node_id (int): Node ID.
    
    Returns:
        ChallengeResponse: Challenge response with timing if successful, None otherwise.
    """
    try:
        start_time = time.time()
        
        # Send request to get next challenge
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{server_address}/challenge",
                headers={"Authorization": f"Bearer {api_key}"},
                params={"validator_hotkey": validator_hotkey},
                timeout=30.0  # 30 second timeout
            )
        
        if response.status_code != 200:
            logger.error(f"Error fetching challenge: {response.status_code} - {response.text}")
            return None
            
        response_data = response.json()
        response_time_ms = int((time.time() - start_time) * 1000)
        
        # Create ChallengeResponse object from response_data with timing information
        challenge_response = ChallengeAPIResponse(
            id=response_data["id"],
            correlation_id=response_data.get("correlation_id"),
            prompt=response_data["prompt"],
            temperature=response_data["temperature"],
            top_p=response_data["top_p"],
            max_tokens=response_data["max_tokens"],
            metadata=response_data["metadata"],
            created_at=datetime.fromisoformat(response_data["created_at"]),
            status=response_data["status"],
            requester=response_data["requester"],
            response_time_ms=response_time_ms
        )
        
        return challenge_response
        
    except Exception as e:
        logger.error(f"Error fetching challenge: {str(e)}")
        return None
    
async def get_next_challenge_with_retry(
    server_address: str,
    api_key: str,
    validator_hotkey: str,
    node_id: int,
    max_retries: int = 2, 
    initial_delay: float = 5.0
) -> Optional[ChallengeAPIResponse]:
    """
    Attempt to fetch the next challenge from the API with retries.
    
    Args:
        server_address (str): The server address to fetch from.
        api_key (str): API key of the requester (sent in headers).
        validator_hotkey (str): The validator's hotkey (sent as request parameter).
        node_id (int): Node ID.
        max_retries (int): Maximum number of retry attempts.
        initial_delay (float): Initial delay in seconds between retries.
    
    Returns:
        Optional[ChallengeResponse]: Challenge response with timing if successful, None otherwise.
    """
    for attempt in range(max_retries + 1):
        try:
            logger.info(f"Fetching next challenge from API (Attempt {attempt + 1}/{max_retries + 1})...")
            challenge_response = await get_next_challenge(server_address, api_key, validator_hotkey, node_id)
            if challenge_response:
                logger.info(f"Successfully fetched challenge: id={challenge_response.id}, response_time={challenge_response.response_time_ms}ms")
                return challenge_response
            else:
                logger.warning("No challenge available from API")
        except Exception as e:
            logger.error(f"Error fetching challenge: {str(e)}")
        
        if attempt < max_retries:
            delay = initial_delay * (2 ** attempt)  # Exponential backoff
            logger.info(f"Retrying in {delay:.2f} seconds...")
            await asyncio.sleep(delay)
    
    logger.warning("Failed to fetch challenge after all retry attempts")
    return None


async def get_next_challenge_with_retry2(config, hotkey, nodeid=0):
    """
    Fetch the next challenge from the API with retries using config and hotkey objects.
    
    Args:
        config: ValidatorConfig object containing challenge API configuration
        hotkey: Bittensor hotkey object with ss58_address
        nodeid (int): Node ID, defaults to 0
    
    Returns:
        Optional[ChallengeAPIResponse]: Challenge response with timing if successful, None otherwise.
    """
    return await get_next_challenge_with_retry(
        server_address=config.challenge_api_url,
        api_key=config.challenge_api_key,
        validator_hotkey=hotkey.ss58_address,
        node_id=nodeid
    )

