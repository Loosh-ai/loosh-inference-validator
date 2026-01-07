from datetime import datetime
from typing import Optional, List, Dict, Any
import time

from loguru import logger
import httpx

import asyncio
from pydantic import BaseModel

from validator.challenge.challenge_types import ChallengeTask
from validator.challenge_api.models import ResponseAPICreate, ResponseAPIRequest, ResponseAPIResult

async def update_challenge_response(
    client: httpx.AsyncClient,
    api_key: str,
    server_address: str,
    response_create: ResponseAPIRequest,
    validator_hotkey: str,
    node_id: int
):
    """
    Update the challenge response to the API.
    
    Args:
        client (httpx.AsyncClient): HTTP client for making requests.
        api_key (str): API key of the requester (sent in headers).
        server_address (str): The server address to fetch from.
        response_create (ResponseCreate): The response data to send.
        validator_hotkey (str): The validator's hotkey (sent as request parameter).
        node_id (int): Node ID.
    
    Returns:
        ResponseAPIResult: Response if successful, None otherwise.
    """
    try:
        start_time = time.time()
        
        # Send request to update challenge response
        response = await client.post(
            f"{server_address}/response",
            json=response_create.model_dump(mode="json"),
            headers={"api-key": api_key},
            params={"validator_hotkey": validator_hotkey},
            timeout=30.0  # 30 second timeout
        )
        
        if response.status_code != 201:
            logger.error(f"Error updating api response: {response.status_code} - {response.text}")
            return None
            
        response_data = response.json()
        response_time_ms = int((time.time() - start_time) * 1000)

        # Create response using available fields
        response_model = ResponseAPIResult(
            id=response_data["id"],
            text=response_data.get("text", ""),
            original_request=response_create.original_request,
            created_at=datetime.now(),
            response_time_ms=response_time_ms
        )

        return response_model
        
    except Exception as e:
        logger.error(f"Error fetching challenge: {str(e)}")
        return None

async def process_challenge_results(
    challenge_tasks: List[ChallengeTask],
    client: httpx.AsyncClient,
    api_key: str,
    server_address: str,
    validator_hotkey: str,
    node_id: int,
    test_mode: bool = False
) -> None:
    """
    Process challenge results without blocking.
    
    In test mode, picks the first response and submits it immediately,
    skipping any evaluation or heatmap generation.
    """
    if test_mode:
        logger.info(f"[TEST MODE] Processing {len(challenge_tasks)} challenge results - will pick first response")
    else:
        logger.info(f"Processing {len(challenge_tasks)} challenge results")
    
    # Wait for all tasks to complete with timeout
    pending = [task.task for task in challenge_tasks]
    timeout = 60  # 60s timeout
    
    first_response_submitted = False
    
    while pending:
        # Wait for the next task to complete, with timeout
        done, pending = await asyncio.wait(
            pending,
            timeout=timeout,  # Check every minute for completed tasks
            return_when=asyncio.FIRST_COMPLETED
        )
        
        # Process completed tasks
        for task in done:
            try:
                response = await task

                challenge_task = next(ct for ct in challenge_tasks if ct.task == task)

                response_create = ResponseAPICreate(
                    id=challenge_task.challenge_id,
                    text=response.response_text,
                    original_request=challenge_task.challenge_orig
                )

                # In test mode, submit first response and exit
                if test_mode and not first_response_submitted:
                    logger.info(
                        f"[TEST MODE] Submitting first response from miner {challenge_task.node_id} "
                        f"for challenge {challenge_task.challenge_id} - skipping evaluation and heatmap generation"
                    )
                    first_response_submitted = True
                elif not test_mode:
                    logger.debug(f"Submitting response from miner {challenge_task.node_id}")

                # Process the response...
                response_api = await update_challenge_response(
                    client=client,
                    api_key=api_key,
                    server_address=server_address,
                    response_create=response_create,
                    validator_hotkey=validator_hotkey,
                    node_id=node_id
                )

                logger.debug(f"Processed challenge response: {response_api}")
                
                # In test mode, exit after first successful response
                if test_mode and first_response_submitted:
                    logger.info(f"[TEST MODE] First response submitted successfully, exiting (remaining tasks cancelled)")
                    # Cancel remaining tasks
                    for remaining_task in pending:
                        remaining_task.cancel()
                    return
                    
            except Exception as e:
                logger.error(f"Error processing challenge result: {str(e)}")
        
        # Log status of remaining tasks (only if not in test mode or first response not submitted)
        if pending and (not test_mode or not first_response_submitted):
            logger.info(f"Still waiting for {len(pending)} challenges to complete")
    
    if not test_mode:
        logger.info("All challenge results processed")
    else:
        logger.info("[TEST MODE] Challenge results processing complete")

