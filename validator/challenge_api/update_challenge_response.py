from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple
import time

from loguru import logger
import httpx

import asyncio
from pydantic import BaseModel

from validator.challenge.challenge_types import ChallengeTask, InferenceResponse
from validator.challenge_api.models import (
    ResponseAPICreate, ResponseAPIRequest, ResponseAPIResult,
    ResponseBatchSubmit, MinerResponseData, EvaluationResult, TokenUsage
)
from validator.network.fiber_client import ValidatorFiberClient
from validator.evaluation.evaluation import InferenceValidator

# Global Fiber client cache (per validator hotkey and Challenge API endpoint)
_fiber_client_cache: Dict[str, ValidatorFiberClient] = {}


async def update_challenge_response(
    client: httpx.AsyncClient,
    api_key: str,
    server_address: str,
    response_create: ResponseAPIRequest,
    validator_hotkey: str,
    node_id: int,
    pipeline_timing: Optional[Any] = None
):
    """
    Update the challenge response to the API using Fiber encryption.
    
    DEPRECATED: Use submit_response_batch for new integrations.
    Kept for backward compatibility and test mode.
    
    Args:
        client (httpx.AsyncClient): HTTP client for making requests.
        api_key (str): API key (deprecated, kept for compatibility but not used with Fiber).
        server_address (str): The server address to fetch from.
        response_create (ResponseCreate): The response data to send.
        validator_hotkey (str): The validator's hotkey (used for Fiber handshake).
        node_id (int): Node ID.
    
    Returns:
        ResponseAPIResult: Response if successful, None otherwise.
    """
    try:
        start_time = time.time()
        
        # Get or create Fiber client for this validator hotkey and Challenge API endpoint
        cache_key = f"{validator_hotkey}:{server_address}"
        if cache_key not in _fiber_client_cache:
            _fiber_client_cache[cache_key] = ValidatorFiberClient(
                validator_hotkey_ss58=validator_hotkey,
                private_key=None,  # TODO: Load from Bittensor wallet
                key_ttl_seconds=3600,
                handshake_timeout_seconds=30
            )
        
        fiber_client = _fiber_client_cache[cache_key]
        
        # Prepare response data
        response_data = response_create.model_dump(mode="json")
        
        # Include timing data in response metadata if available
        if pipeline_timing:
            if response_data.get('original_request') and isinstance(response_data['original_request'], dict):
                if response_data['original_request'].get('metadata') is None:
                    response_data['original_request']['metadata'] = {}
                response_data['original_request']['metadata']['timing_data'] = pipeline_timing.to_dict()
        
        # Send encrypted callback using Fiber
        success = await fiber_client.send_encrypted_callback(
            challenge_api_endpoint=server_address,
            response_data=response_data,
            client=client
        )
        
        if not success:
            logger.error(f"Failed to send encrypted callback for challenge {response_create.id}")
            return None
        
        response_time_ms = int((time.time() - start_time) * 1000)
        
        # Create response model (we don't get full response from encrypted callback,
        # so we construct it from what we sent)
        response_model = ResponseAPIResult(
            id=response_create.id,
            text=response_data.get("text", ""),
            original_request=response_create.original_request,
            created_at=datetime.now(),
            response_time_ms=response_time_ms
        )

        return response_model
        
    except Exception as e:
        logger.error(f"Error sending encrypted callback: {str(e)}", exc_info=True)
        return None


async def submit_response_batch(
    client: httpx.AsyncClient,
    api_key: str,
    server_address: str,
    batch: ResponseBatchSubmit,
    pipeline_timing: Optional[Any] = None
) -> bool:
    """
    Submit a batch of ALL miner responses with evaluation data to Challenge API.
    
    This is the new method for submitting responses (F3).
    Sends all responses to POST /response/batch endpoint.
    
    Args:
        client: HTTP client for making requests
        api_key: API key for Challenge API authentication
        server_address: Challenge API base URL
        batch: ResponseBatchSubmit containing all responses and evaluation
        pipeline_timing: Optional timing data for pipeline tracking
        
    Returns:
        True if batch was submitted successfully, False otherwise
    """
    try:
        start_time = time.time()
        
        # Prepare batch data
        batch_data = batch.model_dump(mode="json")
        
        # Include timing data if available
        if pipeline_timing and batch_data.get('original_request'):
            if batch_data['original_request'].get('metadata') is None:
                batch_data['original_request']['metadata'] = {}
            batch_data['original_request']['metadata']['timing_data'] = pipeline_timing.to_dict()
        
        # Submit batch via HTTP POST
        url = f"{server_address.rstrip('/')}/response/batch"
        headers = {
            "X-API-Key": api_key,
            "x-validator-hotkey": batch.validator_hotkey,
            "Content-Type": "application/json"
        }
        
        logger.info(
            f"[BATCH] Submitting {len(batch.responses)} responses for challenge {batch.challenge_id} "
            f"to {url}"
        )
        
        response = await client.post(url, json=batch_data, headers=headers, timeout=30.0)
        
        if response.status_code == 201:
            elapsed_ms = int((time.time() - start_time) * 1000)
            logger.info(
                f"[BATCH] Successfully submitted response batch for challenge {batch.challenge_id} "
                f"({len(batch.responses)} responses, {elapsed_ms}ms)"
            )
            return True
        elif response.status_code == 409:
            logger.warning(
                f"[BATCH] Responses already exist for challenge {batch.challenge_id} - "
                f"batch submission rejected (409 Conflict)"
            )
            return False
        else:
            logger.error(
                f"[BATCH] Failed to submit response batch for challenge {batch.challenge_id}: "
                f"HTTP {response.status_code} - {response.text}"
            )
            return False
            
    except httpx.TimeoutException:
        logger.error(f"[BATCH] Timeout submitting response batch for challenge {batch.challenge_id}")
        return False
    except Exception as e:
        logger.error(f"[BATCH] Error submitting response batch: {str(e)}", exc_info=True)
        return False

async def process_challenge_results(
    challenge_tasks: List[ChallengeTask],
    client: httpx.AsyncClient,
    api_key: str,
    server_address: str,
    validator_hotkey: str,
    node_id: int,
    test_mode: bool = False,
    validator: Optional[InferenceValidator] = None,
    challenge_prompt: Optional[str] = None,
    challenge_id: Optional[str] = None,
    pipeline_timing: Optional[Any] = None
) -> None:
    """
    Process challenge results without blocking.
    
    In test mode, picks the first response and submits it immediately,
    skipping any evaluation or heatmap generation.
    
    When not in test mode, collects all responses, runs evaluation,
    generates heatmap, and submits the best response.
    """
    if test_mode:
        logger.info(f"[TEST MODE] Processing {len(challenge_tasks)} challenge results - will pick first response (skipping evaluation and heatmap)")
    else:
        logger.info(f"[EVALUATION MODE] Processing {len(challenge_tasks)} challenge results - will evaluate all responses and generate heatmap")
    
    # Wait for all tasks to complete with timeout
    pending = [task.task for task in challenge_tasks]
    timeout = 60  # 60s timeout
    
    # Collect all responses
    collected_responses: List[Tuple[ChallengeTask, InferenceResponse]] = []
    first_response_submitted = False
    
    logger.info(f"Waiting for {len(pending)} miner responses to complete...")
    
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
                
                # Skip if response is None (handshake failed or other error)
                if response is None:
                    challenge_task = next(ct for ct in challenge_tasks if ct.task == task)
                    logger.warning(
                        f"Response from miner {challenge_task.node_id} for challenge {challenge_task.challenge_id} "
                        f"is None (handshake failed or miner error). Skipping."
                    )
                    continue

                challenge_task = next(ct for ct in challenge_tasks if ct.task == task)
                
                # In test mode, submit first response immediately and exit
                if test_mode and not first_response_submitted:
                    logger.info(
                        f"[TEST MODE] Submitting first response from miner {challenge_task.node_id} "
                        f"for challenge {challenge_task.challenge_id} - skipping evaluation and heatmap generation"
                    )
                    
                    # Use challenge_id (UUID) as the ID for response submission
                    # challenge_id is the UUID from the challenge object
                    challenge_id_uuid = challenge_task.challenge_orig.id if hasattr(challenge_task.challenge_orig, 'id') else challenge_task.challenge_id
                    response_create = ResponseAPICreate(
                        id=challenge_id_uuid,
                        text=response.response_text,
                        original_request=challenge_task.challenge_orig
                    )
                    
                    response_api = await update_challenge_response(
                        client=client,
                        api_key=api_key,
                        server_address=server_address,
                        response_create=response_create,
                        validator_hotkey=validator_hotkey,
                        node_id=node_id
                    )
                    
                    logger.info(f"[TEST MODE] First response submitted successfully, exiting (remaining tasks cancelled)")
                    # Cancel remaining tasks
                    for remaining_task in pending:
                        remaining_task.cancel()
                    return
                else:
                    # Collect response for evaluation (non-test mode) or continue collecting (test mode)
                    collected_responses.append((challenge_task, response))
                    logger.info(
                        f"Collected response from miner {challenge_task.node_id} "
                        f"(total collected: {len(collected_responses)}/{len(challenge_tasks)})"
                    )
                    
            except Exception as e:
                logger.error(f"Error processing challenge result: {str(e)}", exc_info=True)
        
        # Log status of remaining tasks
        if pending:
            logger.info(f"Still waiting for {len(pending)} miner responses to complete...")
    
    # If in test mode and we didn't submit first response, something went wrong
    if test_mode:
        if not first_response_submitted:
            logger.warning("[TEST MODE] No responses received - cannot submit")
        else:
            logger.info("[TEST MODE] Challenge results processing complete")
        return
    
    # Non-test mode: Evaluate all collected responses
    if not collected_responses:
        logger.warning("No responses collected - cannot perform evaluation")
        return
    
    logger.info(f"[EVALUATION] Starting evaluation of {len(collected_responses)} responses for challenge {challenge_id or 'unknown'}")
    
    if not validator:
        logger.error("[EVALUATION] InferenceValidator not provided - cannot perform evaluation. Responses will not be submitted.")
        return
    
    if not challenge_prompt:
        # Try to get prompt from first challenge task
        if collected_responses:
            challenge_prompt = collected_responses[0][0].challenge_orig.prompt
        else:
            logger.error("[EVALUATION] Challenge prompt not provided - cannot perform evaluation")
            return
    
    try:
        # Extract responses and miner IDs
        responses = [resp for _, resp in collected_responses]
        miner_ids = [task.node_id for task, _ in collected_responses]
        
        logger.info(
            f"[EVALUATION] Running consensus evaluation on {len(responses)} responses "
            f"from miners {miner_ids} for challenge {challenge_id or 'unknown'}"
        )
        
        # Get challenge ID as integer (if it's a UUID string, we'll use a hash)
        challenge_id_int = hash(challenge_id) if challenge_id and not challenge_id.isdigit() else int(challenge_id) if challenge_id and challenge_id.isdigit() else 0
        
        # Get correlation_id from first challenge task (all should have the same correlation_id)
        correlation_id = None
        if collected_responses:
            correlation_id = getattr(collected_responses[0][0].challenge_orig, 'correlation_id', None) or challenge_id
        
        # Track timing: validator evaluation
        from validator.timing import PipelineStages
        if pipeline_timing:
            validator_eval_stage = pipeline_timing.add_stage(PipelineStages.VALIDATOR_EVALUATION)
        
        # Run evaluation
        consensus_score, heatmap_path, narrative, emissions = await validator.evaluate_responses(
            challenge_id=challenge_id_int,
            prompt=challenge_prompt,
            responses=responses,
            miner_ids=miner_ids,
            correlation_id=correlation_id
        )
        
        if pipeline_timing and validator_eval_stage:
            validator_eval_stage.finish()
        
        logger.info(
            f"[EVALUATION] Evaluation complete for challenge {challenge_id or 'unknown'}: "
            f"consensus_score={consensus_score:.3f}, "
            f"heatmap={'generated' if heatmap_path else 'not generated'}, "
            f"narrative={'generated' if narrative else 'not generated'}, "
            f"emissions={len(emissions)} miners"
        )
        
        # Find the best response (highest emission score)
        best_miner_id = max(emissions.items(), key=lambda x: x[1])[0] if emissions else str(miner_ids[0])
        
        # Get challenge_id UUID for submission
        challenge_id_uuid = collected_responses[0][0].challenge_orig.id if hasattr(collected_responses[0][0].challenge_orig, 'id') else challenge_id
        
        # Build response batch with ALL responses (F3)
        miner_response_data_list = []
        for task, response in collected_responses:
            # Extract usage from miner response if available
            usage = TokenUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0)
            if hasattr(response, 'usage') and response.usage:
                if isinstance(response.usage, dict):
                    usage = TokenUsage(**response.usage)
                elif hasattr(response.usage, 'prompt_tokens'):
                    usage = TokenUsage(
                        prompt_tokens=response.usage.prompt_tokens or 0,
                        completion_tokens=response.usage.completion_tokens or 0,
                        total_tokens=response.usage.total_tokens or 0
                    )
            
            # Get miner identifier (hotkey or UID)
            miner_id_str = str(task.miner_hotkey) if hasattr(task, 'miner_hotkey') and task.miner_hotkey else str(task.node_id)
            
            miner_response_data_list.append(MinerResponseData(
                miner_id=miner_id_str,
                miner_uid=task.node_id,
                text=response.response_text,
                tool_calls=getattr(response, 'tool_calls', None),
                finish_reason=getattr(response, 'finish_reason', 'stop'),
                response_time_ms=response.response_time_ms,
                usage=usage
            ))
        
        # Build evaluation result
        evaluation_result = EvaluationResult(
            consensus_score=consensus_score,
            emissions=emissions,
            best_miner_id=best_miner_id,
            heatmap_path=heatmap_path,
            narrative=narrative
        )
        
        # Build the full batch
        response_batch = ResponseBatchSubmit(
            challenge_id=challenge_id_uuid,
            responses=miner_response_data_list,
            evaluation=evaluation_result,
            validator_hotkey=validator_hotkey,
            original_request=collected_responses[0][0].challenge_orig
        )
        
        logger.info(
            f"[EVALUATION] Submitting batch of {len(miner_response_data_list)} responses "
            f"for challenge {challenge_id_uuid} (best: {best_miner_id})"
        )
        
        # Track timing: validator send to API
        if pipeline_timing:
            validator_send_stage = pipeline_timing.add_stage(PipelineStages.VALIDATOR_SEND_TO_API)
        
        # Submit the entire batch to Challenge API
        success = await submit_response_batch(
            client=client,
            api_key=api_key,
            server_address=server_address,
            batch=response_batch,
            pipeline_timing=pipeline_timing
        )
        
        if pipeline_timing and validator_send_stage:
            validator_send_stage.finish()
        
        if success:
            logger.info(f"[EVALUATION] Successfully submitted response batch to Challenge API for challenge {challenge_id or 'unknown'}")
        else:
            logger.error(
                f"[EVALUATION] Failed to submit response batch to Challenge API for challenge {challenge_id or 'unknown'}. "
                f"Challenge API will use Fluvio to notify gateway of the best response."
            )
            
    except Exception as e:
        logger.error(f"[EVALUATION] Error during evaluation: {str(e)}", exc_info=True)
        # Fallback: submit first response without evaluation using legacy method
        logger.warning("[EVALUATION] Falling back to submitting first response without evaluation (legacy)")
        if collected_responses:
            best_task, best_response = collected_responses[0]
            challenge_id_uuid = best_task.challenge_orig.id if hasattr(best_task.challenge_orig, 'id') else best_task.challenge_id
            response_create = ResponseAPICreate(
                id=challenge_id_uuid,
                text=best_response.response_text,
                original_request=best_task.challenge_orig
            )
            await update_challenge_response(
                client=client,
                api_key=api_key,
                server_address=server_address,
                response_create=response_create,
                validator_hotkey=validator_hotkey,
                node_id=best_task.node_id
            )
    
    logger.info("[EVALUATION] All challenge results processed")

