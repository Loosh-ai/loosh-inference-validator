# InferenceSynapse Implementation

import bittensor as bt
import pydantic
from typing import List, Tuple
from validator.utils.inference_endpoint import inference as endpoint_inference, InferenceRequest, get_config_dependency

bt.trace()

class InferenceSynapse(bt.Synapse):
    """
    Represents the core component of a Synapse for handling LLM inference requests.
    
    The InferenceSynapse captures inference parameters and can generate a completion based on the current state.
    It contains unique hashes for prompts to ensure integrity and uniqueness.
    
    Attributes:
        prompt (str): The input prompt for inference. Immutable post-instantiation.
        model (str): The model to use for inference. Immutable post-instantiation.
        max_tokens (int): Maximum tokens to generate. Immutable post-instantiation.
        temperature (float): Temperature for generation. Immutable post-instantiation.
        top_p (float): Top-p value for generation. Immutable post-instantiation.
        completion (str): A field to store the completion or result after processing.
    """
    class Config: 
        validate_assignment = True
    
    def deserialize(self): 
        return self.completion

    prompt: str = pydantic.Field(..., allow_mutation=False)
    model: str = pydantic.Field(..., allow_mutation=False)
    max_tokens: int = pydantic.Field(..., allow_mutation=False)
    temperature: float = pydantic.Field(..., allow_mutation=False)
    top_p: float = pydantic.Field(..., allow_mutation=False)
    completion: str = None

    #@property
    #def required_hash_fields(self) -> List[str]:
    #    """ Returns the list of fields that are essential for hash computation. """
    #    return ['prompt']

simulate_inference = False

async def inference(synapse: InferenceSynapse) -> InferenceSynapse:
    """
    Process the provided synapse to generate a completion.

    Args:
        synapse (InferenceSynapse): The input synapse to be processed.

    Returns:
        InferenceSynapse: The updated synapse with a completion.
    """
    bt.logging.debug("In inference!")

    validator_hotkey = ""

    if simulate_inference:
        # Call the endpoint inference function
        # TODO: review this
        request = InferenceRequest(prompt=synapse.prompt, model=synapse.model, max_tokens=synapse.max_tokens, temperature=synapse.temperature, top_p=synapse.top_p)
        response = await endpoint_inference(request=request, validator_hotkey=validator_hotkey, config=get_config_dependency())
        #print(response['response_text'])
        synapse.completion = response['response_text']
    else:
        synapse.completion = "I am a bittensor inference node"

    return synapse

def blacklist(synapse: InferenceSynapse) -> Tuple[bool, str]:
    """
    Determines if the provided synapse should be blacklisted.

    Args:
        synapse (InferenceSynapse): The input synapse to be evaluated.

    Returns:
        Tuple[bool, str]: A tuple containing a boolean that indicates whether the synapse is blacklisted,
                          and a string providing the reason.
    """
    return False, ""


def priority(synapse: InferenceSynapse) -> float:
    """
    Determines the priority of the provided synapse.

    Args:
        synapse (InferenceSynapse): The input synapse to be evaluated.

    Returns:
        float: The priority value of the synapse, with higher values indicating higher priority.
    """
    return 0.0
