#!/usr/bin/env python3
import random
import hashlib
import httpx
import asyncio
from datetime import datetime
from typing import Optional, Dict, Any

subjects = ["astronomy", "cooking", "sports", "robotics", "history"]
styles = ["explain like I'm 5", "write a poem", "summarize", "debate", "argue against"]
tones = ["funny", "serious", "technical", "emotional", "sarcastic"]

def generate_prompt():
    subject = random.choice(subjects)
    style = random.choice(styles)
    tone = random.choice(tones)
    return f"{style.capitalize()} the topic of {subject} in a {tone} tone."

def generate_hash64() -> str:
    """Generate a random 64-character hash string."""
    # Generate random bytes and create a hash
    random_bytes = random.getrandbits(256).to_bytes(32, 'big')
    hash_obj = hashlib.sha256(random_bytes)
    return hash_obj.hexdigest()

async def post_challenge(options: Dict[str, Any], api_key: str, url: str = "http://localhost:8080/challenge") -> Optional[Dict[str, Any]]:
    """
    Post a challenge to the challenge API.
    
    Args:
        options: Dictionary containing challenge options with 'model' key
        api_key: API key for authentication
        url: Challenge API URL (default: http://localhost:8080/challenge)
    
    Returns:
        Dict containing the response if successful, None otherwise
    """
    try:
        # Generate random hash64 ID
        challenge_id = generate_hash64()
        
        # Generate random prompt
        prompt = generate_prompt()
        
        # Generate random values if not provided
        temperature = options.get("temperature")
        if temperature is None:
            temperature = round(random.uniform(0.1, 1.0), 2)
        
        top_p = options.get("top_p")
        if top_p is None:
            top_p = round(random.uniform(0.8, 1.0), 2)
        
        max_tokens = options.get("max_tokens")
        if max_tokens is None:
            max_tokens = random.randint(50, 512)
        
        model = options.get("model")
        if model is None:
            models = ["claude-3-haiku", "llama-3-8b"]
            model = random.choice(models)
        
        # Create challenge payload
        challenge_data = {
            "id": challenge_id,
            "prompt": prompt,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "metadata": {
                "model": model
            }
        }
        
        # Prepare headers
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        # Send POST request
        async with httpx.AsyncClient() as client:
            response = await client.post(
                url,
                json=challenge_data,
                headers=headers,
                timeout=30.0
            )
        
        if response.status_code == 201:
            return response.json()
        else:
            print(f"Error posting challenge: {response.status_code} - {response.text}")
            return None
            
    except Exception as e:
        print(f"Error posting challenge: {str(e)}")
        return None

async def example_usage():
    """Example usage of the post_challenge function."""
    # Example with specific options
    options_specific = {
        "model": "microsoft/Phi3-512", #this is just an example, you can use any model so long as the api is OpenAI-compatible
        "temperature": 0.7,
        "top_p": 0.95,
        "max_tokens": 128
    }
    
    # Example with minimal options (will use random values for missing ones)
    options_minimal = {
        "model": "claude-3-haiku"
        # temperature, top_p, max_tokens will be randomly generated
    }
    
    # Example with no options (all values will be random)
    options_empty = {}
    
    # Example API key (replace with actual key)
    api_key = "anonymous"
    
    # Post challenge with specific options
    print("Posting challenge with specific options...")
    result = await post_challenge(options_specific, api_key)
    if result:
        print("Challenge posted successfully:")
        print(result)
    else:
        print("Failed to post challenge")
    
    # Post challenge with minimal options
    print("\nPosting challenge with minimal options (random values)...")
    result = await post_challenge(options_minimal, api_key)
    if result:
        print("Challenge posted successfully:")
        print(result)
    else:
        print("Failed to post challenge")
    
    # Post challenge with no options (all random)
    print("\nPosting challenge with no options (all random values)...")
    result = await post_challenge(options_empty, api_key)
    if result:
        print("Challenge posted successfully:")
        print(result)
    else:
        print("Failed to post challenge")

# Run example if this file is executed directly
if __name__ == "__main__":
    print("Generated prompt:", generate_prompt())
    print("Generated hash64:", generate_hash64())
    
    # Uncomment to test the post_challenge function
    asyncio.run(example_usage())

# POST request to the challenge API like curl:
#
# curl -X POST http://localhost:8080/challenge   -H "Authorization: Bearer your-api-key"   -H "Content-Type: application/json"   -d '{
#     "id": "c-002",
#     "prompt": "Write a haiku about programming",
#     "temperature": 0.7,
#     "top_p": 0.95,
#     "max_tokens": 128,
#     "metadata": {"model": "microsoft/Phi3-512"}
#   }'|jq
# {
#   "id": "c-002",
#   "prompt": "Write a haiku about programming",
#   "temperature": 0.7,
#   "top_p": 0.95,
#   "max_tokens": 128,
#   "metadata": {
#     "model": "microsoft/Phi3-512"
#   },
#   "created_at": "2025-10-03T21:23:58.057959",
#   "status": "available",
#   "requester": null
# }