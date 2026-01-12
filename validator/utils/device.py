"""
Device utilities for validator.
Simple device detection utility.
"""

import torch
from typing import Optional


def get_optimal_device(device: Optional[str] = None) -> str:
    """
    Get the optimal device for computation.
    
    Args:
        device: Optional device string ('cpu', 'cuda', etc.)
        
    Returns:
        Device string ('cpu' or 'cuda')
    """
    if device is not None:
        return device
    
    # Check if CUDA is available
    if torch.cuda.is_available():
        return "cuda"
    
    return "cpu"


