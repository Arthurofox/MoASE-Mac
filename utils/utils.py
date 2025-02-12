import torch
import torch.nn as nn
from typing import Tuple, Dict
from einops import rearrange, reduce, repeat
import math

def get_activation_patterns(x: torch.Tensor, threshold: float = 0.5) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Split activations into high and low based on spatial patterns.

    This function takes an activation tensor (which could be of shape (B, T, F) or (B, C, H, W))
    and returns two boolean masks:
      - high_mask: where activations are above the threshold.
      - low_mask: where activations are at or below the threshold.

    Args:
        x (torch.Tensor): Input activation tensor.
        threshold (float): Threshold value to distinguish high and low activations.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: (high_mask, low_mask) as boolean tensors.
    """
    if x.dim() == 4:
        # Assume shape (B, C, H, W); flatten spatial dimensions to (B, C, H*W)
        B, C, H, W = x.shape
        x_flat = rearrange(x, "b c h w -> b c (h w)")
    elif x.dim() == 3:
        # Assume shape (B, T, F)
        x_flat = x
    else:
        # For tensors with other shapes, operate directly
        x_flat = x

    high_mask = x_flat > threshold
    low_mask = ~high_mask
    return high_mask, low_mask

def compute_homeostatic_stats(x: torch.Tensor, threshold: float = 0.5) -> Dict[str, torch.Tensor]:
    """
    Compute activation statistics for homeostatic regulation.

    This function calculates key statistics such as:
      - Mean activation.
      - Standard deviation.
      - Ratio of high activations.
      - Ratio of low activations.

    These statistics help monitor and regulate the adaptation process
    to avoid catastrophic forgetting during continual test-time adaptation.

    Args:
        x (torch.Tensor): Input activation tensor.
        threshold (float): Threshold to define high and low activations.

    Returns:
        Dict[str, torch.Tensor]: Dictionary with computed statistics:
            - "mean": Mean of activations.
            - "std": Standard deviation of activations.
            - "high_ratio": Fraction of activations above the threshold.
            - "low_ratio": Fraction of activations at or below the threshold.
    """
    mean_activation = torch.mean(x)
    std_activation = torch.std(x)
    high_mask, low_mask = get_activation_patterns(x, threshold)
    high_ratio = high_mask.float().mean()
    low_ratio = low_mask.float().mean()
    return {
        "mean": mean_activation,
        "std": std_activation,
        "high_ratio": high_ratio,
        "low_ratio": low_ratio
    }

def spatial_pool(x: torch.Tensor, pool_type: str = "mean", dim: int = -1) -> torch.Tensor:
    """
    Pool the tensor spatially using a specified method.

    Args:
        x (torch.Tensor): Input tensor.
        pool_type (str): Type of pooling ('mean', 'sum', or 'max').
        dim (int): Dimension over which to pool.

    Returns:
        torch.Tensor: Pooled tensor.
    """
    if pool_type == "mean":
        return torch.mean(x, dim=dim)
    elif pool_type == "sum":
        return torch.sum(x, dim=dim)
    elif pool_type == "max":
        return torch.max(x, dim=dim)[0]
    else:
        raise ValueError(f"Unknown pool_type: {pool_type}")

def reshape_for_attention(x: torch.Tensor, num_heads: int) -> torch.Tensor:
    """
    Reshape tensor for multi-head attention.

    Args:
        x (torch.Tensor): Input tensor of shape (B, T, F).
        num_heads (int): Number of attention heads.

    Returns:
        torch.Tensor: Reshaped tensor of shape (B, num_heads, T, F_head) 
                      where F_head = F // num_heads.
    """
    B, T, F = x.shape
    assert F % num_heads == 0, "Feature dimension must be divisible by num_heads"
    F_head = F // num_heads
    x = rearrange(x, "b t (h f) -> b h t f", h=num_heads, f=F_head)
    return x

def compute_attention_scores(query: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
    """
    Compute scaled dot-product attention scores.

    Args:
        query (torch.Tensor): Query tensor of shape (B, num_heads, T, F_head).
        key (torch.Tensor): Key tensor of shape (B, num_heads, T, F_head).

    Returns:
        torch.Tensor: Attention scores of shape (B, num_heads, T, T).
    """
    d_k = query.shape[-1]
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    return scores

# Additional utility functions for tensor manipulation can be added here as needed for MoASE.