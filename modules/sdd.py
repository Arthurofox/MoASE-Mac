import torch
import torch.nn as nn
from typing import Optional

class SDDSTEFunction(torch.autograd.Function):
    """
    Custom autograd Function for Spatial Differentiate Dropout with
    gradient scaling in the backward pass. (STE = Straight-Through Estimator)
    """
    @staticmethod
    def forward(ctx,
                x: torch.Tensor,
                q: float,
                largest: bool,
                dim: int,
                inplace: bool,
                scale_masked: float,
                scale_unmasked: float):
        """
        Forward pass:
        1. Keeps top-K (or bottom-K) elements along `dim`, zeros out others.
        2. Saves the mask and scaling factors for backward pass.
        """
        # Convert negative indexing
        if dim < 0:
            dim = x.dim() + dim

        # Determine K
        num_elements = x.shape[dim]
        K = int(num_elements * q)

        # Edge case: if K <= 0, everything gets zeroed
        if K <= 0:
            out = torch.zeros_like(x)
            mask = torch.zeros_like(x, dtype=torch.bool)
            ctx.save_for_backward(mask)
            ctx.dim = dim
            ctx.inplace = inplace
            ctx.scale_masked = scale_masked
            ctx.scale_unmasked = scale_unmasked
            return out

        # topk for largest/bottom-k
        topk_vals, _ = torch.topk(x, K, dim=dim, largest=largest)

        if largest:
            boundary_vals = torch.min(topk_vals, dim=dim, keepdim=True)[0]
            mask = x >= boundary_vals
        else:
            boundary_vals = torch.max(topk_vals, dim=dim, keepdim=True)[0]
            mask = x <= boundary_vals

        if inplace:
            out = x
            out = out * mask
        else:
            out = x * mask

        # Save for backward
        ctx.save_for_backward(mask)
        ctx.dim = dim
        ctx.inplace = inplace
        ctx.scale_masked = scale_masked
        ctx.scale_unmasked = scale_unmasked

        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """
        Backward pass:
        - Straight-through approach: pass gradients for both masked and unmasked
          elements, but scale them differently to avoid vanishing gradients.
        """
        (mask,) = ctx.saved_tensors
        dim = ctx.dim
        inplace = ctx.inplace
        scale_masked = ctx.scale_masked
        scale_unmasked = ctx.scale_unmasked

        grad_input = grad_output.clone()

        # Scale for unmasked elements (the ones that survived in forward)
        grad_input[mask] = grad_input[mask] * scale_unmasked
        # Scale for masked elements
        grad_input[~mask] = grad_input[~mask] * scale_masked

        return grad_input, None, None, None, None, None, None

class SpatialDifferentiateDropout(nn.Module):
    """
    Spatial Differentiate Dropout (SDD) with gradient scaling to mitigate vanishing gradients.

    In forward pass, keeps top-K or bottom-K activations along dimension `dim`.
    In backward pass, applies different scaling factors for masked vs. unmasked elements.

    Args:
        q (float): Fraction of elements to keep (0 < q <= 1).
        largest (bool): If True, keep top-K. If False, keep bottom-K.
        dim (int): Dimension along which to apply SDD. Default is 1.
        inplace (bool): If True, zero out in-place.
        scale_masked (float): Gradient scale factor for masked-out elements in backward.
        scale_unmasked (float): Gradient scale factor for unmasked elements in backward.
    """

    def __init__(
        self,
        q: float = 0.5,
        largest: bool = True,
        dim: int = 1,
        inplace: bool = False,
        scale_masked: float = 0.0,
        scale_unmasked: float = 1.0,
    ):
        super().__init__()
        assert 0 < q <= 1.0, "q must be in (0, 1]."
        self.q = q
        self.largest = largest
        self.dim = dim
        self.inplace = inplace
        self.scale_masked = scale_masked
        self.scale_unmasked = scale_unmasked

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return SDDSTEFunction.apply(
            x,
            self.q,
            self.largest,
            self.dim,
            self.inplace,
            self.scale_masked,
            self.scale_unmasked
        )

    def extra_repr(self) -> str:
        return (
            f"q={self.q}, largest={self.largest}, dim={self.dim}, "
            f"inplace={self.inplace}, scale_masked={self.scale_masked}, "
            f"scale_unmasked={self.scale_unmasked}"
        )

