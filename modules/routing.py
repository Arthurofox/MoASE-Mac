import torch
import torch.nn as nn

class Router(nn.Module):
    """
    Router module for Mixture-of-Experts.

    This module routes the input activations to multiple experts based on
    the gating weights produced by the Domain-Aware Gate (DAG). Each expert
    processes the input, and their outputs are weighted (via a softmax distribution)
    and aggregated.

    Args:
        experts (nn.ModuleList): A list (or ModuleList) of expert modules.
    """
    def __init__(self, experts: nn.ModuleList):
        super(Router, self).__init__()
        self.experts = experts

    def forward(self, x: torch.Tensor, gate_weights: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the Router.

        Args:
            x (torch.Tensor): Input tensor to be routed. This could be features from the backbone.
            gate_weights (torch.Tensor): Tensor of shape (B, num_experts) representing the
                soft assignment of each sample to the experts (e.g., from gumbel-softmax).

        Returns:
            torch.Tensor: Aggregated output tensor after routing.
        """
        # Ensure that the number of experts matches the gate_weights' second dimension.
        assert gate_weights.shape[1] == len(self.experts), "Mismatch between number of experts and gate weights"
        outputs = []
        for i, expert in enumerate(self.experts):
            expert_output = expert(x)
            # Expand the corresponding gate weight so that it can be broadcast with expert_output.
            # If expert_output is of shape (B, ...), we unsqueeze gate weight to shape (B, 1, 1, ..., 1)
            weight = gate_weights[:, i]
            for _ in range(expert_output.dim() - 1):
                weight = weight.unsqueeze(-1)
            outputs.append(expert_output * weight)
        # Sum the weighted outputs from all experts.
        routed_output = sum(outputs)
        return routed_output

