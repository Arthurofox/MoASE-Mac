# gates.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class DomainAwareGate(nn.Module):
    """
    Domain-Aware Gate (DAG)

    This gate leverages domain-specific features to produce a routing distribution over experts.
    It includes layer normalization and temperature annealing for better adaptation, and uses
    gumbel-softmax to provide crisp, differentiable gradients.

    Args:
        input_dim (int): The dimension of the domain-related features.
        num_experts (int): Number of experts to route to.
        hidden_dim (int): Hidden dimension for the internal MLP. Defaults to 64.
    """
    def __init__(self, input_dim: int, num_experts: int, hidden_dim: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(hidden_dim, num_experts)
        # Temperature annealing for gumbel-softmax
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, domain_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the Domain-Aware Gate.

        Args:
            domain_features (torch.Tensor): Tensor of shape (B, input_dim) containing domain-specific information.

        Returns:
            torch.Tensor: Routing weights of shape (B, num_experts) computed using gumbel-softmax.
        """
        x = self.fc1(domain_features)
        x = self.layer_norm(x)
        x = self.relu(x)
        logits = self.fc2(x)
        # Use gumbel-softmax with temperature annealing for crisp gradients
        gate_weights = F.gumbel_softmax(logits / self.temperature, tau=1.0, hard=False, dim=-1)
        return gate_weights

class ActivationSparsityGate(nn.Module):
    """
    Activation Sparsity Gate (ASG)

    This gate produces threshold adjustments for each expert, enabling dynamic activation sparsity.
    
    Args:
        input_dim (int): Dimension of the pooled or flattened activation features.
        num_experts (int): Number of experts to generate threshold adjustments for.
        hidden_dim (int): Hidden dimension for the internal MLP. Defaults to 64.
        scale (float): Scaling factor for the threshold adjustments. Defaults to 0.1.
    """
    def __init__(self, input_dim: int, num_experts: int, hidden_dim: int = 64, scale: float = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_experts)
        )
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the Activation Sparsity Gate.

        Args:
            x (torch.Tensor): Activation tensor of shape (B, input_dim), typically pooled features.

        Returns:
            torch.Tensor: Threshold adjustments of shape (B, num_experts).
                         These adjustments can be used to modify the fraction q for SDD.
        """
        raw_thresholds = self.mlp(x)  # (B, num_experts)
        # Bound the output using tanh and apply scaling
        threshold_adjustments = torch.tanh(raw_thresholds) * self.scale
        return threshold_adjustments
