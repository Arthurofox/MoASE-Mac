import torch
import torch.nn as nn
import torch.nn.functional as F

class ExpertConv(nn.Module):
    """
    Convolution-based Expert Module

    This expert processes input features with convolutional layers,
    making it effective at capturing local spatial patterns.
    
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int): Size of the convolutional kernels (default: 3).
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super(ExpertConv, self).__init__()
        padding = kernel_size // 2
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expecting x of shape (B, C, H, W)
        return self.conv_layers(x)


class ExpertMLP(nn.Module):
    """
    MLP-based Expert Module

    This expert processes input features with fully connected layers.
    It is designed to capture global, non-local patterns from flattened features.
    
    Args:
        input_dim (int): Dimension of the flattened input features.
        hidden_dim (int): Dimension of the hidden layer.
        output_dim (int): Dimension of the output features.
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super(ExpertMLP, self).__init__()
        self.mlp_layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # If x has more than 2 dimensions, flatten all dimensions except batch.
        if x.dim() > 2:
            x = x.flatten(start_dim=1)
        return self.mlp_layers(x)