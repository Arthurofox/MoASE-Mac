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


if __name__ == "__main__":
    # Test the experts with dummy data
    
    # For ExpertConv: simulate an image tensor (B, C, H, W)
    conv_expert = ExpertConv(in_channels=3, out_channels=16, kernel_size=3)
    dummy_image = torch.randn(2, 3, 32, 32)
    conv_output = conv_expert(dummy_image)
    print("ExpertConv output shape:", conv_output.shape)  # Expected: (2, 16, 32, 32)
    
    # For ExpertMLP: simulate flattened features
    # Let's assume the feature dimension is 128, and we want to transform it to 64
    mlp_expert = ExpertMLP(input_dim=128, hidden_dim=256, output_dim=64)
    dummy_features = torch.randn(2, 128)
    mlp_output = mlp_expert(dummy_features)
    print("ExpertMLP output shape:", mlp_output.shape)  # Expected: (2, 64)
