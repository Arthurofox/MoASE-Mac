import torch
import torch.nn as nn
import torch.nn.functional as F

class Backbone(nn.Module):
    def __init__(self, num_classes: int = 10):
        """
        A lightweight CNN backbone with test-time adaptation hooks.
        
        Args:
            num_classes (int): Number of output classes.
        """
        super(Backbone, self).__init__()
        self.adapt_mode = False  # Flag to indicate if adaptation mode is enabled
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # (B, 16, 16, 16) for 32x32 input
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # (B, 32, 8, 8)
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))  # (B, 64, 1, 1)
        )
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        """
        Forward pass.
        - When adaptation mode is enabled, the network is set to train() to update BN stats.
        - Otherwise, the network is in eval() mode.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, 3, H, W)
        
        Returns:
            logits (torch.Tensor): Output predictions of shape (B, num_classes)
            features (torch.Tensor): Extracted features (B, 64)
        """
        if self.adapt_mode:
            self.train()  # Allow BN layers to update statistics
        else:
            self.eval()   # Fix BN layers during inference

        features = self.features(x)
        features = features.view(features.size(0), -1)  # Flatten to (B, 64)
        logits = self.classifier(features)
        return logits, features

    def enable_adaptation(self):
        """
        Enable test-time adaptation mode.
        Optionally, freeze classifier parameters so that only BN layers (or other adaptation hooks) update.
        """
        self.adapt_mode = True
        for param in self.classifier.parameters():
            param.requires_grad = False

    def disable_adaptation(self):
        """
        Disable test-time adaptation mode.
        Unfreeze classifier parameters if needed.
        """
        self.adapt_mode = False
        for param in self.classifier.parameters():
            param.requires_grad = True

    def test_time_adaptation_step(self, x: torch.Tensor, loss_fn: nn.Module, optimizer: torch.optim.Optimizer) -> float:
        """
        Perform one test-time adaptation step.
        The method uses the current model predictions as pseudo-labels and performs a gradient update.
        
        Args:
            x (torch.Tensor): Input tensor.
            loss_fn (nn.Module): Loss function to compute the pseudo-label loss (e.g., CrossEntropyLoss).
            optimizer (torch.optim.Optimizer): Optimizer for updating the model parameters.
            
        Returns:
            float: Loss value computed for the adaptation step.
        """
        # Ensure the model is in adaptation mode.
        self.train()
        logits, features = self.forward(x)
        pseudo_labels = torch.argmax(logits, dim=1)
        loss = loss_fn(logits, pseudo_labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        return loss.item()
