"""Relation block for computing relation scores between query and support."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class RelationBlock(nn.Module):
    """
    Relation module that learns to compare feature pairs.
    
    From: Sung et al. "Learning to Compare: Relation Network for Few-Shot Learning" (CVPR 2018)
    
    Takes concatenated feature pairs and outputs relation logits.
    """
    
    def __init__(self, input_size=128, hidden_size=8):
        """
        Args:
            input_size: Input feature dimension (2 * feature_dim from concat)
            hidden_size: Hidden dimension for relation module
        """
        super(RelationBlock, self).__init__()
        
        # Relation module: learns to compare concatenated features
        self.layer1 = nn.Sequential(
            nn.Conv2d(input_size, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Calculate flattened size after convolutions
        # Assuming input is 4x4 from encoder, after 2 maxpools: 4->2->1
        self.fc1 = nn.Linear(64 * 1 * 1, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        """
        Args:
            x: (B, input_size, H, W) concatenated feature pairs
        Returns:
            scores: (B, 1) relation logits (for compatibility with CELoss)
        """
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)  # Flatten
        out = F.relu(self.fc1(out))
        out = self.fc2(out)  # Logits (no sigmoid)
        return out
