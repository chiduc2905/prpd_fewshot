"""Matching Networks encoder - same as ProtoNet paper encoder."""
import torch.nn as nn


class MatchingNetEncoder(nn.Module):
    """
    4-layer CNN encoder for Matching Networks.
    
    Architecture from: Vinyals et al. "Matching Networks for One Shot Learning" (NIPS 2016)
    - Uses same CNN structure as Prototypical Networks
    - 4 conv blocks: Conv(3->64) + BatchNorm + ReLU + MaxPool(2x2)
    - Input: (B, 3, 64, 64) -> Output: (B, 1024) flattened features
    
    Used by: MatchingNet
    """
    
    def __init__(self):
        super(MatchingNetEncoder, self).__init__()
        
        def conv_block(in_channels, out_channels):
            """Standard conv block: Conv -> BatchNorm -> ReLU -> MaxPool"""
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )
        
        self.features = nn.Sequential(
            conv_block(3, 64),      # 64x64 -> 32x32
            conv_block(64, 64),     # 32x32 -> 16x16
            conv_block(64, 64),     # 16x16 -> 8x8
            conv_block(64, 64),     # 8x8 -> 4x4
        )
        
    def forward(self, x):
        """
        Args:
            x: (B, 3, 64, 64) input images
        Returns:
            (B, 1024) flattened features (64 * 4 * 4)
        """
        feat = self.features(x)  # (B, 64, 4, 4)
        return feat.view(feat.size(0), -1)  # (B, 1024)
