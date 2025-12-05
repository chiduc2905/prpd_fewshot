"""Shared CNN backbone for few-shot models."""
import torch.nn as nn
import functools


def get_norm_layer(norm_type='group'):
    """Get normalization layer by name."""
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=True)
    elif norm_type == 'group':
        # Use 8 groups for 64 channels (8 channels per group)
        norm_layer = functools.partial(nn.GroupNorm, 8)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer

class Conv64F_Encoder(nn.Module):
    """4-layer CNN encoder. Input: 3x64x64 -> Output: 64x16x16."""
    
    def __init__(self, norm_layer=functools.partial(nn.GroupNorm, 8)):
        super(Conv64F_Encoder, self).__init__()
        # GroupNorm and BatchNorm use bias=False in conv, others use bias=True
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.features = nn.Sequential(                       
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=use_bias),
            norm_layer(64),
            nn.LeakyReLU(0.2, True),
            nn.MaxPool2d(kernel_size=2, stride=2),           
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=use_bias),
            norm_layer(64),
            nn.LeakyReLU(0.2, True),
            nn.MaxPool2d(kernel_size=2, stride=2),           
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=use_bias),
            norm_layer(64),
            nn.LeakyReLU(0.2, True),                         
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=use_bias),
            norm_layer(64),
            nn.LeakyReLU(0.2, True),
        )
        
    def forward(self, x):
        return self.features(x)


class Conv64F_Paper_Encoder(nn.Module):
    """
    Standard 4-layer CNN encoder matching official ProtoNet paper.
    
    Architecture from: Snell et al. "Prototypical Networks for Few-shot Learning" (NeurIPS 2017)
    - 4 conv blocks: Conv(3->64) + BatchNorm + ReLU + MaxPool(2x2)
    - Input: (B, 3, 64, 64) -> Output: (B, 1024) flattened features
    
    This matches the official implementation:
    https://github.com/jakesnell/prototypical-networks
    """
    
    def __init__(self):
        super(Conv64F_Paper_Encoder, self).__init__()
        
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
