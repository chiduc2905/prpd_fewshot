"""Prototypical Network for few-shot learning."""
import torch
import torch.nn as nn
from net.encoders.base_encoder import Conv64F_Encoder
from net.encoders.protonet_encoder import Conv64F_Paper_Encoder
from net.utils import init_weights


class ProtoNet(nn.Module):
    """Few-shot classifier using prototype-based Euclidean distance."""
    
    def __init__(self, encoder_type='default', init_type='kaiming', device='cuda'):
        """Initialize ProtoNet with encoder selection.
        
        Args:
            encoder_type: 'default' (Conv64F_Encoder with GroupNorm) or 
                         'paper' (Conv64F_Paper_Encoder matching official implementation)
            init_type: Weight initialization type
            device: Device to use
        """
        super(ProtoNet, self).__init__()
        
        if encoder_type == 'paper':
            self.encoder = Conv64F_Paper_Encoder()  # Output: (B, 1024) flattened
            self.use_pooling = False  # Paper encoder already flattens
        else:
            self.encoder = Conv64F_Encoder()  # Output: (B, 64, H, W) feature maps
            self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
            self.use_pooling = True  # Need to pool and flatten
        
        init_weights(self, init_type=init_type)
        self.to(device)

    def forward(self, query, support):
        """Compute negative Euclidean distance to prototypes.
        
        Args:
            query: (B, NQ, C, H, W) query images
            support: (B, Way, Shot, C, H, W) support images
        Returns:
            scores: (B*NQ, Way) negative distance
        """
        B, NQ, C, H, W = query.size()
        B_s, Way, Shot, C_s, H_s, W_s = support.size()
        
        # Flatten and encode
        query_flat = query.view(-1, C, H, W)
        support_flat = support.view(-1, C, H, W)
        
        q_feat = self.encoder(query_flat)
        s_feat = self.encoder(support_flat)
        
        # Flatten features (pooling if needed)
        if self.use_pooling:
            q_feat = self.avg_pool(q_feat).view(q_feat.size(0), -1)
            s_feat = self.avg_pool(s_feat).view(s_feat.size(0), -1)
        # else: paper encoder already returns flattened features
        
        # Compute prototypes
        s_feat = s_feat.view(B, Way, Shot, -1)
        prototypes = s_feat.mean(dim=2)
        
        # Compute squared Euclidean distances (as in official ProtoNet paper)
        q_feat = q_feat.view(B, NQ, -1)
        dists = torch.cdist(q_feat, prototypes).pow(2)  # Squared Euclidean distance
        
        # Return negative distance as scores
        scores = -dists.view(-1, Way)
        return scores
