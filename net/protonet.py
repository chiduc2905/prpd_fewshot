import torch
import torch.nn as nn
from net.encoder import Conv64F_Encoder
from net.utils import init_weights

class ProtoNet(nn.Module):
    def __init__(self, init_type='normal', use_gpu=True):
        super(ProtoNet, self).__init__()
        self.encoder = Conv64F_Encoder()
        # ProtoNet usually uses the output of the encoder directly or with a flattened layer
        # The encoder outputs (B, 64, H', W'). We need a vector.
        # Typically ProtoNet does global average pooling or flattening.
        # Conv64F_Encoder output is 64 channels, spatial dims depend on input.
        # We'll add a global average pooling and maybe a linear layer if needed.
        # But standard ProtoNet on Conv64 is just Flatten -> Vector.
        # Let's add an AdaptiveAvgPool to get 1x1 spatial, then flatten.
        
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, 64) # Optional, often just the features are used. keeping for compatibility with 64-dim embedding.
        
        init_weights(self, init_type=init_type)
        if use_gpu and torch.cuda.is_available():
            self.cuda()

    def forward(self, query, support):
        """
        Args:
            query: (B, NQ, C, H, W)
            support: (B, Way, Shot, C, H, W)
        Returns:
            scores: (B * NQ, Way) - Negative Euclidean distance
        """
        B, NQ, C, H, W = query.size()
        B_s, Way, Shot, C_s, H_s, W_s = support.size()
        
        # Flatten query and support to pass through encoder
        # query: (B*NQ, C, H, W)
        query_flat = query.view(-1, C, H, W)
        # support: (B*Way*Shot, C, H, W)
        support_flat = support.view(-1, C, H, W)
        
        # Encode
        q_feat = self.encoder(query_flat) # (B*NQ, 64, h, w)
        s_feat = self.encoder(support_flat) # (B*Way*Shot, 64, h, w)
        
        # Pool and Flatten
        q_feat = self.avg_pool(q_feat).view(q_feat.size(0), -1) # (B*NQ, 64)
        s_feat = self.avg_pool(s_feat).view(s_feat.size(0), -1) # (B*Way*Shot, 64)
        
        # Optional FC
        # q_feat = self.fc(q_feat)
        # s_feat = self.fc(s_feat)
        
        # Reshape support to compute prototypes
        # s_feat: (B, Way, Shot, Feature_Dim)
        s_feat = s_feat.view(B, Way, Shot, -1)
        
        # Compute prototypes: Mean over Shot dimension
        prototypes = s_feat.mean(dim=2) # (B, Way, Feature_Dim)
        
        # Reshape query to (B, NQ, Feature_Dim)
        q_feat = q_feat.view(B, NQ, -1)
        
        # Calculate distances
        # We want dists[b, i, j] = ||q[b, i] - proto[b, j]||^2
        # q_feat: (B, NQ, D)
        # prototypes: (B, Way, D)
        
        dists = torch.cdist(q_feat, prototypes) # (B, NQ, Way)
        
        # Flatten output to (B*NQ, Way) for compatibility with loss function
        scores = -dists.view(-1, Way)
        
        return scores
