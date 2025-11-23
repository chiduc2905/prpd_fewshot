import torch
import torch.nn as nn
import functools
from net.encoder import Conv64F_Encoder, get_norm_layer
from net.cova_block import CovaBlock
from net.utils import init_weights

class CovarianceNet(nn.Module):
    def __init__(self, norm_layer=nn.BatchNorm2d, num_classes=5, init_type='normal', use_gpu=True, input_size=224):
        super(CovarianceNet, self).__init__()
        
        if type(norm_layer) == str:
             norm_layer = get_norm_layer(norm_layer)
        elif norm_layer is None:
             norm_layer = nn.BatchNorm2d
             
        # Check norm_layer to decide bias
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.features = Conv64F_Encoder(norm_layer=norm_layer)
        
        self.covariance = CovaBlock()
        
        # Determine kernel size for classifier based on input size
        # Encoder has 2 max pooling layers (stride 2), so downsample factor is 4
        self.feature_h = input_size // 4
        self.feature_w = input_size // 4
        kernel_size = self.feature_h * self.feature_w
        
        self.classifier = nn.Sequential(
            nn.LeakyReLU(0.2, True),
            nn.Dropout(),
            nn.Conv1d(1, 1, kernel_size=kernel_size, stride=kernel_size, bias=use_bias),
        )
        
        init_weights(self, init_type=init_type)
        if use_gpu and torch.cuda.is_available():
            self.cuda()

    def forward(self, query, support):
        """
        Args:
            query: (B, NQ, C, H, W) 
            support: (B, Way, Shot, C, H, W)
        Returns:
            scores: (B * NQ, Way)
        """
        B, NQ, C, H, W = query.shape
        B_s, Way, Shot, C_s, H_s, W_s = support.shape
        
        # Flatten query to (B*NQ, C, H, W)
        input1 = query.view(-1, C, H, W)
        
        # Flatten support to list of tensors per class per batch?
        # My CovaBlock expects:
        # x1 (query): (Batch, C, h, w) -> here Batch is B*NQ
        # x2 (support): List of (Batch, Shot, C, h, w) ??
        # Let's check CovaBlock.
        # cal_covariance(input): input is list of tensors. 
        # for i in range(len(input)): support_set_sam = input[i] ... (B, C, h, w)
        # Wait, CovaBlock implementation in cova_block.py:
        # support_set_sam = input[i]
        # B, C, h, w = support_set_sam.size()
        # ... covariance_matrix = ... div(..., h*w*B-1)
        # Here 'B' in CovaBlock refers to the Shot dimension (number of samples per class).
        
        # So CovaBlock expects `input` to be a list of tensors, where each tensor is (Shot, C, h, w).
        # AND `x1` (query) to be (Total_Query, C, h, w).
        
        # But wait, if I process multiple batches (B > 1), CovaBlock might mix them if I just flatten B.
        # CovaBlock `cal_similarity`:
        # B, C, h, w = input.size() -> input is query.
        # for i in range(B): ...
        # It computes similarity for each query against the covariance matrices.
        
        # If I have batching, the covariance matrices for batch b=0 are different from b=1.
        # So I cannot simply flatten B and NQ together unless I also duplicate/arrange support covariances accordingly.
        
        # Current CovaBlock implementation seems to assume ONE set of support classes (one episode).
        # If B > 1, we have multiple episodes.
        # CovaBlock `cal_covariance` returns `CovaMatrix_list` (one per class).
        # If we have B episodes, we need B sets of covariance matrices.
        
        # The provided CovaBlock code is:
        # def cal_covariance(self, input):
        #    for i in range(len(input)): ...
        #    input[i] is (Shot, C, H, W).
        
        # It does NOT seem to handle batching of episodes. It handles batching of Queries against ONE support set.
        
        # So if I want to support B > 1 in `main.py`, I must iterate over B in `CovarianceNet.forward` or modify `CovaBlock`.
        # Given `CovaBlock` is from the "author" (user provided), I should try to keep it or wrap it.
        
        # I'll wrap it in `CovarianceNet.forward`: iterate over the batch dimension.
        
        scores_list = []
        
        for b in range(B):
            # Query for this batch: (NQ, C, H, W)
            q_b = query[b] 
            
            # Support for this batch: (Way, Shot, C, H, W)
            s_b = support[b]
            
            # Prepare support for CovaBlock: List of (Shot, C, H, W)
            s_input = [s_b[w] for w in range(Way)]
            
            # Extract features
            q_feat = self.features(q_b) # (NQ, 64, h, w)
            
            s_feats = []
            for w in range(Way):
                sf = self.features(s_input[w]) # (Shot, 64, h, w)
                s_feats.append(sf)
            
            # Calculate scores
            # x = self.covariance(q_feat, s_feats) 
            # But wait, CovaBlock.forward calls cal_covariance then cal_similarity.
            # cal_similarity returns Cova_Sim: (NQ, 1, h*w*Way)
            
            x = self.covariance(q_feat, s_feats) 
            
            # Classifier
            x = self.classifier(x) # (NQ, 1, Way)
            x = x.squeeze(1) # (NQ, Way)
            
            scores_list.append(x)
            
        scores = torch.cat(scores_list, dim=0) # (B*NQ, Way)
        
        return scores

# Expose CovaMNet as CovarianceNet
CovaMNet = CovarianceNet
