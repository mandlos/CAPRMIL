import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# from nystrom_attention import NystromAttention      # Nystrom attention by the authors
from custom_utils.NystromAttentionSkywalker import NystromAttention

from math import ceil
import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, reduce



class TransLayer(nn.Module):

    def __init__(self, norm_layer=nn.LayerNorm, dim=512):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = NystromAttention(
            dim = dim,
            dim_head = dim//8,
            heads = 8,
            num_landmarks = dim//2,    # number of landmarks
            pinv_iterations = 6,    # number of moore-penrose iterations for approximating pinverse. 6 was recommended by the paper
            residual = True,         # whether to do an extra residual with the value or not. supposedly faster convergence if turned on
            dropout=0.1,
            return_attn = True,
            head_fusion = 'mean'
        )

    def forward(self, x):
        # print(x)
        norm_x, attn = self.attn(self.norm(x))
        x = x + norm_x

        return x, attn


class TransLayer1(nn.Module):

    def __init__(self, norm_layer=nn.LayerNorm, dim=512, head_fusion='mean'):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = NystromAttention(
            dim = dim,
            dim_head = dim//8,
            heads = 8,
            num_landmarks = dim//2,    # number of landmarks
            pinv_iterations = 6,    # number of moore-penrose iterations for approximating pinverse. 6 was recommended by the paper
            residual = True,         # whether to do an extra residual with the value or not. supposedly faster convergence if turned on
            dropout=0.1,
            # return_attn= True,
            # head_fusion='max',
        )

    def forward(self, x):
        x = x + self.attn(self.norm(x))
        return x



class PPEG(nn.Module):
    def __init__(self, dim=512):
        super(PPEG, self).__init__()
        self.proj = nn.Conv2d(dim, dim, 7, 1, 7//2, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5//2, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3//2, groups=dim)

    def forward(self, x, H, W):
        B, _, C = x.shape
        cls_token, feat_token = x[:, 0], x[:, 1:]
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        x = self.proj(cnn_feat)+cnn_feat+self.proj1(cnn_feat)+self.proj2(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        x = torch.cat((cls_token.unsqueeze(1), x), dim=1)
        return x


class TransMIL(nn.Module):
    def __init__(self, n_classes:int=2, in_size=1024):
        super(TransMIL, self).__init__()
        self.pos_layer = PPEG(dim=512)
        self._fc1 = nn.Sequential(nn.Linear(in_size, 512), nn.ReLU())
        self.cls_token = nn.Parameter(torch.randn(1, 1, 512))
        self.n_classes = n_classes
        self.layer1 = TransLayer(dim=512)
        self.layer2 = TransLayer(dim=512)
        self.norm = nn.LayerNorm(512)
        self._fc2 = nn.Linear(512, self.n_classes)


    def forward(self, x):

        h = x.unsqueeze(dim=0) #[B, n, 1024] i.e. Batch_size/number of bags, number of instances/patches, feature_dim

        h = self._fc1(h) #[B, n, 512]
        
        #---->pad
        H = h.shape[1]
        _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
        add_length = _H * _W - H
        h = torch.cat([h, h[:,:add_length,:]],dim = 1) #[B, N, 512]

        #---->cls_token
        B = h.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        h0 = torch.cat((cls_tokens, h), dim=1)

        #---->Translayer x1
        h1, attn0 = self.layer1(h0) #[B, N, 512]

        #---->PPEG
        h2 = self.pos_layer(h1, _H, _W) #[B, N, 512]

        #---->Translayer x2
        h3, attn1 = self.layer2(h2) #[B, N, 512]

        # print(cls_attn.shape)

        #---->cls_token
        h3 = self.norm(h3)[:,0]
        
        
        #---->predict
        logits = self._fc2(h3) #[B, n_classes]
        Y_hat = torch.argmax(logits, dim=1)
        Y_prob = F.softmax(logits, dim = 1)
        return {'logits': logits, 'Y_prob': Y_prob, 'Y_hat': Y_hat, 'cls_token':h, 'A':attn1, 'A0':attn0}

if __name__ == "__main__":
    data = torch.randn((1, 6000, 1024)).cuda()
    model = TransMIL(n_classes=2).cuda()
    print(model.eval())
    results_dict = model(data = data)
    print(results_dict)
