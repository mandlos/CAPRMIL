import torch
import torch.nn as nn
from einops import rearrange

class MILAggregator(nn.Module):
    """
    MIL aggregation heads:
      - mean
      - attn
      - gated_attn
    """
    def __init__(self, mode='mean', dim=None, attn_hidden_dim=128, dropout=False):
        super().__init__()
        self.mode = mode
        if mode == 'mean':
            self.attn = None

        elif mode == 'attn':
            assert dim is not None, "dim must be provided for attention aggregation"
            self.attn = Attn_Net(L=dim, D=attn_hidden_dim,
                                 dropout=dropout, n_classes=1)

        elif mode == 'gated_attn':
            assert dim is not None, "dim must be provided for gated attention aggregation"
            self.attn = DAttn_Net_Gated(L=dim, D=attn_hidden_dim,
                                       dropout=dropout, n_classes=1)
        else:
            raise ValueError(f"Unknown aggregator mode: {mode}")

    def forward(self, x):
        """
        x: [B, N, D]
        returns: [B, D]
        """
        if self.mode == 'mean':
            return x.mean(dim=1)

        # Attention-based aggregation
        B, N, D = x.shape
        bag_reprs = []

        for b in range(B):
            A, xb = self.attn(x[b])        # A: [N, 1], xb: [N, D]
            A = torch.softmax(A, dim=0)    # normalize over instances
            bag = torch.sum(A * xb, dim=0)
            bag_reprs.append(bag)

        return torch.stack(bag_reprs, dim=0)  # [B, D]



"""
Attention Network without Gating (2 fc layers)
args:
    L: input feature dimension
    D: hidden layer dimension
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes 
"""
class Attn_Net(nn.Module):

    def __init__(self, L = 1024, D = 256, dropout = False, n_classes = 1):
        super(Attn_Net, self).__init__()
        self.module = [
            nn.Linear(L, D),
            nn.Tanh()]

        if dropout:
            self.module.append(nn.Dropout(0.25))

        self.module.append(nn.Linear(D, n_classes))
        
        self.module = nn.Sequential(*self.module)
    
    def forward(self, x):
        return self.module(x), x # N x n_classes

"""
Attention Network with Gating (3 fc layers)
"""
class DAttn_Net_Gated(nn.Module):
    def __init__(self, L = 1024, D = 256, dropout = False, n_classes = 1):
        super(DAttn_Net_Gated, self).__init__()
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()]

        self.attention_b = [nn.Linear(L, D),
                            nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)

        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes
        # print(x.shape)
        return A, x


class TransolverAttention(nn.Module):
    """
    Transolver-style attention for MIL (irregular mesh version).
    Performs:
        1. Slice (soft clustering)
        2. Slice-level attention
        3. De-slice (map back to original tokens)
    Input:
        x: [B, N, D]
    Output:
        out: [B, N, D]
    """
    def __init__(self, input_dim, head_num=8, head_dim=64, 
                 cluster_num=32, dropout=0.1, use_temperature=True):
        super().__init__()

        # dimensions
        self.input_dim = input_dim
        self.head_num = head_num
        self.head_dim = head_dim
        self.cluster_num = cluster_num
        inner_dim = head_num * head_dim

        # projection to per-head channels
        self.in_project_x  = nn.Linear(input_dim, inner_dim)
        self.in_project_fx = nn.Linear(input_dim, inner_dim)
        # proto/cluster/slice assignment layer
        self.in_project_slice = nn.Linear(head_dim, cluster_num)
        nn.init.orthogonal_(self.in_project_slice.weight)

        # QKV projections for slice tokens
        self.to_q = nn.Linear(head_dim, head_dim, bias=False)
        self.to_k = nn.Linear(head_dim, head_dim, bias=False)
        self.to_v = nn.Linear(head_dim, head_dim, bias=False)

        # output projection
        self.to_out = nn.Sequential(nn.Linear(inner_dim, input_dim),
                                    nn.Dropout(dropout))

        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.scale = head_dim ** -0.5
        
        self.use_temperature = use_temperature

        # learnable temperature (per head)
        if self.use_temperature:
            self.temperature = nn.Parameter(torch.ones(1, head_num, 1, 1) * 0.5)
        else:
            self.register_buffer('temperature', torch.ones(1, head_num, 1, 1))

    def forward(self, x):
        """
        x: [B, N, D]
        returns: [B, N, D]
        """
        B, N, D = x.shape

        # ---- (1) Slice ------------------------------------------------------
        # Project input to per-head channels: [B, N, H*D_head] → [B, H, N, D_head]
        x_mid  = self.in_project_x(x).reshape(B, N, self.head_num, self.head_dim).permute(0, 2, 1, 3)
        # print(f"[Slice] x_mid shape: {x_mid.shape}")
        fx_mid = self.in_project_fx(x).reshape(B, N, self.head_num, self.head_dim).permute(0, 2, 1, 3)
        # print(f"[Slice] fx_mid shape: {fx_mid.shape}")

        # Soft assignment of each patch to M clusters/slices
        slice_logits = self.in_project_slice(x_mid) / self.temperature  # [B, H, N, M]
        # print(f"[Slice] slice_logits shape: {slice_logits.shape}")
        slice_weights = self.softmax(slice_logits) # each of the B*H matrices NxM is row-stochastic

        # Normalize each of the M slices/clusters (avoid division by zero)
        slice_norm = slice_weights.sum(dim=2)  # [B, H, M]

        # Compute slice tokens/clusters via weighted average
        slice_token = torch.einsum("bhnd,bhnm->bhmd", fx_mid, slice_weights)
        # print(f"[Slice] prototypes shape: {slice_token.shape}")
        slice_token = slice_token / (slice_norm[:, :, :, None] + 1e-5)

        # ---- (2) Self-attention between slice tokens ------------------------
        q = self.to_q(slice_token)
        # print(f"[Attention] q shape: {q.shape}")
        k = self.to_k(slice_token)
        v = self.to_v(slice_token)

        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.softmax(attn)
        attn = self.dropout(attn)

        out_slice_token = torch.matmul(attn, v)  # [B, H, M, D_head]
        # print(f"[Attention] out_slice_token shape: {out_slice_token.shape}")

        # ---- (3) De-slice ---------------------------------------------------
        out_x = torch.einsum("bhmd,bhnm->bhnd", out_slice_token, slice_weights)
        out_x = rearrange(out_x, 'b h n d -> b n (h d)')  # concat heads

        return self.to_out(out_x)


class TransolverBlock(nn.Module):
    """
    A single Transolver block:
      LN → TransolverAttention → residual
      LN → MLP → residual
    """
    def __init__(self, dim, num_heads, head_dim, 
                 cluster_num=32, mlp_ratio=4, dropout=0.1, 
                 last_layer=False, out_dim=1, use_temperature=True):
        super().__init__()

        self.last_layer = last_layer
        self.ln1 = nn.LayerNorm(dim)
        self.attn = TransolverAttention(
            input_dim=dim,
            head_num=num_heads,
            head_dim=head_dim,
            cluster_num=cluster_num,
            dropout=dropout,
            use_temperature=use_temperature
        )

        self.ln2 = nn.LayerNorm(dim)

        # MLP from the original Transolver paper
        hidden_dim = dim * mlp_ratio
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

        # Only last layer predicts logits
        if last_layer:
            self.ln3 = nn.LayerNorm(dim)
            self.out_proj = nn.Linear(dim, out_dim)
        
        self.residual_dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Might need dropout here for residuals (as per vanilla implementation)
        # Attn block
        x = x + self.residual_dropout(self.attn(self.ln1(x)))
        # print(f"[Post Attn] x shape: {x.shape}")

        # MLP block
        x = x + self.residual_dropout(self.mlp(self.ln2(x)))
        # print(f"[Post MLP] x shape: {x.shape}")
        
        if self.last_layer:
            return self.out_proj(self.ln3(x))  # [B, N, out_dim]

        return x

class CAPRMIL(nn.Module):
    """
    Full MIL model:
        Patch embeddings → L × TransolverBlock → MIL aggregator → Bag logits
    """
    def __init__(self, input_dim, hidden_dim, num_heads=8, head_dim=64, cluster_num=32, T_depth=1, 
                 mlp_ratio=4, dropout=0.1, aggregator = 'mean', num_classes=2, use_temperature=True):
        super().__init__()

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout))

        current_dim = hidden_dim
        head_dim = hidden_dim // num_heads
        self.blocks = nn.ModuleList()

        for i in range(T_depth):
            self.blocks.append(
                TransolverBlock(
                    dim=current_dim,
                    num_heads=num_heads,
                    head_dim=head_dim,
                    cluster_num=cluster_num,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    last_layer=False,  # last layer is MIL head, not block
                    use_temperature=use_temperature
                )
            )

        # MIL aggregator
        self.aggregator = MILAggregator(mode=aggregator, dim=current_dim, attn_hidden_dim=current_dim, dropout=True)

        # Final classifier
        self.classifier = nn.Linear(current_dim, num_classes)

    def forward(self, x):
        """
        Args:
            x: [B, N, D] frozen patch embeddings
        """
        x = self.input_proj(x)  # [B, N, hidden_dim]
        # Pass through Transolver blocks
        for blk in self.blocks:
            x = blk(x)

        # MIL aggregation
        bag_repr = self.aggregator(x)   # [B, D]

        # Bag-level prediction
        logits = self.classifier(bag_repr)         # [B, num_classes]
        return logits
