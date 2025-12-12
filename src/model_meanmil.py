import torch
import torch.nn as nn

class MeanMIL(nn.Module):
    """
    Mean MIL aggregation head.
    """
    def __init__(self, input_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

        self.phi = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU())
        self.classifier = nn.Linear(hidden_dim, out_dim)
    
    def forward(self, x):
        """
        x: [B, N, D]
        returns: [B, D]
        """
        x = self.phi(x)
        if x.ndim == 2:
            x = x.mean(dim=0).unsqueeze(0)
        elif x.ndim == 3:
            x = x.mean(dim=1)
        else:
            raise ValueError('Input x must be 2D or 3D tensor')
        logits = self.classifier(x)
        Y_prob = torch.softmax(logits, dim=-1)
        Y_hat = Y_prob.argmax(dim=-1)
        return {"Y_prob": Y_prob, "Y_hat": Y_hat, "Y_logits": logits, 'A': None}