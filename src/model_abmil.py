import torch
import torch.nn as nn
import torch.nn.functional as F

class ABMIL(nn.Module):
    def __init__(self, in_features:int=1024, L:int=512, M:int=128, n_classes:int=2, attn_branches:int=1, gated:bool=True, dropout:float=0.):
        super().__init__()
        self.in_features = in_features
        self.M = M
        self.L = L
        self.num_classes = n_classes
        self.attn_branches = attn_branches
        self.gated = gated

        self.MLP = nn.Sequential(
            nn.Linear(in_features, L),
            nn.ReLU()
            )
        
        if self.gated:
            self.A_V = nn.Sequential(nn.Linear(L, M), 
                                     nn.Tanh())
            self.A_U = nn.Sequential(nn.Linear(L, M),
                                     nn.Sigmoid())
        else:
            self.A_V = nn.Sequential(nn.Linear(L, M), 
                                     nn.Tanh())
            self.A_U = nn.Identity()
        
        self.w = nn.Linear(M, self.attn_branches)
        self.classifier = nn.Sequential(nn.Dropout(dropout),
                                        nn.Linear(self.L,self.num_classes))
        self.normalize = nn.Softmax(dim=-1)

    def forward(self, x):
        # print(f'Shape of x before anything: {x.shape}')
        H = self.MLP(x) # NxL
        # print(f'Shape of h after MLP: {H.shape}')
        V = self.A_V(H) # NxM
        # print(f'Shape of v after A_V: {V.shape}')
        U = self.A_U(H) # NxM
        # print(f'Shape of u after A_U: {U.shape}')
        if self.gated:
            E = torch.mul(V, U)
        else:
            E = V
        A = self.w(E) # Nxbranches
        A = torch.transpose(A, 1, 0) # branchesxN
        A = F.softmax(A, dim=1) # branchesxN
        # print(f'Shape of a.T: {A.shape}') 
        Z = torch.mm(A, H) # branchesxL
        # print(f'Shape of z: {Z.shape}')
        Y_logits = self.classifier(Z)
        Y_prob = self.normalize(Y_logits)
        Y_hat = Y_prob.argmax(dim=-1)
        # print(f'Shape of y_prob: {Y_prob.shape}, y_hat: {Y_hat.shape}')
        return {"Y_prob": Y_prob, "Y_hat": Y_hat, "Y_logits": Y_logits, "A": A}

if __name__ == "__main__":
    model = ABMIL()
    print(model)

    x = torch.randn(500, 1024)
    out = model(x)
    print(f'Model output:')
    print(f'Y_prob: {out["Y_prob"].shape}, Y_hat: {out["Y_hat"].shape}, A: {out["A"].shape}')
    print('Done!')