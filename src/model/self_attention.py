import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, masked=False):
        self.masked=masked
        self.W_Q = nn.Linear(512, 64)
        self.W_K = nn.Linear(512, 64)
        self.W_V = nn.Linear(512, 64)

    def forward(self, input, encoder_K=None, encoder_V=None):
        Q = self.W_Q(input)
        K = self.W_K(input) if encoder_K is None else encoder_K
        V = self.W_V(input) if encoder_V is None else encoder_V
        output = torch.matmul(torch.softmax(torch.matmul(Q,torch.t(K))/8), V)
        return output
        