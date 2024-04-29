import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self):
        self.W_Q = nn.Linear(512, 64)
        self.W_K = nn.Linear(512, 64)
        self.W_V = nn.Linear(512, 64)

    def forward(self, input):
        Q = self.W_Q(input)
        K = self.W_K(input)
        V = self.W_V(input)
        output = torch.matmul(torch.softmax(torch.matmul(Q,torch.t(K))/8), V)
        return output
        