import torch.nn as nn
from multihead_attention import MultiheadAttention
import torch.nn.functional as fn

class Encoder(nn.Module):
    def __init__(self):
        self.multihead_attention = MultiheadAttention()
        self.layer_norm = nn.LayerNorm(512)
        self.FFN = nn.Sequential(nn.Linear(512, 2048),
                                 nn.ReLU(),
                                 nn.Linear(2048, 512))
    
    def forward(self, input):
        x1 = self.multihead_attention(input)
        x2 = fn.dropout(x1, 0.1)
        x3 = self.layer_norm(x2 + input)
        x4 = self.FFN(x3)
        output = self.layer_norm(x4 + x3)
        return output
