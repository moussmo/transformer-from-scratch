import torch.nn as nn
from multihead_attention import MultiheadAttention

class Encoder(nn.Module):
    def __init__(self):
        self.multihead_attention = MultiheadAttention()
        self.layer_norm = nn.LayerNorm(512)
        self.FFN = nn.Sequential(nn.Linear(512, 2048),
                                 nn.Linear(2048, 512))
    
    def forward(self, input):
        x1 = self.multihead_attention(input)
        x2 = self.layer_norm(x1 + input)
        x3 = self.FFN(x2)
        output = self.layer_norm(x3 + x2)
        return output
