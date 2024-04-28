import torch.nn as nn
from multihead_attention import Multihead_attention

class Encoder(nn.Module):
    def __init__(self):
        self.multihead_attention = Multihead_attention()
        self.FFN = nn.Sequential(nn.Linear(512, 2048),
                                 nn.Linear(2048, 512))

    def _normalize(self, x):
        return x
    
    def forward(self, input):
        x1 = self.multihead_attention(input)
        x2 = self._normalize(x1 + input)
        x3 = self.FFN(x2)
        x4 = self._normalize(x3 + x2)
        return x4
