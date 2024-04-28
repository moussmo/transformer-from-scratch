import torch
import torch.nn as nn
from self_attention import SelfAttention

class MultiheadAttention(nn.Module):
    def __init__(self):
        self.attention_heads = [SelfAttention() for i in range(8)]
        self.W = nn.Linear(512,512)

    def forward(self, input):
        attention_outputs = [attention_head(input) for attention_head in self.attention_heads]
        concatenated_outputs = torch.concat(attention_outputs, dim=1)
        output = self.W(concatenated_outputs)
        return output