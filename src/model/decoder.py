import torch.nn as nn
from multihead_attention import MultiheadAttention

class Decoder(nn.Module):
    def __init__(self):
        self.masked_multihead_attention = MultiheadAttention(masked=True)
        self.layer_norm = nn.LayerNorm(512)
        self.encoder_decoder_attention = MultiheadAttention()
        self.FFN = nn.Sequential(nn.Linear(512, 2048),
                                 nn.Linear(2048, 512))
        
    def forward(self, input, encoder_output):
        x1 = self.masked_multihead_attention(input)
        x2 = self.layer_norm(x1 + input)
        x3 = self.encoder_decoder_attention(x2, encoder_output)
        x4 = self.layer_norm(x3 + x2)
        x5 = self.FFN(x4)
        output = self.layer_norm(x5 + x4)
        return output

        