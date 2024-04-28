import torch.nn as nn
from src.model.encoder import Encoder
from src.model.decoder import Decoder
from src.model.embedding import Embedding

class Transformer(nn.Module):
    def __init__(self):
        self.embedding_model = Embedding()
        self.encoder_stack = nn.Sequential(*[Encoder() for i in range(6)])
        self.decoder_stack = nn.Sequential(*[Decoder() for i in range(6)])
        self.linear_layer = nn.Linear()

    def _add_positional_encoding(self, sequence):
        return sequence

    def _shift_right(self, sequence):
        return sequence

    def forward(self, input):
        encoder_sequence, decoder_sequence = input
        
        encoder_sequence = self.embedding_model.embed(encoder_sequence)
        encoder_sequence = self._add_positional_encoding(encoder_sequence)
        encoder_output = self.encoder_stack(encoder_sequence)
        K, V = encoder_output

        decoder_sequence = self.embedding_model.embed(decoder_sequence)
        decoder_sequence = self._shift_right(decoder_sequence)
        decoder_sequence = self._add_positional_encoding(decoder_sequence)
        decoder_output = self.decoder_stack(decoder_sequence, K, V)     

        linear_output = self.linear_layer(decoder_output)
        output_probabilities = nn.Softmax(linear_output)

        return output_probabilities