import numpy as np
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
        def positional_encoding_builder(_, row, column):
            if column%2 == 0 :
                return np.sin(row/10000**(2*column/512))
            else:
                return np.cos(row/10000**(2*column/512))
        positional_encoding_builder_vectorized = np.vectorize(positional_encoding_builder)
        positional_encoding = np.fromfunction(positional_encoding_builder_vectorized, shape=sequence.shape)

        sequence_with_positionals= sequence + positional_encoding
        return sequence_with_positionals

    def _shift_right(self, sequence):
        return sequence

    def forward(self, input):
        encoder_sequence, decoder_sequence = input
        
        encoder_sequence = self.embedding_model.embed(encoder_sequence)
        encoder_sequence = self._add_positional_encoding(encoder_sequence)
        encoder_output = self.encoder_stack(encoder_sequence)

        decoder_sequence = self.embedding_model.embed(decoder_sequence)
        decoder_sequence = self._shift_right(decoder_sequence)
        decoder_sequence = self._add_positional_encoding(decoder_sequence)
        decoder_output = self.decoder_stack(decoder_sequence, encoder_output)     

        linear_output = self.linear_layer(decoder_output)
        output_probabilities = nn.Softmax(linear_output)

        return output_probabilities