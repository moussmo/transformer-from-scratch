import numpy as np
import torch.nn as nn
from src.model.encoder import Encoder
from src.model.decoder import Decoder
import torch.nn.functional as fn

class Transformer(nn.Module):
    def __init__(self, vocabulary_size_org, vocabulary_size_dst):
        self.embedding_model = nn.Embedding(vocabulary_size_org, 512)
        self.encoder_stack = nn.Sequential(*[Encoder() for i in range(6)])
        self.decoder_stack = nn.Sequential(*[Decoder() for i in range(6)])
        self.linear_layer = nn.Linear(512,vocabulary_size_dst)

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

    def _encode(self, encoder_sequence):
        encoder_sequence = self.embedding_model(encoder_sequence)*np.sqrt(512)
        encoder_sequence = self._add_positional_encoding(encoder_sequence)
        encoder_sequence = fn.dropout(encoder_sequence, 0.1)
        encoder_output = self.encoder_stack(encoder_sequence)
        return encoder_output
    
    def _decode(self, decoder_sequence, encoder_output):
        decoder_sequence = self.embedding_model.embed(decoder_sequence)
        decoder_sequence = self._shift_right(decoder_sequence)
        decoder_sequence = self._add_positional_encoding(decoder_sequence)
        decoder_output = self.decoder_stack(decoder_sequence, encoder_output)     
        linear_output = self.linear_layer(decoder_output)
        output_probabilities = fn.softmax(linear_output)
        return output_probabilities
    
    def forward(self, encoder_sequence, decoder_sequence):   
        encoder_output = self._encode(encoder_sequence)
        output_probabilities = self._decode(decoder_sequence, encoder_output)
        return output_probabilities
    
    def predict(self, input_sequence):
        return 1