import torch.nn as nn
from src.model.encoder import Encoder
from src.model.decoder import Decoder
from src.model.embedding import Embedding

class Transformer(nn.Module):
    N_ENCODERS = 6
    N_DECODERS = 6

    def __init__(self):
        self.embedding_model = Embedding()
        self.encoder_stack = nn.Sequential(*[Encoder() for i in range(self.N_ENCODERS)])
        self.decoder_stack = nn.Sequential(*[Decoder() for i in range(self.N_DECODERS)])
        self.linear_layer = nn.Linear()

    def _add_positional_encoding(self, sequence):
        return 1

    def _shift_right(self, sequence):
        return 1

    def forward(self, input):
        encoder_sequence, decoder_sequence = input
        
        encoder_sequence = self.embedding_model.embed(encoder_sequence)
        encoder_sequence = self._add_positional_encoding(encoder_sequence)
        encoder_output = self.encoder_stack(encoder_sequence)
        K, V = encoder_output

        decoder_sequence = self.embedding_model.embed(decoder_sequence)
        decoder_sequence = self._shift_right(decoder_sequence)
        decoder_sequence = self.add_positional_encoding(decoder_sequence)
        decoder_output = self.decoder_stack(decoder_sequence, K, V)     

        linear_output = self.linear_layer(decoder_output)
        output_probabilities = nn.Softmax(linear_output)

        return output_probabilities

    def predict(self, input):
        pass