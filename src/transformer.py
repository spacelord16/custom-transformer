# from src.utils import positional_encoding
from utils import positional_encoding
from attention import multi_head_attention
import numpy as np


def feed_forward_network(d_model, d_ff):
    def ff_network(x):
        W1 = np.random.rand(d_model, d_ff) * 0.1
        b1 = np.zeros(d_ff)
        x = np.dot(x, W1) + b1
        x = np.maximum(0, x)  # ReLU

        W2 = np.random.rand(d_ff, d_model) * 0.1
        b2 = np.zeros(d_model)
        x = np.dot(x, W2) + b2
        return x
    return ff_network

class EncoderLayer:
    def __init__(self, d_model, num_heads, d_ff):
        self.num_heads = num_heads
        self.multi_head_attention = multi_head_attention
        self.feed_forward = feed_forward_network(d_model, d_ff)

    def forward(self, x, mask):
        attn_output = self.multi_head_attention(x, x, x, self.num_heads, mask)
        x = x + attn_output
        x = x / np.linalg.norm(x, axis=-1, keepdims=True)

        ff_output = self.feed_forward(x)
        x = x + ff_output
        x = x / np.linalg.norm(x, axis=-1, keepdims=True)
        return x
class DecoderLayer:
    def __init__(self, d_model, num_heads, d_ff):
        self.num_heads = num_heads
        self.masked_multi_head_attention = multi_head_attention
        self.multi_head_attention = multi_head_attention
        self.feed_forward = feed_forward_network(d_model, d_ff)

    def forward(self, x, enc_output, look_ahead_mask, padding_mask):
        attn1 = self.masked_multi_head_attention(x, x, x, self.num_heads, look_ahead_mask)
        x = x + attn1
        x = x / np.linalg.norm(x, axis=-1, keepdims=True)

        attn2 = self.multi_head_attention(x, enc_output, enc_output, self.num_heads, padding_mask)
        x = x + attn2
        x = x / np.linalg.norm(x, axis=-1, keepdims=True)

        ff_output = self.feed_forward(x)
        x = x + ff_output
        x = x / np.linalg.norm(x, axis=-1, keepdims=True)
        return x
        return x

class Transformer:
    def __init__(self, num_layers, d_model, num_heads, d_ff, input_vocab_size, target_vocab_size, max_seq_length):
        self.encoder_layers = [EncoderLayer(d_model, num_heads, d_ff) for _ in range(num_layers)]
        self.decoder_layers = [DecoderLayer(d_model, num_heads, d_ff) for _ in range(num_layers)]
        self.input_embedding = np.random.rand(input_vocab_size, d_model)
        self.target_embedding = np.random.rand(target_vocab_size, d_model)
        self.output_linear = np.random.rand(d_model, target_vocab_size)

    def forward(self, input_seq, target_seq):
        input_embeddings = self.input_embedding[input_seq]
        target_embeddings = self.target_embedding[target_seq]

        if input_embeddings.ndim == 2:
            input_embeddings = np.expand_dims(input_embeddings, axis=0)

        if target_embeddings.ndim == 2:
            target_embeddings = np.expand_dims(target_embeddings, axis=0)

        encoder_input = input_embeddings
        for layer in self.encoder_layers:
            encoder_input = layer.forward(encoder_input, None)

        decoder_input = target_embeddings
        for layer in self.decoder_layers:
            decoder_input = layer.forward(decoder_input, encoder_input, None, None)

        logits = np.dot(decoder_input, self.output_linear)
        return logits

# Example usage
transformer_model = Transformer(num_layers=6, d_model=512, num_heads=8, d_ff=2048, input_vocab_size=10000, target_vocab_size=10000, max_seq_length=40)
input_seq = np.array([np.random.randint(0, 10000, 40)])  # Ensuring correct shape
target_seq = np.array([np.random.randint(0, 10000, 40)])
output = transformer_model.forward(input_seq, target_seq)
print("Transformer output shape:", output.shape)
    
# # Example usage
# encoder_layer = EncoderLayer(d_model=64, num_heads=8)
# input_embedding = np.random.rand(10, 64)  # Example input embedding
# encoder_output = encoder_layer.forward(input_embedding, mask=None)


