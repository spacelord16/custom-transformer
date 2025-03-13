import numpy as np

def scaled_dot_product_attention(Q, K, V, mask=None):
    matmul_qk = np.matmul(Q, K.swapaxes(-1, -2))  # (..., seq_len_q, seq_len_k)
    
    # scale matmul_qk
    dk = K.shape[-1]
    scaled_attention_logits = matmul_qk / np.sqrt(dk)

    # add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)  

    # softmax is applied to the last axis (seq_len_k) so that scores add to 1.
    attention_weights = np.exp(scaled_attention_logits)
    attention_weights /= np.sum(attention_weights, axis=-1, keepdims=True)

    output = np.matmul(attention_weights, V)  # (..., seq_len_q, depth_v)

    return output, attention_weights

def multi_head_attention(Q, K, V, num_heads, mask=None):
    if len(Q.shape) != 3 or len(K.shape) != 3 or len(V.shape) != 3:
        raise ValueError("Inputs Q, K, V must all have three dimensions [batch_size, seq_length, model_dim]")

    depth = Q.shape[2]
    assert depth % num_heads == 0, "Depth must be divisible by number of heads."
    depth_per_head = depth // num_heads
    
    def split_head(x):
        # Split the last dimension into (num_heads, depth_per_head)
        # and transpose the result to (num_heads, seq_len, depth_per_head)
        # return np.reshape(x, (x.shape[0], x.shape[1], num_heads, depth_per_head)).swapaxes(1, 2)  
        return x.reshape(x.shape[0], x.shape[1], num_heads, depth_per_head).transpose(0, 2, 1, 3)
    
    Q, K, V = split_head(Q), split_head(K), split_head(V)
    
    attention_outputs, _ = zip(*[scaled_dot_product_attention(Q[i], K[i], V[i], mask) for i in range(num_heads)])
    
    concatenated = np.concatenate(attention_outputs, axis=-1)  # (batch_size, seq_len_q, depth)
    
    return concatenated

