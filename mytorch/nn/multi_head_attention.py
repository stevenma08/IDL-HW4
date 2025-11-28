from .linear import Linear
from .scaled_dot_product_attention import ScaledDotProductAttention
import numpy as np

class MultiHeadAttention:
    """
    Multi Head Attention
    """ 
    def __init__(self, embed_dim, num_heads):
        """
        :param embed_dim: Embedding dimension
        :param num_heads: Number of attention heads
        """
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")

        # Initialize parameters and layers
        # DO NOT MODIFY
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        # Initialize your scaled dot product attention layer
        self.attention = ScaledDotProductAttention()
        
        # Initialize your linear layer
        #  embed_dim -> embed_dim
        self.q_proj   = Linear(embed_dim, embed_dim)
        self.k_proj   = Linear(embed_dim, embed_dim)
        self.v_proj   = Linear(embed_dim, embed_dim)
        self.out_proj = Linear(embed_dim, embed_dim)

    def init_weights(self, Wq, bq, Wk, bk, Wv, bv, Wo, bo):
        """
        Initialize the weights and biases with the given values.
        """
        # Initialize your linear layers (DO NOT MODIFY)
        self.q_proj.init_weights(Wq, bq)
        self.k_proj.init_weights(Wk, bk)
        self.v_proj.init_weights(Wv, bv)
        self.out_proj.init_weights(Wo, bo)

    def forward(self, query, key, value, key_padding_mask=None, attn_mask=None):
        """
        :param query: (N, L, E)
        :param key: (N, S, E)
        :param value: (N, S, E)
        :param key_padding_mask: (N, S) where 1/True indicates positions to ignore
        :param attn_mask: (L, S) where 1/True indicates positions to ignore
        :return: (N, L, E)
        """
        
        self.N = query.shape[0]
        self.L = query.shape[1]
        self.S = key.shape[1]
        self.E = query.shape[2]
        
        # Project the query, key, and value inputs into query, key, and value
        q = self.q_proj.forward(query)
        k = self.k_proj.forward(key)
        v = self.v_proj.forward(value)

        # Split the query, key, and value into multiple heads
        q = self._split_heads(q)
        k = self._split_heads(k)
        v = self._split_heads(v)

        # Merge the masks
        mask = self._merge_masks(key_padding_mask, attn_mask)

        # Apply the attention mechanism
        attn_outputs = self.attention.forward(q, k, v, mask)

        # Merge the attention outputs   
        attn_output = self._concat_heads(attn_outputs)

        # Project the attention outputs
        output = self.out_proj.forward(attn_output)

        return output

    def backward(self, d_output):
        """
        :param d_output: Gradient of loss wrt output of shape (N, L, E)
        :return: Gradient of loss wrt input query, key, value of shapes (N, L, E), (N, S, E), (N, S, E)
        """

        # Backpropagate through the output projection   
        d_attn_output = self.out_proj.backward(d_output)

        # Split the gradients into multiple heads
        d_attn_outputs = self._split_heads(d_attn_output)

        # Backpropagate through the attention mechanism
        d_q, d_k, d_v = self.attention.backward(d_attn_outputs)

        # Merge the gradients   
        d_q = self._concat_heads(d_q)
        d_k = self._concat_heads(d_k)
        d_v = self._concat_heads(d_v)

        # Backpropagate through the input projections   
        d_q = self.q_proj.backward(d_q)
        d_k = self.k_proj.backward(d_k)
        d_v = self.v_proj.backward(d_v)

        return d_q, d_k, d_v

    def _merge_masks(self, key_padding_mask, attn_mask):
        """
        Merge key_padding_mask and attn_mask into a single mask.
        :param key_padding_mask: (N, S)
        :param attn_mask: (L, S)
        :return: (N, H, L, S)
        """
        if key_padding_mask is None and attn_mask is None:
            return None
        
        combined_mask = None
        
        if key_padding_mask is not None:
            combined_mask = key_padding_mask.reshape(key_padding_mask.shape[0], 1, 1, key_padding_mask.shape[1])
        
        if attn_mask is not None:
            a_mask = attn_mask.reshape(1, 1, attn_mask.shape[0], attn_mask.shape[1])
            if combined_mask is None:
                combined_mask = a_mask
            else:
                combined_mask = np.logical_or(combined_mask, a_mask)
        
        if combined_mask is not None:
            combined_mask = np.repeat(combined_mask, self.num_heads, axis=1)
        
        return combined_mask

    def _split_heads(self, x):
        """
        Split the last dimension into (num_heads, d_k).
        Transpose to move num_heads dimension to the front.
        :param x: (N, L, embed_dim)
        :return: (N, num_heads, L, embed_dim // num_heads)
        """
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        head_dim = self.embed_dim // self.num_heads
        
        x = x.reshape(batch_size, seq_len, self.num_heads, head_dim)
        x = np.transpose(x, (0, 2, 1, 3))
        
        return x

    def _concat_heads(self, x):
        """
        Concatenate the last dimension into (num_heads, d_k).
        Transpose to move num_heads dimension to the back.
        :param x: (N, num_heads, L, embed_dim // num_heads)
        :return: (N, L, embed_dim)
        """
        batch_size = x.shape[0]
        num_heads = x.shape[1]
        seq_len = x.shape[2]
        head_dim = x.shape[3]
        
        x = np.transpose(x, (0, 2, 1, 3))
        x = x.reshape(batch_size, seq_len, num_heads * head_dim)
        
        return x