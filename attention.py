'''import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # Initialize the linear transformation layers for key, value, query.
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        # Dropout for attention weights
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transform(self, x, linear_layer):
        """ Projects input x using linear_layer and reshapes for multi-head attention. """
        proj = linear_layer(x)
        proj = rearrange(proj, 'b t (h d) -> b h t d', h=self.num_attention_heads)
        return proj

    def attention(self, key, query, value, attention_mask):
        """
        Implements standard masked multi-head self-attention.
        Applies scaled dot-product attention with causal masking.
        """
        # Compute attention scores
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) * self.attention_head_size ** -0.5

        # Apply causal mask (upper-triangular)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        # Softmax normalization
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        # Weighted sum of values
        attention_output = torch.matmul(attention_probs, value)
        return attention_output

    def forward(self, hidden_states, attention_mask):
        """
        hidden_states: [bs, seq_len, hidden_state]
        attention_mask: [bs, 1, 1, seq_len]
        output: [bs, seq_len, hidden_state]
        """
        # Generate key, value, query
        key_layer = self.transform(hidden_states, self.key)
        value_layer = self.transform(hidden_states, self.value)
        query_layer = self.transform(hidden_states, self.query)

        # Compute attention
        attn_value = self.attention(key_layer, query_layer, value_layer, attention_mask)

        # Restore shape: [batch_size, seq_len, hidden_dim]
        attn_value = rearrange(attn_value, 'b h t d -> b t (h d)')

        return attn_value
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # Initialize the linear transformation layers for key, value, query
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        # Dropout for attention weights
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

        # Register causal mask buffer
        mask = torch.triu(torch.ones(config.max_position_embeddings, 
                                   config.max_position_embeddings), diagonal=1).bool()
        self.register_buffer("causal_mask", mask)

    def transform(self, x, linear_layer):
        """ Projects input x using linear_layer and reshapes for multi-head attention. """
        proj = linear_layer(x)
        proj = rearrange(proj, 'b t (h d) -> b h t d', h=self.num_attention_heads)
        return proj

    def attention(self, key, query, value, attention_mask):
        batch_size, seq_length = query.shape[0], query.shape[2]
        
        # Compute attention scores
        attention_scores = torch.matmul(query, key.transpose(-2, -1))
        attention_scores = attention_scores * (self.attention_head_size ** -0.5)
        
        # Apply causal mask
        causal_mask = self.causal_mask[:seq_length, :seq_length]
        attention_scores.masked_fill_(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        
        # Apply padding mask if provided
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        # Softmax normalization and dropout
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        # Compute context vector
        return torch.matmul(attention_probs, value)

    def forward(self, hidden_states, attention_mask):
        """
        hidden_states: [bs, seq_len, hidden_state]
        attention_mask: [bs, 1, 1, seq_len]
        output: [bs, seq_len, hidden_state]
        """
        # Generate key, value, query
        key_layer = self.transform(hidden_states, self.key)
        value_layer = self.transform(hidden_states, self.value)
        query_layer = self.transform(hidden_states, self.query)

        # Compute attention
        attn_value = self.attention(key_layer, query_layer, value_layer, attention_mask)

        # Restore shape: [batch_size, seq_len, hidden_dim]
        attn_value = rearrange(attn_value, 'b h t d -> b t (h d)')

        return attn_value


class LogSparseAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # Linear layers for query, key, value.
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

        # Register causal mask (upper triangular mask, True where j > i).
        mask = torch.triu(torch.ones(config.max_position_embeddings, config.max_position_embeddings), diagonal=1).bool()
        self.register_buffer("causal_mask", mask)

    def transform(self, x, linear_layer):
        """Projects x with linear_layer and reshapes to [batch, heads, seq_len, head_dim]."""
        proj = linear_layer(x)
        proj = rearrange(proj, 'b t (h d) -> b h t d', h=self.num_attention_heads)
        return proj

    def _build_logsparse_mask(self, seq_length, device):
        """
        Create a [seq_length, seq_length] boolean mask where for each query position i,
        only a logarithmically sparse set of key positions is allowed. Allowed positions
        for query token i: i (or i-1 if you prefer) and i-2^k for k >= 1, if available.
        """
        mask = torch.ones(seq_length, seq_length, dtype=torch.bool, device=device)  # True = mask out
        for i in range(seq_length):
            allowed = set()
            # Always allow the immediate past (or current position)
            if i >= 0:
                allowed.add(i)
            if i - 1 >= 0:
                allowed.add(i - 1)
            k = 1
            while (i - 2**k) >= 0:
                allowed.add(i - 2**k)
                k += 1
            # For row i, set allowed indices to False (i.e. do not mask)
            for j in allowed:
                mask[i, j] = False
        return mask

    def attention(self, key, query, value, attention_mask):
        batch_size, seq_length = query.shape[0], query.shape[2]

        # Compute scaled dot-product attention scores.
        attention_scores = torch.matmul(query, key.transpose(-2, -1))
        attention_scores = attention_scores * (self.attention_head_size ** -0.5)

        # Retrieve the standard causal mask (True where j > i).
        causal_mask = self.causal_mask[:seq_length, :seq_length]  # shape [seq_length, seq_length]

        # Build the log-sparse mask.
        logsparse_mask = self._build_logsparse_mask(seq_length, query.device)
        # Final mask: mask out keys that violate causality OR are not in the allowed log-sparse set.
        final_mask = causal_mask | logsparse_mask  # True means masked out.

        # Expand final_mask to match attention_scores: [1, 1, seq_length, seq_length]
        final_mask = final_mask.unsqueeze(0).unsqueeze(0)
        attention_scores.masked_fill_(final_mask, float('-inf'))

        # Apply additional attention_mask (e.g., padding) if provided.
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        # Softmax normalization and dropout.
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        # Compute context vector.
        return torch.matmul(attention_probs, value)

    def forward(self, hidden_states, attention_mask):
        """
        hidden_states: [batch_size, seq_len, hidden_dim]
        attention_mask: [batch_size, 1, 1, seq_len]
        Returns: [batch_size, seq_len, hidden_dim]
        """
        # Project hidden_states into key, value, query.
        key_layer = self.transform(hidden_states, self.key)
        value_layer = self.transform(hidden_states, self.value)
        query_layer = self.transform(hidden_states, self.query)

        # Compute attention with log-sparse + causal masking.
        attn_value = self.attention(key_layer, query_layer, value_layer, attention_mask)
        # Restore shape: [batch_size, seq_len, hidden_dim]
        attn_value = rearrange(attn_value, 'b h t d -> b t (h d)')
        return attn_value
