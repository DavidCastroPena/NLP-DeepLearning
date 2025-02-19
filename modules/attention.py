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