import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        # Use getattr with default values for safety
        max_pos_emb = getattr(config, "max_position_embeddings", 512)
        num_heads = getattr(config, "num_attention_heads", 12)
        hidden_size = getattr(config, "hidden_size", 768)
        dropout_prob = getattr(config, "attention_probs_dropout_prob", 0.1)
        
        # Print which values we're using for debugging
        print(f"Using max_position_embeddings: {max_pos_emb}")
        
        self.num_attention_heads = num_heads
        self.attention_head_size = int(hidden_size / num_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # Initialize the linear transformation layers for key, value, query
        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        # Dropout for attention weights
        self.dropout = nn.Dropout(dropout_prob)
        
        # Register causal mask buffer
        mask = torch.triu(torch.ones(max_pos_emb, max_pos_emb), diagonal=1).bool()
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