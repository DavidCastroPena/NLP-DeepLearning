import torch
from modules.attention import CausalSelfAttention  # Import your implementation

# Dummy config
class Config:
    hidden_size = 768
    num_attention_heads = 12
    attention_probs_dropout_prob = 0.1

# Initialize attention module
config = Config()
attention_layer = CausalSelfAttention(config)

# Generate dummy input
batch_size = 2
seq_length = 10
hidden_dim = config.hidden_size

dummy_input = torch.randn(batch_size, seq_length, hidden_dim)
dummy_mask = torch.ones((batch_size, 1, 1, seq_length))  # No masking

# Forward pass
output = attention_layer(dummy_input, dummy_mask)

# Verify output shape
assert output.shape == dummy_input.shape, f"Output shape mismatch: {output.shape}"
print("✅ CausalSelfAttention test passed!")
