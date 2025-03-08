import torch
from modules.new_attention import CausalSelfAttention  # Import from the new file

# Config class
class Config:
    def __init__(self):
        self.hidden_size = 768
        self.num_attention_heads = 12
        self.attention_probs_dropout_prob = 0.1
        self.max_position_embeddings = 512

# Initialize config
config = Config()

# Print all attributes for debugging
print("Config attributes:", dir(config))  

# Initialize attention module
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