# Create a new file named "new_test_attention.py" with this content
import torch
import os
import sys

# Print detailed debugging information
print(f"Current working directory: {os.getcwd()}")
print(f"Python version: {sys.version}")
print(f"Python executable: {sys.executable}")

# Verify the module path
modules_path = os.path.join(os.getcwd(), 'modules')
print(f"Modules directory exists: {os.path.exists(modules_path)}")
print(f"Files in modules directory: {os.listdir(modules_path) if os.path.exists(modules_path) else 'N/A'}")

# Import the attention module with detailed error handling
try:
    from modules.attention import CausalSelfAttention
    print("Successfully imported CausalSelfAttention")
except Exception as e:
    print(f"Error importing CausalSelfAttention: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Create a very explicit config class
class MyConfig:
    def __init__(self):
        # Set all required attributes explicitly
        self._hidden_size = 768
        self._num_attention_heads = 12
        self._attention_probs_dropout_prob = 0.1
        self._max_position_embeddings = 512
    
    # Use property getters to ensure attributes are accessible
    @property
    def hidden_size(self):
        return self._hidden_size
    
    @property
    def num_attention_heads(self):
        return self._num_attention_heads
    
    @property
    def attention_probs_dropout_prob(self):
        return self._attention_probs_dropout_prob
    
    @property
    def max_position_embeddings(self):
        return self._max_position_embeddings

# Create the config and print its contents
config = MyConfig()
print("Config attributes:")
print(f"- hidden_size: {config.hidden_size}")
print(f"- num_attention_heads: {config.num_attention_heads}")
print(f"- attention_probs_dropout_prob: {config.attention_probs_dropout_prob}")
print(f"- max_position_embeddings: {config.max_position_embeddings}")

# Verify the attribute existence explicitly
print(f"hasattr(config, 'max_position_embeddings'): {hasattr(config, 'max_position_embeddings')}")

# Now create the attention layer
try:
    print("Creating attention layer...")
    attention_layer = CausalSelfAttention(config)
    print("Successfully created attention layer")
except Exception as e:
    print(f"Error creating attention layer: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test with dummy input
print("Creating dummy input...")
batch_size = 2
seq_length = 10
hidden_dim = config.hidden_size

dummy_input = torch.randn(batch_size, seq_length, hidden_dim)
dummy_mask = torch.ones((batch_size, 1, 1, seq_length))

# Forward pass
print("Running forward pass...")
output = attention_layer(dummy_input, dummy_mask)

# Verify output
print(f"Output shape: {output.shape}")
print(f"Expected shape: {dummy_input.shape}")
assert output.shape == dummy_input.shape, f"Output shape mismatch: {output.shape}"
print("✅ CausalSelfAttention test passed!")