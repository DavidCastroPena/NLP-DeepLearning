''''
from torch import nn
import torch.nn.functional as F
from modules.attention import CausalSelfAttention

class GPT2Layer(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Multi-head attention.
        self.self_attention = CausalSelfAttention(config)
        # Add-norm for multi-head attention.
        self.attention_dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.attention_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attention_dropout = nn.Dropout(config.hidden_dropout_prob)
        # Feed forward.
        self.interm_dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.interm_af = F.gelu
        # Add-norm for feed forward.
        self.out_dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.out_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.out_dropout = nn.Dropout(config.hidden_dropout_prob)

 
    def add(self, input, output, dense_layer, dropout):
        """
        Applies residual connection and dropout before adding back to input.
        """
        transformed_output = dense_layer(output)  # Ensure output is projected properly
        print("Transformed Output Shape after Dense:", transformed_output.shape)  # check dimensions
        return input + dropout(transformed_output)


    def forward(self, hidden_states, attention_mask):
        """
        Forward pass for a GPT-2 Transformer layer.
        """
        
        normed_hidden_states = self.attention_layer_norm(hidden_states)

        
        attention_output = self.self_attention(normed_hidden_states, attention_mask)

        
        hidden_states = self.add(hidden_states, attention_output, self.attention_dense, self.attention_dropout)

        
        normed_hidden_states = self.out_layer_norm(hidden_states)

        
        ff_output = self.interm_af(self.interm_dense(normed_hidden_states))
        print("FF Output Shape after interm_dense:", ff_output.shape)
        ff_output = self.out_dense(ff_output)
        print("FF Output Shape after out_dense:", ff_output.shape)
        
        hidden_states = self.add(hidden_states, ff_output, self.out_layer_norm, self.out_dropout)

        return hidden_states
'''
from torch import nn
import torch.nn.functional as F
from modules.attention import CausalSelfAttention

class GPT2Layer(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Multi-head attention.
        self.self_attention = CausalSelfAttention(config)
        # Add-norm for multi-head attention.
        self.attention_dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.attention_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attention_dropout = nn.Dropout(config.hidden_dropout_prob)
        
        # Feed forward.
        self.interm_dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.interm_af = F.gelu
        # Add-norm for feed forward.
        self.out_dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.out_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.out_dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, attention_mask):
        """
        Forward pass for a GPT-2 Transformer layer.
        Pre-norm architecture: LayerNorm -> Sublayer -> Residual
        """
        # Attention block
        normed_hidden_states = self.attention_layer_norm(hidden_states)
        attention_output = self.self_attention(normed_hidden_states, attention_mask)
        attention_output = self.attention_dropout(self.attention_dense(attention_output))
        hidden_states = hidden_states + attention_output

        # Feed-forward block
        normed_hidden_states = self.out_layer_norm(hidden_states)
        ff_output = self.interm_af(self.interm_dense(normed_hidden_states))
        ff_output = self.out_dropout(self.out_dense(ff_output))
        hidden_states = hidden_states + ff_output

        return hidden_states