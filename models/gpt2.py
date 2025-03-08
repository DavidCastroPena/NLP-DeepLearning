

###############totally new for impossible destillation
import torch
from torch import nn
from transformers import GPT2Model as OpenAIGPT2Model

from config import GPT2Config  
from models.base_gpt import GPTPreTrainedModel
from modules.gpt2_layer import GPT2Layer
from utils import get_extended_attention_mask


class GPT2Model(GPTPreTrainedModel):
    """
    The GPT model returns the final embeddings for each token in a sentence.

    The model consists of:
    1. Embedding layers (used in self.embed).
    2. A stack of n GPT layers (used in self.encode).
    3. A linear transformation layer for the [CLS] token (used in self.forward, as given).
    """

    def __init__(self, config=None):
        if isinstance(config, dict):  # 🔹 Convert dictionary to GPT2Config
            config = GPT2Config(**config)
        super().__init__(config)
        self.config = config

        # ✅ Embedding layers
        self.word_embedding = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.pos_embedding = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.embed_dropout = nn.Dropout(config.hidden_dropout_prob)

        # ✅ Register position_ids (1, len position emb) to buffer because it is a constant.
        position_ids = torch.arange(config.max_position_embeddings).unsqueeze(0)
        self.register_buffer('position_ids', position_ids)

        # ✅ GPT-2 layers
        self.gpt_layers = nn.ModuleList([GPT2Layer(config) for _ in range(config.num_hidden_layers)])

        # ✅ [CLS] token transformations
        self.pooler_dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.pooler_af = nn.Tanh()

        # ✅ Final layer norm
        self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.init_weights()

    def embed(self, input_ids):
        input_shape = input_ids.size()
        seq_length = input_shape[1]

        # ✅ Get token embeddings
        inputs_embeds = self.word_embedding(input_ids)

        # ✅ Get positional embeddings
        pos_ids = self.position_ids[:, :seq_length]
        pos_embeds = self.pos_embedding(pos_ids)

        # ✅ Add embeddings and apply dropout
        embeddings = self.embed_dropout(inputs_embeds + pos_embeds)

        return embeddings

    def encode(self, hidden_states, attention_mask):
        """
        hidden_states: the output from the embedding layer [batch_size, seq_len, hidden_size]
        attention_mask: [batch_size, seq_len]
        """
        # ✅ Get the extended attention mask for self-attention.
        extended_attention_mask: torch.Tensor = get_extended_attention_mask(attention_mask, self.dtype)

        # ✅ Pass the hidden states through the encoder layers.
        for layer_module in self.gpt_layers:
            hidden_states = layer_module(hidden_states, extended_attention_mask)

        return hidden_states

    def forward(self, input_ids, attention_mask):
        """
        input_ids: [batch_size, seq_len], seq_len is the max length of the batch
        attention_mask: same size as input_ids, 1 represents non-padding tokens, 0 represents padding tokens
        """
        # ✅ Get the embedding for each input token.
        embedding_output = self.embed(input_ids=input_ids)

        # ✅ Feed to a transformer (a stack of GPTLayers).
        sequence_output = self.encode(embedding_output, attention_mask=attention_mask)
        sequence_output = self.final_layer_norm(sequence_output)

        # ✅ Compute last non-padding token index safely
        last_non_pad_idx = attention_mask.sum(dim=1) - 1  # Get index of last non-pad token
        last_non_pad_idx = last_non_pad_idx.clamp(min=0)  # Prevent negative indices
        last_non_pad_idx = last_non_pad_idx.to(torch.long)  # Convert to long type

        # ✅ Get the hidden state of the final token.
        last_token = sequence_output[torch.arange(sequence_output.shape[0]), last_non_pad_idx]

        return {'last_hidden_state': sequence_output, 'last_token': last_token}

    def hidden_state_to_token(self, hidden_state):
        """
        GPT-2 uses weight tying with the input word embeddings. The logits are the dot product between output hidden states
        and the word embedding weights:
        """
        return torch.matmul(hidden_state, self.word_embedding.weight.T)

    @classmethod
    def from_pretrained(cls, model='gpt2', d=768, l=12, num_heads=12):
        gpt_model = OpenAIGPT2Model.from_pretrained(model).eval()

        # ✅ Ensure the new GPT2Config includes max_position_embeddings
        config = GPT2Config(hidden_size=d, num_hidden_layers=l, num_attention_heads=num_heads,
                            intermediate_size=d * 3, max_position_embeddings=1024)

        our_model = GPT2Model(config).eval()

        # ✅ Load word and positional embeddings.
        our_model.word_embedding.load_state_dict(gpt_model.wte.state_dict())
        our_model.pos_embedding.load_state_dict(gpt_model.wpe.state_dict())

        print("Weights loaded successfully from OpenAI GPT-2")

        for i in range(l):
            layer = our_model.gpt_layers[i]
            # ✅ Remap Q,K,V weights from conv1d to 3 linear projections
            layer.self_attention.query.weight.data = gpt_model.state_dict()[f'h.{i}.attn.c_attn.weight'][:, :d].T
            layer.self_attention.query.bias.data = gpt_model.state_dict()[f'h.{i}.attn.c_attn.bias'][:d]
            layer.self_attention.key.weight.data = gpt_model.state_dict()[f'h.{i}.attn.c_attn.weight'][:, d:d*2].T
            layer.self_attention.key.bias.data = gpt_model.state_dict()[f'h.{i}.attn.c_attn.bias'][d:d*2]
            layer.self_attention.value.weight.data = gpt_model.state_dict()[f'h.{i}.attn.c_attn.weight'][:, d*2:].T
            layer.self_attention.value.bias.data = gpt_model.state_dict()[f'h.{i}.attn.c_attn.bias'][d*2:]

            # ✅ Remap other layers
            layer.attention_dense.weight.data = gpt_model.state_dict()[f'h.{i}.attn.c_proj.weight'].T
            layer.attention_dense.bias.data = gpt_model.state_dict()[f'h.{i}.attn.c_proj.bias']
            layer.attention_layer_norm.weight.data = gpt_model.state_dict()[f'h.{i}.ln_1.weight']
            layer.attention_layer_norm.bias.data = gpt_model.state_dict()[f'h.{i}.ln_1.bias']
            layer.out_layer_norm.weight.data = gpt_model.state_dict()[f'h.{i}.ln_2.weight']
            layer.out_layer_norm.bias.data = gpt_model.state_dict()[f'h.{i}.ln_2.bias']

        # ✅ Remap final layer norm values.
        our_model.final_layer_norm.weight.data = gpt_model.state_dict()['ln_f.weight']
        our_model.final_layer_norm.bias.data = gpt_model.state_dict()['ln_f.bias']

        return our_model
