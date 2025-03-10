'''
Sonnet generation starter code.

Running:
  `python sonnet_generation.py --use_gpu`

trains your SonnetGPT model and writes the required submission files.
'''

import argparse
import random
import torch

import numpy as np
import torch.nn.functional as F

from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import GPT2Tokenizer
from einops import rearrange

from datasets import (
  SonnetsDataset,
)

from models.gpt2 import GPT2Model, GPT2ModelWithLMHead
from config import GPT2Config
from optimizer import AdamW

# LoRA Configuration
from peft import get_peft_model, LoraConfig, TaskType
lora_config = LoraConfig(
  task_type=TaskType.CAUSAL_LM,  # Language modeling task
  r=8,  # LoRA rank (low-rank adaptation)
  lora_alpha=16,  # Scaling factor
  lora_dropout=0.1,  # Dropout for LoRA layers
  target_modules=["self_attention.query", "self_attention.key", "self_attention.value"], 
)

TQDM_DISABLE = False

def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"🔢 Total Parameters: {total_params:,}")
    print(f"🎯 Trainable Parameters (LoRA): {trainable_params:,}")

# Fix the random seed.
def seed_everything(seed=11711):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.benchmark = False
  torch.backends.cudnn.deterministic = True

class SonnetGPT(nn.Module):
  """Your GPT-2 Model designed for paraphrase detection."""

  def __init__(self, args):
    super().__init__()
    self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2', pad_token='<|endoftext|>')
    self.tokenizer.pad_token = self.tokenizer.eos_token
    
    self.gpt = GPT2ModelWithLMHead.from_pretrained(model=args.model_size, d=args.d, l=args.l, num_heads=args.num_heads)
    self.gpt.config = GPT2Config(hidden_size=args.d, num_hidden_layers=args.l, num_attention_heads=args.num_heads)

    # Apply LoRA
    self.gpt = get_peft_model(self.gpt, lora_config)
    self.gpt.print_trainable_parameters()
    
    # for name, module in self.gpt.named_modules():
    #   if isinstance(module, LoRALinear):
    #       print(f"✅ LoRA applied to: {name} -> {module}")
    # count_parameters(self.gpt)
    # print(self.gpt)  # Ensure LoRA layers are part of the model
    
    # # Linear layer to map hidden states to vocab
    # self.language_modeling_head = nn.Linear(args.d, self.tokenizer.vocab_size, bias=False)
    # self.language_modeling_head.weight = self.gpt.word_embedding.weight

  def forward(self, input_ids, attention_mask):
    """
    This is similar to the forward for ParaphraseGPT, but we now want to produce a logit for each token in our sequence;
    not just the last token! This will allow our model to learn the natural language distribution that composes sonnets,
    not just the distribution over next tokens for the last token!
    """
    outputs = self.gpt.base_model(input_ids=input_ids, attention_mask=attention_mask)
    # outputs = self.gpt(input_ids, attention_mask=attention_mask)
    # hidden_states = outputs['last_hidden_state']
    # logits = self.language_modeling_head(hidden_states)
    return outputs["logits"]


  def get_device(self):
    for param in self.gpt.parameters():
      return param.device

  @torch.no_grad()
  def generate(self, encoding, temperature=0.7, top_p=0.9, max_length=128, num_beams=5):
    """
    Generates an original sonnet using top-p sampling and softmax temperature.

    TODO: this is probably not ideal. You can look at hugging face's model.generate(...) function for inspiration.
    In particular, generating multiple sequences and choosing the best with beam search is one avenue. Top_k is another;
    there are many.
    """
    token_ids = encoding.to(self.get_device())
    attention_mask = torch.ones(token_ids.shape, dtype=torch.int64).to(self.get_device())

    ################################################################################################ 
    ## Decoding methods begins
    ################################################################################################
    def greedy_search(probs):
      return torch.argmax(probs, dim=-1, keepdim=True)

    def top_p_sampling(probs, top_p=0.9):
      sorted_probs, sorted_indices = torch.sort(probs, descending=True)
      cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
      top_p_mask = cumulative_probs <= top_p
      top_p_mask[..., 1:] = top_p_mask[..., :-1].clone()  # Shift mask right for proper thresholding
      top_p_mask[..., 0] = True  # Always include the highest probability token
      filtered_probs = sorted_probs * top_p_mask  # Zero out unlikely tokens
      filtered_probs /= filtered_probs.sum(dim=-1, keepdim=True)  # Normalize probabilities

      # Sample from filtered distribution
      sampled_index = torch.multinomial(filtered_probs, 1)
      return sorted_indices.gather(dim=-1, index=sampled_index)
    
    def top_k_sampling(probs, k=50):
      top_values, top_indices = torch.topk(probs, k)
      top_values /= torch.sum(top_values)  # Normalize
      sampled_index = torch.multinomial(top_values, 1)
      return top_indices.gather(dim=-1, index=sampled_index)

    def beam_search(logits_sequence, num_beams=5, max_length=10):
      _, _, vocab_size = logits_sequence.shape
      input_ids               = token_ids.expand(num_beams, -1)
      attention_mask_expanded = attention_mask.expand(num_beams, -1)
      beam_scores = torch.zeros(num_beams, device=self.get_device())

      # Probe max_length times into the future to get the best beam.
      for _ in range(max_length):
        # probe
        logits_sequence = self.forward(input_ids, attention_mask_expanded)
        logits = logits_sequence[:, -1, :] / temperature  # Apply temperature scaling
        probs = torch.nn.functional.log_softmax(logits, dim=-1)

        # get new beam scores
        scores = (beam_scores.unsqueeze(-1) + probs).view(-1)
        top_scores, top_indices = torch.topk(scores, num_beams)
        beam_scores = top_scores

        # process top_indices
        next_token_ids = top_indices % vocab_size
        beam_indices = top_indices // vocab_size

        # get new inputs and beam
        input_ids = torch.cat([input_ids[beam_indices], next_token_ids.unsqueeze(-1)], dim=-1)
        new_attention_tokens = torch.ones((num_beams, 1), dtype=torch.int64, device=self.get_device())
        attention_mask_expanded = torch.cat([attention_mask_expanded[beam_indices], new_attention_tokens], dim=-1)

        # stop if EOS
        if (next_token_ids == self.tokenizer.eos_token_id).all():
            break

      # Choose best sequence
      best_sequence_idx = beam_scores.argmax()
      return input_ids[best_sequence_idx, -1].unsqueeze(0).unsqueeze(1)
    
    ################################################################################################
    ## Decoding methods ends
    ################################################################################################

    for _ in range(max_length):
      # Forward pass to get logits
      logits_sequence = self.forward(token_ids, attention_mask)
      logits_last_token = logits_sequence[:, -1, :] / temperature  # Apply temperature scaling

      # Convert logits to probabilities
      probs = torch.nn.functional.softmax(logits_last_token, dim=-1)

      # sampled_token = greedy_search(probs)
      # sampled_token = top_p_sampling(probs)
      sampled_token = top_k_sampling(probs)
      # sampled_token = beam_search(logits_sequence, num_beams)

      # Stop if end-of-sequence token is reached
      if sampled_token.item() == self.tokenizer.eos_token_id:
        break

      # Append sampled token
      token_ids = torch.cat([token_ids, sampled_token], dim=1)
      attention_mask = torch.cat(
        [attention_mask, torch.ones((1, 1), dtype=torch.int64).to(self.get_device())], dim=1
      )

    generated_output = self.tokenizer.decode(token_ids[0].cpu().numpy().tolist(), skip_special_tokens=True)[3:]
    return token_ids, generated_output


def save_model(model, optimizer, args, filepath):
  save_info = {
    'model': model.state_dict(),
    'optim': optimizer.state_dict(),
    'args': args,
    'system_rng': random.getstate(),
    'numpy_rng': np.random.get_state(),
    'torch_rng': torch.random.get_rng_state(),
  }

  torch.save(save_info, filepath)
  print(f"save the model to {filepath}")


def train(args):
  """Train GPT-2 for paraphrase detection on the Quora dataset."""
  device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
  # Create the data and its corresponding datasets and dataloader.
  sonnet_dataset = SonnetsDataset(args.sonnet_path)
  sonnet_dataloader = DataLoader(sonnet_dataset, shuffle=True, batch_size=args.batch_size,
                                 collate_fn=sonnet_dataset.collate_fn)

  # Create the held-out dataset: these only have the first 3 lines. Your job is to fill in the rest!
  held_out_sonnet_dataset = SonnetsDataset(args.held_out_sonnet_path)

  args = add_arguments(args)
  model = SonnetGPT(args)
  model = model.to(device)

  lr = args.lr
  optimizer = AdamW(model.parameters(), lr=lr)

  patience = 3  # Stop if no improvement after 3 epochs.
  best_loss = float('inf')
  epochs_no_improve = 0

  # Run for the specified number of epochs.
  for epoch in range(args.epochs):
    model.train()
    train_loss = 0
    num_batches = 0

    for batch in tqdm(sonnet_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE):
      # Get the input and move it to the gpu (I do not recommend training this model on CPU).
      b_ids, b_mask = batch['token_ids'], batch['attention_mask']
      b_ids = b_ids.to(device)
      b_mask = b_mask.to(device)

      # Compute the loss, gradients, and update the model's parameters.
      optimizer.zero_grad()
      logits = model(b_ids, b_mask)
      logits = rearrange(logits[:, :-1].contiguous(), 'b t d -> (b t) d')  # Ignore the last prediction in the sequence.
      labels = b_ids[:, 1:].contiguous().flatten()  # Ignore the first token to compose the labels.
      loss = F.cross_entropy(logits, labels, reduction='mean')
      loss.backward()
      optimizer.step()

      train_loss += loss.item()
      num_batches += 1

    train_loss = train_loss / num_batches
    print(f"Epoch {epoch}: train loss :: {train_loss :.3f}.")

    print('Generating several output sonnets...')
    model.eval()
    for batch in held_out_sonnet_dataset:
      encoding = model.tokenizer(batch[1], return_tensors='pt', padding=True, truncation=True).to(device)
      output = model.generate(encoding['input_ids'], temperature=args.temperature, top_p=args.top_p)
      print(f'{batch[1]}{output[1]}\n\n')

    # Early stopping to prevent overfitting on the small dataset of sonnets.
    if train_loss < best_loss:
        best_loss = train_loss
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1

    if epochs_no_improve >= patience:
        print(f"Stopping early at epoch {epoch}. No improvement for {patience} epochs.")
        break
    
    # save_model(model, optimizer, args, f'{epoch}_{args.filepath}')
  save_model(model, optimizer, args, f'final_{args.filepath}')


@torch.no_grad()
def generate_submission_sonnets(args):
  device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
  saved = torch.load(f'final_{args.filepath}', weights_only=False)

  model = SonnetGPT(saved['args'])
  model.load_state_dict(saved['model'])
  model = model.to(device)
  model.eval()

  # Create the held-out dataset: these only have the first 3 lines. Your job is to fill in the rest!
  held_out_sonnet_dataset = SonnetsDataset(args.held_out_sonnet_path)

  generated_sonnets = []
  for batch in held_out_sonnet_dataset:
    sonnet_id = batch[0]
    encoding = model.tokenizer(batch[1], return_tensors='pt', padding=False, truncation=True).to(device)
    output = model.generate(encoding['input_ids'], temperature=args.temperature, top_p=args.top_p)[0][0]
    decoded_output = model.tokenizer.decode(output)
    full_sonnet = f'{decoded_output}\n\n'
    generated_sonnets.append((sonnet_id, full_sonnet))

    print(f'{decoded_output}\n\n')

  with open(args.sonnet_out, "w+") as f:
    f.write(f"--Generated Sonnets-- \n\n")
    for sonnet in generated_sonnets:
      f.write(f"\n{sonnet[0]}\n")
      f.write(sonnet[1])

def get_args():
  parser = argparse.ArgumentParser()

  parser.add_argument("--sonnet_path", type=str, default="data/sonnets.txt")
  parser.add_argument("--held_out_sonnet_path", type=str, default="data/sonnets_held_out.txt")
  parser.add_argument("--sonnet_out", type=str, default="predictions/generated_sonnets.txt")

  parser.add_argument("--seed", type=int, default=11711)
  parser.add_argument("--epochs", type=int, default=10)
  parser.add_argument("--use_gpu", action='store_true')

  # Generation parameters.
  parser.add_argument("--temperature", type=float, help="softmax temperature.", default=1.2)
  parser.add_argument("--top_p", type=float, help="Cumulative probability distribution for nucleus sampling.",
                      default=0.9)

  parser.add_argument("--batch_size", help='The training batch size.', type=int, default=8)
  parser.add_argument("--lr", type=float, help="learning rate", default=1e-5)
  parser.add_argument("--model_size", type=str, help="The model size as specified on hugging face.",
                      choices=['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'], default='gpt2')

  args = parser.parse_args()
  return args


def add_arguments(args):
  """Add arguments that are deterministic on model size."""
  if args.model_size == 'gpt2':
    args.d = 768
    args.l = 12
    args.num_heads = 12
  elif args.model_size == 'gpt2-medium':
    args.d = 1024
    args.l = 24
    args.num_heads = 16
  elif args.model_size == 'gpt2-large':
    args.d = 1280
    args.l = 36
    args.num_heads = 20
  else:
    raise Exception(f'{args.model_size} is not supported.')
  return args


if __name__ == "__main__":
  args = get_args()
  args.filepath = f'{args.epochs}-{args.lr}-sonnet.pt'  # Save path.
  seed_everything(args.seed)  # Fix the seed for reproducibility.
  train(args)
  generate_submission_sonnets(args)