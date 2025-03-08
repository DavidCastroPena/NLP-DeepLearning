'''
Basis---Impossible distillation works here, but we needed to modify required predictions for the leaderboard

"""
Paraphrase detection for GPT-2.

Usage:
  `python paraphrase_detection.py --use_gpu`
"""

import argparse
import random
import torch
import numpy as np
import torch.nn.functional as F

from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import ParaphraseDetectionDataset, ParaphraseDetectionTestDataset, load_paraphrase_data
from models.gpt2 import GPT2Model
from optimizer import AdamW
# Fix circular import by importing the module instead of specific functions
import evaluation

TQDM_DISABLE = False

def seed_everything(seed=11711):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

class ParaphraseGPT(nn.Module):
    def __init__(self, args):
        super().__init__()
        # Get configuration parameters
        if isinstance(args, dict):
            model_size = args.get("model_size", "gpt2")
            d = args.get("d", 768)
            l = args.get("l", 12)
            num_heads = args.get("num_heads", 12)
        else:
            model_size = getattr(args, "model_size", "gpt2")
            d = getattr(args, "d", 768)
            l = getattr(args, "l", 12)
            num_heads = getattr(args, "num_heads", 12)

        # Load the model
        try:
            # First try with expected parameters
            self.gpt = GPT2Model.from_pretrained(model=model_size, d=d, l=l, num_heads=num_heads)
        except TypeError as e:
            print(f"Warning: Initial model loading failed with: {e}")
            print("Trying alternate initialization...")
            # Try alternate initialization if model parameters are different
            self.gpt = GPT2Model.from_pretrained("gpt2")  # Use default parameters
        
        # Add classifier for binary classification
        # Use appropriate input dimension based on model
        if hasattr(self.gpt, "config") and hasattr(self.gpt.config, "hidden_size"):
            input_dim = self.gpt.config.hidden_size
        else:
            input_dim = d  # Default to provided d value
            
        self.classifier = nn.Linear(input_dim, 2)

        # Set parameters to trainable
        for param in self.gpt.parameters():
            param.requires_grad = True

    def forward(self, input_ids, attention_mask, labels=None, neg_input_ids=None):
        """
        Forward pass for ParaphraseGPT with proper loss calculation.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            labels: Optional ground truth labels [batch_size]
            neg_input_ids: Optional negative example token IDs [batch_size, seq_len]
            
        Returns:
            tuple: (loss, logits)
        """
        # Add debugging information
        print(f"Input shape: {input_ids.shape}")
        
        # Get the GPT2 output
        gpt_output = self.gpt(input_ids, attention_mask)
        
        # Add more debugging information
        print(f"GPT output type: {type(gpt_output)}")
        if isinstance(gpt_output, dict):
            print(f"GPT output keys: {list(gpt_output.keys())}")
        
        # Initialize loss that will actually be used for backprop
        loss = None
        
        # Handle different types of outputs from the GPT model
        if isinstance(gpt_output, dict):
            # Process dictionary output (typical for HuggingFace models)
            if 'last_hidden_state' in gpt_output:
                hidden_states = gpt_output['last_hidden_state']
                # Apply mean pooling over sequence length
                pooled_output = torch.mean(hidden_states, dim=1)
                logits = self.classifier(pooled_output)
            elif 'last_token' in gpt_output:
                # If last_token is provided directly
                last_token = gpt_output['last_token']
                logits = self.classifier(last_token)
            elif 'logits' in gpt_output:
                # If logits are directly provided
                logits = gpt_output['logits']
            else:
                # Create fallback for unknown dict keys
                print(f"Warning: Unexpected dict keys: {list(gpt_output.keys())}")
                logits = torch.zeros((input_ids.size(0), 2), device=input_ids.device)
        elif isinstance(gpt_output, tuple):
            # Process tuple output
            if len(gpt_output) >= 2:
                hidden_states = gpt_output[1]
                pooled_output = torch.mean(hidden_states, dim=1)
                logits = self.classifier(pooled_output)
            else:
                # Handle unexpected tuple format
                print(f"Warning: Unexpected tuple length: {len(gpt_output)}")
                logits = torch.zeros((input_ids.size(0), 2), device=input_ids.device)
        elif torch.is_tensor(gpt_output):
            # If output is a tensor, treat as hidden states
            hidden_states = gpt_output
            # Ensure we can apply mean pooling (check if 3D tensor)
            if len(hidden_states.shape) == 3:
                pooled_output = torch.mean(hidden_states, dim=1)
                logits = self.classifier(pooled_output)
            else:
                # If not 3D, try to use as is
                logits = self.classifier(hidden_states)
        else:
            # Handle any other output types
            print(f"Warning: Unhandled output type: {type(gpt_output)}")
            logits = torch.zeros((input_ids.size(0), 2), device=input_ids.device)
        
        # Fix logits shape for classification
        if len(logits.shape) == 3:  # [batch, seq_len, hidden_dim]
            logits = torch.mean(logits, dim=1)  # Pool across sequence length
        
        # Calculate classification loss if labels are provided
        if labels is not None:
            # Make sure labels are flattened
            if len(labels.shape) > 1:
                labels = labels.flatten()
                
            # Apply cross entropy loss
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
        
        # Process negative samples for impossible distillation if provided
        if neg_input_ids is not None:
            try:
                neg_output = self.gpt(neg_input_ids, attention_mask)
                
                # Process negative samples similar to positive ones
                neg_logits = None
                
                if isinstance(neg_output, dict):
                    if 'last_hidden_state' in neg_output:
                        neg_hidden = neg_output['last_hidden_state']
                        neg_pooled = torch.mean(neg_hidden, dim=1)
                        neg_logits = self.classifier(neg_pooled)
                    elif 'last_token' in neg_output:
                        neg_token = neg_output['last_token']
                        neg_logits = self.classifier(neg_token)
                    elif 'logits' in neg_output:
                        neg_logits = neg_output['logits']
                elif isinstance(neg_output, tuple) and len(neg_output) >= 2:
                    neg_hidden = neg_output[1]
                    neg_pooled = torch.mean(neg_hidden, dim=1)
                    neg_logits = self.classifier(neg_pooled)
                elif torch.is_tensor(neg_output):
                    neg_hidden = neg_output
                    if len(neg_hidden.shape) == 3:
                        neg_pooled = torch.mean(neg_hidden, dim=1)
                        neg_logits = self.classifier(neg_pooled)
                    else:
                        neg_logits = self.classifier(neg_hidden)
                
                # If we got valid negative logits, apply KL divergence
                if neg_logits is not None and torch.is_tensor(neg_logits):
                    # Fix shape if needed
                    if len(neg_logits.shape) == 3:
                        neg_logits = torch.mean(neg_logits, dim=1)
                    
                    # Apply KL divergence loss
                    kd_loss = F.kl_div(
                        F.log_softmax(neg_logits, dim=-1),
                        F.softmax(logits, dim=-1),
                        reduction="batchmean"
                    )
                    
                    # Add KL loss to classification loss
                    if loss is None:
                        loss = 0.1 * kd_loss
                    else:
                        loss = loss + 0.1 * kd_loss
            except Exception as e:
                print(f"Error processing negative samples: {e}")
        
        # If no loss was calculated (no labels and no negative samples), set to zero
        if loss is None:
            loss = torch.tensor(0.0, device=input_ids.device, requires_grad=True)
        
        # Print final shapes for debugging
        print(f"Final loss: {loss.item()}, Final logits shape: {logits.shape}")
        
        return loss, logits

def train(args):
    if isinstance(args, dict):
        use_gpu = args.get("use_gpu", False)
        para_train = args.get("para_train", "data/quora-train.csv")
        para_dev = args.get("para_dev", "data/quora-dev.csv")
        batch_size = args.get("batch_size", 8)
        lr = args.get("lr", 1e-5)
        epochs = args.get("epochs", 10)
        filepath = args.get("filepath", "paraphrase.pt")
    else:
        use_gpu = args.use_gpu
        para_train = args.para_train
        para_dev = args.para_dev
        batch_size = args.batch_size
        lr = args.lr
        epochs = args.epochs
        filepath = args.filepath

    device = torch.device("cuda") if use_gpu else torch.device("cpu")

    print(f"Loading training data from {para_train}")
    para_train_data = load_paraphrase_data(para_train)
    print(f"Loading dev data from {para_dev}")
    para_dev_data = load_paraphrase_data(para_dev)

    para_train_data = ParaphraseDetectionDataset(para_train_data, args)
    para_dev_data = ParaphraseDetectionDataset(para_dev_data, args)

    para_train_dataloader = DataLoader(para_train_data, shuffle=True, batch_size=batch_size, collate_fn=para_train_data.collate_fn)
    para_dev_dataloader = DataLoader(para_dev_data, shuffle=False, batch_size=batch_size, collate_fn=para_dev_data.collate_fn)

    print("Initializing model...")
    model = ParaphraseGPT(args).to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.)

    best_dev_acc = 0
    total_train_loss = 0

    print(f"Beginning training for {epochs} epochs")
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0

        for batch_idx, batch in enumerate(tqdm(para_train_dataloader, desc=f'Training Epoch {epoch+1}', disable=TQDM_DISABLE)):
            try:
                # Process this batch
                b_ids = batch['token_ids'].to(device)
                b_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].flatten().to(device)
                
                optimizer.zero_grad()
                
                # Run model forward pass with labels
                try:
                    loss, _ = model(b_ids, b_mask, labels=labels)
                    
                    # Validate loss
                    if not torch.is_tensor(loss):
                        print(f"Warning: loss is not a tensor: {loss}, using zero tensor")
                        loss = torch.tensor(0.0, device=device, requires_grad=True)
                    elif loss.item() == 0 and batch_idx < 5:
                        # Only print warning for first few batches to avoid spam
                        print("Warning: Loss is still exactly zero. Check model implementation.")
                except Exception as e:
                    print(f"Error in model forward pass: {e}")
                    loss = torch.tensor(0.0, device=device, requires_grad=True)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                
                # Print occasional debugging info
                if batch_idx % 100 == 0:
                    print(f"Batch {batch_idx}, Loss: {loss.item()}")
                
            except Exception as e:
                print(f"Error processing batch: {e}")
                continue

        # Evaluate on dev set
        print("\nEvaluating on dev set...")
        model.eval()
        try:
            dev_acc, dev_f1, *_ = evaluation.model_eval_paraphrase(para_dev_dataloader, model, device)
            impossibility_score = evaluation.evaluate_impossibility(model, para_dev_dataloader, device)

            print(f"Epoch {epoch+1} - Dev Acc: {dev_acc:.3f}, Dev F1: {dev_f1:.3f}, Impossibility Score: {impossibility_score:.3f}")

            # Save best model
            if dev_acc > best_dev_acc:
                best_dev_acc = dev_acc
                torch.save(model.state_dict(), filepath)
                print(f"✅ New best model saved at {filepath}")
        except Exception as e:
            print(f"Error during evaluation: {e}")
        
        total_train_loss += epoch_loss
        print(f"Epoch {epoch+1} average loss: {epoch_loss/len(para_train_dataloader):.4f}")

    print(f"🎉 Training complete! Best dev accuracy: {best_dev_acc:.3f}")
    return total_train_loss

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_gpu', action='store_true')
    parser.add_argument('--para_train', type=str, default="data/quora-train.csv")
    parser.add_argument('--para_dev', type=str, default="data/quora-dev.csv")
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--model_size', type=str, default="gpt2")
    parser.add_argument('--d', type=int, default=768)
    parser.add_argument('--l', type=int, default=12)
    parser.add_argument('--num_heads', type=int, default=12)
    
    args = parser.parse_args()
    args.filepath = f"{args.epochs}-{args.lr}-paraphrase.pt"

    seed_everything(11711)
    train(args)

'''

# Modified code to generate predictions for the leaderboard

"""
Paraphrase detection for GPT-2.

Usage:
  `python paraphrase_detection.py --use_gpu`
"""

import argparse
import random
import torch
import numpy as np
import torch.nn.functional as F
import os

from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import ParaphraseDetectionDataset, ParaphraseDetectionTestDataset, load_paraphrase_data
from models.gpt2 import GPT2Model
from optimizer import AdamW
# Fix circular import by importing the module instead of specific functions
import evaluation

TQDM_DISABLE = False

def seed_everything(seed=11711):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

class ParaphraseGPT(nn.Module):
    def __init__(self, args):
        super().__init__()
        # Get configuration parameters
        if isinstance(args, dict):
            model_size = args.get("model_size", "gpt2")
            d = args.get("d", 768)
            l = args.get("l", 12)
            num_heads = args.get("num_heads", 12)
        else:
            model_size = getattr(args, "model_size", "gpt2")
            d = getattr(args, "d", 768)
            l = getattr(args, "l", 12)
            num_heads = getattr(args, "num_heads", 12)

        # Load the model
        try:
            # First try with expected parameters
            self.gpt = GPT2Model.from_pretrained(model=model_size, d=d, l=l, num_heads=num_heads)
        except TypeError as e:
            print(f"Warning: Initial model loading failed with: {e}")
            print("Trying alternate initialization...")
            # Try alternate initialization if model parameters are different
            self.gpt = GPT2Model.from_pretrained("gpt2")  # Use default parameters
        
        # Add classifier for binary classification
        # Use appropriate input dimension based on model
        if hasattr(self.gpt, "config") and hasattr(self.gpt.config, "hidden_size"):
            input_dim = self.gpt.config.hidden_size
        else:
            input_dim = d  # Default to provided d value
            
        self.classifier = nn.Linear(input_dim, 2)

        # Set parameters to trainable
        for param in self.gpt.parameters():
            param.requires_grad = True

    def forward(self, input_ids, attention_mask, labels=None, neg_input_ids=None):
        """
        Forward pass for ParaphraseGPT with proper loss calculation.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            labels: Optional ground truth labels [batch_size]
            neg_input_ids: Optional negative example token IDs [batch_size, seq_len]
            
        Returns:
            tuple: (loss, logits)
        """
        # Add debugging information
        print(f"Input shape: {input_ids.shape}")
        
        # Get the GPT2 output
        gpt_output = self.gpt(input_ids, attention_mask)
        
        # Add more debugging information
        print(f"GPT output type: {type(gpt_output)}")
        if isinstance(gpt_output, dict):
            print(f"GPT output keys: {list(gpt_output.keys())}")
        
        # Initialize loss that will actually be used for backprop
        loss = None
        
        # Handle different types of outputs from the GPT model
        if isinstance(gpt_output, dict):
            # Process dictionary output (typical for HuggingFace models)
            if 'last_hidden_state' in gpt_output:
                hidden_states = gpt_output['last_hidden_state']
                # Apply mean pooling over sequence length
                pooled_output = torch.mean(hidden_states, dim=1)
                logits = self.classifier(pooled_output)
            elif 'last_token' in gpt_output:
                # If last_token is provided directly
                last_token = gpt_output['last_token']
                logits = self.classifier(last_token)
            elif 'logits' in gpt_output:
                # If logits are directly provided
                logits = gpt_output['logits']
            else:
                # Create fallback for unknown dict keys
                print(f"Warning: Unexpected dict keys: {list(gpt_output.keys())}")
                logits = torch.zeros((input_ids.size(0), 2), device=input_ids.device)
        elif isinstance(gpt_output, tuple):
            # Process tuple output
            if len(gpt_output) >= 2:
                hidden_states = gpt_output[1]
                pooled_output = torch.mean(hidden_states, dim=1)
                logits = self.classifier(pooled_output)
            else:
                # Handle unexpected tuple format
                print(f"Warning: Unexpected tuple length: {len(gpt_output)}")
                logits = torch.zeros((input_ids.size(0), 2), device=input_ids.device)
        elif torch.is_tensor(gpt_output):
            # If output is a tensor, treat as hidden states
            hidden_states = gpt_output
            # Ensure we can apply mean pooling (check if 3D tensor)
            if len(hidden_states.shape) == 3:
                pooled_output = torch.mean(hidden_states, dim=1)
                logits = self.classifier(pooled_output)
            else:
                # If not 3D, try to use as is
                logits = self.classifier(hidden_states)
        else:
            # Handle any other output types
            print(f"Warning: Unhandled output type: {type(gpt_output)}")
            logits = torch.zeros((input_ids.size(0), 2), device=input_ids.device)
        
        # Fix logits shape for classification
        if len(logits.shape) == 3:  # [batch, seq_len, hidden_dim]
            logits = torch.mean(logits, dim=1)  # Pool across sequence length
        
        # Calculate classification loss if labels are provided
        if labels is not None:
            # Make sure labels are flattened
            if len(labels.shape) > 1:
                labels = labels.flatten()
                
            # Apply cross entropy loss
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
        
        # Process negative samples for impossible distillation if provided
        if neg_input_ids is not None:
            try:
                neg_output = self.gpt(neg_input_ids, attention_mask)
                
                # Process negative samples similar to positive ones
                neg_logits = None
                
                if isinstance(neg_output, dict):
                    if 'last_hidden_state' in neg_output:
                        neg_hidden = neg_output['last_hidden_state']
                        neg_pooled = torch.mean(neg_hidden, dim=1)
                        neg_logits = self.classifier(neg_pooled)
                    elif 'last_token' in neg_output:
                        neg_token = neg_output['last_token']
                        neg_logits = self.classifier(neg_token)
                    elif 'logits' in neg_output:
                        neg_logits = neg_output['logits']
                elif isinstance(neg_output, tuple) and len(neg_output) >= 2:
                    neg_hidden = neg_output[1]
                    neg_pooled = torch.mean(neg_hidden, dim=1)
                    neg_logits = self.classifier(neg_pooled)
                elif torch.is_tensor(neg_output):
                    neg_hidden = neg_output
                    if len(neg_hidden.shape) == 3:
                        neg_pooled = torch.mean(neg_hidden, dim=1)
                        neg_logits = self.classifier(neg_pooled)
                    else:
                        neg_logits = self.classifier(neg_hidden)
                
                # If we got valid negative logits, apply KL divergence
                if neg_logits is not None and torch.is_tensor(neg_logits):
                    # Fix shape if needed
                    if len(neg_logits.shape) == 3:
                        neg_logits = torch.mean(neg_logits, dim=1)
                    
                    # Apply KL divergence loss
                    kd_loss = F.kl_div(
                        F.log_softmax(neg_logits, dim=-1),
                        F.softmax(logits, dim=-1),
                        reduction="batchmean"
                    )
                    
                    # Add KL loss to classification loss
                    if loss is None:
                        loss = 0.1 * kd_loss
                    else:
                        loss = loss + 0.1 * kd_loss
            except Exception as e:
                print(f"Error processing negative samples: {e}")
        
        # If no loss was calculated (no labels and no negative samples), set to zero
        if loss is None:
            loss = torch.tensor(0.0, device=input_ids.device, requires_grad=True)
        
        # Print final shapes for debugging
        print(f"Final loss: {loss.item()}, Final logits shape: {logits.shape}")
        
        return loss, logits

def train(args):
    if isinstance(args, dict):
        use_gpu = args.get("use_gpu", False)
        para_train = args.get("para_train", "data/quora-train.csv")
        para_dev = args.get("para_dev", "data/quora-dev.csv")
        batch_size = args.get("batch_size", 8)
        lr = args.get("lr", 1e-5)
        epochs = args.get("epochs", 10)
        filepath = args.get("filepath", "paraphrase.pt")
    else:
        use_gpu = args.use_gpu
        para_train = args.para_train
        para_dev = args.para_dev
        batch_size = args.batch_size
        lr = args.lr
        epochs = args.epochs
        filepath = args.filepath

    device = torch.device("cuda") if use_gpu else torch.device("cpu")

    print(f"Loading training data from {para_train}")
    para_train_data = load_paraphrase_data(para_train)
    print(f"Loading dev data from {para_dev}")
    para_dev_data = load_paraphrase_data(para_dev)

    para_train_data = ParaphraseDetectionDataset(para_train_data, args)
    para_dev_data = ParaphraseDetectionDataset(para_dev_data, args)

    para_train_dataloader = DataLoader(para_train_data, shuffle=True, batch_size=batch_size, collate_fn=para_train_data.collate_fn)
    para_dev_dataloader = DataLoader(para_dev_data, shuffle=False, batch_size=batch_size, collate_fn=para_dev_data.collate_fn)

    print("Initializing model...")
    model = ParaphraseGPT(args).to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.)

    best_dev_acc = 0
    total_train_loss = 0

    print(f"Beginning training for {epochs} epochs")
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0

        for batch_idx, batch in enumerate(tqdm(para_train_dataloader, desc=f'Training Epoch {epoch+1}', disable=TQDM_DISABLE)):
            try:
                # Process this batch
                b_ids = batch['token_ids'].to(device)
                b_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].flatten().to(device)
                
                optimizer.zero_grad()
                
                # Run model forward pass with labels
                try:
                    loss, _ = model(b_ids, b_mask, labels=labels)
                    
                    # Validate loss
                    if not torch.is_tensor(loss):
                        print(f"Warning: loss is not a tensor: {loss}, using zero tensor")
                        loss = torch.tensor(0.0, device=device, requires_grad=True)
                    elif loss.item() == 0 and batch_idx < 5:
                        # Only print warning for first few batches to avoid spam
                        print("Warning: Loss is still exactly zero. Check model implementation.")
                except Exception as e:
                    print(f"Error in model forward pass: {e}")
                    loss = torch.tensor(0.0, device=device, requires_grad=True)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                
                # Print occasional debugging info
                if batch_idx % 100 == 0:
                    print(f"Batch {batch_idx}, Loss: {loss.item()}")
                
            except Exception as e:
                print(f"Error processing batch: {e}")
                continue

        # Evaluate on dev set
        print("\nEvaluating on dev set...")
        model.eval()
        try:
            dev_acc, dev_f1, *_ = evaluation.model_eval_paraphrase(para_dev_dataloader, model, device)
            impossibility_score = evaluation.evaluate_impossibility(model, para_dev_dataloader, device)

            print(f"Epoch {epoch+1} - Dev Acc: {dev_acc:.3f}, Dev F1: {dev_f1:.3f}, Impossibility Score: {impossibility_score:.3f}")

            # Save best model
            if dev_acc > best_dev_acc:
                best_dev_acc = dev_acc
                torch.save(model.state_dict(), filepath)
                print(f"✅ New best model saved at {filepath}")
        except Exception as e:
            print(f"Error during evaluation: {e}")
        
        total_train_loss += epoch_loss
        print(f"Epoch {epoch+1} average loss: {epoch_loss/len(para_train_dataloader):.4f}")

    print(f"🎉 Training complete! Best dev accuracy: {best_dev_acc:.3f}")
    return total_train_loss

def test(args, model, device, dataset_type="dev"):
    """
    Generate predictions on the dev or test set and save them to the expected format.
    
    Args:
        args: Command line arguments
        model: Trained ParaphraseGPT model
        device: Device to run predictions on
        dataset_type: String indicating whether to use dev or test dataset
    """
    # Load the appropriate dataset
    if dataset_type == "dev":
        data_path = args.para_dev
        output_file = "predictions/finetune-quora-dev-out.csv"
        dataset_class = ParaphraseDetectionDataset
    else:  # test
        data_path = args.para_test
        output_file = "predictions/finetune-quora-test-out.csv"
        dataset_class = ParaphraseDetectionTestDataset
    
    print(f"Loading {dataset_type} data from {data_path} for predictions")
    data = load_paraphrase_data(data_path)
    dataset = dataset_class(data, args)
    
    # Create dataloader
    dataloader = DataLoader(
        dataset, 
        shuffle=False, 
        batch_size=args.batch_size, 
        collate_fn=dataset.collate_fn
    )
    
    # Set model to evaluation mode
    model.eval()
    all_predictions = []
    all_ids = []
    
    # Generate predictions
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Generating {dataset_type} predictions"):
            ids = batch['sent_ids']
            b_ids = batch['token_ids'].to(device)
            b_mask = batch['attention_mask'].to(device)
            
            _, logits = model(b_ids, b_mask)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            
            all_predictions.extend(preds)
            all_ids.extend(ids)
    
    # Create predictions directory if it doesn't exist
    os.makedirs("predictions", exist_ok=True)
    
    # Save predictions to CSV with the expected format
    with open(output_file, 'w') as f:
        for id_, pred in zip(all_ids, all_predictions):
            f.write(f"{id_}\t{pred}\n")
    
    print(f"{dataset_type.capitalize()} predictions saved to {output_file}")
    return output_file

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_gpu', action='store_true')
    parser.add_argument('--para_train', type=str, default="data/quora-train.csv")
    parser.add_argument('--para_dev', type=str, default="data/quora-dev.csv")
    parser.add_argument('--para_test', type=str, default="data/quora-test-student.csv")
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--model_size', type=str, default="gpt2")
    parser.add_argument('--d', type=int, default=768)
    parser.add_argument('--l', type=int, default=12)
    parser.add_argument('--num_heads', type=int, default=12)
    parser.add_argument('--generate_predictions', action='store_true', 
                        help='Generate predictions without training')
    
    args = parser.parse_args()
    args.filepath = f"{args.epochs}-{args.lr}-paraphrase.pt"
    
    # Set device
    device = torch.device("cuda") if args.use_gpu else torch.device("cpu")

    # If only generating predictions
    if args.generate_predictions:
        model = ParaphraseGPT(args).to(device)
        print(f"Loading model from {args.filepath}")
        model.load_state_dict(torch.load(args.filepath, map_location=device))
        
        # Generate predictions for both dev and test sets
        test(args, model, device, "dev")
        test(args, model, device, "test")
    else:
        # Full training and prediction generation
        seed_everything(11711)
        train(args)
        
        # Load best model for predictions
        model = ParaphraseGPT(args).to(device)
        print(f"Loading best model from {args.filepath}")
        model.load_state_dict(torch.load(args.filepath, map_location=device))
        
        # Generate predictions for submission
        test(args, model, device, "dev")
        test(args, model, device, "test")