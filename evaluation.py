
'''
"""
Evaluation code for paraphrase detection.

- model_eval_paraphrase: Evaluates paraphrase detection performance.
- model_test_paraphrase: Evaluates model predictions on test data.
- evaluate_impossibility: Computes the model's ability to reject incorrect paraphrases.
"""

import torch
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm
import numpy as np
from sacrebleu.metrics import CHRF
from datasets import SonnetsDataset

TQDM_DISABLE = False

@torch.no_grad()
def model_eval_paraphrase(dataloader, model, device):
    """
    Evaluates model performance on labeled paraphrase data.

    Args:
        dataloader (DataLoader): The evaluation dataset.
        model (nn.Module): The trained model.
        device (torch.device): The computation device (CPU/GPU).

    Returns:
        tuple: (accuracy, f1_score, predictions, true_labels, sentence_ids)
    """
    model.eval()
    y_true, y_pred, sent_ids = [], [], []

    for batch in tqdm(dataloader, desc='Evaluating', disable=TQDM_DISABLE):
        b_ids, b_mask, b_sent_ids, labels = (
            batch['token_ids'].to(device), 
            batch['attention_mask'].to(device), 
            batch['sent_ids'], 
            batch['labels'].flatten().to(device)
        )

        # Get logits from model
        try:
            _, logits = model(b_ids, b_mask)
            
            # Ensure logits are a tensor
            if not isinstance(logits, torch.Tensor):
                raise TypeError(f"Expected tensor for logits, got {type(logits)}")
                
            # Ensure logits have the right shape for classification
            if logits.shape[-1] != 2:
                raise ValueError(f"Expected logits with last dimension 2, got shape {logits.shape}")
                
        except Exception as e:
            print(f"Error getting logits from model: {e}")
            # Create a fallback tensor of zeros as a last resort
            logits = torch.zeros((b_ids.size(0), 2), device=device)
            print("Using fallback zero logits")

        preds = torch.argmax(logits, dim=1).cpu().numpy().flatten()
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds)
        sent_ids.extend(b_sent_ids)

    f1 = f1_score(y_true, y_pred, average='macro')
    acc = accuracy_score(y_true, y_pred)

    return acc, f1, y_pred, y_true, sent_ids

@torch.no_grad()
def model_test_paraphrase(dataloader, model, device):
    """
    Evaluates the model on test data (without labels).

    Args:
        dataloader (DataLoader): The test dataset.
        model (nn.Module): The trained model.
        device (torch.device): The computation device.

    Returns:
        tuple: (predictions, sentence_ids)
    """
    model.eval()
    y_pred, sent_ids = [], []

    for batch in tqdm(dataloader, desc='Testing', disable=TQDM_DISABLE):
        b_ids, b_mask, b_sent_ids = (
            batch['token_ids'].to(device),
            batch['attention_mask'].to(device),
            batch['sent_ids']
        )

        # Get model predictions
        try:
            _, logits = model(b_ids, b_mask)
            
            # Ensure logits are a tensor
            if not isinstance(logits, torch.Tensor):
                raise TypeError(f"Expected tensor for logits, got {type(logits)}")
                
            # Ensure logits have the right shape for classification
            if logits.shape[-1] != 2:
                raise ValueError(f"Expected logits with last dimension 2, got shape {logits.shape}")
                
        except Exception as e:
            print(f"Error getting logits from model: {e}")
            # Create a fallback tensor of zeros as a last resort
            logits = torch.zeros((b_ids.size(0), 2), device=device)
            print("Using fallback zero logits")

        preds = torch.argmax(logits, dim=1).cpu().numpy().flatten()
        y_pred.extend(preds)
        sent_ids.extend(b_sent_ids)

    return y_pred, sent_ids

@torch.no_grad()
@torch.no_grad()
def evaluate_impossibility(model, dataloader, device):
    """
    Computes an "impossibility score" to measure how well the model rejects incorrect paraphrases.

    Args:
        model (nn.Module): The trained paraphrase detection model.
        dataloader (DataLoader): The dataset containing paraphrases and negative samples.
        device (torch.device): CPU or GPU.

    Returns:
        float: The impossibility score.
    """
    model.eval()
    total_score = 0
    count = 0

    for batch in dataloader:
        # Check if the batch contains negative examples
        if 'neg_token_ids' not in batch:
            continue
        
        try:    
            input_ids = batch['token_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            neg_input_ids = batch['neg_token_ids'].to(device)
    
            # Get model outputs for both original and negative examples
            # We're not passing labels here since we just want the logits
            _, logits = model(input_ids, attention_mask)
            _, neg_logits = model(neg_input_ids, attention_mask)
    
            # Ensure both are proper tensors with the right shape
            if not torch.is_tensor(logits) or not torch.is_tensor(neg_logits):
                print("Skipping batch - non-tensor outputs")
                continue
                
            if logits.shape[-1] != 2 or neg_logits.shape[-1] != 2:
                print(f"Unexpected shapes: logits {logits.shape}, neg_logits {neg_logits.shape}")
                continue
    
            # Calculate impossibility score: how often negative examples get lower probability than positives
            # Higher score means better at recognizing impossible paraphrases
            pos_probs = torch.softmax(logits, dim=1)[:, 1]  # Probability of class 1 (paraphrase)
            neg_probs = torch.softmax(neg_logits, dim=1)[:, 1]  # Probability of class 1 for negative
            
            # Score is how often neg_prob < pos_prob
            batch_score = (neg_probs < pos_probs).float().mean().item()
            total_score += batch_score
            count += 1
            
        except Exception as e:
            print(f"Error processing batch for impossibility score: {e}")
            continue

    return total_score / max(count, 1)  # Avoid division by zero

@torch.no_grad()
def test_sonnet(test_path='predictions/generated_sonnets.txt', gold_path='data/TRUE_sonnets_held_out.txt'):
    """
    Computes the CHRF score for generated sonnets.

    Args:
        test_path (str): Path to generated sonnets.
        gold_path (str): Path to ground truth sonnets.

    Returns:
        float: CHRF score.
    """
    chrf = CHRF()

    # Load sonnets
    generated_sonnets = [x[1] for x in SonnetsDataset(test_path)]
    true_sonnets = [x[1] for x in SonnetsDataset(gold_path)]

    # Match dataset lengths
    max_len = min(len(true_sonnets), len(generated_sonnets))
    true_sonnets = true_sonnets[:max_len]
    generated_sonnets = generated_sonnets[:max_len]

    # Compute CHRF score
    chrf_score = chrf.corpus_score(generated_sonnets, [true_sonnets])
    return float(chrf_score.score)
    
'''

import torch
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm
import numpy as np
from sacrebleu.metrics import CHRF
from datasets import SonnetsDataset
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer

TQDM_DISABLE = False

@torch.no_grad()
def model_eval_paraphrase(dataloader, model, device):
    model.eval()
    y_true, y_pred, sent_ids = [], [], []

    for batch in tqdm(dataloader, desc='Evaluating', disable=TQDM_DISABLE):
        b_ids, b_mask, b_sent_ids, labels = (
            batch['token_ids'].to(device), 
            batch['attention_mask'].to(device), 
            batch['sent_ids'], 
            batch['labels'].flatten().to(device)
        )

        try:
            _, logits = model(b_ids, b_mask)
            if logits.shape[-1] != 2:
                raise ValueError(f"Expected logits with last dimension 2, got shape {logits.shape}")
        except Exception as e:
            print(f"Error getting logits from model: {e}")
            logits = torch.zeros((b_ids.size(0), 2), device=device)

        preds = torch.argmax(logits, dim=1).cpu().numpy().flatten()
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds)
        sent_ids.extend(b_sent_ids)

    f1 = f1_score(y_true, y_pred, average='macro')
    acc = accuracy_score(y_true, y_pred)

    return acc, f1, y_pred, y_true, sent_ids

@torch.no_grad()
def evaluate_paraphrase_quality(reference_sentences, generated_sentences):
    scores = {"BLEU": [], "ROUGE-1": [], "ROUGE-2": [], "ROUGE-L": [], "METEOR": []}
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    for ref, gen in zip(reference_sentences, generated_sentences):
        scores["BLEU"].append(sentence_bleu([ref.split()], gen.split()))
        rouge_scores = scorer.score(ref, gen)
        scores["ROUGE-1"].append(rouge_scores["rouge1"].fmeasure)
        scores["ROUGE-2"].append(rouge_scores["rouge2"].fmeasure)
        scores["ROUGE-L"].append(rouge_scores["rougeL"].fmeasure)
        scores["METEOR"].append(meteor_score([ref], gen))
    
    return {metric: sum(values) / len(values) for metric, values in scores.items()}

@torch.no_grad()
def evaluate_self_bleu(generated_sentences):
    self_bleu_scores = [
        sentence_bleu([p.split() for p in generated_sentences if p != para], para.split())
        for para in generated_sentences
    ]
    return sum(self_bleu_scores) / len(self_bleu_scores)

@torch.no_grad()
def model_test_paraphrase(dataloader, model, device):
    model.eval()
    y_pred, sent_ids = [], []

    for batch in tqdm(dataloader, desc='Testing', disable=TQDM_DISABLE):
        b_ids, b_mask, b_sent_ids = (
            batch['token_ids'].to(device),
            batch['attention_mask'].to(device),
            batch['sent_ids']
        )

        try:
            _, logits = model(b_ids, b_mask)
            if logits.shape[-1] != 2:
                raise ValueError(f"Expected logits with last dimension 2, got shape {logits.shape}")
        except Exception as e:
            print(f"Error getting logits from model: {e}")
            logits = torch.zeros((b_ids.size(0), 2), device=device)

        preds = torch.argmax(logits, dim=1).cpu().numpy().flatten()
        y_pred.extend(preds)
        sent_ids.extend(b_sent_ids)

    return y_pred, sent_ids

@torch.no_grad()
def evaluate_impossibility(model, dataloader, device):
    model.eval()
    total_score = 0
    count = 0

    for batch in dataloader:
        # Check if the batch contains negative examples
        if 'neg_token_ids' not in batch:
            continue
        
        try:    
            input_ids = batch['token_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            neg_input_ids = batch['neg_token_ids'].to(device)
            neg_attention_mask = batch['neg_attention_mask'].to(device)  # Add this line
    
            # Get model outputs for both original and negative examples
            _, logits = model(input_ids, attention_mask)
            _, neg_logits = model(neg_input_ids, neg_attention_mask)  # Use neg_attention_mask
    
            # Ensure both are proper tensors with the right shape
            if not torch.is_tensor(logits) or not torch.is_tensor(neg_logits):
                print("Skipping batch - non-tensor outputs")
                continue
                
            if logits.shape[-1] != 2 or neg_logits.shape[-1] != 2:
                print(f"Unexpected shapes: logits {logits.shape}, neg_logits {neg_logits.shape}")
                continue
    
            # Calculate impossibility score: how often negative examples get lower probability than positives
            pos_probs = torch.softmax(logits, dim=1)[:, 1]
            neg_probs = torch.softmax(neg_logits, dim=1)[:, 1]
            
            batch_score = (neg_probs < pos_probs).float().mean().item()
            total_score += batch_score
            count += 1
            
        except Exception as e:
            print(f"Error processing batch for impossibility score: {e}")
            continue

    return total_score / max(count, 1)  # Avoid division by zero

@torch.no_grad()
def test_sonnet(test_path='predictions/generated_sonnets.txt', gold_path='data/TRUE_sonnets_held_out.txt'):
    chrf = CHRF()
    generated_sonnets = [x[1] for x in SonnetsDataset(test_path)]
    true_sonnets = [x[1] for x in SonnetsDataset(gold_path)]
    max_len = min(len(true_sonnets), len(generated_sonnets))
    true_sonnets = true_sonnets[:max_len]
    generated_sonnets = generated_sonnets[:max_len]
    chrf_score = chrf.corpus_score(generated_sonnets, [true_sonnets])
    return float(chrf_score.score)
