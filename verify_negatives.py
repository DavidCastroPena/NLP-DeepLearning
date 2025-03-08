# verify_negatives.py

import random
import torch
from datasets import ParaphraseDetectionDataset, load_paraphrase_data

# Set seed for reproducibility
random.seed(42)
torch.manual_seed(42)

def print_separator():
    print("\n" + "="*80 + "\n")

print("Testing hard negative sample generation...")

# Create a simple test dataset
test_data = [
    ("What is the capital of France?", "What's France's capital city?", 1, "id1"),
    ("How tall is Mount Everest?", "What's the height of Everest?", 1, "id2"),
    ("What time is it?", "How to bake bread?", 0, "id3"),
    ("Where can I find a good pizza?", "Where to eat pizza?", 1, "id4"),
]

# Mock args object
class Args:
    pass

args = Args()

# Initialize dataset
print("Creating dataset...")
dataset = ParaphraseDetectionDataset(test_data, args)

# Manually test the hard negative generation
print_separator()
print("TESTING HARD NEGATIVE GENERATION:")
batch_indices = [0, 1]  # First two examples
original_sent1 = [test_data[i][0] for i in batch_indices]
original_sent2 = [test_data[i][1] for i in batch_indices]

print("Original pairs:")
for i, (s1, s2) in enumerate(zip(original_sent1, original_sent2)):
    print(f"{i+1}. \"{s1}\" | \"{s2}\"")

# Generate hard negatives
print("\nGenerating hard negatives...")
hard_negatives = dataset.create_hard_negatives(batch_indices, original_sent1, original_sent2)

print("\nHard negative pairs:")
for i, (s1, s2) in enumerate(hard_negatives):
    print(f"{i+1}. \"{s1}\" | \"{s2}\"")

# Test the full collate_fn
print_separator()
print("TESTING FULL COLLATE FUNCTION:")

batch = [test_data[0], test_data[1]]
print("Creating batch with collate_fn...")
try:
    collated = dataset.collate_fn(batch)
    print("Successfully collated batch!")
    
    if 'neg_token_ids' in collated:
        print("\nNegative examples were generated successfully!")
        print(f"neg_token_ids shape: {collated['neg_token_ids'].shape}")
        print(f"neg_attention_mask shape: {collated['neg_attention_mask'].shape}")
        
        # Decode one example
        print("\nSample decoded negative example:")
        neg_text = dataset.tokenizer.decode(collated['neg_token_ids'][0], skip_special_tokens=True)
        print(neg_text)
    else:
        print("No negative examples were generated!")
        
except Exception as e:
    print(f"Error in collate_fn: {str(e)}")

print_separator()
print("Verification complete!")