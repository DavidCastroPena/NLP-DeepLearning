'''
#With this version, the csv were accepted, but the ID's were malformed; got 0 accuracy in the leaderboard"

import os
import torch
import csv
import pandas as pd
import hashlib
from transformers import GPT2Tokenizer
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from paraphrase_detection import ParaphraseGPT  # Import your model class

# Define the token mappings
TOKEN_MAPPING = {0: 3919, 1: 8505}  # 0 → 3919 ("No"), 1 → 8505 ("Yes")

# Dataset class for test data
class TestDataset(Dataset):
    def __init__(self, data_file):
        self.data = []
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load test data
        with open(data_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='\t')
            header = next(reader)  # Skip header
            for row in reader:
                if len(row) >= 3:
                    # Extract parts for ID processing
                    id_parts = row[0].strip().split()
                    
                    # Generate 25-digit hexadecimal ID - if there's a UUID, use that
                    if len(id_parts) >= 2:
                        uuid = id_parts[1]  # e.g., "641bc8de-b905-4ae4-a0ee-abea15fbc88e"
                        # Remove hyphens and take first 25 chars
                        hex_id = uuid.replace('-', '')[:25]
                    else:
                        # If there's no UUID, hash the entire ID
                        hex_id = hashlib.md5(row[0].strip().encode()).hexdigest()[:25]
                    
                    self.data.append((hex_id, row[1].strip(), row[2].strip()))  # hex_id, sentence1, sentence2
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    def collate_fn(self, all_data):
        ids = [x[0] for x in all_data]
        sent1 = [x[1] for x in all_data]
        sent2 = [x[2] for x in all_data]
        
        # Create cloze-style prompts
        prompts = [
            f'Question 1: "{s1}"\nQuestion 2: "{s2}"\nAre these questions asking the same thing?'
            for s1, s2 in zip(sent1, sent2)
        ]
        
        # Tokenize
        encoding = self.tokenizer(prompts, return_tensors='pt', padding=True, truncation=True)
        
        return {
            'token_ids': encoding['input_ids'],
            'attention_mask': encoding['attention_mask'],
            'sent_ids': ids
        }

# Main prediction function
def generate_predictions(model_path, test_file, is_dev=False, batch_size=8, use_gpu=True):
    # Set device
    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
    
    # Define model arguments
    class Args:
        def __init__(self):
            self.model_size = "gpt2"
            self.d = 768
            self.l = 12
            self.num_heads = 12
    
    args = Args()
    
    # Load model
    print(f"🔄 Loading model from {model_path}...")
    model = ParaphraseGPT(args).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Load test data
    print(f"📂 Loading data from {test_file}...")
    dataset = TestDataset(test_file)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=dataset.collate_fn)
    
    # Generate predictions
    all_preds = []
    all_ids = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="🔍 Generating Predictions"):
            ids = batch['sent_ids']
            token_ids = batch['token_ids'].to(device)
            attn_mask = batch['attention_mask'].to(device)
            
            _, logits = model(token_ids, attn_mask)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            
            # Map predictions to token IDs for "Yes" and "No"
            token_preds = [TOKEN_MAPPING[p] for p in preds]
            
            all_preds.extend(token_preds)
            all_ids.extend(ids)
    
    # Create the predictions directory if it doesn't exist
    os.makedirs("predictions", exist_ok=True)
    
    # Determine the correct output filename based on dev or test set
    output_file = "predictions/para-dev-output.csv" if is_dev else "predictions/para-test-output.csv"
    
    # Create DataFrame with the required format
    df = pd.DataFrame({"id": all_ids, "Predicted_Is_Paraphrase": all_preds})
    
    # Save to CSV with exactly the required format
    print(f"✅ Saving predictions to {output_file}")
    with open(output_file, 'w', newline='') as f:
        f.write("id,Predicted_Is_Paraphrase\n")  # Exact header format
        for _, row in df.iterrows():
            f.write(f"{row['id']},{row['Predicted_Is_Paraphrase']}\n")
    
    # Verify the output format
    print(f"🔍 Verifying output format...")
    with open(output_file, 'r') as f:
        first_lines = [next(f) for _ in range(min(5, len(all_ids) + 1))]
        print("First few lines of the output file:")
        for line in first_lines:
            print(line.strip())

if __name__ == "__main__":
    # Generate predictions for test set
    generate_predictions(
        model_path="10-1e-05-paraphrase.pt",
        test_file="data/quora-test-student.csv",
        is_dev=False,
        batch_size=8,
        use_gpu=True
    )
    
    generate_predictions(
        model_path="10-1e-05-paraphrase.pt",
        test_file="data/quora-dev-student.csv",  # Change to dev file path
        is_dev=True,
        batch_size=8,
        use_gpu=True
    )
    '''
    
######TRY TWO
import os
import torch
import csv
import pandas as pd
from transformers import GPT2Tokenizer
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from paraphrase_detection import ParaphraseGPT  # Import your model class

# Define the token mappings
TOKEN_MAPPING = {0: 3919, 1: 8505}  # 0 → 3919 ("No"), 1 → 8505 ("Yes")

# Dataset class for test data
class TestDataset(Dataset):
    def __init__(self, data_file):
        self.data = []
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load test data
        with open(data_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='\t')
            header = next(reader)  # Skip header
            for row in reader:
                if len(row) >= 3:
                    # Use the original ID exactly as it appears in the file
                    original_id = row[0].strip()
                    self.data.append((original_id, row[1].strip(), row[2].strip()))  # original_id, sentence1, sentence2
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    def collate_fn(self, all_data):
        ids = [x[0] for x in all_data]
        sent1 = [x[1] for x in all_data]
        sent2 = [x[2] for x in all_data]
        
        # Create cloze-style prompts
        prompts = [
            f'Question 1: "{s1}"\nQuestion 2: "{s2}"\nAre these questions asking the same thing?'
            for s1, s2 in zip(sent1, sent2)
        ]
        
        # Tokenize
        encoding = self.tokenizer(prompts, return_tensors='pt', padding=True, truncation=True)
        
        return {
            'token_ids': encoding['input_ids'],
            'attention_mask': encoding['attention_mask'],
            'sent_ids': ids
        }

# Main prediction function
def generate_predictions(model_path, test_file, is_dev=False, batch_size=8, use_gpu=True):
    # Set device
    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
    
    # Define model arguments
    class Args:
        def __init__(self):
            self.model_size = "gpt2"
            self.d = 768
            self.l = 12
            self.num_heads = 12
    
    args = Args()
    
    # Load model
    print(f"🔄 Loading model from {model_path}...")
    model = ParaphraseGPT(args).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Load test data
    print(f"📂 Loading data from {test_file}...")
    dataset = TestDataset(test_file)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=dataset.collate_fn)
    
    # Generate predictions
    all_preds = []
    all_ids = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="🔍 Generating Predictions"):
            ids = batch['sent_ids']
            token_ids = batch['token_ids'].to(device)
            attn_mask = batch['attention_mask'].to(device)
            
            _, logits = model(token_ids, attn_mask)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            
            # Map predictions to token IDs for "Yes" and "No"
            token_preds = [TOKEN_MAPPING[p] for p in preds]
            
            all_preds.extend(token_preds)
            all_ids.extend(ids)
    
    # Create the predictions directory if it doesn't exist
    os.makedirs("predictions", exist_ok=True)
    
    # Determine the correct output filename based on dev or test set
    output_file = "predictions/para-dev-output.csv" if is_dev else "predictions/para-test-output.csv"
    
    # Create DataFrame with the required format
    df = pd.DataFrame({"id": all_ids, "Predicted_Is_Paraphrase": all_preds})
    
    # Save to CSV with exactly the required format
    print(f"✅ Saving predictions to {output_file}")
    with open(output_file, 'w', newline='') as f:
        f.write("id,Predicted_Is_Paraphrase\n")  # Exact header format
        for _, row in df.iterrows():
            f.write(f"{row['id']},{row['Predicted_Is_Paraphrase']}\n")
    
    # Verify the output format
    print(f"🔍 Verifying output format...")
    with open(output_file, 'r') as f:
        first_lines = [next(f) for _ in range(min(5, len(all_ids) + 1))]
        print("First few lines of the output file:")
        for line in first_lines:
            print(line.strip())

# Find and list available files in the data directory
def list_available_files(directory='data'):
    print(f"\n📁 Checking available files in {directory}:")
    try:
        files = os.listdir(directory)
        for file in files:
            print(f"- {file}")
        return files
    except Exception as e:
        print(f"Error accessing directory: {e}")
        return []

if __name__ == "__main__":
    # List available files to help identify correct filenames
    data_files = list_available_files()
    
    # Find the test file - looking for patterns in available files
    test_file = None
    for file in data_files:
        if 'test' in file.lower() and ('quora' in file.lower() or 'para' in file.lower()):
            test_file = os.path.join('data', file)
            print(f"Found test file: {test_file}")
            break
    
    if not test_file:
        test_file = "data/quora-test-student.csv"  # Default fallback
        print(f"No test file found in directory. Using default: {test_file}")
    
    # Find the dev file - looking for patterns in available files
    dev_file = None
    for file in data_files:
        if 'dev' in file.lower() and ('quora' in file.lower() or 'para' in file.lower()):
            dev_file = os.path.join('data', file)
            print(f"Found dev file: {dev_file}")
            break
    
    if not dev_file:
        dev_file = "data/quora-dev-student.csv"  # Default fallback
        print(f"No dev file found in directory. Using default: {dev_file}")
    
    # Process test file if it exists
    if os.path.exists(test_file):
        print(f"\n🔍 Processing test file: {test_file}")
        generate_predictions(
            model_path="10-1e-05-paraphrase.pt",
            test_file=test_file,
            is_dev=False,
            batch_size=8,
            use_gpu=True
        )
    else:
        print(f"⚠️ Test file not found: {test_file}")
    
    # Process dev file if it exists
    if os.path.exists(dev_file):
        print(f"\n🔍 Processing dev file: {dev_file}")
        generate_predictions(
            model_path="10-1e-05-paraphrase.pt",
            test_file=dev_file,
            is_dev=True,
            batch_size=8,
            use_gpu=True
        )
    else:
        print(f"⚠️ Dev file not found: {dev_file}")
        # Create an empty predictions file for dev to satisfy the submission requirement
        os.makedirs("predictions", exist_ok=True)
        with open("predictions/para-dev-output.csv", 'w') as f:
            f.write("id,Predicted_Is_Paraphrase\n")
        print("Created empty dev predictions file to satisfy submission requirements.")