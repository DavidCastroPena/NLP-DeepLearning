'''import csv
import re
import torch
import random
from torch.utils.data import Dataset
from transformers import GPT2Tokenizer

def preprocess_string(s):
    """Clean and normalize input strings."""
    return ' '.join(s.lower()
                    .replace('.', ' .')
                    .replace('?', ' ?')
                    .replace(',', ' ,')
                    .replace('\'', ' \'')
                    .split())

def load_paraphrase_data(paraphrase_filename, split='train'):
    """Load paraphrase data from CSV files."""
    paraphrase_data = []
    
    try:
        with open(paraphrase_filename, 'r', encoding='utf-8') as fp:
            reader = csv.DictReader(fp, delimiter='\t')
            
            if split == 'test':
                for record in reader:
                    try:
                        sent_id = record['id'].strip()
                        sentence1 = record['sentence1'].strip()
                        sentence2 = record['sentence2'].strip()
                        
                        # Debugging output to ensure correct parsing
                        print(f"✅ Parsed test record: {sent_id}, {sentence1}, {sentence2}")
                        
                        paraphrase_data.append((
                            preprocess_string(sentence1),
                            preprocess_string(sentence2),
                            sent_id
                        ))
                    except KeyError as e:
                        print(f"⚠️ Skipping malformed test record: {e}")
            else:
                for record in reader:
                    try:
                        sent_id = record['id'].strip()
                        is_duplicate = int(float(record['is_duplicate']))
                        sentence1 = record['sentence1'].strip()
                        sentence2 = record['sentence2'].strip()
                        
                        paraphrase_data.append((
                            preprocess_string(sentence1),
                            preprocess_string(sentence2),
                            is_duplicate,
                            sent_id
                        ))
                    except (KeyError, ValueError) as e:
                        print(f"⚠️ Skipping malformed train record: {e}")
        
        print(f"✅ Loaded {len(paraphrase_data)} {split} examples from {paraphrase_filename}")
        return paraphrase_data
    
    except Exception as e:
        print(f"❌ Error loading data from {paraphrase_filename}: {e}")
        return []

class ParaphraseDetectionDataset(Dataset):
    def __init__(self, dataset, args, use_negative_samples=True):
        """
        Dataset for paraphrase detection with impossible distillation.
        
        Args:
            dataset: List of examples (sentence1, sentence2, label, id)
            args: Configuration arguments
            use_negative_samples: Whether to generate negative samples for training
        """
        self.dataset = dataset
        self.args = args
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.use_negative_samples = use_negative_samples
        
        # Create a list of positive and negative examples for sampling
        self.positive_examples = [i for i, example in enumerate(dataset) if example[2] == 1]
        self.negative_examples = [i for i, example in enumerate(dataset) if example[2] == 0]
        
        print(f"Dataset initialized with {len(self.positive_examples)} positive and {len(self.negative_examples)} negative examples")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]
        
    def create_hard_negatives(self, batch_indices, original_sent1, original_sent2):
        """
        Create challenging negative examples using multiple strategies.
        
        Args:
            batch_indices: Indices of examples in the current batch
            original_sent1: First sentences from the batch
            original_sent2: Second sentences from the batch
            
        Returns:
            List of hard negative examples (sent1, sent2)
        """
        hard_negatives = []
        
        for i, (idx, s1, s2) in enumerate(zip(batch_indices, original_sent1, original_sent2)):
            strategy = random.random()
            
            # Strategy 1: Sentence swap (30% of cases)
            # Use the same sentence twice, which definitely doesn't form a paraphrase pair
            if strategy < 0.3:
                hard_negatives.append((s1, s1))
            
            # Strategy 2: Label-flipping (30% of cases)
            # Find an example with opposite label to create confusion
            elif strategy < 0.6:
                current_label = self.dataset[idx][2]
                opposite_pool = self.negative_examples if current_label == 1 else self.positive_examples
                
                if opposite_pool:
                    opposite_idx = random.choice(opposite_pool)
                    opposite_example = self.dataset[opposite_idx]
                    # Mix sentences from examples with opposite labels
                    hard_negatives.append((s1, opposite_example[1]))
                else:
                    # Fallback to random sampling
                    hard_negatives.append((s1, random.choice(original_sent2)))
            
            # Strategy 3: Entity/keyword substitution (20% of cases)
            # Replace key terms with different ones to create semantic shift
            elif strategy < 0.8:
                words = s2.split()
                if len(words) > 4:
                    # Replace a random word (not first or last) with a random word from another example
                    random_idx = random.randint(1, len(words) - 2)
                    random_example = self.dataset[random.randint(0, len(self.dataset) - 1)]
                    random_words = random_example[1].split()
                    if random_words:
                        replacement = random.choice(random_words)
                        words[random_idx] = replacement
                        modified_s2 = " ".join(words)
                        hard_negatives.append((s1, modified_s2))
                    else:
                        hard_negatives.append((s1, random.choice(original_sent2)))
                else:
                    hard_negatives.append((s1, random.choice(original_sent2)))
            
            # Strategy 4: Semantically similar but different questions (20% of cases)
            # Find questions that are lexically similar but have different meanings
            else:
                # Simple word overlap similarity
                most_similar_idx = None
                highest_similarity = -1
                
                # Search through 20 random examples for efficiency
                for _ in range(20):
                    random_idx = random.randint(0, len(self.dataset) - 1)
                    if random_idx != idx and self.dataset[random_idx][2] != self.dataset[idx][2]:
                        # Calculate word overlap similarity
                        s1_words = set(s1.split())
                        random_words = set(self.dataset[random_idx][0].split())
                        overlap = len(s1_words.intersection(random_words)) / max(len(s1_words.union(random_words)), 1)
                        
                        if overlap > highest_similarity:
                            highest_similarity = overlap
                            most_similar_idx = random_idx
                
                if most_similar_idx is not None:
                    hard_negatives.append((s1, self.dataset[most_similar_idx][1]))
                else:
                    hard_negatives.append((s1, random.choice(original_sent2)))
        
        return hard_negatives

    def collate_fn(self, all_data):
        """Collate function with support for impossible distillation."""
        try:
            sent1 = [x[0] for x in all_data]
            sent2 = [x[1] for x in all_data]
            labels = torch.LongTensor([x[2] for x in all_data])
            sent_ids = [x[3] for x in all_data]
            
            # Create input format for model
            cloze_style_sents = [
                f'Question 1: "{s1}"\nQuestion 2: "{s2}"\nAre these questions asking the same thing?' 
                for s1, s2 in zip(sent1, sent2)
            ]
            
            encoding = self.tokenizer(cloze_style_sents, return_tensors='pt', padding=True, truncation=True)
            token_ids = torch.LongTensor(encoding['input_ids'])
            attention_mask = torch.LongTensor(encoding['attention_mask'])
            
            batch = {
                'token_ids': token_ids,
                'attention_mask': attention_mask,
                'labels': labels,
                'sent_ids': sent_ids
            }
            
            # Add negative samples for impossible distillation if enabled
            if self.use_negative_samples and len(self.dataset) > 1:
                # Get the indices of the current batch
                batch_indices = []
                for item in all_data:
                    for i, example in enumerate(self.dataset):
                        if item == example:
                            batch_indices.append(i)
                            break
                    else:
                        # If not found, use a random index as fallback
                        batch_indices.append(random.randint(0, len(self.dataset) - 1))
                
                # Generate hard negative samples using various strategies
                neg_samples = self.create_hard_negatives(batch_indices, sent1, sent2)
                
                # Get sentences from negative samples
                neg_sent1 = [x[0] for x in neg_samples]
                neg_sent2 = [x[1] for x in neg_samples]
                
                # Create negative cloze format
                neg_cloze_sents = [
                    f'Question 1: "{s1}"\nQuestion 2: "{s2}"\nAre these questions asking the same thing?'
                    for s1, s2 in zip(neg_sent1, neg_sent2)
                ]
                
                neg_encoding = self.tokenizer(neg_cloze_sents, return_tensors='pt', padding=True, truncation=True)
                batch['neg_token_ids'] = torch.LongTensor(neg_encoding['input_ids'])
                batch['neg_attention_mask'] = torch.LongTensor(neg_encoding['attention_mask'])

            return batch
                   
        except Exception as e:
            print(f"Error in collate_fn: {e}")
            # Return a minimal valid batch to avoid crashes
            return {
                'token_ids': torch.zeros((1, 10), dtype=torch.long),
                'attention_mask': torch.ones((1, 10), dtype=torch.long),
                'labels': torch.zeros(1, dtype=torch.long),
                'sent_ids': ['error'],
                'neg_token_ids': torch.zeros((1, 10), dtype=torch.long),
                'neg_attention_mask': torch.ones((1, 10), dtype=torch.long)
            }
        
class ParaphraseDetectionTestDataset(Dataset):
    """Dataset for evaluating on test data (no labels)."""
    def __init__(self, dataset, args):
        self.dataset = dataset
        self.args = args
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def collate_fn(self, all_data):
        sent1 = [x[0] for x in all_data]
        sent2 = [x[1] for x in all_data]
        sent_ids = [x[2] for x in all_data]

        cloze_style_sents = [
            f'Question 1: "{s1}"\nQuestion 2: "{s2}"\nAre these questions asking the same thing?' 
            for s1, s2 in zip(sent1, sent2)
        ]

        encoding = self.tokenizer(cloze_style_sents, return_tensors='pt', padding=True, truncation=True)
        token_ids = torch.LongTensor(encoding['input_ids'])
        attention_mask = torch.LongTensor(encoding['attention_mask'])

        batched_data = {
            'token_ids': token_ids,
            'attention_mask': attention_mask,
            'sent_ids': sent_ids
        }

        return batched_data

class SonnetsDataset(Dataset):
    """Dataset for loading sonnets."""
    def __init__(self, file_path):
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.sonnets = self._load_sonnets(file_path)

    def _load_sonnets(self, file_path):
        """Reads the file and extracts individual sonnets."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()

            # Split sonnets based on numbering pattern
            sonnets = re.split(r'\n\s*\d+\s*\n', text)[1:]  # Remove header text
            return [s.strip() for s in sonnets]
        except Exception as e:
            print(f"Error loading sonnets from {file_path}: {e}")
            return []

    def __len__(self):
        return len(self.sonnets)

    def __getitem__(self, idx):
        return (idx, self.sonnets[idx])

    def collate_fn(self, all_data):
        idx = [example[0] for example in all_data]
        sonnets = [example[1] for example in all_data]

        encoding = self.tokenizer(sonnets, return_tensors='pt', padding=True, truncation=True)
        token_ids = torch.LongTensor(encoding['input_ids'])
        attention_mask = torch.LongTensor(encoding['attention_mask'])

        batched_data = {
            'token_ids': token_ids,
            'attention_mask': attention_mask,
            'sent_ids': idx
        }

        return batched_data
        
'''

"""
Dataset classes for paraphrase detection and impossible distillation.
"""

import csv
import re
import torch
import random
from torch.utils.data import Dataset
from transformers import GPT2Tokenizer

def preprocess_string(s):
    """Clean and normalize input strings."""
    return ' '.join(s.lower()
                    .replace('.', ' .')
                    .replace('?', ' ?')
                    .replace(',', ' ,')
                    .replace('\'', ' \'')
                    .split())

def load_paraphrase_data(paraphrase_filename, split='train'):
    """Load paraphrase data from CSV files."""
    paraphrase_data = []
    
    try:
        with open(paraphrase_filename, 'r', encoding='utf-8') as fp:
            reader = csv.DictReader(fp, delimiter='\t')
            
            if split == 'test':
                for record in reader:
                    try:
                        sent_id = record['id'].strip()
                        sentence1 = record['sentence1'].strip()
                        sentence2 = record['sentence2'].strip()
                        
                        # Debugging output to ensure correct parsing
                        print(f"✅ Parsed test record: {sent_id}, {sentence1}, {sentence2}")
                        
                        paraphrase_data.append((
                            preprocess_string(sentence1),
                            preprocess_string(sentence2),
                            sent_id
                        ))
                    except KeyError as e:
                        print(f"⚠️ Skipping malformed test record: {e}")
            else:
                for record in reader:
                    try:
                        sent_id = record['id'].strip()
                        is_duplicate = int(float(record['is_duplicate']))
                        sentence1 = record['sentence1'].strip()
                        sentence2 = record['sentence2'].strip()
                        
                        paraphrase_data.append((
                            preprocess_string(sentence1),
                            preprocess_string(sentence2),
                            is_duplicate,
                            sent_id
                        ))
                    except (KeyError, ValueError) as e:
                        print(f"⚠️ Skipping malformed train record: {e}")
        
        print(f"✅ Loaded {len(paraphrase_data)} {split} examples from {paraphrase_filename}")
        return paraphrase_data
    
    except Exception as e:
        print(f"❌ Error loading data from {paraphrase_filename}: {e}")
        return []

class ParaphraseDetectionDataset(Dataset):
    def __init__(self, dataset, args, use_negative_samples=True):
        """
        Dataset for paraphrase detection with impossible distillation.
        
        Args:
            dataset: List of examples (sentence1, sentence2, label, id)
            args: Configuration arguments
            use_negative_samples: Whether to generate negative samples for training
        """
        self.dataset = dataset
        self.args = args
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.use_negative_samples = use_negative_samples
        
        # Create a list of positive and negative examples for sampling
        self.positive_examples = [i for i, example in enumerate(dataset) if example[2] == 1]
        self.negative_examples = [i for i, example in enumerate(dataset) if example[2] == 0]
        
        # Define multiple prompt templates for better performance
        self.prompt_templates = [
            # Original format
            'Question 1: "{s1}"\nQuestion 2: "{s2}"\nAre these questions asking the same thing?',
            
            # More structured format with clear instructions
            'Task: Determine if these two questions are paraphrases (asking for the same information).\n\nFirst question: {s1}\nSecond question: {s2}\n\nAre these questions paraphrases?',
            
            # Format emphasizing semantic equivalence
            'Compare the meaning of these questions:\n"{s1}"\n"{s2}"\nDo these questions have the same semantic meaning?',
            
            # Format with examples
            'Some questions may ask for the same information using different words. For example, "What is the capital of France?" and "What city serves as France\'s capital?" are paraphrases.\n\nQuestion 1: "{s1}"\nQuestion 2: "{s2}"\nAre these questions paraphrases?'
        ]
        
        print(f"Dataset initialized with {len(self.positive_examples)} positive and {len(self.negative_examples)} negative examples")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]
        
    def create_hard_negatives(self, batch_indices, original_sent1, original_sent2):
        """
        Create challenging negative examples using multiple strategies.
        
        Args:
            batch_indices: Indices of examples in the current batch
            original_sent1: First sentences from the batch
            original_sent2: Second sentences from the batch
            
        Returns:
            List of hard negative examples (sent1, sent2)
        """
        hard_negatives = []
        
        for i, (idx, s1, s2) in enumerate(zip(batch_indices, original_sent1, original_sent2)):
            strategy = random.random()
            
            # Strategy 1: Sentence swap (30% of cases)
            # Use the same sentence twice, which definitely doesn't form a paraphrase pair
            if strategy < 0.3:
                hard_negatives.append((s1, s1))
            
            # Strategy 2: Label-flipping (30% of cases)
            # Find an example with opposite label to create confusion
            elif strategy < 0.6:
                current_label = self.dataset[idx][2]
                opposite_pool = self.negative_examples if current_label == 1 else self.positive_examples
                
                if opposite_pool:
                    opposite_idx = random.choice(opposite_pool)
                    opposite_example = self.dataset[opposite_idx]
                    # Mix sentences from examples with opposite labels
                    hard_negatives.append((s1, opposite_example[1]))
                else:
                    # Fallback to random sampling
                    hard_negatives.append((s1, random.choice(original_sent2)))
            
            # Strategy 3: Entity/keyword substitution (20% of cases)
            # Replace key terms with different ones to create semantic shift
            elif strategy < 0.8:
                words = s2.split()
                if len(words) > 4:
                    # Replace a random word (not first or last) with a random word from another example
                    random_idx = random.randint(1, len(words) - 2)
                    random_example = self.dataset[random.randint(0, len(self.dataset) - 1)]
                    random_words = random_example[1].split()
                    if random_words:
                        replacement = random.choice(random_words)
                        words[random_idx] = replacement
                        modified_s2 = " ".join(words)
                        hard_negatives.append((s1, modified_s2))
                    else:
                        hard_negatives.append((s1, random.choice(original_sent2)))
                else:
                    hard_negatives.append((s1, random.choice(original_sent2)))
            
            # Strategy 4: Semantically similar but different questions (20% of cases)
            # Find questions that are lexically similar but have different meanings
            else:
                # Simple word overlap similarity
                most_similar_idx = None
                highest_similarity = -1
                
                # Search through 20 random examples for efficiency
                for _ in range(20):
                    random_idx = random.randint(0, len(self.dataset) - 1)
                    if random_idx != idx and self.dataset[random_idx][2] != self.dataset[idx][2]:
                        # Calculate word overlap similarity
                        s1_words = set(s1.split())
                        random_words = set(self.dataset[random_idx][0].split())
                        overlap = len(s1_words.intersection(random_words)) / max(len(s1_words.union(random_words)), 1)
                        
                        if overlap > highest_similarity:
                            highest_similarity = overlap
                            most_similar_idx = random_idx
                
                if most_similar_idx is not None:
                    hard_negatives.append((s1, self.dataset[most_similar_idx][1]))
                else:
                    hard_negatives.append((s1, random.choice(original_sent2)))
        
        return hard_negatives

    def collate_fn(self, all_data):
        """Collate function with support for impossible distillation."""
        try:
            sent1 = [x[0] for x in all_data]
            sent2 = [x[1] for x in all_data]
            labels = torch.LongTensor([x[2] for x in all_data])
            sent_ids = [x[3] for x in all_data]
            
            # Randomly select a prompt template for this batch
            selected_template = random.choice(self.prompt_templates)
            
            # Create input format for model using the selected template
            cloze_style_sents = [
                selected_template.format(s1=s1, s2=s2)
                for s1, s2 in zip(sent1, sent2)
            ]
            
            encoding = self.tokenizer(cloze_style_sents, return_tensors='pt', padding=True, truncation=True)
            token_ids = torch.LongTensor(encoding['input_ids'])
            attention_mask = torch.LongTensor(encoding['attention_mask'])
            
            batch = {
                'token_ids': token_ids,
                'attention_mask': attention_mask,
                'labels': labels,
                'sent_ids': sent_ids,
                'template_idx': self.prompt_templates.index(selected_template)
            }
            
            # Add negative samples for impossible distillation if enabled
            if self.use_negative_samples and len(self.dataset) > 1:
                # Get the indices of the current batch
                batch_indices = []
                for item in all_data:
                    for i, example in enumerate(self.dataset):
                        if item == example:
                            batch_indices.append(i)
                            break
                    else:
                        # If not found, use a random index as fallback
                        batch_indices.append(random.randint(0, len(self.dataset) - 1))
                
                # Generate hard negative samples using various strategies
                neg_samples = self.create_hard_negatives(batch_indices, sent1, sent2)
                
                # Get sentences from negative samples
                neg_sent1 = [x[0] for x in neg_samples]
                neg_sent2 = [x[1] for x in neg_samples]
                
                # Create negative cloze format using the same template for consistency
                neg_cloze_sents = [
                    selected_template.format(s1=s1, s2=s2)
                    for s1, s2 in zip(neg_sent1, neg_sent2)
                ]
                
                neg_encoding = self.tokenizer(neg_cloze_sents, return_tensors='pt', padding=True, truncation=True)
                batch['neg_token_ids'] = torch.LongTensor(neg_encoding['input_ids'])
                batch['neg_attention_mask'] = torch.LongTensor(neg_encoding['attention_mask'])

            return batch
               
        except Exception as e:
            print(f"Error in collate_fn: {e}")
            # Return a minimal valid batch to avoid crashes
            return {
                'token_ids': torch.zeros((1, 10), dtype=torch.long),
                'attention_mask': torch.ones((1, 10), dtype=torch.long),
                'labels': torch.zeros(1, dtype=torch.long),
                'sent_ids': ['error'],
                'neg_token_ids': torch.zeros((1, 10), dtype=torch.long),
                'neg_attention_mask': torch.ones((1, 10), dtype=torch.long),
                'template_idx': 0
            }
        
class ParaphraseDetectionTestDataset(Dataset):
    """Dataset for evaluating on test data (no labels)."""
    def __init__(self, dataset, args):
        self.dataset = dataset
        self.args = args
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Add the same prompt templates as in the training dataset
        self.prompt_templates = [
            'Question 1: "{s1}"\nQuestion 2: "{s2}"\nAre these questions asking the same thing?',
            'Task: Determine if these two questions are paraphrases (asking for the same information).\n\nFirst question: {s1}\nSecond question: {s2}\n\nAre these questions paraphrases?',
            'Compare the meaning of these questions:\n"{s1}"\n"{s2}"\nDo these questions have the same semantic meaning?',
            'Some questions may ask for the same information using different words. For example, "What is the capital of France?" and "What city serves as France\'s capital?" are paraphrases.\n\nQuestion 1: "{s1}"\nQuestion 2: "{s2}"\nAre these questions paraphrases?'
        ]
        
        # During testing, we use a single consistent template (the first one)
        self.selected_template = self.prompt_templates[0]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def collate_fn(self, all_data):
        sent1 = [x[0] for x in all_data]
        sent2 = [x[1] for x in all_data]
        sent_ids = [x[2] for x in all_data]

        # Use the selected template for consistency in testing
        cloze_style_sents = [
            self.selected_template.format(s1=s1, s2=s2)
            for s1, s2 in zip(sent1, sent2)
        ]

        encoding = self.tokenizer(cloze_style_sents, return_tensors='pt', padding=True, truncation=True)
        token_ids = torch.LongTensor(encoding['input_ids'])
        attention_mask = torch.LongTensor(encoding['attention_mask'])

        batched_data = {
            'token_ids': token_ids,
            'attention_mask': attention_mask,
            'sent_ids': sent_ids
        }

        return batched_data

class SonnetsDataset(Dataset):
    """Dataset for loading sonnets."""
    def __init__(self, file_path):
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.sonnets = self._load_sonnets(file_path)

    def _load_sonnets(self, file_path):
        """Reads the file and extracts individual sonnets."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()

            # Split sonnets based on numbering pattern
            sonnets = re.split(r'\n\s*\d+\s*\n', text)[1:]  # Remove header text
            return [s.strip() for s in sonnets]
        except Exception as e:
            print(f"Error loading sonnets from {file_path}: {e}")
            return []

    def __len__(self):
        return len(self.sonnets)

    def __getitem__(self, idx):
        return (idx, self.sonnets[idx])

    def collate_fn(self, all_data):
        idx = [example[0] for example in all_data]
        sonnets = [example[1] for example in all_data]

        encoding = self.tokenizer(sonnets, return_tensors='pt', padding=True, truncation=True)
        token_ids = torch.LongTensor(encoding['input_ids'])
        attention_mask = torch.LongTensor(encoding['attention_mask'])

        batched_data = {
            'token_ids': token_ids,
            'attention_mask': attention_mask,
            'sent_ids': idx
        }

        return batched_data
