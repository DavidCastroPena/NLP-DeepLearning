import unittest
import torch
from evaluation import evaluate_impossibility
from models.gpt2 import GPT2Model

class TestEvaluation(unittest.TestCase):
    def setUp(self):
        self.args = {
            "model_size": "gpt2",
            "d": 768,
            "l": 12,
            "num_heads": 12
        }
        # Fix: Use appropriate arguments based on what the model expects
        self.model = GPT2Model.from_pretrained(model="gpt2", d=768, l=12, num_heads=12)
        self.device = torch.device("cpu")
    
    def test_impossibility_score(self):
        # Create a simulated dataloader with one batch
        fake_dataloader = [{
            "token_ids": torch.randint(0, 50256, (2, 10)),
            "attention_mask": torch.ones((2, 10)),
            "neg_token_ids": torch.randint(0, 50256, (2, 10))
        }]
        
        score = evaluate_impossibility(self.model, fake_dataloader, self.device)
        self.assertTrue(0 <= score <= 1, "Impossibility score should be between 0 and 1.")

if __name__ == '__main__':
    unittest.main()