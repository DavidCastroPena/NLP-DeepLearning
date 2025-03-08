import unittest
import torch
from paraphrase_detection import ParaphraseGPT

class TestParaphraseGPT(unittest.TestCase):
    def setUp(self):
        self.args = {
            "model_size": "gpt2",
            "d": 768,
            "l": 12,
            "num_heads": 12
        }
        # Initialize model with a try-except to catch any initialization errors
        try:
            self.model = ParaphraseGPT(self.args)
        except Exception as e:
            print(f"Error initializing model: {e}")
            raise
        self.device = torch.device("cpu")
    
    def test_model_forward(self):
        # Create dummy inputs
        batch_size = 2
        seq_len = 10
        input_ids = torch.randint(0, 50256, (batch_size, seq_len))
        attention_mask = torch.ones((batch_size, seq_len))
        
        # Run model
        loss, logits = self.model(input_ids, attention_mask)
        
        # Check output types and shapes
        self.assertIsInstance(logits, torch.Tensor, "Output should be a tensor, not a string")
        self.assertEqual(logits.shape, (batch_size, 2), "Output logits should have shape [batch_size, 2].")
        
        # Test with negative samples
        neg_input_ids = torch.randint(0, 50256, (batch_size, seq_len))
        loss, logits = self.model(input_ids, attention_mask, neg_input_ids=neg_input_ids)
        
        # Check outputs are still correct
        self.assertIsInstance(logits, torch.Tensor)
        self.assertEqual(logits.shape, (batch_size, 2))

if __name__ == '__main__':
    unittest.main()