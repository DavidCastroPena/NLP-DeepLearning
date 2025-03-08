import unittest
import torch
from datasets import ParaphraseDetectionDataset

class TestDataset(unittest.TestCase):
    def setUp(self):
        self.sample_data = [
            ("What is AI?", "Can you explain AI?", 1),
            ("Where is the Eiffel Tower?", "How tall is the Eiffel Tower?", 0)
        ]
        self.args = {}
    
    def test_negative_sampling(self):
        dataset = ParaphraseDetectionDataset(self.sample_data, self.args, use_negative_samples=True)
        sample = dataset[0]
        self.assertIn("neg_token_ids_1", sample, "Negative sample should be included.")
        self.assertTrue(sample["neg_token_ids_1"] is None or isinstance(sample["neg_token_ids_1"], torch.Tensor))

if __name__ == '__main__':
    unittest.main()

