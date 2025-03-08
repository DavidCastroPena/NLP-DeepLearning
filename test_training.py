import unittest
import os
import torch
from paraphrase_detection import train, seed_everything

class TestTraining(unittest.TestCase):
    def setUp(self):
        # Setup minimal training config
        seed_everything(11711)  # Ensure reproducibility
        
        self.args = {
            "use_gpu": torch.cuda.is_available(),
            "para_train": "data/quora-train.csv",  # Make sure this path exists
            "para_dev": "data/quora-dev.csv",      # Make sure this path exists
            "batch_size": 2,                       # Small batch for quick test
            "lr": 1e-5,
            "epochs": 1,                           # Just one epoch for testing
            "model_size": "gpt2",
            "d": 768,
            "l": 12, 
            "num_heads": 12,
            "filepath": "test_model.pt"            # Temporary file for test
        }
        
        # Check if test data exists
        for data_file in [self.args["para_train"], self.args["para_dev"]]:
            self.assertTrue(os.path.exists(data_file), 
                           f"Test data file {data_file} not found. Ensure test data is available.")
    
    def test_training_runs(self):
        try:
            # Run a single epoch of training
            train_loss = train(self.args)
            
            # Basic check that training completed
            self.assertIsNotNone(train_loss, "Training should return a loss value")
            
            # Check that model file was created (if training succeeded)
            if train_loss is not None:
                self.assertTrue(
                    os.path.exists(self.args["filepath"]), 
                    "Training should save a model checkpoint"
                )
            
        except Exception as e:
            self.fail(f"Training failed with error: {e}")
            
        finally:
            # Clean up - always try to remove test file
            if os.path.exists(self.args["filepath"]):
                os.remove(self.args["filepath"])

if __name__ == '__main__':
    unittest.main()