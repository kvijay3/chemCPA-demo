#!/usr/bin/env python3
"""
Test script to verify that the dataset configuration fixes work
"""

import sys
import os
sys.path.append('.')

# Test the configuration loading
def test_config():
    """Test that the dataset configurations are properly loaded"""
    
    # Import the trainer class
    from train_chemcpa_simple import SimplifiedChemCPATrainer
    
    # Test each dataset configuration
    datasets = ['sciplex', 'broad', 'lincs', 'biolord']
    
    for dataset in datasets:
        print(f"\n🧪 Testing {dataset} configuration...")
        
        # Create a mock args object
        class MockArgs:
            def __init__(self, dataset):
                self.dataset = dataset
                self.epochs = 50
                self.batch_size = 128
                self.learning_rate = 1e-3
                self.output_dir = './outputs'
                self.use_wandb = False
                self.pretrained_path = None
                self.embedding_model = 'rdkit'
                self.embedding_dim = 256
                self.hidden_dim = 512
                self.num_layers = 3
                self.dropout = 0.1
                self.weight_decay = 1e-4
                self.scheduler = 'cosine'
                self.warmup_epochs = 5
                self.patience = 10
                self.min_delta = 1e-4
                self.gradient_clip_val = 1.0
                self.accumulate_grad_batches = 1
                self.precision = 32
                self.num_workers = 4
                self.pin_memory = True
                self.persistent_workers = True
        
        args = MockArgs(dataset)
        trainer = SimplifiedChemCPATrainer(args)
        
        # Check if degs_key is properly set
        degs_key = trainer.config['dataset'].get('degs_key', 'NOT_SET')
        print(f"   Dataset: {dataset}")
        print(f"   Dataset path: {trainer.config['dataset']['dataset_path']}")
        print(f"   DEGs key: {degs_key}")
        
        # Verify the degs_key is appropriate for each dataset
        expected_degs_keys = {
            'sciplex': 'all_DEGs',
            'broad': 'rank_genes_groups_cov_all', 
            'lincs': 'rank_genes_groups_cov',
            'biolord': 'rank_genes_groups_cov_all'
        }
        
        expected = expected_degs_keys[dataset]
        if degs_key == expected:
            print(f"   ✅ Correct DEGs key: {degs_key}")
        else:
            print(f"   ❌ Wrong DEGs key: got {degs_key}, expected {expected}")
    
    print(f"\n🎉 Configuration test completed!")

if __name__ == '__main__':
    test_config()
