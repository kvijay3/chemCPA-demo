#!/usr/bin/env python3
"""
Simplified ChemCPA Training Script for Stem Cell Perturbation Prediction

This script trains a model that takes:
- Input: scRNA-seq data + unseen drug (SMILES)
- Output: Predicted perturbed scRNA-seq data

Usage:
    python train_chemcpa_simple.py --dataset sciplex --epochs 100 --batch_size 32
    python train_chemcpa_simple.py --dataset broad --epochs 200 --batch_size 64
"""

import argparse
import os
import sys
from pathlib import Path
import torch
import torch.nn as nn
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger, TensorBoardLogger
import numpy as np
import pandas as pd
import scanpy as sc
from omegaconf import OmegaConf

# Add the chemCPA module to path
sys.path.append(str(Path(__file__).parent))

from chemCPA.data.data import load_dataset_splits, PerturbationDataModule
from chemCPA.lightning_module import ChemCPA
from chemCPA.embedding import get_chemical_representation


class SimplifiedChemCPATrainer:
    """Simplified trainer for ChemCPA model focused on stem cell applications"""
    
    def __init__(self, args):
        self.args = args
        self.setup_config()
        
    def setup_config(self):
        """Setup configuration for training"""
        
        # Dataset configurations for different datasets
        dataset_configs = {
            'sciplex': {
                'dataset_path': 'project_folder/datasets/sciplex_complete_v2.h5ad',
                'split_key': 'split_cellcycle_ood',
                'perturbation_key': 'condition',
                'pert_category': 'cov_drug_dose_name',
                'dose_key': 'dose',
                'smiles_key': 'SMILES',  # Sciplex has SMILES
                'covariate_keys': 'cell_type',
                'description': 'Sciplex dataset - good for initial testing'
            },
            'broad': {
                'dataset_path': 'project_folder/datasets/adata_biolord_split_30_subset.h5ad', 
                'split_key': 'split_cellcycle_ood',
                'perturbation_key': 'condition',
                'pert_category': 'cov_drug_dose_name',
                'dose_key': 'dose',
                'smiles_key': 'SMILES',  # Broad has SMILES
                'covariate_keys': 'cell_type',
                'degs_key': 'rank_genes_groups_cov_all',  # Broad uses same as biolord
                'description': 'Broad dataset - larger scale training'
            },
            'lincs': {
                'dataset_path': 'project_folder/datasets/lincs_full_smiles.h5ad',  # Use SMILES-enabled dataset
                'split_key': 'split',
                'perturbation_key': 'condition',  # Preprocessed file uses 'condition'
                'pert_category': 'cov_drug_dose_name',  # Standard ChemCPA format
                'dose_key': 'dose_val',  # Standard ChemCPA format
                'smiles_key': 'canonical_smiles',  # SMILES data now available!
                'covariate_keys': ['cell_type'],  # Standard ChemCPA format
                'degs_key': 'rank_genes_groups_cov',  # LINCS uses this key instead of all_DEGs
                'description': 'LINCS L1000 dataset - large drug screening (with SMILES)'
            },
            'biolord': {
                'dataset_path': 'project_folder/datasets/adata_biolord_split_30.h5ad',
                'split_key': 'split_cellcycle_ood',
                'perturbation_key': 'condition',
                'pert_category': 'cov_drug_dose_name',
                'dose_key': 'dose',
                'smiles_key': 'SMILES',  # Biolord has SMILES
                'covariate_keys': 'cell_type',
                'degs_key': 'rank_genes_groups_cov_all',  # Biolord uses this key
                'description': 'Biolord dataset - high-quality biological data for stem cells'
            }
        }
        
        # Get dataset-specific configuration
        dataset_config = dataset_configs[self.args.dataset]
        
        # Base configuration
        self.config = {
            'dataset': {
                'use_drugs_idx': True,
                'degs_key': 'all_DEGs',
                **dataset_config
            },
            'model': {
                'hparams': {
                    'batch_size': self.args.batch_size,
                    'lr': self.args.learning_rate,
                    'wd': 1e-7,
                    'dim': 256,  # Latent dimension for encoder/decoder
                    'autoencoder_width': 512,
                    'autoencoder_depth': 4,
                    'autoencoder_lr': self.args.learning_rate,
                    'autoencoder_wd': 1e-6,
                    'adversary_width': 128,
                    'adversary_depth': 3,
                    'adversary_lr': 3e-4,
                    'adversary_wd': 1e-4,
                    'adversary_steps': 3,
                    'reg_adversary': 5.0,
                    'penalty_adversary': 3.0,
                    'reg_adversary_cov': 15.0,
                    'reg_adversary_drugs': 5.0,
                    'dosers_width': 64,
                    'dosers_depth': 2,
                    'dosers_lr': self.args.learning_rate,
                    'dosers_wd': 1e-7,
                    'step_size_lr': 45,
                    'embedding_encoder_width': 512,
                    'embedding_encoder_depth': 0,
                },
                'embedding': 'zeros',  # Use zero embeddings as fallback (will be learned during training)
                'embedding_datapath': None,  # No pre-computed embeddings
                'additional_params': {
                    'seed': 42,
                    'doser_type': 'mlp',
                },
                'append_ae_layer': False,
                'load_pretrained': self.args.pretrained_path is not None,
                'pretrained_model_path': self.args.pretrained_path or '',
                'pretrained_model_hashes': {'model': 'latest'} if self.args.pretrained_path else {}
            },
            'training': {
                'num_epochs': self.args.epochs,
                'max_minutes': '12:00:00',
                'save_dir': self.args.output_dir,
                'checkpoint_freq': 5,
            },
            'wandb': {
                'project': f'chemcpa-stem-cells-{self.args.dataset}',
                'name': f'chemcpa-{self.args.dataset}-{self.args.epochs}epochs',
                'tags': ['stem-cells', 'drug-perturbation', self.args.dataset],
            } if self.args.use_wandb else None
        }
    
    def prepare_data(self):
        """Load and prepare the dataset"""
        print(f"Loading {self.args.dataset} dataset...")
        print(f"Description: {self.config['dataset']['description']}")
        
        # Load dataset splits
        try:
            # Remove description from config before passing to load_dataset_splits
            dataset_config = {k: v for k, v in self.config['dataset'].items() if k != 'description'}
            datasets, dataset = load_dataset_splits(**dataset_config, return_dataset=True)
            
            # Create data module
            dm = PerturbationDataModule(
                datasplits=datasets, 
                train_bs=self.config['model']['hparams']['batch_size']
            )
            
            # Dataset configuration for model
            dataset_config = {
                'num_genes': datasets['training'].num_genes,
                'num_drugs': datasets['training'].num_drugs, 
                'num_covariates': datasets['training'].num_covariates,
                'use_drugs_idx': dataset.use_drugs_idx,
                'canon_smiles_unique_sorted': dataset.canon_smiles_unique_sorted,
            }
            
            print(f"Dataset loaded successfully!")
            print(f"- Number of genes: {dataset_config['num_genes']}")
            print(f"- Number of drugs: {dataset_config['num_drugs']}")
            print(f"- Number of covariates: {dataset_config['num_covariates']}")
            print(f"- Available splits: {list(datasets.keys())}")
            print(f"- Training samples: {len(datasets['training'])}")
            if 'validation' in datasets:
                print(f"- Validation samples: {len(datasets['validation'])}")
            else:
                print("- Validation samples: Not available")
            if 'test' in datasets:
                print(f"- Test samples: {len(datasets['test'])}")
            else:
                print("- Test samples: Not available")
            
            return dm, dataset_config, dataset
            
        except Exception as e:
            print(f"Error loading dataset: {e}")
            print("Make sure the dataset files are available. You may need to run preprocessing first.")
            print("Check the preprocessing/ folder for data preparation scripts.")
            raise
    
    def create_model(self, dataset_config):
        """Create the ChemCPA model"""
        print("Creating ChemCPA model...")
        
        model = ChemCPA(self.config, dataset_config)
        
        # Print model summary
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"Model created successfully!")
        print(f"- Total parameters: {total_params:,}")
        print(f"- Trainable parameters: {trainable_params:,}")
        
        return model
    
    def setup_callbacks(self):
        """Setup training callbacks"""
        callbacks = []
        
        # Model checkpointing
        checkpoint_callback = ModelCheckpoint(
            dirpath=Path(self.args.output_dir) / 'checkpoints',
            filename='chemcpa-{epoch:02d}-{val_loss:.2f}',
            monitor='val_loss',
            mode='min',
            save_top_k=3,
            save_last=True,
            verbose=True
        )
        callbacks.append(checkpoint_callback)
        
        # Early stopping
        if self.args.early_stopping:
            early_stop_callback = EarlyStopping(
                monitor='val_loss',
                min_delta=0.001,
                patience=10,
                verbose=True,
                mode='min'
            )
            callbacks.append(early_stop_callback)
        
        # Learning rate monitoring
        lr_monitor = LearningRateMonitor(logging_interval='step')
        callbacks.append(lr_monitor)
        
        return callbacks
    
    def setup_logger(self):
        """Setup experiment logger"""
        if self.args.use_wandb:
            logger = WandbLogger(**self.config['wandb'], save_dir=self.args.output_dir)
        else:
            logger = TensorBoardLogger(
                save_dir=self.args.output_dir,
                name='chemcpa_logs',
                version=f'{self.args.dataset}_{self.args.epochs}epochs'
            )
        return logger
    
    def train(self):
        """Main training loop"""
        print("="*60)
        print("ChemCPA Training for Stem Cell Drug Perturbation Prediction")
        print("="*60)
        
        # Prepare data
        dm, dataset_config, dataset = self.prepare_data()
        
        # Create model
        model = self.create_model(dataset_config)
        
        # Setup callbacks and logger
        callbacks = self.setup_callbacks()
        logger = self.setup_logger()
        
        # Create trainer
        trainer = L.Trainer(
            accelerator='cuda' if torch.cuda.is_available() else 'cpu',
            devices=1,
            max_epochs=self.args.epochs,
            callbacks=callbacks,
            logger=logger,
            check_val_every_n_epoch=self.config['training']['checkpoint_freq'],
            gradient_clip_val=1.0,  # Prevent gradient explosion
            deterministic=True,  # For reproducibility
        )
        
        print(f"Starting training for {self.args.epochs} epochs...")
        print(f"Using device: {trainer.accelerator}")
        
        # Train the model
        trainer.fit(model, datamodule=dm)
        
        print("Training completed!")
        print(f"Best model saved at: {callbacks[0].best_model_path}")
        
        return trainer, model
    
    def predict_perturbation(self, model, gene_expression, drug_smiles, cell_type=None):
        """
        Predict perturbation response for new drug
        
        Args:
            model: Trained ChemCPA model
            gene_expression: scRNA-seq data (genes x cells or cells x genes)
            drug_smiles: SMILES string of the drug
            cell_type: Cell type information (optional)
        
        Returns:
            Predicted perturbed gene expression
        """
        model.eval()
        with torch.no_grad():
            # This is a placeholder - you'll need to implement the actual prediction logic
            # based on how the model processes inputs
            print(f"Predicting perturbation for drug: {drug_smiles}")
            print(f"Input expression shape: {gene_expression.shape}")
            
            # Convert inputs to appropriate format for the model
            # This will depend on your specific model architecture
            
            return None  # Placeholder


def main():
    parser = argparse.ArgumentParser(description='Train ChemCPA for stem cell drug perturbation prediction')
    
    # Dataset arguments
    parser.add_argument('--dataset', type=str, choices=['sciplex', 'broad', 'lincs', 'biolord'], 
                       default='sciplex', help='Dataset to use for training')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--early_stopping', action='store_true', help='Use early stopping')
    
    # Model arguments
    parser.add_argument('--pretrained_path', type=str, default=None, 
                       help='Path to pretrained model weights')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='./outputs', 
                       help='Directory to save outputs')
    
    # Logging arguments
    parser.add_argument('--use_wandb', action='store_true', help='Use Weights & Biases logging')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize trainer
    trainer = SimplifiedChemCPATrainer(args)
    
    # Train model
    lightning_trainer, model = trainer.train()
    
    print("\n" + "="*60)
    print("Training Summary:")
    print(f"- Dataset: {args.dataset}")
    print(f"- Epochs: {args.epochs}")
    print(f"- Batch size: {args.batch_size}")
    print(f"- Learning rate: {args.learning_rate}")
    print(f"- Output directory: {args.output_dir}")
    print("="*60)
    
    # Example of how to use the trained model for prediction
    print("\nTo use the trained model for prediction:")
    print("1. Load your scRNA-seq data")
    print("2. Provide SMILES string of your drug of interest")
    print("3. Call trainer.predict_perturbation(model, gene_expression, drug_smiles)")


if __name__ == '__main__':
    main()
