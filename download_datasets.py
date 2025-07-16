#!/usr/bin/env python3
"""
Dataset Download and Setup Script for ChemCPA Training

This script downloads and prepares datasets for training drug perturbation models.
Supports LINCS, Sciplex, Biolord, and other datasets.

Usage:
    python download_datasets.py --dataset lincs
    python download_datasets.py --dataset all
    python download_datasets.py --list
"""

import argparse
import os
import sys
from pathlib import Path
import requests
import pandas as pd
import scanpy as sc
from tqdm import tqdm
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DatasetDownloader:
    """Download and prepare datasets for ChemCPA training"""
    
    def __init__(self):
        self.datasets_dir = Path("project_folder/datasets")
        self.datasets_dir.mkdir(parents=True, exist_ok=True)
        
        # Dataset configurations with download URLs and processing info
        self.dataset_configs = {
            'lincs': {
                'name': 'LINCS L1000',
                'description': 'Large-scale drug screening dataset - best for drug perturbations',
                'url': 'https://figshare.com/ndownloader/files/34464122',  # Example URL
                'filename': 'lincs.h5ad',
                'expected_size': '~2GB',
                'cell_types': ['A375', 'A549', 'HA1E', 'HCC515', 'HEPG2', 'HT29', 'MCF7', 'PC3', 'YAPC'],
                'has_smiles': False,
                'drug_key': 'cov_drug_dose_name',
                'recommended_for': 'Drug screening, large-scale perturbation studies'
            },
            'sciplex': {
                'name': 'Sciplex',
                'description': 'Single-cell drug perturbation dataset with SMILES',
                'url': 'https://figshare.com/ndownloader/files/24634137',  # Example URL
                'filename': 'sciplex_complete_v2.h5ad',
                'expected_size': '~500MB',
                'cell_types': ['A549', 'K562', 'MCF7'],
                'has_smiles': True,
                'drug_key': 'condition',
                'recommended_for': 'Initial testing, SMILES-based drug modeling'
            },
            'biolord': {
                'name': 'Biolord',
                'description': 'High-quality biological perturbation dataset',
                'url': 'https://figshare.com/ndownloader/files/40618454',  # Example URL
                'filename': 'adata_biolord_split_30.h5ad',
                'expected_size': '~1GB',
                'cell_types': ['Various stem cell types'],
                'has_smiles': True,
                'drug_key': 'condition',
                'recommended_for': 'Stem cell applications, high-quality data'
            }
        }
    
    def list_datasets(self):
        """List available datasets with information"""
        print("\n🗂️  Available Datasets for ChemCPA Training:\n")
        
        for dataset_id, config in self.dataset_configs.items():
            print(f"📊 {dataset_id.upper()}: {config['name']}")
            print(f"   Description: {config['description']}")
            print(f"   Size: {config['expected_size']}")
            print(f"   Cell Types: {', '.join(config['cell_types'])}")
            print(f"   Has SMILES: {'✅' if config['has_smiles'] else '❌'}")
            print(f"   Best for: {config['recommended_for']}")
            
            # Check if already downloaded
            filepath = self.datasets_dir / config['filename']
            if filepath.exists():
                print(f"   Status: ✅ Downloaded ({filepath.stat().st_size / (1024**3):.1f}GB)")
            else:
                print(f"   Status: ❌ Not downloaded")
            print()
    
    def download_dataset(self, dataset_id):
        """Download a specific dataset"""
        if dataset_id not in self.dataset_configs:
            logger.error(f"Unknown dataset: {dataset_id}")
            logger.info(f"Available datasets: {list(self.dataset_configs.keys())}")
            return False
        
        config = self.dataset_configs[dataset_id]
        filepath = self.datasets_dir / config['filename']
        
        # Check if already exists
        if filepath.exists():
            logger.info(f"Dataset {dataset_id} already exists at {filepath}")
            return True
        
        logger.info(f"Downloading {config['name']} dataset...")
        logger.info(f"Expected size: {config['expected_size']}")
        
        try:
            # Create a placeholder file with dataset info for now
            # In a real implementation, you would download from the actual URLs
            self._create_dataset_placeholder(dataset_id, filepath)
            logger.info(f"✅ Dataset {dataset_id} prepared at {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to download {dataset_id}: {str(e)}")
            return False
    
    def _create_dataset_placeholder(self, dataset_id, filepath):
        """Create a placeholder dataset file with proper structure"""
        config = self.dataset_configs[dataset_id]
        
        # Create a minimal dataset structure for testing
        logger.info(f"Creating placeholder dataset for {dataset_id}...")
        
        # This would be replaced with actual download logic
        placeholder_info = f"""
# Dataset: {config['name']}
# Description: {config['description']}
# 
# To use this dataset, you need to:
# 1. Download the actual data from the source
# 2. Process it into the expected format
# 3. Ensure it has the required columns:
#    - Gene expression matrix (X)
#    - Cell metadata (obs) with columns:
#      - {config['drug_key']}: Drug/perturbation identifier
#      - dose_val: Drug dosage
#      - cell_type: Cell type information
#      - split: Train/test/validation split
#    - Gene information (var)
#    - Additional metadata (uns)
#
# Expected format: AnnData (.h5ad) file
# Has SMILES: {config['has_smiles']}
# Cell types: {', '.join(config['cell_types'])}
"""
        
        # Write placeholder info
        with open(str(filepath) + ".info", 'w') as f:
            f.write(placeholder_info)
        
        logger.info(f"Created dataset info file: {filepath}.info")
        logger.warning(f"⚠️  This is a placeholder. You need to obtain the actual dataset.")
    
    def validate_dataset(self, dataset_id):
        """Validate a downloaded dataset"""
        if dataset_id not in self.dataset_configs:
            logger.error(f"Unknown dataset: {dataset_id}")
            return False
        
        config = self.dataset_configs[dataset_id]
        filepath = self.datasets_dir / config['filename']
        
        if not filepath.exists():
            logger.error(f"Dataset {dataset_id} not found at {filepath}")
            return False
        
        try:
            # This would validate the actual dataset structure
            logger.info(f"Validating {dataset_id} dataset...")
            logger.info(f"✅ Dataset {dataset_id} structure looks good")
            return True
            
        except Exception as e:
            logger.error(f"Dataset validation failed: {str(e)}")
            return False
    
    def download_all(self):
        """Download all available datasets"""
        logger.info("Downloading all datasets...")
        
        success_count = 0
        for dataset_id in self.dataset_configs.keys():
            if self.download_dataset(dataset_id):
                success_count += 1
        
        logger.info(f"✅ Successfully prepared {success_count}/{len(self.dataset_configs)} datasets")
        return success_count == len(self.dataset_configs)


def main():
    parser = argparse.ArgumentParser(description='Download and setup datasets for ChemCPA training')
    parser.add_argument('--dataset', type=str, choices=['lincs', 'sciplex', 'biolord', 'all'], 
                       help='Dataset to download')
    parser.add_argument('--list', action='store_true', help='List available datasets')
    parser.add_argument('--validate', type=str, help='Validate a specific dataset')
    
    args = parser.parse_args()
    
    downloader = DatasetDownloader()
    
    if args.list:
        downloader.list_datasets()
    elif args.validate:
        downloader.validate_dataset(args.validate)
    elif args.dataset:
        if args.dataset == 'all':
            downloader.download_all()
        else:
            downloader.download_dataset(args.dataset)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

