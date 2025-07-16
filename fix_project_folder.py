#!/usr/bin/env python3
"""
Script to fix the project folder structure for ChemCPA training
"""

import os
import sys

def create_project_structure():
    """Create the required project folder structure"""
    
    print("🏗️  Creating project folder structure...")
    
    # Required directories
    directories = [
        'project_folder/datasets',
        'project_folder/embeddings/rdkit',
        'project_folder/embeddings/chemCPA',
        'project_folder/binaries',
        'outputs/checkpoints',
        'outputs/logs',
        'plots',
        'logs'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"   ✅ Created: {directory}")
    
    print("\n🎉 Project structure created successfully!")
    
    # Check for existing datasets
    print("\n📊 Checking for existing datasets...")
    dataset_files = [
        'project_folder/datasets/sciplex_complete_v2.h5ad',
        'project_folder/datasets/lincs.h5ad',
        'project_folder/datasets/adata_biolord_split_30.h5ad',
        'project_folder/datasets/adata_biolord_split_30_subset.h5ad'
    ]
    
    found_datasets = []
    missing_datasets = []
    
    for dataset_file in dataset_files:
        if os.path.exists(dataset_file):
            size = os.path.getsize(dataset_file) / (1024 * 1024)  # MB
            found_datasets.append((dataset_file, size))
            print(f"   ✅ Found: {dataset_file} ({size:.1f} MB)")
        else:
            missing_datasets.append(dataset_file)
            print(f"   ❌ Missing: {dataset_file}")
    
    if found_datasets:
        print(f"\n🎉 Found {len(found_datasets)} dataset(s)!")
        print("You can start training with these datasets.")
    
    if missing_datasets:
        print(f"\n⚠️  Missing {len(missing_datasets)} dataset(s).")
        print("To download datasets, run:")
        print("   python raw_data/datasets.py --dataset all")
        print("   # or for specific datasets:")
        print("   python raw_data/datasets.py --dataset lincs_full")
        print("   python raw_data/datasets.py --dataset sciplex")
        print("   python raw_data/datasets.py --dataset biolord")
    
    return len(found_datasets), len(missing_datasets)

def main():
    """Main function"""
    print("=" * 80)
    print("🧬 CHEMCPA PROJECT FOLDER SETUP")
    print("=" * 80)
    
    found, missing = create_project_structure()
    
    print("\n" + "=" * 80)
    print("📋 SUMMARY")
    print("=" * 80)
    print(f"✅ Project structure: Created")
    print(f"📊 Datasets found: {found}")
    print(f"❌ Datasets missing: {missing}")
    
    if missing == 0:
        print("\n🚀 Ready to train! Try:")
        print("   python train_chemcpa_simple.py --dataset sciplex --epochs 50")
    else:
        print("\n📥 Next step: Download datasets")
        print("   python raw_data/datasets.py --dataset all")
    
    print("=" * 80)

if __name__ == '__main__':
    main()

