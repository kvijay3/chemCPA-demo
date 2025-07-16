#!/usr/bin/env python3
"""
ChemCPA Setup Script

This script sets up the ChemCPA environment and downloads essential datasets.
It's designed to get you up and running quickly for stem cell drug perturbation prediction.

Usage:
    python setup_chemcpa.py                    # Full setup
    python setup_chemcpa.py --quick            # Quick setup (minimal datasets)
    python setup_chemcpa.py --datasets-only    # Only download datasets
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
import time


class ChemCPASetup:
    """Setup class for ChemCPA environment"""
    
    def __init__(self):
        self.root_dir = Path.cwd()
        self.project_folder = self.root_dir / "project_folder"
        
    def print_banner(self):
        """Print setup banner"""
        print("\n" + "="*80)
        print("🧬 CHEMCPA SETUP FOR STEM CELL DRUG PERTURBATION PREDICTION")
        print("="*80)
        print("This script will set up your environment for training ChemCPA models")
        print("to predict how unseen drugs affect stem cell gene expression.")
        print("="*80)
    
    def check_python_version(self):
        """Check if Python version is compatible"""
        print("\n🐍 Checking Python version...")
        
        version = sys.version_info
        if version.major < 3 or (version.major == 3 and version.minor < 8):
            print(f"❌ Python {version.major}.{version.minor} detected.")
            print("❌ ChemCPA requires Python 3.8 or higher.")
            print("Please upgrade Python and try again.")
            return False
        
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} - Compatible!")
        return True
    
    def install_dependencies(self):
        """Install required Python packages"""
        print("\n📦 Installing Python dependencies...")
        
        # Core dependencies
        core_deps = [
            "torch>=1.12.0",
            "lightning>=2.0.0", 
            "numpy>=1.21.0",
            "scipy>=1.7.0",
            "pandas>=1.3.0",
            "scanpy>=1.9.0",
            "anndata>=0.8.0",
            "omegaconf>=2.1.0",
            "hydra-core>=1.1.0",
            "tqdm>=4.62.0",
            "matplotlib>=3.5.0",
            "seaborn>=0.11.0",
            "h5py>=3.6.0",
            "scikit-learn>=1.0.0"
        ]
        
        # Download dependencies
        download_deps = [
            "gdown",
            "requests"
        ]
        
        # Chemistry dependencies (may need special handling)
        chem_deps = [
            "rdkit-pypi>=2022.3.0"
        ]
        
        # Optional dependencies
        optional_deps = [
            "wandb>=0.12.0",
            "tensorboard>=2.8.0"
        ]
        
        all_deps = core_deps + download_deps + chem_deps + optional_deps
        
        print(f"Installing {len(all_deps)} packages...")
        
        failed_packages = []
        for i, package in enumerate(all_deps, 1):
            try:
                print(f"   [{i:2d}/{len(all_deps)}] Installing {package.split('>=')[0]}...")
                subprocess.run([
                    sys.executable, "-m", "pip", "install", package
                ], check=True, capture_output=True, text=True)
                
            except subprocess.CalledProcessError as e:
                print(f"   ❌ Failed to install {package}")
                failed_packages.append(package)
                continue
        
        if failed_packages:
            print(f"\n⚠️  Some packages failed to install:")
            for pkg in failed_packages:
                print(f"   • {pkg}")
            print("You may need to install these manually.")
        else:
            print("✅ All dependencies installed successfully!")
        
        return len(failed_packages) == 0
    
    def setup_project_structure(self):
        """Create project directory structure"""
        print("\n🏗️  Setting up project structure...")
        
        directories = [
            "project_folder",
            "project_folder/datasets",
            "project_folder/embeddings", 
            "project_folder/embeddings/rdkit",
            "project_folder/embeddings/rdkit/data",
            "project_folder/embeddings/rdkit/data/embeddings",
            "project_folder/embeddings/chemCPA",
            "project_folder/binaries",
            "outputs",
            "outputs/checkpoints",
            "outputs/logs",
            "plots",
            "logs"
        ]
        
        for directory in directories:
            dir_path = Path(directory)
            
            # Handle existing files/symlinks
            if dir_path.exists():
                if dir_path.is_dir():
                    print(f"   📁 {directory} (already exists)")
                    continue
                elif dir_path.is_symlink():
                    print(f"   🔗 {directory} (symlink exists)")
                    # Create subdirectories if it's a symlink to a directory
                    if dir_path.is_dir():
                        continue
                else:
                    print(f"   ⚠️  {directory} exists but is not a directory")
                    continue
            
            try:
                dir_path.mkdir(parents=True, exist_ok=True)
                print(f"   📁 {directory}")
            except FileExistsError:
                # Handle race condition or symlink issues
                if dir_path.exists() and dir_path.is_dir():
                    print(f"   📁 {directory} (created)")
                else:
                    print(f"   ⚠️  Could not create {directory}")
        
        print("✅ Project structure setup complete!")
    
    def download_essential_datasets(self, quick_mode=False):
        """Download essential datasets for stem cell work"""
        print(f"\n📥 Downloading {'essential' if quick_mode else 'comprehensive'} datasets...")
        
        if quick_mode:
            # Quick mode: just download Sciplex for testing
            datasets_to_download = ['cpa_binaries']
            print("Quick mode: Downloading only Sciplex dataset for testing")
        else:
            # Full mode: download stem cell essentials
            datasets_to_download = ['cpa_binaries', 'adata_biolord_split_30', 'drugbank_all']
            print("Full mode: Downloading essential datasets for stem cell work")
        
        try:
            # Import and use the download script
            sys.path.append(str(self.root_dir))
            from download_datasets import ChemCPADatasetDownloader
            
            downloader = ChemCPADatasetDownloader()
            
            success_count = 0
            for dataset in datasets_to_download:
                try:
                    print(f"\n{'='*50}")
                    print(f"📦 Downloading: {dataset}")
                    print(f"{'='*50}")
                    
                    downloader.download_single_dataset(dataset)
                    success_count += 1
                    
                except Exception as e:
                    print(f"❌ Error downloading {dataset}: {e}")
                    continue
            
            if success_count == len(datasets_to_download):
                print(f"\n✅ All {len(datasets_to_download)} datasets downloaded successfully!")
                return True
            else:
                print(f"\n⚠️  Downloaded {success_count}/{len(datasets_to_download)} datasets")
                return False
                
        except Exception as e:
            print(f"❌ Error setting up dataset downloader: {e}")
            print("You can download datasets manually using: python download_datasets.py")
            return False
    
    def install_chemcpa_package(self):
        """Install ChemCPA package in development mode"""
        print("\n🔧 Installing ChemCPA package...")
        
        try:
            subprocess.run([
                sys.executable, "-m", "pip", "install", "-e", "."
            ], check=True, capture_output=True, text=True)
            print("✅ ChemCPA package installed!")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Error installing ChemCPA package: {e}")
            print("You may need to install manually with: pip install -e .")
            return False
    
    def run_test_example(self):
        """Run a quick test to verify setup"""
        print("\n🧪 Running setup verification test...")
        
        try:
            # Try to import key modules
            import torch
            import lightning
            import scanpy
            import pandas
            import numpy
            
            print("✅ Core dependencies imported successfully!")
            
            # Check if datasets exist
            sciplex_path = self.project_folder / "datasets" / "sciplex_raw_chunk_0.h5ad"
            if sciplex_path.exists():
                print("✅ Sciplex dataset found!")
            else:
                print("⚠️  Sciplex dataset not found - you may need to download it")
            
            return True
            
        except ImportError as e:
            print(f"❌ Import error: {e}")
            return False
    
    def print_next_steps(self, quick_mode=False):
        """Print next steps for the user"""
        print("\n" + "="*80)
        print("🎉 SETUP COMPLETE!")
        print("="*80)
        
        print("\n📋 NEXT STEPS:")
        
        if quick_mode:
            print("\n1️⃣  Download more datasets (optional):")
            print("   python download_datasets.py --stem-cell-essentials")
            print("   python download_datasets.py --dataset broad")
        
        print("\n2️⃣  Train your first model:")
        print("   python train_chemcpa_simple.py --dataset sciplex --epochs 50")
        
        print("\n3️⃣  Make predictions on unseen drugs:")
        print("   python predict_perturbation.py \\")
        print("       --model_path ./outputs/checkpoints/best_model.ckpt \\")
        print("       --input_data your_stem_cells.h5ad \\")
        print("       --drug_smiles \"CCO\"")
        
        print("\n4️⃣  Run the example workflow:")
        print("   python example_workflow.py")
        
        print("\n📚 DOCUMENTATION:")
        print("   • Read: STEM_CELL_TRAINING_README.md")
        print("   • Configuration: config_stem_cells.yaml")
        print("   • Download more data: python download_datasets.py --help")
        
        print("\n💡 TIPS:")
        print("   • Start with Sciplex dataset for quick testing")
        print("   • Use Broad dataset for comprehensive training")
        print("   • Check dataset status: python download_datasets.py --status")
        
        print("="*80)
        print("🧬 Ready to predict stem cell drug perturbations! 🧬")
        print("="*80)


def main():
    parser = argparse.ArgumentParser(
        description='Setup ChemCPA for stem cell drug perturbation prediction',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--quick', action='store_true',
                       help='Quick setup with minimal datasets (faster)')
    parser.add_argument('--datasets-only', action='store_true', 
                       help='Only download datasets (skip package installation)')
    parser.add_argument('--no-datasets', action='store_true',
                       help='Skip dataset download (only setup environment)')
    
    args = parser.parse_args()
    
    setup = ChemCPASetup()
    
    # Print banner
    setup.print_banner()
    
    # Check Python version
    if not setup.check_python_version():
        sys.exit(1)
    
    success = True
    
    if not args.datasets_only:
        # Install dependencies
        if not setup.install_dependencies():
            print("⚠️  Some dependencies failed to install, but continuing...")
        
        # Setup project structure
        setup.setup_project_structure()
        
        # Install ChemCPA package
        if not setup.install_chemcpa_package():
            success = False
    
    if not args.no_datasets:
        # Download datasets
        if not setup.download_essential_datasets(quick_mode=args.quick):
            print("⚠️  Some datasets failed to download, but continuing...")
    
    if not args.datasets_only:
        # Run verification test
        if not setup.run_test_example():
            success = False
    
    # Print next steps
    setup.print_next_steps(quick_mode=args.quick)
    
    if success:
        print("\n🎉 Setup completed successfully!")
    else:
        print("\n⚠️  Setup completed with some issues. Check messages above.")
        sys.exit(1)


if __name__ == '__main__':
    main()
