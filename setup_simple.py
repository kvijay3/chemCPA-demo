#!/usr/bin/env python3
"""
Simple ChemCPA Setup Script

A simplified setup script that works around common issues like existing symlinks.
This script focuses on the essential setup steps.

Usage:
    python setup_simple.py
"""

import os
import sys
import subprocess
from pathlib import Path


def install_dependencies():
    """Install essential dependencies"""
    print("📦 Installing essential dependencies...")
    
    # Essential packages only
    packages = [
        "torch",
        "lightning", 
        "numpy",
        "pandas",
        "scanpy",
        "anndata",
        "omegaconf",
        "hydra-core",
        "matplotlib",
        "seaborn",
        "h5py",
        "gdown",
        "requests",
        "rdkit-pypi"
    ]
    
    for package in packages:
        try:
            print(f"   Installing {package}...")
            subprocess.run([sys.executable, "-m", "pip", "install", package], 
                         check=True, capture_output=True)
        except subprocess.CalledProcessError:
            print(f"   ⚠️  Failed to install {package} - you may need to install manually")
    
    print("✅ Dependencies installed!")


def fix_project_folder():
    """Fix broken project_folder symlink and create proper directory"""
    print("🔧 Fixing project_folder...")
    
    project_folder = Path("project_folder")
    
    # Check if it's a broken symlink
    if project_folder.is_symlink():
        target = project_folder.readlink()
        if not target.exists():
            print(f"   ❌ Broken symlink to {target} - removing")
            project_folder.unlink()
        else:
            print(f"   🔗 Valid symlink to {target}")
            return True
    
    # Create local directory if needed
    if not project_folder.exists():
        project_folder.mkdir()
        print("   📁 Created local project_folder")
    
    return True


def create_directories():
    """Create necessary directories"""
    print("🏗️  Creating directories...")
    
    # Fix project_folder first
    if not fix_project_folder():
        print("❌ Failed to fix project_folder")
        return False
    
    # Create subdirectories in project_folder
    project_subdirs = [
        "project_folder/datasets",
        "project_folder/embeddings",
        "project_folder/embeddings/rdkit",
        "project_folder/embeddings/rdkit/data",
        "project_folder/embeddings/rdkit/data/embeddings", 
        "project_folder/embeddings/chemCPA",
        "project_folder/binaries"
    ]
    
    for subdir in project_subdirs:
        try:
            Path(subdir).mkdir(parents=True, exist_ok=True)
            print(f"   📁 {subdir}")
        except Exception as e:
            print(f"   ⚠️  Could not create {subdir}: {e}")
    
    # Create other directories
    other_dirs = [
        "outputs",
        "outputs/checkpoints", 
        "outputs/logs",
        "plots",
        "logs"
    ]
    
    for directory in other_dirs:
        try:
            Path(directory).mkdir(parents=True, exist_ok=True)
            print(f"   📁 {directory}")
        except Exception as e:
            print(f"   ⚠️  Could not create {directory}: {e}")
    
    print("✅ Directories created!")
    return True


def install_chemcpa():
    """Install ChemCPA package"""
    print("🔧 Installing ChemCPA package...")
    
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-e", "."], 
                      check=True, capture_output=True)
        print("✅ ChemCPA package installed!")
        return True
    except subprocess.CalledProcessError:
        print("⚠️  ChemCPA package installation failed - you may need to run manually:")
        print("   pip install -e .")
        return False


def main():
    print("🧬 Simple ChemCPA Setup")
    print("="*50)
    
    # Check Python version
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python {version.major}.{version.minor} detected.")
        print("❌ ChemCPA requires Python 3.8 or higher.")
        return
    
    print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
    
    # Install dependencies
    install_dependencies()
    
    # Create directories
    if not create_directories():
        print("❌ Directory creation failed")
        return
    
    # Install ChemCPA
    install_chemcpa()
    
    print("\n🎉 Setup complete!")
    print("\nNext steps:")
    print("1. Download datasets: python download_datasets.py --stem-cell-essentials")
    print("2. Train a model: python train_chemcpa_simple.py --dataset sciplex")
    print("3. Read the guide: STEM_CELL_TRAINING_README.md")


if __name__ == '__main__':
    main()
