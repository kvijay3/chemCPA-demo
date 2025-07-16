#!/usr/bin/env python3
"""
Fix Project Folder Script

This script fixes the broken project_folder symlink and creates a proper local directory structure.
The original symlink points to a path that doesn't exist on your system.

Usage:
    python fix_project_folder.py
"""

import os
import sys
from pathlib import Path
import shutil


def fix_project_folder():
    """Fix the broken project_folder symlink"""
    print("🔧 Fixing project_folder setup...")
    
    project_folder = Path("project_folder")
    
    # Check current state
    if project_folder.is_symlink():
        target = project_folder.readlink()
        print(f"   Current symlink points to: {target}")
        
        if not target.exists():
            print("   ❌ Symlink target doesn't exist - removing broken symlink")
            project_folder.unlink()
        else:
            print("   ✅ Symlink target exists - keeping it")
            return True
    
    # Create local project_folder if it doesn't exist
    if not project_folder.exists():
        print("   📁 Creating local project_folder directory")
        project_folder.mkdir()
    
    # Create subdirectories
    subdirs = [
        "datasets",
        "embeddings",
        "embeddings/rdkit",
        "embeddings/rdkit/data", 
        "embeddings/rdkit/data/embeddings",
        "embeddings/chemCPA",
        "binaries"
    ]
    
    for subdir in subdirs:
        subdir_path = project_folder / subdir
        subdir_path.mkdir(parents=True, exist_ok=True)
        print(f"   📁 project_folder/{subdir}")
    
    print("✅ project_folder fixed!")
    return True


def create_other_directories():
    """Create other necessary directories"""
    print("🏗️  Creating other directories...")
    
    other_dirs = [
        "outputs",
        "outputs/checkpoints",
        "outputs/logs", 
        "plots",
        "logs"
    ]
    
    for directory in other_dirs:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"   📁 {directory}")
    
    print("✅ Additional directories created!")


def main():
    print("🧬 ChemCPA Project Folder Fix")
    print("="*50)
    print("This script fixes the broken project_folder symlink")
    print("and creates a proper local directory structure.")
    print("="*50)
    
    # Fix project folder
    if not fix_project_folder():
        print("❌ Failed to fix project_folder")
        return False
    
    # Create other directories
    create_other_directories()
    
    # Verify setup
    print("\n📋 Verifying setup...")
    project_folder = Path("project_folder")
    
    if project_folder.exists() and project_folder.is_dir():
        print("✅ project_folder is now a proper directory")
        
        # List contents
        subdirs = list(project_folder.iterdir())
        if subdirs:
            print("   Contents:")
            for subdir in sorted(subdirs):
                print(f"   📁 {subdir.name}")
        else:
            print("   (empty - this is normal)")
    else:
        print("❌ project_folder still has issues")
        return False
    
    print("\n🎉 Setup fixed successfully!")
    print("\nNext steps:")
    print("1. Download datasets: python download_datasets.py --stem-cell-essentials")
    print("2. Train a model: python train_chemcpa_simple.py --dataset sciplex")
    
    return True


if __name__ == '__main__':
    success = main()
    if not success:
        sys.exit(1)

