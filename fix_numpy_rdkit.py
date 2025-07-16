#!/usr/bin/env python3
"""
Quick fix for NumPy/RDKit compatibility issues in ChemCPA
Run this script to downgrade NumPy and fix the training script
"""

import subprocess
import sys
import os

def run_command(cmd):
    """Run a command and return success status"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {cmd}")
            return True
        else:
            print(f"❌ {cmd}")
            print(f"Error: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ {cmd}")
        print(f"Exception: {e}")
        return False

def fix_training_script():
    """Fix the training script to remove description parameter"""
    script_path = "train_chemcpa_simple.py"
    
    if not os.path.exists(script_path):
        print(f"❌ {script_path} not found")
        return False
    
    # Read the file
    with open(script_path, 'r') as f:
        content = f.read()
    
    # Fix the load_dataset_splits call
    old_line = "datasets, dataset = load_dataset_splits(**self.config['dataset'], return_dataset=True)"
    new_lines = """# Remove description from config before passing to load_dataset_splits
            dataset_config = {k: v for k, v in self.config['dataset'].items() if k != 'description'}
            datasets, dataset = load_dataset_splits(**dataset_config, return_dataset=True)"""
    
    if old_line in content:
        content = content.replace(old_line, new_lines)
        
        # Write back
        with open(script_path, 'w') as f:
            f.write(content)
        
        print("✅ Fixed training script parameter issue")
        return True
    else:
        print("⚠️  Training script already appears to be fixed")
        return True

def main():
    print("🔧 ChemCPA NumPy/RDKit Compatibility Fix")
    print("=" * 50)
    
    # Step 1: Downgrade NumPy
    print("\n1. Downgrading NumPy to fix RDKit compatibility...")
    if not run_command('pip install "numpy<2.0" --force-reinstall'):
        print("❌ Failed to downgrade NumPy")
        return False
    
    # Step 2: Reinstall RDKit
    print("\n2. Reinstalling RDKit...")
    run_command('pip uninstall rdkit -y')
    if not run_command('pip install rdkit'):
        print("❌ Failed to reinstall RDKit")
        return False
    
    # Step 3: Fix training script
    print("\n3. Fixing training script...")
    if not fix_training_script():
        return False
    
    # Step 4: Test imports
    print("\n4. Testing imports...")
    if not run_command('python -c "from rdkit import Chem; print(\\"RDKit OK\\")"'):
        print("❌ RDKit import still failing")
        return False
    
    if not run_command('python -c "import train_chemcpa_simple; print(\\"Training script OK\\")"'):
        print("❌ Training script import still failing")
        return False
    
    print("\n🎉 All fixes applied successfully!")
    print("\nYou can now run:")
    print("python train_chemcpa_simple.py --dataset biolord --epochs 50")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

