#!/usr/bin/env python3
"""
Upload necessary files to Modal volume for ChemCPA training
"""

import modal
import os
import shutil
import tarfile
from pathlib import Path

# Connect to the same app and volume
app = modal.App("chemcpa-training")
volume = modal.Volume.from_name("chemcpa-data", create_if_missing=True)

# Simple image for file operations
image = modal.Image.debian_slim(python_version="3.10")

@app.function(
    image=image,
    volumes={"/data": volume},
    timeout=1800,  # 30 minutes
)
def upload_and_extract(tar_data: bytes, filename: str):
    """Extract uploaded tar file to Modal volume"""
    import tarfile
    import io
    from pathlib import Path
    
    print(f"📤 Extracting {filename} to Modal volume...")
    
    data_dir = Path("/data")
    
    # Extract tar file
    with tarfile.open(fileobj=io.BytesIO(tar_data), mode='r:gz') as tar:
        tar.extractall(data_dir)
    
    # List what was uploaded
    print("📂 Files uploaded:")
    for item in data_dir.rglob("*"):
        if item.is_file():
            print(f"  {item.relative_to(data_dir)}")
    
    volume.commit()
    print("✅ Upload completed and volume committed!")
    return {"status": "success", "message": f"Uploaded {filename}"}

def create_upload_package():
    """Create a tar.gz package with necessary files"""
    print("📦 Creating upload package...")
    
    # Files and directories to include
    include_items = [
        "chemCPA/",
        "train_chemcpa_simple.py", 
        "config/",
        "datasets/",  # This includes your downloaded datasets
    ]
    
    # Create tar.gz file
    tar_path = Path("chemcpa_upload.tar.gz")
    
    with tarfile.open(tar_path, "w:gz") as tar:
        for item in include_items:
            item_path = Path(item)
            if item_path.exists():
                print(f"  Adding {item}...")
                tar.add(item_path, arcname=item)
            else:
                print(f"  ⚠️  Skipping {item} (not found)")
    
    print(f"✅ Package created: {tar_path} ({tar_path.stat().st_size / 1024 / 1024:.1f} MB)")
    return tar_path

def main():
    """Main upload function"""
    print("🚀 ChemCPA Modal Upload Script")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not Path("train_chemcpa_simple.py").exists():
        print("❌ Please run this script from the chemCPA-demo directory")
        return
    
    # Create upload package
    tar_path = create_upload_package()
    
    # Read the tar file
    with open(tar_path, "rb") as f:
        tar_data = f.read()
    
    print(f"📤 Uploading {len(tar_data) / 1024 / 1024:.1f} MB to Modal...")
    
    # Upload to Modal
    with app.run():
        result = upload_and_extract.remote(tar_data, tar_path.name)
        print(f"Result: {result}")
    
    # Clean up local tar file
    tar_path.unlink()
    print("🧹 Cleaned up local tar file")
    
    print("\n✅ Upload completed! You can now run training with:")
    print("modal run modal_train.py::train_chemcpa")

if __name__ == "__main__":
    main()
