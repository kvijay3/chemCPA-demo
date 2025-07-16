#!/usr/bin/env python3
"""
Download training results from Modal volume
"""

import modal
import tarfile
from pathlib import Path

# Connect to the same app and volume
app = modal.App("chemcpa-training")
volume = modal.Volume.from_name("chemcpa-data")

# Simple image for file operations
image = modal.Image.debian_slim(python_version="3.10")

@app.function(
    image=image,
    volumes={"/data": volume},
    timeout=600,  # 10 minutes
)
def create_results_package():
    """Create a tar.gz package with training results"""
    import tarfile
    import io
    from pathlib import Path
    
    print("📦 Creating results package...")
    
    data_dir = Path("/data")
    results_dir = data_dir / "results"
    
    if not results_dir.exists():
        print("❌ No results directory found")
        return None
    
    # Create tar.gz in memory
    tar_buffer = io.BytesIO()
    
    with tarfile.open(fileobj=tar_buffer, mode='w:gz') as tar:
        for item in results_dir.rglob("*"):
            if item.is_file():
                print(f"  Adding {item.relative_to(results_dir)}...")
                tar.add(item, arcname=item.relative_to(results_dir))
    
    tar_data = tar_buffer.getvalue()
    print(f"✅ Results package created ({len(tar_data) / 1024 / 1024:.1f} MB)")
    
    return tar_data

def main():
    """Main download function"""
    print("🚀 ChemCPA Results Download Script")
    print("=" * 50)
    
    # Download results from Modal
    with app.run():
        tar_data = create_results_package.remote()
        
        if tar_data is None:
            print("❌ No results to download")
            return
    
    # Save results locally
    results_path = Path("training_results.tar.gz")
    with open(results_path, "wb") as f:
        f.write(tar_data)
    
    print(f"📥 Downloaded results to {results_path}")
    
    # Extract results
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    with tarfile.open(results_path, "r:gz") as tar:
        tar.extractall(results_dir)
    
    print(f"📂 Extracted results to {results_dir}/")
    
    # List downloaded files
    print("\n📋 Downloaded files:")
    for item in results_dir.rglob("*"):
        if item.is_file():
            print(f"  {item}")
    
    # Clean up tar file
    results_path.unlink()
    print("\n✅ Download completed!")

if __name__ == "__main__":
    main()
