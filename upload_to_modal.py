import modal
import shutil
from pathlib import Path

app = modal.App("chemcpa-upload")
volume = modal.Volume.from_name("chemcpa-data", create_if_missing=True)

@app.function(volumes={"/data": volume}, timeout=1800)
def upload_updated_code():
    """Upload the updated ChemCPA code to Modal volume"""
    import os
    import subprocess
    
    data_dir = Path("/data")
    
    print("📁 Current contents of /data:")
    for item in data_dir.rglob("*"):
        if item.is_file():
            print(f"   {item}")
    
    # Remove old code
    if (data_dir / "chemCPA").exists():
        shutil.rmtree(data_dir / "chemCPA")
        print("🗑️ Removed old chemCPA code")
    
    if (data_dir / "train_chemcpa_simple.py").exists():
        (data_dir / "train_chemcpa_simple.py").unlink()
        print("🗑️ Removed old training script")
    
    # Clone the latest code with fixes
    print("📥 Cloning updated code...")
    result = subprocess.run([
        "git", "clone", "--branch", "codegen-bot/fix-dataset-configuration-degs-key",
        "https://github.com/kvijay3/chemCPA-demo.git", "/tmp/chemcpa-updated"
    ], capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"❌ Git clone failed: {result.stderr}")
        return {"error": "Git clone failed"}
    
    # Copy updated files to volume
    updated_dir = Path("/tmp/chemcpa-updated")
    
    # Copy chemCPA module
    if (updated_dir / "chemCPA").exists():
        shutil.copytree(updated_dir / "chemCPA", data_dir / "chemCPA")
        print("✅ Copied updated chemCPA module")
    
    # Copy training script
    if (updated_dir / "train_chemcpa_simple.py").exists():
        shutil.copy2(updated_dir / "train_chemcpa_simple.py", data_dir / "train_chemcpa_simple.py")
        print("✅ Copied updated training script")
    
    # Copy config
    if (updated_dir / "config").exists():
        if (data_dir / "config").exists():
            shutil.rmtree(data_dir / "config")
        shutil.copytree(updated_dir / "config", data_dir / "config")
        print("✅ Copied updated config")
    
    print("🎉 Code update complete!")
    return {"status": "success"}

if __name__ == "__main__":
    with app.run():
        result = upload_updated_code.remote()
        print("Upload result:", result)
