import modal
import shutil
from pathlib import Path
import os

app = modal.App("chemcpa-upload-simple")
volume = modal.Volume.from_name("chemcpa-data", create_if_missing=True)

@app.function(volumes={"/data": volume}, timeout=1800)
def upload_local_code():
    """Upload the local ChemCPA code to Modal volume"""
    
    data_dir = Path("/data")
    
    print("📁 Current contents of /data:")
    for item in data_dir.rglob("*"):
        if item.is_file():
            print(f"   {item}")
    
    # Create directories if they don't exist
    (data_dir / "results").mkdir(exist_ok=True)
    (data_dir / "results" / "outputs").mkdir(exist_ok=True)
    (data_dir / "results" / "outputs" / "checkpoints").mkdir(exist_ok=True)
    
    print("✅ Created necessary directories")
    return {"status": "directories_created"}

# Function to copy files from local to Modal (run this locally)
def copy_files_to_modal():
    """This function helps copy files from local to Modal volume"""
    
    # This is a helper - the actual file copying needs to be done
    # by mounting the volume and copying files
    
    print("📋 To upload your local code to Modal:")
    print("1. Make sure your local code has all the fixes")
    print("2. The Modal training script will copy code from working directory")
    print("3. Or use the modal_train.py which already handles code copying")
    
    return {"status": "helper_info"}

if __name__ == "__main__":
    with app.run():
        result = upload_local_code.remote()
        print("Upload result:", result)
        
        # Show helper info
        copy_files_to_modal()

