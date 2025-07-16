#!/usr/bin/env python3
"""
Modal deployment script for ChemCPA training on A100 GPU
"""

import modal
import os
from pathlib import Path

# Create Modal app
app = modal.App("chemcpa-training")

# Define the image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install([
        "torch>=2.0.0",
        "lightning>=2.0.0", 
        "numpy",
        "pandas",
        "scipy",
        "scikit-learn",
        "h5py",
        "anndata",
        "scanpy",
        "rdkit",
        "hydra-core",
        "omegaconf",
        "tqdm",
        "matplotlib",
        "seaborn",
        "tensorboard",
        "wandb",
        "sympy",
    ])
    .apt_install(["git"])
)

# Create volume for persistent storage
volume = modal.Volume.from_name("chemcpa-data", create_if_missing=True)

@app.function(
    image=image,
    gpu="A100-40GB",
    volumes={"/data": volume},
    timeout=3600 * 4,  # 4 hours timeout
    memory=32768,  # 32GB RAM
)
def train_chemcpa(
    dataset: str = "lincs",
    epochs: int = 50,
    batch_size: int = 128,
    learning_rate: float = 1e-3
):
    """Train ChemCPA model on Modal with A100 GPU"""
    import subprocess
    import sys
    import os
    import shutil
    from pathlib import Path
    
    print("🚀 Starting ChemCPA training on A100 GPU...")
    print(f"Dataset: {dataset}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    
    # Set up working directory
    work_dir = Path("/tmp/chemcpa")
    work_dir.mkdir(exist_ok=True)
    os.chdir(work_dir)
    
    # Check if data exists in volume, if not download it
    data_dir = Path("/data")
    if not (data_dir / "datasets").exists():
        print("📥 Downloading datasets...")
        # We'll need to copy the dataset download script
        # For now, let's assume data is uploaded separately
        print("⚠️  Please upload datasets to Modal volume first")
        return {"error": "Datasets not found in volume"}
    
    # Copy datasets from volume to working directory
    print("📂 Copying datasets from volume...")
    # Create project_folder structure to match expected paths
    project_folder = work_dir / "project_folder"
    project_folder.mkdir(exist_ok=True)
    shutil.copytree(data_dir / "datasets", project_folder / "datasets")
    
    # Copy source code from volume
    if (data_dir / "chemCPA").exists():
        shutil.copytree(data_dir / "chemCPA", work_dir / "chemCPA")
    else:
        print("⚠️  ChemCPA source code not found in volume")
        return {"error": "Source code not found"}
    
    # Copy training script
    if (data_dir / "train_chemcpa_simple.py").exists():
        shutil.copy2(data_dir / "train_chemcpa_simple.py", work_dir / "train_chemcpa_simple.py")
    else:
        print("⚠️  Training script not found in volume")
        return {"error": "Training script not found"}
    
    # Copy config files
    if (data_dir / "config").exists():
        shutil.copytree(data_dir / "config", work_dir / "config")
    
    # Copy embeddings directory to project_folder
    if (data_dir / "embeddings").exists():
        shutil.copytree(data_dir / "embeddings", project_folder / "embeddings")
        print("✅ Copied embeddings directory")
    else:
        print("⚠️  Embeddings directory not found in volume")
    
    # Run training
    print("🏋️ Starting training...")
    print(f"📊 Training configuration:")
    print(f"   Dataset: {dataset}")
    print(f"   Epochs: {epochs}")
    print(f"   Batch size: {batch_size}")
    print(f"   Learning rate: {learning_rate}")
    print("="*60)
    
    # Set output directory to save models in working directory
    output_dir = work_dir / "outputs"
    output_dir.mkdir(exist_ok=True)
    
    cmd = [
        sys.executable, "train_chemcpa_simple.py",
        "--dataset", dataset,
        "--epochs", str(epochs),
        "--batch_size", str(batch_size),
        "--learning_rate", str(learning_rate),
        "--output_dir", str(output_dir)
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("✅ Training completed successfully!")
        print("STDOUT:", result.stdout)
        
        # Save results back to volume
        results_dir = data_dir / "results"
        results_dir.mkdir(exist_ok=True)
        
        # Copy any generated files back to volume
        print("📁 Copying training outputs to volume...")
        
        # Copy the entire outputs directory
        if output_dir.exists():
            shutil.copytree(output_dir, results_dir / "outputs", dirs_exist_ok=True)
            print(f"✅ Copied outputs directory to volume: {results_dir / 'outputs'}")
        
        # Also copy any other generated files
        for pattern in ["*.ckpt", "*.log", "lightning_logs/"]:
            for file_path in work_dir.glob(pattern):
                if file_path.is_file():
                    shutil.copy2(file_path, results_dir)
                elif file_path.is_dir():
                    shutil.copytree(file_path, results_dir / file_path.name, dirs_exist_ok=True)
        
        # Print model locations
        checkpoints_dir = results_dir / "outputs" / "checkpoints"
        if checkpoints_dir.exists():
            print(f"🎯 Model checkpoints saved to volume at: {checkpoints_dir}")
            ckpt_files = list(checkpoints_dir.glob("*.ckpt"))
            if ckpt_files:
                print(f"📦 Found {len(ckpt_files)} checkpoint files:")
                for ckpt in ckpt_files:
                    print(f"   - {ckpt.name}")
        else:
            print("⚠️  No checkpoints directory found")
        
        return {
            "status": "success",
            "stdout": result.stdout,
            "stderr": result.stderr
        }
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Training failed with exit code {e.returncode}")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        return {
            "status": "error",
            "exit_code": e.returncode,
            "stdout": e.stdout,
            "stderr": e.stderr
        }

@app.function(
    image=image,
    volumes={"/data": volume},
    timeout=1800,  # 30 minutes for upload
)
def upload_files():
    """Upload necessary files to Modal volume"""
    import subprocess
    import shutil
    from pathlib import Path
    
    print("📤 Uploading files to Modal volume...")
    
    # This function would be called locally to upload files
    # But since we can't access local files from Modal function,
    # we'll create a separate upload script
    
    data_dir = Path("/data")
    
    # Create directories
    (data_dir / "datasets").mkdir(exist_ok=True)
    (data_dir / "results").mkdir(exist_ok=True)
    
    print("✅ Volume directories created")
    return {"status": "ready"}

@app.function(
    image=image,
    volumes={"/data": volume},
    timeout=60,  # 1 minute timeout
)
def list_saved_models():
    """List all saved model checkpoints in the volume"""
    from pathlib import Path
    
    data_dir = Path("/data")
    results_dir = data_dir / "results"
    
    if not results_dir.exists():
        return {"error": "No training results found"}
    
    models = []
    
    # Check for models in outputs/checkpoints
    checkpoints_dir = results_dir / "outputs" / "checkpoints"
    if checkpoints_dir.exists():
        for ckpt_file in checkpoints_dir.glob("*.ckpt"):
            models.append({
                "path": str(ckpt_file),
                "name": ckpt_file.name,
                "size_mb": round(ckpt_file.stat().st_size / (1024 * 1024), 2)
            })
    
    # Check for any other checkpoint files
    for ckpt_file in results_dir.rglob("*.ckpt"):
        if ckpt_file not in [Path(m["path"]) for m in models]:
            models.append({
                "path": str(ckpt_file),
                "name": ckpt_file.name,
                "size_mb": round(ckpt_file.stat().st_size / (1024 * 1024), 2)
            })
    
    return {
        "models": models,
        "total_models": len(models),
        "results_dir": str(results_dir)
    }

if __name__ == "__main__":
    # Example usage
    with app.run():
        # Train model
        result = train_chemcpa.remote(
            dataset="lincs",
            epochs=50,
            batch_size=128,
            learning_rate=1e-3
        )
        print("Training result:", result)
        
        # List saved models
        models = list_saved_models.remote()
        print("Saved models:", models)
