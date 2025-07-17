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
    timeout=3600 * 20,  # 20 hours timeout
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
    
    # Copy source code from current repo (mounted via Modal)
    # Modal automatically syncs the current directory
    import subprocess
    
    print("📥 Cloning latest code with fixes...")
    result = subprocess.run([
        "git", "clone", "--branch", "codegen-bot/fix-dataset-configuration-degs-key",
        "https://github.com/kvijay3/chemCPA-demo.git", "/tmp/chemcpa-source"
    ], capture_output=True, text=True)
    
    if result.returncode == 0:
        source_dir = Path("/tmp/chemcpa-source")
        
        # Copy chemCPA module
        if (source_dir / "chemCPA").exists():
            shutil.copytree(source_dir / "chemCPA", work_dir / "chemCPA")
            print("✅ Copied chemCPA module with fixes")
        
        # Copy training script
        if (source_dir / "train_chemcpa_simple.py").exists():
            shutil.copy2(source_dir / "train_chemcpa_simple.py", work_dir / "train_chemcpa_simple.py")
            print("✅ Copied training script with fixes")
        
        # Copy config files
        if (source_dir / "config").exists():
            shutil.copytree(source_dir / "config", work_dir / "config")
            print("✅ Copied config files")
    else:
        print(f"❌ Failed to clone repo: {result.stderr}")
        # Fallback to volume if available
        if (data_dir / "chemCPA").exists():
            shutil.copytree(data_dir / "chemCPA", work_dir / "chemCPA")
            print("⚠️  Using code from volume (may be outdated)")
        else:
            return {"error": "Could not get source code"}
    
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
        sys.executable, "-u", "train_chemcpa_simple.py",  # -u for unbuffered output
        "--dataset", dataset,
        "--epochs", str(epochs),
        "--batch_size", str(batch_size),
        "--learning_rate", str(learning_rate),
        "--output_dir", str(output_dir)
    ]
    
    try:
        # Use Popen for real-time output streaming
        print("🚀 Starting training with real-time output...")
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT,  # Merge stderr into stdout
            text=True,
            bufsize=1,  # Line buffered
            universal_newlines=True
        )
        
        # Stream output in real-time
        output_lines = []
        while True:
            line = process.stdout.readline()
            if line == '' and process.poll() is not None:
                break
            if line:
                print(line.rstrip())  # Print to Modal logs immediately
                output_lines.append(line.rstrip())
        
        # Wait for process to complete
        return_code = process.wait()
        
        if return_code == 0:
            print("✅ Training completed successfully!")
        
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
                "output": "\n".join(output_lines),
                "model_path": str(results_dir / "outputs" / "checkpoints")
            }
        else:
            print(f"❌ Training failed with exit code {return_code}")
            return {
                "status": "error",
                "exit_code": return_code,
                "output": "\n".join(output_lines)
            }
        
    except Exception as e:
        print(f"❌ Training failed with exception: {e}")
        return {
            "status": "error",
            "exception": str(e)
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
