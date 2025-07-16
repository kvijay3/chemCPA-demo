# ChemCPA Training on Modal A100 GPU

This guide shows how to train ChemCPA models on Modal's A100 GPUs for much faster training than local Mac.

## Prerequisites

1. **Install Modal CLI:**
   ```bash
   pip install modal
   ```

2. **Set up Modal account:**
   ```bash
   modal setup
   ```
   Follow the prompts to create an account and authenticate.

3. **Make sure you have the latest fixes:**
   ```bash
   git pull origin codegen-bot/fix-dataset-configuration-degs-key
   ```

## Step 1: Upload Files to Modal

Upload your code and datasets to Modal's persistent volume:

```bash
python upload_to_modal.py
```

This will:
- ✅ Package necessary files (`chemCPA/`, `train_chemcpa_simple.py`, `config/`, `datasets/`)
- ✅ Upload to Modal volume (persistent storage)
- ✅ Extract files on Modal's infrastructure

**Expected output:**
```
🚀 ChemCPA Modal Upload Script
==================================================
📦 Creating upload package...
  Adding chemCPA/...
  Adding train_chemcpa_simple.py...
  Adding config/...
  Adding datasets/...
✅ Package created: chemcpa_upload.tar.gz (XXX.X MB)
📤 Uploading XXX.X MB to Modal...
📤 Extracting chemcpa_upload.tar.gz to Modal volume...
📂 Files uploaded:
  chemCPA/...
  train_chemcpa_simple.py
  config/...
  datasets/...
✅ Upload completed and volume committed!
🧹 Cleaned up local tar file

✅ Upload completed! You can now run training with:
modal run modal_train.py::train_chemcpa
```

## Step 2: Start Training on A100

Run training on Modal's A100 GPU:

```bash
modal run modal_train.py::train_chemcpa
```

**With custom parameters:**
```bash
modal run modal_train.py::train_chemcpa --dataset lincs --epochs 50 --batch_size 256 --learning_rate 1e-3
```

**What happens:**
- 🚀 Spins up A100 GPU instance (much faster than Mac!)
- 📂 Copies files from volume to working directory
- 🏋️ Runs training with your parameters
- 💾 Saves results back to persistent volume
- 💰 Automatically shuts down when complete (pay per use)

**Expected training time:** ~30-60 minutes (vs hours on Mac)

## Step 3: Download Results

After training completes, download your results:

```bash
python download_results.py
```

This will:
- ✅ Download trained model checkpoints
- ✅ Download training logs
- ✅ Download any generated files
- ✅ Extract to local `results/` directory

## File Structure

```
chemCPA-demo/
├── modal_train.py          # Modal training script
├── upload_to_modal.py      # Upload files to Modal
├── download_results.py     # Download results from Modal
├── MODAL_TRAINING.md       # This guide
├── chemCPA/               # Source code (uploaded)
├── datasets/              # Your datasets (uploaded)
├── train_chemcpa_simple.py # Training script (uploaded)
├── config/                # Config files (uploaded)
└── results/               # Downloaded results
    ├── *.ckpt            # Model checkpoints
    ├── lightning_logs/   # Training logs
    └── checkpoints/      # Additional checkpoints
```

## Cost Estimation

Modal A100 pricing (approximate):
- **A100 GPU:** ~$2.50/hour
- **Training time:** ~1 hour for 50 epochs
- **Total cost:** ~$2.50 for full training

Much cheaper and faster than running locally!

## Troubleshooting

### Upload Issues
```bash
# If upload fails, check you're in the right directory
ls train_chemcpa_simple.py  # Should exist

# Re-run upload
python upload_to_modal.py
```

### Training Issues
```bash
# Check Modal logs
modal logs chemcpa-training

# Re-run training with different parameters
modal run modal_train.py::train_chemcpa --epochs 10 --batch_size 64
```

### Download Issues
```bash
# Check if results exist on Modal
modal volume ls chemcpa-data

# Re-run download
python download_results.py
```

## Advanced Usage

### Monitor Training
```bash
# View real-time logs
modal logs chemcpa-training --follow
```

### Custom Training Parameters
```bash
# Smaller batch size for memory constraints
modal run modal_train.py::train_chemcpa --batch_size 64

# Shorter training for testing
modal run modal_train.py::train_chemcpa --epochs 5

# Different dataset
modal run modal_train.py::train_chemcpa --dataset sciplex
```

### Clean Up
```bash
# Delete Modal volume (saves storage costs)
modal volume delete chemcpa-data
```

## Benefits of Modal Training

✅ **Speed:** A100 GPU is ~10x faster than Mac  
✅ **Cost:** Pay only for compute time used  
✅ **Memory:** 32GB RAM + GPU memory  
✅ **Reliability:** No local crashes or thermal throttling  
✅ **Scalability:** Easy to run multiple experiments  
✅ **Reproducibility:** Consistent environment  

Happy training! 🚀
