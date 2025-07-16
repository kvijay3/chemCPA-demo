# 🧬 ChemCPA Training Guide: Drug Perturbation Prediction for Stem Cells

This guide provides complete instructions for training models that predict cellular responses to drug perturbations using single-cell RNA sequencing data.

## 🎯 **Model Overview**

**Input**: scRNA-seq data + unseen drug (SMILES or drug name)  
**Output**: Predicted perturbed scRNA-seq expression profile

Perfect for stem cell applications and drug discovery!

## 🚀 **Quick Start**

### 1. Setup Environment

```bash
# Install dependencies
pip install -r requirements_simple.txt

# Setup the package
pip install -e .
```

### 2. Download Datasets

```bash
# List available datasets
python download_datasets.py --list

# Download LINCS (recommended for drug perturbations)
python download_datasets.py --dataset lincs

# Or download all datasets
python download_datasets.py --dataset all
```

### 3. Train Your First Model

```bash
# Train on LINCS dataset (best for drug perturbations)
python train_chemcpa_simple.py --dataset lincs --epochs 50 --batch_size 128

# Train on Sciplex (has SMILES data)
python train_chemcpa_simple.py --dataset sciplex --epochs 100 --batch_size 64

# Train on Biolord (great for stem cells)
python train_chemcpa_simple.py --dataset biolord --epochs 200 --batch_size 32
```

## 📊 **Available Datasets**

| Dataset | Best For | Has SMILES | Cell Types | Size |
|---------|----------|------------|------------|------|
| **LINCS** | Drug screening, large-scale studies | ❌ | 9 cell lines | ~2GB |
| **Sciplex** | Initial testing, SMILES-based modeling | ✅ | 3 cell lines | ~500MB |
| **Biolord** | Stem cell applications | ✅ | Various stem cells | ~1GB |

### Dataset Details

#### 🔬 **LINCS L1000** (Recommended)
- **Best for**: Drug perturbation prediction, large-scale screening
- **Advantages**: Largest dataset, diverse drugs, well-validated
- **Cell types**: A375, A549, HA1E, HCC515, HEPG2, HT29, MCF7, PC3, YAPC
- **Drug representation**: Drug names/IDs (no SMILES required)

#### 🧪 **Sciplex**
- **Best for**: Initial testing, SMILES-based drug modeling
- **Advantages**: Has SMILES data, well-processed
- **Cell types**: A549, K562, MCF7
- **Drug representation**: SMILES strings

#### 🌱 **Biolord**
- **Best for**: Stem cell applications, high-quality data
- **Advantages**: High-quality biological data, stem cell focus
- **Cell types**: Various stem cell types
- **Drug representation**: SMILES strings

## 🎛️ **Training Configuration**

### Basic Training Command

```bash
python train_chemcpa_simple.py \
    --dataset lincs \
    --epochs 100 \
    --batch_size 128 \
    --learning_rate 1e-3 \
    --output_dir ./models/lincs_model
```

### Advanced Configuration

```bash
python train_chemcpa_simple.py \
    --dataset biolord \
    --epochs 200 \
    --batch_size 64 \
    --learning_rate 5e-4 \
    --autoencoder_width 512 \
    --autoencoder_depth 4 \
    --adversary_width 128 \
    --reg_adversary 5.0 \
    --output_dir ./models/stem_cell_model \
    --use_wandb \
    --project_name "stem_cell_perturbations"
```

### Key Parameters

| Parameter | Description | Default | Recommended Range |
|-----------|-------------|---------|-------------------|
| `--epochs` | Training epochs | 100 | 50-200 |
| `--batch_size` | Batch size | 128 | 32-256 |
| `--learning_rate` | Learning rate | 1e-3 | 1e-4 to 1e-2 |
| `--autoencoder_width` | AE hidden size | 512 | 256-1024 |
| `--autoencoder_depth` | AE layers | 4 | 3-6 |
| `--reg_adversary` | Adversarial regularization | 5.0 | 1.0-10.0 |

## 🔮 **Making Predictions**

### Predict Single Drug Response

```bash
python predict_new_drug.py \
    --model_path ./models/lincs_model/best_model.ckpt \
    --drug_name "Imatinib" \
    --cell_type "A549" \
    --dose 1.0 \
    --output_file predictions.h5ad
```

### Predict with SMILES

```bash
python predict_new_drug.py \
    --model_path ./models/sciplex_model/best_model.ckpt \
    --smiles "CC1=CC=C(C=C1)C2=CC(=NN2C3=CC=C(C=C3)S(=O)(=O)N)C(F)(F)F" \
    --cell_type "K562" \
    --dose 10.0 \
    --output_file drug_response.h5ad
```

### Batch Predictions

```bash
python predict_new_drug.py \
    --model_path ./models/biolord_model/best_model.ckpt \
    --drug_list drugs_to_test.csv \
    --cell_types "stem_cell_type1,stem_cell_type2" \
    --doses "0.1,1.0,10.0" \
    --output_dir ./predictions/
```

## 📈 **Model Evaluation**

### Training Metrics

The model tracks several metrics during training:

- **Reconstruction Loss**: How well the model reconstructs gene expression
- **Adversarial Loss**: How well the model handles confounders
- **Drug Loss**: How well the model learns drug representations
- **R² Score**: Correlation between predicted and actual expression

### Validation

```bash
# Evaluate trained model
python evaluate_model.py \
    --model_path ./models/lincs_model/best_model.ckpt \
    --test_data project_folder/datasets/lincs.h5ad \
    --output_dir ./evaluation/
```

## 🛠️ **Troubleshooting**

### Common Issues

#### 1. SMILES Data Loading Error
```
KeyError: 'SMILES' not found in obs
```
**Solution**: Use a dataset without SMILES (like LINCS) or ensure your dataset has SMILES data.

#### 2. Memory Issues
```
CUDA out of memory
```
**Solutions**:
- Reduce batch size: `--batch_size 32`
- Reduce model size: `--autoencoder_width 256`
- Use CPU: `--device cpu`

#### 3. Dataset Not Found
```
FileNotFoundError: Dataset not found
```
**Solution**: Download the dataset first:
```bash
python download_datasets.py --dataset lincs
```

#### 4. Poor Model Performance
**Solutions**:
- Increase training epochs: `--epochs 200`
- Adjust learning rate: `--learning_rate 5e-4`
- Increase model capacity: `--autoencoder_width 1024`
- Use a larger dataset (LINCS)

### Performance Tips

1. **For stem cells**: Use Biolord dataset with longer training (200+ epochs)
2. **For drug screening**: Use LINCS dataset with larger batch sizes
3. **For SMILES modeling**: Use Sciplex dataset with chemical embeddings
4. **For production**: Train on multiple datasets and ensemble models

## 🧬 **Stem Cell Specific Applications**

### Recommended Workflow for Stem Cells

1. **Dataset**: Use Biolord for stem cell-specific data
2. **Training**: Longer training with smaller batches
3. **Validation**: Focus on stem cell-relevant metrics
4. **Prediction**: Test on stem cell differentiation drugs

```bash
# Stem cell optimized training
python train_chemcpa_simple.py \
    --dataset biolord \
    --epochs 300 \
    --batch_size 32 \
    --learning_rate 3e-4 \
    --autoencoder_width 1024 \
    --autoencoder_depth 6 \
    --output_dir ./models/stem_cell_specialist
```

### Stem Cell Drug Testing

```bash
# Test differentiation drugs
python predict_new_drug.py \
    --model_path ./models/stem_cell_specialist/best_model.ckpt \
    --drug_name "Retinoic acid" \
    --cell_type "embryonic_stem_cell" \
    --dose 1.0 \
    --output_file stem_cell_differentiation.h5ad
```

## 📚 **Advanced Usage**

### Custom Dataset Integration

To use your own dataset:

1. **Format**: Convert to AnnData (.h5ad) format
2. **Required columns in obs**:
   - Drug/perturbation identifier
   - Dose information
   - Cell type
   - Train/test split
3. **Update config**: Add your dataset to `train_chemcpa_simple.py`

### Model Architecture Customization

Modify the model architecture by editing the config:

```python
'model': {
    'hparams': {
        'autoencoder_width': 1024,  # Increase for more capacity
        'autoencoder_depth': 6,     # Deeper networks
        'adversary_width': 256,     # Stronger adversary
        'dosers_width': 128,        # Drug embedding size
        # ... other parameters
    }
}
```

### Multi-GPU Training

```bash
# Use multiple GPUs
python train_chemcpa_simple.py \
    --dataset lincs \
    --epochs 100 \
    --batch_size 256 \
    --devices 4 \
    --strategy ddp
```

## 🎯 **Best Practices**

1. **Start small**: Begin with Sciplex for testing, then scale to LINCS
2. **Monitor training**: Use Weights & Biases for experiment tracking
3. **Validate thoroughly**: Test on held-out drugs and cell types
4. **Ensemble models**: Combine predictions from multiple models
5. **Domain adaptation**: Fine-tune on your specific cell types

## 📞 **Support**

For issues and questions:
- Check the troubleshooting section above
- Review the dataset documentation
- Ensure all dependencies are installed correctly

Happy training! 🚀

