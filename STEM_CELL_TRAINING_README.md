# ChemCPA for Stem Cell Drug Perturbation Prediction

This guide provides simplified scripts for training and using ChemCPA models to predict how unseen drugs will affect stem cell gene expression.

## Overview

**What this does:**
- **Input**: scRNA-seq data + drug SMILES string
- **Output**: Predicted perturbed scRNA-seq data
- **Use case**: Predict how new drugs will affect stem cell gene expression without doing experiments

## Quick Start

### 1. Setup Environment

```bash
# Create conda environment
conda env create -f environment.yml
conda activate chemCPA

# Install package
python setup.py install -e .
```

### 2. Prepare Data

First, you need to prepare your datasets. The preprocessing scripts are in the `preprocessing/` folder:

```bash
# Run all preprocessing steps
python preprocessing/run_notebooks.py

# Or run individual preprocessing notebooks:
# - preprocessing/sciplex.ipynb (for Sciplex dataset)
# - preprocessing/biolord.ipynb (for broad dataset)
```

### 3. Train Model

#### Option A: Quick Training (Sciplex dataset)
```bash
python train_chemcpa_simple.py \
    --dataset sciplex \
    --epochs 100 \
    --batch_size 32 \
    --output_dir ./outputs/sciplex_training \
    --use_wandb
```

#### Option B: Large-scale Training (Broad dataset)
```bash
python train_chemcpa_simple.py \
    --dataset broad \
    --epochs 200 \
    --batch_size 64 \
    --learning_rate 1e-3 \
    --output_dir ./outputs/broad_training \
    --use_wandb \
    --early_stopping
```

#### Option C: Using Configuration File
```bash
# Edit config_stem_cells.yaml for your needs, then:
python chemCPA/train_hydra.py --config-path . --config-name config_stem_cells
```

### 4. Make Predictions

Once you have a trained model, predict perturbation effects:

```bash
python predict_perturbation.py \
    --model_path ./outputs/sciplex_training/checkpoints/best_model.ckpt \
    --input_data your_stem_cell_data.h5ad \
    --drug_smiles "CCO" \
    --dose 1.0 \
    --output predictions.csv \
    --plot_dir ./plots
```

## Available Datasets

### 1. Sciplex Dataset
- **Best for**: Initial testing and development
- **Size**: Medium (~100K cells)
- **Cell types**: Various cancer cell lines
- **Drugs**: ~200 compounds
- **Use**: `--dataset sciplex`

### 2. Broad Dataset (BioLord)
- **Best for**: Large-scale training
- **Size**: Large (~1M+ cells)
- **Cell types**: Diverse primary cells
- **Drugs**: ~1000+ compounds
- **Use**: `--dataset broad`

### 3. LINCS Dataset
- **Best for**: Comprehensive drug screening
- **Size**: Very large
- **Cell types**: Multiple cell lines
- **Drugs**: ~2000+ compounds
- **Use**: `--dataset lincs`

## Model Architecture

The ChemCPA model consists of:

1. **Autoencoder**: Learns gene expression representations
2. **Drug Embeddings**: Chemical structure representations (SMILES → vectors)
3. **Dosage Response**: Models dose-dependent effects
4. **Adversarial Training**: Ensures robust predictions across cell types

## Key Features for Stem Cells

### Stem Cell Specific Configuration
The `config_stem_cells.yaml` includes:
- Optimized hyperparameters for stem cell data
- Expected stem cell markers and pathways
- Quality control metrics for stem cell datasets

### Supported Cell Types
- Embryonic stem cells (ESCs)
- Induced pluripotent stem cells (iPSCs)
- Neural stem cells (NSCs)
- Mesenchymal stem cells (MSCs)
- Hematopoietic stem cells (HSCs)

## Training Tips

### For Stem Cell Data:
1. **Batch Size**: Start with 32-64, increase if you have enough GPU memory
2. **Learning Rate**: 1e-3 works well, reduce to 1e-4 for fine-tuning
3. **Epochs**: 100-200 epochs usually sufficient
4. **Early Stopping**: Recommended to prevent overfitting

### Hardware Requirements:
- **Minimum**: 8GB GPU memory, 16GB RAM
- **Recommended**: 16GB+ GPU memory, 32GB+ RAM
- **Training Time**: 2-12 hours depending on dataset size

## Output Files

After training, you'll get:
- `checkpoints/`: Model checkpoints
- `logs/`: Training logs and metrics
- `predictions.csv`: Gene perturbation predictions
- `plots/`: Visualization of results

## Example Workflow

```python
# 1. Train model
python train_chemcpa_simple.py --dataset sciplex --epochs 100

# 2. Load your stem cell data
import scanpy as sc
adata = sc.read_h5ad("my_stem_cells.h5ad")

# 3. Predict drug effects
python predict_perturbation.py \
    --model_path ./outputs/checkpoints/best_model.ckpt \
    --input_data my_stem_cells.h5ad \
    --drug_smiles "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O" \  # Ibuprofen
    --output ibuprofen_effects.csv
```

## Troubleshooting

### Common Issues:

1. **CUDA out of memory**: Reduce batch size
2. **Dataset not found**: Run preprocessing scripts first
3. **Model loading error**: Check checkpoint path and compatibility
4. **Poor predictions**: Try longer training or different hyperparameters

### Data Format Requirements:
- **Input**: AnnData (.h5ad) format
- **Required columns**: 
  - `obs['condition']`: Drug/perturbation names
  - `obs['SMILES']`: Drug SMILES strings
  - `obs['cell_type']`: Cell type information
  - `obs['dose']`: Drug dosages

## Advanced Usage

### Custom Drug Embeddings:
```python
# Use your own drug embeddings
from chemCPA.embedding import get_chemical_representation

embeddings = get_chemical_representation(
    smiles=your_smiles_list,
    embedding_model={'model': 'your_method'},
    data_path='path/to/embeddings'
)
```

### Transfer Learning:
```python
# Use pretrained weights
python train_chemcpa_simple.py \
    --dataset your_data \
    --pretrained_path ./pretrained_models/chemcpa_base.ckpt \
    --epochs 50  # Fewer epochs for fine-tuning
```

## Citation

If you use this code, please cite the original ChemCPA paper:

```bibtex
@inproceedings{hetzel2022predicting,
  title={Predicting Cellular Responses to Novel Drug Perturbations at a Single-Cell Resolution},
  author={Hetzel, Leon and Böhm, Simon and Kilbertus, Niki and Günnemann, Stephan and Lotfollahi, Mohammad and Theis, Fabian J},
  booktitle={NeurIPS 2022},
  year={2022}
}
```

## Support

For questions or issues:
1. Check the troubleshooting section above
2. Look at the original repository documentation
3. Open an issue with your specific problem and dataset details

