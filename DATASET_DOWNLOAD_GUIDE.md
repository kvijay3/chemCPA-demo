# ChemCPA Dataset Download Guide

This guide shows you exactly how to download the datasets needed for training ChemCPA models on stem cell data.

## Quick Start

### 🚀 One-Command Setup
```bash
# Complete setup with essential datasets
python setup_chemcpa.py

# Quick setup (minimal datasets for testing)
python setup_chemcpa.py --quick

# Only download datasets (if environment already set up)
python setup_chemcpa.py --datasets-only
```

### 📊 Available Datasets

| Dataset | Size | Cells | Drugs | Description | Use Case |
|---------|------|-------|-------|-------------|----------|
| **Sciplex** | ~2GB | ~100K | ~200 | Good for initial testing | Training/Testing |
| **Broad (BioLord)** | ~5GB | ~1M+ | ~1000+ | Large-scale comprehensive | Large-scale training |
| **LINCS** | ~8GB | ~2M+ | ~2000+ | Comprehensive drug screening | Full-scale screening |
| **DrugBank** | ~50MB | N/A | ~13K | Drug information/metadata | Analysis/Annotation |

## Detailed Download Commands

### 📋 List Available Datasets
```bash
# See all available datasets
python download_datasets.py --list

# Get detailed information about datasets
python download_datasets.py --info

# Check what's already downloaded
python download_datasets.py --status
```

### 📥 Download Specific Datasets

#### Essential Datasets for Stem Cells
```bash
# Download the most important datasets for stem cell work
python download_datasets.py --stem-cell-essentials
```
This downloads:
- Sciplex dataset (for testing)
- Broad dataset (for comprehensive training)
- DrugBank (for drug information)

#### Individual Datasets
```bash
# Sciplex dataset (good for testing)
python download_datasets.py --dataset sciplex

# Broad dataset (best for comprehensive training)
python download_datasets.py --dataset broad

# LINCS dataset (largest, most comprehensive)
python download_datasets.py --dataset lincs

# DrugBank information
python download_datasets.py --dataset drugbank

# Chemical embeddings
python download_datasets.py --dataset rdkit
```

#### Dataset Groups
```bash
# All main training datasets
python download_datasets.py --training-datasets

# Pre-computed chemical embeddings
python download_datasets.py --embeddings

# Everything (warning: very large!)
python download_datasets.py --dataset all
```

## Dataset Details

### 🧪 Sciplex Dataset
- **File**: `project_folder/datasets/sciplex_raw_chunk_*.h5ad` (5 chunks)
- **Source**: Part of CPA binaries
- **Best for**: Initial testing, learning the system
- **Download**: `python download_datasets.py --dataset sciplex`

### 🏢 Broad Dataset (BioLord)
- **File**: `project_folder/datasets/adata_biolord_split_30.h5ad`
- **Source**: Broad Institute
- **Best for**: Large-scale training, comprehensive results
- **Download**: `python download_datasets.py --dataset broad`

### 🔬 LINCS Dataset
- **File**: `project_folder/datasets/lincs_full.h5ad`
- **Source**: NIH LINCS program
- **Best for**: Comprehensive drug screening, maximum coverage
- **Download**: `python download_datasets.py --dataset lincs`

### 💊 DrugBank Dataset
- **File**: `project_folder/datasets/drug_bank/drugbank_all.csv`
- **Source**: DrugBank database
- **Best for**: Drug annotations, metadata analysis
- **Download**: `python download_datasets.py --dataset drugbank`

## File Structure After Download

```
project_folder/
├── datasets/
│   ├── sciplex_raw_chunk_0.h5ad          # Sciplex data (chunk 0)
│   ├── sciplex_raw_chunk_1.h5ad          # Sciplex data (chunk 1)
│   ├── sciplex_raw_chunk_2.h5ad          # Sciplex data (chunk 2)
│   ├── sciplex_raw_chunk_3.h5ad          # Sciplex data (chunk 3)
│   ├── sciplex_raw_chunk_4.h5ad          # Sciplex data (chunk 4)
│   ├── norman.h5ad                       # Norman dataset
│   ├── adata_biolord_split_30.h5ad       # Broad dataset
│   ├── lincs_full.h5ad                   # LINCS dataset
│   └── drug_bank/
│       └── drugbank_all.csv              # DrugBank info
├── embeddings/
│   └── rdkit/
│       └── data/
│           └── embeddings/
│               └── rdkit2D_embedding_biolord.parquet
└── binaries/
    └── cpa_binaries.tar                  # Original CPA binaries
```

## Usage Examples

### Training with Different Datasets

```bash
# Train on Sciplex (quick testing)
python train_chemcpa_simple.py --dataset sciplex --epochs 50

# Train on Broad dataset (comprehensive)
python train_chemcpa_simple.py --dataset broad --epochs 150 --batch_size 64

# Train on LINCS (full scale)
python train_chemcpa_simple.py --dataset lincs --epochs 200 --batch_size 32
```

### Checking Dataset Status

```bash
# See what's downloaded
python download_datasets.py --status

# Output example:
# ✅ cpa_binaries                    |    1.8 GB | project_folder/binaries/cpa_binaries.tar
# ✅ adata_biolord_split_30          |    4.2 GB | project_folder/datasets/adata_biolord_split_30.h5ad
# ❌ lincs_full                      | Not found | project_folder/datasets/lincs_full.h5ad
```

## Troubleshooting

### Common Issues

1. **Download fails with "Permission denied"**
   ```bash
   # Make sure you have write permissions
   chmod +w project_folder/
   ```

2. **Google Drive download quota exceeded**
   ```bash
   # Try again later, or download manually from the links in the error message
   ```

3. **Not enough disk space**
   ```bash
   # Check available space
   df -h .
   
   # Download datasets one by one instead of all at once
   python download_datasets.py --dataset sciplex  # Start with smallest
   ```

4. **Network timeout**
   ```bash
   # Retry the download - it will resume where it left off
   python download_datasets.py --dataset <dataset_name>
   ```

### Manual Download Links

If automatic download fails, you can download manually:

- **CPA Binaries**: https://dl.fbaipublicfiles.com/dlp/cpa_binaries.tar
- **Broad Dataset**: https://drive.google.com/uc?export=download&id=18QkyADzuM8b7lMxRg94jufHaKRPkzEFw
- **LINCS Dataset**: https://f003.backblazeb2.com/file/chemCPA-datasets/lincs_full.h5ad.gz
- **DrugBank**: https://drive.google.com/uc?export=download&id=18MYC6ykf2CxxFIRrGYigPNfjvF8Mu6jL

Place downloaded files in the appropriate `project_folder/` subdirectories.

## Recommendations

### For Different Use Cases

**🧪 Just Testing/Learning:**
```bash
python download_datasets.py --dataset sciplex
```

**🧬 Stem Cell Research:**
```bash
python download_datasets.py --stem-cell-essentials
```

**🔬 Comprehensive Drug Screening:**
```bash
python download_datasets.py --training-datasets
```

**💻 Full Research Setup:**
```bash
python download_datasets.py --dataset all
```

### Storage Requirements

- **Minimal setup**: ~2GB (Sciplex only)
- **Stem cell essentials**: ~7GB (Sciplex + Broad + DrugBank)
- **Full training datasets**: ~15GB (All main datasets)
- **Complete setup**: ~20GB (Everything including embeddings)

Make sure you have sufficient disk space before downloading!

## Next Steps

After downloading datasets:

1. **Verify downloads**: `python download_datasets.py --status`
2. **Train a model**: `python train_chemcpa_simple.py --dataset sciplex`
3. **Read the guide**: `STEM_CELL_TRAINING_README.md`
4. **Try the example**: `python example_workflow.py`

