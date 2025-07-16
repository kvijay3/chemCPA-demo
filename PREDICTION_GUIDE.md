# ChemCPA Drug Effect Prediction Guide

This guide shows how to predict drug effects on gene expression using your trained ChemCPA model.

## Quick Start

### 1. Train Model (if not done already)
```bash
modal run modal_train.py::train_chemcpa --dataset lincs --epochs 50 --batch_size 256 --learning_rate 1e-3
```

### 2. Download Trained Model
```bash
python download_results.py
```

### 3. Make Predictions
```bash
python predict_drug_effect.py --tsv_file sample_gene_expression.tsv --smiles "CCO" --output predictions.csv
```

## Input Format

### TSV File Format
Your gene expression file should be tab-separated with these columns:

```
gene_symbol    expression
GAPDH         12.5
ACTB          11.2
TP53          8.9
MYC           7.3
...
```

**Requirements:**
- ✅ Tab-separated values (TSV)
- ✅ Header row with `gene_symbol` and `expression`
- ✅ Gene symbols should match standard HGNC symbols
- ✅ Expression values as numbers (log2 counts, TPM, etc.)

### SMILES String
Provide a valid SMILES string for your drug of interest:

**Examples:**
- Ethanol: `CCO`
- Aspirin: `CC(=O)OC1=CC=CC=C1C(=O)O`
- Caffeine: `CN1C=NC2=C1C(=O)N(C(=O)N2C)C`

## Usage Examples

### Basic Prediction
```bash
python predict_drug_effect.py \
    --tsv_file my_expression_data.tsv \
    --smiles "CCO" \
    --output ethanol_predictions.csv
```

### Custom Model Checkpoint
```bash
python predict_drug_effect.py \
    --tsv_file my_expression_data.tsv \
    --smiles "CC(=O)OC1=CC=CC=C1C(=O)O" \
    --checkpoint results/epoch_30.ckpt \
    --output aspirin_predictions.csv
```

### Different Dataset
```bash
python predict_drug_effect.py \
    --tsv_file my_expression_data.tsv \
    --smiles "CN1C=NC2=C1C(=O)N(C(=O)N2C)C" \
    --dataset sciplex \
    --output caffeine_predictions.csv
```

## Output Format

The script generates a CSV file with predicted gene expression changes:

```csv
gene_symbol,predicted_expression,log2_fold_change
MYC,2.34,2.34
TP53,-1.87,-1.87
EGFR,1.23,1.23
...
```

**Columns:**
- `gene_symbol`: Gene name
- `predicted_expression`: Predicted expression change
- `log2_fold_change`: Log2 fold change (same as predicted_expression)

Results are sorted by absolute fold change (most affected genes first).

## Sample Files

### Create Sample Gene Expression File
```bash
cat > my_genes.tsv << 'EOF'
gene_symbol	expression
GAPDH	12.5
ACTB	11.2
TP53	8.9
MYC	7.3
EGFR	6.8
BRCA1	5.4
KRAS	9.1
PIK3CA	7.8
AKT1	8.2
MTOR	6.9
EOF
```

### Test with Sample Data
```bash
python predict_drug_effect.py \
    --tsv_file sample_gene_expression.tsv \
    --smiles "CCO" \
    --output test_predictions.csv
```

## Advanced Usage

### Batch Predictions
Create a script to predict multiple drugs:

```bash
#!/bin/bash
# predict_multiple.sh

DRUGS=(
    "CCO"                                    # Ethanol
    "CC(=O)OC1=CC=CC=C1C(=O)O"             # Aspirin  
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"        # Caffeine
)

for i in "${!DRUGS[@]}"; do
    echo "Predicting drug $((i+1)): ${DRUGS[i]}"
    python predict_drug_effect.py \
        --tsv_file sample_gene_expression.tsv \
        --smiles "${DRUGS[i]}" \
        --output "drug_${i}_predictions.csv"
done
```

### Custom Analysis
```python
import pandas as pd
import matplotlib.pyplot as plt

# Load predictions
df = pd.read_csv('predictions.csv')

# Find top upregulated genes
upregulated = df[df['log2_fold_change'] > 1].head(10)
print("Top upregulated genes:")
print(upregulated)

# Find top downregulated genes  
downregulated = df[df['log2_fold_change'] < -1].head(10)
print("Top downregulated genes:")
print(downregulated)

# Plot distribution
plt.figure(figsize=(10, 6))
plt.hist(df['log2_fold_change'], bins=50, alpha=0.7)
plt.xlabel('Log2 Fold Change')
plt.ylabel('Number of Genes')
plt.title('Distribution of Predicted Gene Expression Changes')
plt.savefig('prediction_distribution.png')
```

## Troubleshooting

### Model Not Found
```bash
# Download results from Modal
python download_results.py

# Or specify custom checkpoint path
python predict_drug_effect.py --checkpoint path/to/model.ckpt ...
```

### Invalid SMILES
```bash
# Validate SMILES online: https://www.daylight.com/daycgi/depict
# Or use RDKit to check:
python -c "from rdkit import Chem; print(Chem.MolFromSmiles('YOUR_SMILES'))"
```

### Gene Symbol Issues
```bash
# Check gene symbols against HGNC database
# Common issues:
# - Use HGNC symbols (e.g., 'TP53' not 'p53')
# - Avoid Ensembl IDs (use gene symbols)
# - Check for typos in gene names
```

### Memory Issues
```bash
# For large gene expression files, process in chunks
# Or use a machine with more RAM
```

## Important Notes

⚠️ **Drug Limitations**: Currently, the script uses random embeddings for new drugs. For real predictions, the drug should be in your training dataset, or you need a pre-trained molecular encoder.

⚠️ **Gene Coverage**: Missing genes are filled with 0. For best results, include as many genes as possible from the training dataset.

⚠️ **Cell Context**: The model uses dummy covariates. For cell-type-specific predictions, you'd need to provide appropriate covariate information.

## Next Steps

1. **Improve Drug Encoding**: Integrate a pre-trained molecular encoder (e.g., ChemBERTa, MolT5)
2. **Add Cell Context**: Include cell line or tissue type information
3. **Validation**: Compare predictions with experimental data
4. **Batch Processing**: Create pipeline for multiple drugs/conditions

Happy predicting! 🔮
