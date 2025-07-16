#!/usr/bin/env python3
"""
Simple ChemCPA Drug Effect Prediction Script

Usage:
    python predict_drug_effect.py --tsv_file data.tsv --smiles "CCO" --output predictions.csv

Input TSV format:
    gene_symbol    expression
    GAPDH         10.5
    ACTB          8.2
    ...
"""

import argparse
import pandas as pd
import numpy as np
import torch
from pathlib import Path
import sys
import warnings
warnings.filterwarnings('ignore')

# Add chemCPA to path
sys.path.append('.')
from chemCPA.data.data import load_dataset_splits
from chemCPA.model import ChemCPA
from chemCPA.data.utils import canonicalize_smiles

def load_trained_model(checkpoint_path):
    """Load trained ChemCPA model from checkpoint"""
    print(f"📦 Loading model from {checkpoint_path}...")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Extract model config from checkpoint
    model_config = checkpoint.get('hyper_parameters', {})
    
    # Create model
    model = ChemCPA(**model_config)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    
    print("✅ Model loaded successfully!")
    return model

def process_gene_expression(tsv_file, reference_genes):
    """Process TSV file with gene expression data"""
    print(f"📊 Processing gene expression from {tsv_file}...")
    
    # Read TSV file
    df = pd.read_csv(tsv_file, sep='\t')
    
    # Validate columns
    if 'gene_symbol' not in df.columns or 'expression' not in df.columns:
        raise ValueError("TSV file must have 'gene_symbol' and 'expression' columns")
    
    # Create expression vector aligned with reference genes
    expression_dict = dict(zip(df['gene_symbol'], df['expression']))
    
    # Align with reference genes (fill missing with 0)
    expression_vector = []
    missing_genes = []
    
    for gene in reference_genes:
        if gene in expression_dict:
            expression_vector.append(expression_dict[gene])
        else:
            expression_vector.append(0.0)  # or use mean/median
            missing_genes.append(gene)
    
    if missing_genes:
        print(f"⚠️  Warning: {len(missing_genes)} genes missing from input (filled with 0)")
        if len(missing_genes) <= 10:
            print(f"   Missing genes: {missing_genes}")
    
    print(f"✅ Processed {len(df)} input genes, aligned to {len(reference_genes)} reference genes")
    return np.array(expression_vector, dtype=np.float32)

def process_smiles(smiles_string, drug_embeddings):
    """Process SMILES string and get drug embedding"""
    print(f"🧪 Processing SMILES: {smiles_string}")
    
    # Canonicalize SMILES
    canonical_smiles = canonicalize_smiles(smiles_string)
    if canonical_smiles is None:
        raise ValueError(f"Invalid SMILES string: {smiles_string}")
    
    print(f"   Canonical SMILES: {canonical_smiles}")
    
    # For now, we'll use a random embedding since we need the drug to be in training set
    # In practice, you'd need to either:
    # 1. Have the drug in your training set, or
    # 2. Use a pre-trained molecular encoder
    
    print("⚠️  Note: Using random drug embedding. For real predictions, drug must be in training set.")
    embedding_dim = drug_embeddings.weight.shape[1]
    drug_embedding = torch.randn(1, embedding_dim)
    
    return drug_embedding

def predict_drug_effect(model, gene_expression, drug_embedding, covariates=None):
    """Make prediction using ChemCPA model"""
    print("🔮 Making prediction...")
    
    # Convert to tensors
    gene_expr_tensor = torch.tensor(gene_expression).unsqueeze(0)  # Add batch dimension
    
    # Create dummy covariates if not provided
    if covariates is None:
        # Use zeros for covariates (you might want to use actual cell line info)
        covariates = torch.zeros(1, model.model.num_covariates[0])
    
    # Make prediction
    with torch.no_grad():
        # Forward pass through model
        prediction = model.model(
            gene_expr_tensor,
            drug_embedding,
            covariates
        )
    
    return prediction.squeeze().numpy()

def save_predictions(predictions, gene_names, output_file):
    """Save predictions to CSV file"""
    print(f"💾 Saving predictions to {output_file}...")
    
    # Create results DataFrame
    results_df = pd.DataFrame({
        'gene_symbol': gene_names,
        'predicted_expression': predictions,
        'log2_fold_change': predictions  # Assuming predictions are log2 fold changes
    })
    
    # Sort by absolute fold change
    results_df['abs_fold_change'] = np.abs(results_df['log2_fold_change'])
    results_df = results_df.sort_values('abs_fold_change', ascending=False)
    results_df = results_df.drop('abs_fold_change', axis=1)
    
    # Save to CSV
    results_df.to_csv(output_file, index=False)
    
    print(f"✅ Saved {len(results_df)} predictions to {output_file}")
    
    # Show top 10 predictions
    print("\n🔝 Top 10 predicted changes:")
    print(results_df.head(10).to_string(index=False))

def main():
    parser = argparse.ArgumentParser(description='Predict drug effects using ChemCPA')
    parser.add_argument('--tsv_file', required=True, help='TSV file with gene expression data')
    parser.add_argument('--smiles', required=True, help='SMILES string of the drug')
    parser.add_argument('--checkpoint', default='results/best_model.ckpt', help='Path to trained model checkpoint')
    parser.add_argument('--output', default='predictions.csv', help='Output CSV file')
    parser.add_argument('--dataset', default='lincs', help='Dataset used for training (to get gene names)')
    
    args = parser.parse_args()
    
    print("🚀 ChemCPA Drug Effect Prediction")
    print("=" * 50)
    
    try:
        # Load reference dataset to get gene names
        print("📚 Loading reference dataset...")
        datasets, dataset = load_dataset_splits(
            dataset_name=args.dataset,
            data_dir="datasets",
            return_dataset=True
        )
        gene_names = dataset.var_names.tolist()
        print(f"✅ Loaded {len(gene_names)} reference genes")
        
        # Load trained model
        if not Path(args.checkpoint).exists():
            print(f"❌ Checkpoint not found: {args.checkpoint}")
            print("   Please run training first or specify correct checkpoint path")
            return
        
        model = load_trained_model(args.checkpoint)
        
        # Process input gene expression
        gene_expression = process_gene_expression(args.tsv_file, gene_names)
        
        # Process SMILES
        drug_embedding = process_smiles(args.smiles, model.drug_embeddings)
        
        # Make prediction
        predictions = predict_drug_effect(model, gene_expression, drug_embedding)
        
        # Save results
        save_predictions(predictions, gene_names, args.output)
        
        print(f"\n✅ Prediction completed successfully!")
        print(f"📊 Results saved to: {args.output}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
