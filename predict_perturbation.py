#!/usr/bin/env python3
"""
Drug Perturbation Prediction Script

This script demonstrates how to use a trained ChemCPA model to predict
the effects of unseen drugs on stem cell gene expression.

Usage:
    python predict_perturbation.py --model_path ./outputs/checkpoints/best_model.ckpt \
                                   --input_data stem_cell_data.h5ad \
                                   --drug_smiles "CCO" \
                                   --output predictions.csv
"""

import argparse
import sys
from pathlib import Path
import torch
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as adata
from typing import List, Optional, Union
import matplotlib.pyplot as plt
import seaborn as sns

# Add the chemCPA module to path
sys.path.append(str(Path(__file__).parent))

from chemCPA.lightning_module import ChemCPA
from chemCPA.embedding import get_chemical_representation
from chemCPA.helper import canonicalize_smiles


class DrugPerturbationPredictor:
    """Class for predicting drug perturbation effects using trained ChemCPA model"""
    
    def __init__(self, model_path: str):
        """
        Initialize predictor with trained model
        
        Args:
            model_path: Path to trained ChemCPA checkpoint
        """
        self.model_path = model_path
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.load_model()
    
    def load_model(self):
        """Load trained ChemCPA model from checkpoint"""
        print(f"Loading model from {self.model_path}")
        
        try:
            # Load the checkpoint
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            # Extract model configuration from checkpoint
            if 'hyper_parameters' in checkpoint:
                config = checkpoint['hyper_parameters']['config']
                dataset_config = checkpoint['hyper_parameters']['dataset_config']
            else:
                raise ValueError("Checkpoint missing hyperparameters")
            
            # Initialize model
            self.model = ChemCPA(config, dataset_config)
            self.model.load_state_dict(checkpoint['state_dict'])
            self.model.to(self.device)
            self.model.eval()
            
            print("Model loaded successfully!")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def preprocess_expression_data(self, adata_obj: adata.AnnData) -> torch.Tensor:
        """
        Preprocess scRNA-seq expression data for model input
        
        Args:
            adata_obj: AnnData object containing expression data
            
        Returns:
            Preprocessed expression tensor
        """
        # Basic preprocessing
        sc.pp.filter_cells(adata_obj, min_genes=200)
        sc.pp.filter_genes(adata_obj, min_cells=3)
        
        # Normalize and log transform
        sc.pp.normalize_total(adata_obj, target_sum=1e4)
        sc.pp.log1p(adata_obj)
        
        # Convert to tensor
        if hasattr(adata_obj.X, 'toarray'):
            expression_matrix = adata_obj.X.toarray()
        else:
            expression_matrix = adata_obj.X
            
        return torch.FloatTensor(expression_matrix).to(self.device)
    
    def encode_drug(self, smiles: str) -> torch.Tensor:
        """
        Encode drug SMILES into embedding vector
        
        Args:
            smiles: SMILES string of the drug
            
        Returns:
            Drug embedding tensor
        """
        # Canonicalize SMILES
        canonical_smiles = canonicalize_smiles(smiles)
        
        # Get chemical representation (this will depend on your embedding method)
        # For now, this is a placeholder - you'll need to implement based on your embedding approach
        drug_embedding = self.model.drug_embeddings.weight[0:1]  # Placeholder
        
        return drug_embedding
    
    def predict_perturbation(self, 
                           expression_data: Union[adata.AnnData, torch.Tensor],
                           drug_smiles: str,
                           dose: float = 1.0,
                           cell_type: Optional[str] = None) -> dict:
        """
        Predict perturbation effects of a drug on gene expression
        
        Args:
            expression_data: scRNA-seq data (AnnData or tensor)
            drug_smiles: SMILES string of the drug
            dose: Drug dosage (default: 1.0)
            cell_type: Cell type information (optional)
            
        Returns:
            Dictionary containing predictions and metadata
        """
        print(f"Predicting perturbation for drug: {drug_smiles}")
        
        with torch.no_grad():
            # Preprocess expression data
            if isinstance(expression_data, adata.AnnData):
                expr_tensor = self.preprocess_expression_data(expression_data)
                gene_names = expression_data.var_names.tolist()
            else:
                expr_tensor = expression_data.to(self.device)
                gene_names = [f"Gene_{i}" for i in range(expr_tensor.shape[1])]
            
            # Encode drug
            drug_embedding = self.encode_drug(drug_smiles)
            
            # Prepare model inputs
            batch_size = expr_tensor.shape[0]
            
            # Create drug indices (placeholder - adjust based on your model)
            drug_idx = torch.zeros(batch_size, dtype=torch.long).to(self.device)
            
            # Create dose tensor
            dose_tensor = torch.full((batch_size,), dose).to(self.device)
            
            # Create covariate tensor (cell type)
            if cell_type:
                # This is a placeholder - you'll need to map cell types to indices
                covariate_idx = torch.zeros(batch_size, dtype=torch.long).to(self.device)
            else:
                covariate_idx = torch.zeros(batch_size, dtype=torch.long).to(self.device)
            
            # Make prediction using the model
            # Note: This is a simplified version - you'll need to adjust based on your model's forward method
            try:
                predictions = self.model.model(
                    genes=expr_tensor,
                    drugs_idx=drug_idx,
                    dosages=dose_tensor,
                    covariates_idx=covariate_idx
                )
                
                # Extract predicted expression
                if isinstance(predictions, tuple):
                    predicted_expression = predictions[0]  # Usually the first output
                else:
                    predicted_expression = predictions
                
                # Convert back to numpy
                predicted_expression = predicted_expression.cpu().numpy()
                
                # Calculate perturbation effect (difference from baseline)
                baseline_expression = expr_tensor.cpu().numpy()
                perturbation_effect = predicted_expression - baseline_expression
                
                results = {
                    'predicted_expression': predicted_expression,
                    'baseline_expression': baseline_expression,
                    'perturbation_effect': perturbation_effect,
                    'gene_names': gene_names,
                    'drug_smiles': drug_smiles,
                    'dose': dose,
                    'cell_type': cell_type,
                    'n_cells': batch_size,
                    'n_genes': len(gene_names)
                }
                
                print(f"Prediction completed for {batch_size} cells and {len(gene_names)} genes")
                return results
                
            except Exception as e:
                print(f"Error during prediction: {e}")
                raise
    
    def analyze_perturbation_effects(self, results: dict, top_n: int = 50) -> pd.DataFrame:
        """
        Analyze and rank perturbation effects
        
        Args:
            results: Results from predict_perturbation
            top_n: Number of top affected genes to return
            
        Returns:
            DataFrame with top affected genes
        """
        perturbation_effect = results['perturbation_effect']
        gene_names = results['gene_names']
        
        # Calculate mean perturbation effect across cells
        mean_effect = np.mean(perturbation_effect, axis=0)
        std_effect = np.std(perturbation_effect, axis=0)
        
        # Create results dataframe
        effect_df = pd.DataFrame({
            'gene': gene_names,
            'mean_perturbation': mean_effect,
            'std_perturbation': std_effect,
            'abs_mean_perturbation': np.abs(mean_effect)
        })
        
        # Sort by absolute effect size
        effect_df = effect_df.sort_values('abs_mean_perturbation', ascending=False)
        
        return effect_df.head(top_n)
    
    def visualize_results(self, results: dict, output_dir: str = './plots'):
        """
        Create visualizations of perturbation results
        
        Args:
            results: Results from predict_perturbation
            output_dir: Directory to save plots
        """
        Path(output_dir).mkdir(exist_ok=True)
        
        perturbation_effect = results['perturbation_effect']
        gene_names = results['gene_names']
        
        # 1. Histogram of perturbation effects
        plt.figure(figsize=(10, 6))
        plt.hist(perturbation_effect.flatten(), bins=50, alpha=0.7)
        plt.xlabel('Perturbation Effect (log fold change)')
        plt.ylabel('Frequency')
        plt.title(f'Distribution of Perturbation Effects\nDrug: {results["drug_smiles"]}')
        plt.savefig(f'{output_dir}/perturbation_histogram.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Top affected genes
        top_genes = self.analyze_perturbation_effects(results, top_n=20)
        
        plt.figure(figsize=(12, 8))
        sns.barplot(data=top_genes, x='abs_mean_perturbation', y='gene')
        plt.xlabel('Absolute Mean Perturbation Effect')
        plt.title(f'Top 20 Most Affected Genes\nDrug: {results["drug_smiles"]}')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/top_affected_genes.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Heatmap of perturbation effects (subset of genes and cells)
        n_genes_plot = min(50, len(gene_names))
        n_cells_plot = min(100, perturbation_effect.shape[0])
        
        # Select top variable genes
        gene_var = np.var(perturbation_effect, axis=0)
        top_var_genes = np.argsort(gene_var)[-n_genes_plot:]
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(
            perturbation_effect[:n_cells_plot, top_var_genes].T,
            cmap='RdBu_r',
            center=0,
            yticklabels=[gene_names[i] for i in top_var_genes],
            xticklabels=False
        )
        plt.xlabel('Cells')
        plt.ylabel('Genes')
        plt.title(f'Perturbation Effects Heatmap\nDrug: {results["drug_smiles"]}')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/perturbation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Plots saved to {output_dir}/")


def main():
    parser = argparse.ArgumentParser(description='Predict drug perturbation effects using trained ChemCPA model')
    
    # Model arguments
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained ChemCPA checkpoint')
    
    # Input arguments
    parser.add_argument('--input_data', type=str, required=True,
                       help='Path to input scRNA-seq data (h5ad format)')
    parser.add_argument('--drug_smiles', type=str, required=True,
                       help='SMILES string of the drug to test')
    parser.add_argument('--dose', type=float, default=1.0,
                       help='Drug dosage (default: 1.0)')
    parser.add_argument('--cell_type', type=str, default=None,
                       help='Cell type information (optional)')
    
    # Output arguments
    parser.add_argument('--output', type=str, default='predictions.csv',
                       help='Output file for predictions')
    parser.add_argument('--plot_dir', type=str, default='./plots',
                       help='Directory to save plots')
    
    args = parser.parse_args()
    
    # Initialize predictor
    predictor = DrugPerturbationPredictor(args.model_path)
    
    # Load input data
    print(f"Loading input data from {args.input_data}")
    adata_obj = sc.read_h5ad(args.input_data)
    
    # Make predictions
    results = predictor.predict_perturbation(
        expression_data=adata_obj,
        drug_smiles=args.drug_smiles,
        dose=args.dose,
        cell_type=args.cell_type
    )
    
    # Analyze results
    top_affected_genes = predictor.analyze_perturbation_effects(results)
    
    # Save results
    top_affected_genes.to_csv(args.output, index=False)
    print(f"Top affected genes saved to {args.output}")
    
    # Create visualizations
    predictor.visualize_results(results, args.plot_dir)
    
    # Print summary
    print("\n" + "="*60)
    print("Prediction Summary:")
    print(f"- Drug SMILES: {args.drug_smiles}")
    print(f"- Dose: {args.dose}")
    print(f"- Number of cells: {results['n_cells']}")
    print(f"- Number of genes: {results['n_genes']}")
    print(f"- Top affected gene: {top_affected_genes.iloc[0]['gene']}")
    print(f"- Max perturbation effect: {top_affected_genes.iloc[0]['abs_mean_perturbation']:.4f}")
    print("="*60)


if __name__ == '__main__':
    main()

