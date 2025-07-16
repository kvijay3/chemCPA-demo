#!/usr/bin/env python3
"""
Complete Example Workflow for ChemCPA Stem Cell Drug Perturbation Prediction

This script demonstrates the complete workflow from data loading to prediction.
It's designed to be educational and show you exactly how the model works.

Usage:
    python example_workflow.py
"""

import os
import sys
from pathlib import Path
import torch
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as adata
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import seaborn as sns

# Add the chemCPA module to path
sys.path.append(str(Path(__file__).parent))

def create_synthetic_stem_cell_data(n_cells: int = 1000, n_genes: int = 2000) -> adata.AnnData:
    """
    Create synthetic stem cell data for demonstration purposes
    
    Args:
        n_cells: Number of cells to generate
        n_genes: Number of genes to generate
        
    Returns:
        AnnData object with synthetic stem cell expression data
    """
    print(f"Creating synthetic stem cell data: {n_cells} cells x {n_genes} genes")
    
    # Generate synthetic expression data
    np.random.seed(42)
    
    # Create base expression levels (log-normal distribution)
    base_expression = np.random.lognormal(mean=1.0, sigma=1.5, size=(n_cells, n_genes))
    
    # Add some structure to make it more realistic
    # Create cell type-specific expression patterns
    cell_types = ['embryonic_stem_cell', 'neural_stem_cell', 'mesenchymal_stem_cell']
    n_cells_per_type = n_cells // len(cell_types)
    
    cell_type_labels = []
    for i, cell_type in enumerate(cell_types):
        start_idx = i * n_cells_per_type
        end_idx = (i + 1) * n_cells_per_type if i < len(cell_types) - 1 else n_cells
        
        # Add cell type-specific expression patterns
        type_specific_genes = np.random.choice(n_genes, size=100, replace=False)
        base_expression[start_idx:end_idx, type_specific_genes] *= np.random.uniform(2, 5)
        
        cell_type_labels.extend([cell_type] * (end_idx - start_idx))
    
    # Create gene names
    gene_names = [f"Gene_{i:04d}" for i in range(n_genes)]
    
    # Add some known stem cell markers
    stem_cell_markers = {
        'POU5F1': 0, 'SOX2': 1, 'NANOG': 2, 'KLF4': 3, 'MYC': 4,  # Pluripotency
        'NES': 5, 'SOX1': 6, 'PAX6': 7,  # Neural
        'CD44': 8, 'CD90': 9, 'CD105': 10  # Mesenchymal
    }
    
    for marker, idx in stem_cell_markers.items():
        if idx < n_genes:
            gene_names[idx] = marker
    
    # Create cell names
    cell_names = [f"Cell_{i:04d}" for i in range(n_cells)]
    
    # Create AnnData object
    adata_obj = adata.AnnData(
        X=base_expression,
        obs=pd.DataFrame({
            'cell_type': cell_type_labels,
            'condition': ['control'] * n_cells,  # All control for now
            'dose': [0.0] * n_cells,
            'SMILES': [''] * n_cells,  # Empty for control
        }, index=cell_names),
        var=pd.DataFrame({
            'gene_name': gene_names,
            'highly_variable': [True] * n_genes,
        }, index=gene_names)
    )
    
    print(f"Created synthetic data with cell types: {cell_types}")
    return adata_obj

def demonstrate_data_preprocessing(adata_obj: adata.AnnData) -> adata.AnnData:
    """
    Demonstrate typical scRNA-seq preprocessing steps
    
    Args:
        adata_obj: Input AnnData object
        
    Returns:
        Preprocessed AnnData object
    """
    print("\nPerforming data preprocessing...")
    
    # Make a copy to avoid modifying original
    adata_processed = adata_obj.copy()
    
    # Basic filtering
    print("- Filtering cells and genes")
    sc.pp.filter_cells(adata_processed, min_genes=200)
    sc.pp.filter_genes(adata_processed, min_cells=3)
    
    # Calculate QC metrics
    print("- Calculating QC metrics")
    adata_processed.var['mt'] = adata_processed.var_names.str.startswith('MT-')
    sc.pp.calculate_qc_metrics(adata_processed, percent_top=None, log1p=False, inplace=True)
    
    # Normalization
    print("- Normalizing and log-transforming")
    sc.pp.normalize_total(adata_processed, target_sum=1e4)
    sc.pp.log1p(adata_processed)
    
    # Find highly variable genes
    print("- Finding highly variable genes")
    sc.pp.highly_variable_genes(adata_processed, min_mean=0.0125, max_mean=3, min_disp=0.5)
    
    print(f"Preprocessing complete: {adata_processed.n_obs} cells x {adata_processed.n_vars} genes")
    return adata_processed

def simulate_drug_perturbation(adata_obj: adata.AnnData, 
                             drug_smiles: str = "CCO",
                             drug_name: str = "ethanol",
                             dose: float = 1.0,
                             effect_strength: float = 0.5) -> adata.AnnData:
    """
    Simulate drug perturbation effects on gene expression
    
    Args:
        adata_obj: Input AnnData object
        drug_smiles: SMILES string of the drug
        drug_name: Name of the drug
        dose: Drug dosage
        effect_strength: Strength of the perturbation effect
        
    Returns:
        AnnData object with simulated perturbation
    """
    print(f"\nSimulating perturbation with {drug_name} (SMILES: {drug_smiles})")
    
    # Make a copy
    adata_perturbed = adata_obj.copy()
    
    # Simulate perturbation effects
    n_affected_genes = int(0.1 * adata_perturbed.n_vars)  # Affect 10% of genes
    affected_genes = np.random.choice(adata_perturbed.n_vars, size=n_affected_genes, replace=False)
    
    # Create perturbation effects (some up, some down)
    perturbation_effects = np.random.normal(0, effect_strength, size=n_affected_genes)
    
    # Apply effects to expression data
    if hasattr(adata_perturbed.X, 'toarray'):
        expression_matrix = adata_perturbed.X.toarray()
    else:
        expression_matrix = adata_perturbed.X.copy()
    
    expression_matrix[:, affected_genes] += perturbation_effects
    
    # Ensure non-negative values (since we're working with log-transformed data)
    expression_matrix = np.maximum(expression_matrix, 0)
    
    # Update the AnnData object
    adata_perturbed.X = expression_matrix
    
    # Update metadata
    adata_perturbed.obs['condition'] = drug_name
    adata_perturbed.obs['SMILES'] = drug_smiles
    adata_perturbed.obs['dose'] = dose
    
    print(f"Applied perturbation to {n_affected_genes} genes")
    return adata_perturbed

def analyze_perturbation_effects(adata_control: adata.AnnData, 
                               adata_perturbed: adata.AnnData,
                               top_n: int = 20) -> pd.DataFrame:
    """
    Analyze the differences between control and perturbed conditions
    
    Args:
        adata_control: Control condition data
        adata_perturbed: Perturbed condition data
        top_n: Number of top affected genes to return
        
    Returns:
        DataFrame with top affected genes
    """
    print(f"\nAnalyzing perturbation effects...")
    
    # Calculate mean expression for each condition
    if hasattr(adata_control.X, 'toarray'):
        control_mean = np.mean(adata_control.X.toarray(), axis=0)
        perturbed_mean = np.mean(adata_perturbed.X.toarray(), axis=0)
    else:
        control_mean = np.mean(adata_control.X, axis=0)
        perturbed_mean = np.mean(adata_perturbed.X, axis=0)
    
    # Calculate log fold change
    log_fold_change = perturbed_mean - control_mean
    
    # Create results dataframe
    results_df = pd.DataFrame({
        'gene': adata_control.var_names,
        'control_mean': control_mean,
        'perturbed_mean': perturbed_mean,
        'log_fold_change': log_fold_change,
        'abs_log_fold_change': np.abs(log_fold_change)
    })
    
    # Sort by absolute fold change
    results_df = results_df.sort_values('abs_log_fold_change', ascending=False)
    
    print(f"Top {top_n} most affected genes identified")
    return results_df.head(top_n)

def create_visualizations(adata_control: adata.AnnData,
                         adata_perturbed: adata.AnnData,
                         results_df: pd.DataFrame,
                         output_dir: str = './example_plots'):
    """
    Create visualizations of the perturbation analysis
    
    Args:
        adata_control: Control condition data
        adata_perturbed: Perturbed condition data
        results_df: Results from analyze_perturbation_effects
        output_dir: Directory to save plots
    """
    print(f"\nCreating visualizations...")
    Path(output_dir).mkdir(exist_ok=True)
    
    # 1. Distribution of log fold changes
    plt.figure(figsize=(10, 6))
    plt.hist(results_df['log_fold_change'], bins=50, alpha=0.7, edgecolor='black')
    plt.xlabel('Log Fold Change')
    plt.ylabel('Number of Genes')
    plt.title('Distribution of Gene Expression Changes')
    plt.axvline(x=0, color='red', linestyle='--', alpha=0.7)
    plt.savefig(f'{output_dir}/log_fold_change_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Top affected genes
    top_genes = results_df.head(20)
    plt.figure(figsize=(12, 8))
    colors = ['red' if x > 0 else 'blue' for x in top_genes['log_fold_change']]
    plt.barh(range(len(top_genes)), top_genes['log_fold_change'], color=colors, alpha=0.7)
    plt.yticks(range(len(top_genes)), top_genes['gene'])
    plt.xlabel('Log Fold Change')
    plt.title('Top 20 Most Affected Genes')
    plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/top_affected_genes.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Scatter plot: control vs perturbed expression
    plt.figure(figsize=(8, 8))
    plt.scatter(results_df['control_mean'], results_df['perturbed_mean'], 
               alpha=0.6, s=20)
    
    # Add diagonal line
    min_val = min(results_df['control_mean'].min(), results_df['perturbed_mean'].min())
    max_val = max(results_df['control_mean'].max(), results_df['perturbed_mean'].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.7)
    
    plt.xlabel('Control Mean Expression')
    plt.ylabel('Perturbed Mean Expression')
    plt.title('Control vs Perturbed Expression')
    plt.savefig(f'{output_dir}/control_vs_perturbed_scatter.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Cell type-specific effects (if multiple cell types)
    if len(adata_control.obs['cell_type'].unique()) > 1:
        plt.figure(figsize=(12, 6))
        
        # Calculate mean expression by cell type
        cell_types = adata_control.obs['cell_type'].unique()
        top_gene = results_df.iloc[0]['gene']
        gene_idx = list(adata_control.var_names).index(top_gene)
        
        control_by_type = []
        perturbed_by_type = []
        
        for cell_type in cell_types:
            control_cells = adata_control.obs['cell_type'] == cell_type
            perturbed_cells = adata_perturbed.obs['cell_type'] == cell_type
            
            if hasattr(adata_control.X, 'toarray'):
                control_expr = adata_control.X.toarray()[control_cells, gene_idx]
                perturbed_expr = adata_perturbed.X.toarray()[perturbed_cells, gene_idx]
            else:
                control_expr = adata_control.X[control_cells, gene_idx]
                perturbed_expr = adata_perturbed.X[perturbed_cells, gene_idx]
            
            control_by_type.append(control_expr)
            perturbed_by_type.append(perturbed_expr)
        
        # Create box plot
        data_for_plot = []
        labels_for_plot = []
        
        for i, cell_type in enumerate(cell_types):
            data_for_plot.extend([control_by_type[i], perturbed_by_type[i]])
            labels_for_plot.extend([f'{cell_type}\nControl', f'{cell_type}\nPerturbed'])
        
        plt.boxplot(data_for_plot, labels=labels_for_plot)
        plt.ylabel('Expression Level')
        plt.title(f'Expression of {top_gene} by Cell Type')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/cell_type_specific_effects.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"Visualizations saved to {output_dir}/")

def main():
    """Main workflow demonstration"""
    print("="*80)
    print("ChemCPA Stem Cell Drug Perturbation Prediction - Example Workflow")
    print("="*80)
    
    # Step 1: Create synthetic stem cell data
    print("\n1. Creating synthetic stem cell data...")
    adata_control = create_synthetic_stem_cell_data(n_cells=1000, n_genes=2000)
    
    # Step 2: Preprocess the data
    print("\n2. Preprocessing data...")
    adata_processed = demonstrate_data_preprocessing(adata_control)
    
    # Step 3: Simulate drug perturbation
    print("\n3. Simulating drug perturbation...")
    adata_perturbed = simulate_drug_perturbation(
        adata_processed,
        drug_smiles="CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",  # Ibuprofen
        drug_name="ibuprofen",
        dose=1.0,
        effect_strength=0.8
    )
    
    # Step 4: Analyze perturbation effects
    print("\n4. Analyzing perturbation effects...")
    results = analyze_perturbation_effects(adata_processed, adata_perturbed, top_n=50)
    
    # Step 5: Create visualizations
    print("\n5. Creating visualizations...")
    create_visualizations(adata_processed, adata_perturbed, results)
    
    # Step 6: Save results
    print("\n6. Saving results...")
    results.to_csv('example_perturbation_results.csv', index=False)
    adata_processed.write('example_control_data.h5ad')
    adata_perturbed.write('example_perturbed_data.h5ad')
    
    # Step 7: Print summary
    print("\n" + "="*80)
    print("WORKFLOW SUMMARY")
    print("="*80)
    print(f"✓ Created synthetic stem cell data: {adata_processed.n_obs} cells x {adata_processed.n_vars} genes")
    print(f"✓ Cell types: {', '.join(adata_processed.obs['cell_type'].unique())}")
    print(f"✓ Simulated perturbation with ibuprofen")
    print(f"✓ Top affected gene: {results.iloc[0]['gene']}")
    print(f"✓ Max log fold change: {results.iloc[0]['abs_log_fold_change']:.3f}")
    print(f"✓ Results saved to: example_perturbation_results.csv")
    print(f"✓ Plots saved to: ./example_plots/")
    print("\nThis demonstrates the type of analysis ChemCPA performs.")
    print("In practice, ChemCPA would predict these perturbation effects")
    print("for unseen drugs without needing experimental data!")
    print("="*80)
    
    # Step 8: Show how to use with real ChemCPA model
    print("\n8. Next steps with real ChemCPA model:")
    print("   a) Train model: python train_chemcpa_simple.py --dataset sciplex")
    print("   b) Make predictions: python predict_perturbation.py --model_path <checkpoint> --input_data <data.h5ad> --drug_smiles <smiles>")
    print("   c) The model will learn to predict these perturbation effects automatically!")

if __name__ == '__main__':
    main()

