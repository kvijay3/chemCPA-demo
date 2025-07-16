#!/usr/bin/env python3
"""
Debug script to examine the LINCS dataset structure
"""

import scanpy as sc
import pandas as pd
import numpy as np

def examine_lincs_dataset():
    """Examine the structure of the LINCS dataset"""
    
    print("🔍 Examining LINCS dataset structure...")
    
    # Load the dataset
    dataset_path = 'project_folder/datasets/lincs.h5ad'
    try:
        adata = sc.read_h5ad(dataset_path)
        print(f"✅ Loaded dataset: {dataset_path}")
        print(f"   Shape: {adata.shape}")
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return
    
    # Examine obs (observations/metadata)
    print(f"\n📊 Observations (obs) columns:")
    for col in adata.obs.columns:
        unique_count = adata.obs[col].nunique()
        dtype = adata.obs[col].dtype
        sample_values = adata.obs[col].head(3).tolist()
        print(f"   {col}: {dtype}, {unique_count} unique values")
        print(f"      Sample values: {sample_values}")
    
    # Check specific columns mentioned in config
    print(f"\n🎯 Checking specific columns:")
    
    # Check perturbation_key candidates
    perturbation_candidates = ['condition', 'cov_drug_dose_name', 'drug', 'perturbation']
    for key in perturbation_candidates:
        if key in adata.obs.columns:
            values = adata.obs[key]
            print(f"   ✅ {key}: {values.dtype}")
            print(f"      Sample values: {values.head(5).tolist()}")
            print(f"      Unique count: {values.nunique()}")
        else:
            print(f"   ❌ {key}: Not found")
    
    # Check uns (unstructured data)
    print(f"\n📚 Unstructured data (uns) keys:")
    for key in adata.uns.keys():
        print(f"   {key}: {type(adata.uns[key])}")
    
    # Check if rank_genes_groups_cov exists
    if 'rank_genes_groups_cov' in adata.uns:
        print(f"   ✅ rank_genes_groups_cov found")
        degs_data = adata.uns['rank_genes_groups_cov']
        print(f"      Type: {type(degs_data)}")
        if hasattr(degs_data, 'keys'):
            print(f"      Keys: {list(degs_data.keys())}")
    else:
        print(f"   ❌ rank_genes_groups_cov not found")
    
    # Check var (variables/genes)
    print(f"\n🧬 Variables (var) info:")
    print(f"   Shape: {adata.var.shape}")
    print(f"   Columns: {list(adata.var.columns)}")

if __name__ == '__main__':
    examine_lincs_dataset()

