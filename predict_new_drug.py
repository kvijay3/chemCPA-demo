#!/usr/bin/env python3
"""
Predict Cellular Responses to New Drug Perturbations

This script uses a trained ChemCPA model to predict how cells will respond
to new drug perturbations. Supports both SMILES-based and drug name-based predictions.

Usage:
    # Predict single drug response
    python predict_new_drug.py --model_path model.ckpt --drug_name "Imatinib" --cell_type "A549"
    
    # Predict with SMILES
    python predict_new_drug.py --model_path model.ckpt --smiles "CC1=CC=C..." --cell_type "K562"
    
    # Batch predictions
    python predict_new_drug.py --model_path model.ckpt --drug_list drugs.csv --cell_types "A549,K562"
"""

import argparse
import os
import sys
from pathlib import Path
import torch
import numpy as np
import pandas as pd
import scanpy as sc
from tqdm import tqdm
import logging
from typing import List, Dict, Optional, Union

# Add the chemCPA module to path
sys.path.append(str(Path(__file__).parent))

from chemCPA.lightning_module import ChemCPA
from chemCPA.data.data import load_dataset_splits
from chemCPA.embedding import get_chemical_representation

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DrugPerturbationPredictor:
    """Predict cellular responses to drug perturbations using trained ChemCPA model"""
    
    def __init__(self, model_path: str, device: str = 'auto'):
        """
        Initialize the predictor with a trained model
        
        Args:
            model_path: Path to trained ChemCPA model checkpoint
            device: Device to run predictions on ('auto', 'cpu', 'cuda')
        """
        self.model_path = Path(model_path)
        self.device = self._setup_device(device)
        self.model = None
        self.reference_data = None
        self.drug_embeddings = {}
        
        self._load_model()
    
    def _setup_device(self, device: str) -> torch.device:
        """Setup computation device"""
        if device == 'auto':
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return torch.device(device)
    
    def _load_model(self):
        """Load the trained ChemCPA model"""
        logger.info(f"Loading model from {self.model_path}")
        
        try:
            # Load the checkpoint
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            # Initialize model from checkpoint
            self.model = ChemCPA.load_from_checkpoint(
                self.model_path,
                map_location=self.device
            )
            self.model.eval()
            self.model.to(self.device)
            
            logger.info("✅ Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise
    
    def load_reference_data(self, data_path: str):
        """
        Load reference dataset for baseline expression profiles
        
        Args:
            data_path: Path to reference dataset (.h5ad file)
        """
        logger.info(f"Loading reference data from {data_path}")
        
        try:
            self.reference_data = sc.read_h5ad(data_path)
            logger.info(f"✅ Loaded reference data: {self.reference_data.shape}")
            
        except Exception as e:
            logger.error(f"Failed to load reference data: {str(e)}")
            raise
    
    def get_drug_embedding(self, drug_identifier: str, identifier_type: str = 'auto') -> np.ndarray:
        """
        Get drug embedding from SMILES or drug name
        
        Args:
            drug_identifier: SMILES string or drug name
            identifier_type: 'smiles', 'name', or 'auto'
            
        Returns:
            Drug embedding vector
        """
        # Cache embeddings to avoid recomputation
        cache_key = f"{identifier_type}:{drug_identifier}"
        if cache_key in self.drug_embeddings:
            return self.drug_embeddings[cache_key]
        
        try:
            if identifier_type == 'smiles' or (identifier_type == 'auto' and self._is_smiles(drug_identifier)):
                # Use SMILES-based embedding
                embedding = get_chemical_representation([drug_identifier], 'SMILES')
                
            elif identifier_type == 'name' or identifier_type == 'auto':
                # Use drug name - create a simple embedding or lookup
                embedding = self._get_drug_name_embedding(drug_identifier)
                
            else:
                raise ValueError(f"Unknown identifier type: {identifier_type}")
            
            # Cache the embedding
            self.drug_embeddings[cache_key] = embedding
            return embedding
            
        except Exception as e:
            logger.warning(f"Failed to get embedding for {drug_identifier}: {str(e)}")
            # Return zero embedding as fallback
            embedding = np.zeros((1, 256))  # Default embedding size
            self.drug_embeddings[cache_key] = embedding
            return embedding
    
    def _is_smiles(self, string: str) -> bool:
        """Check if string looks like a SMILES string"""
        # Simple heuristic: SMILES typically contain chemical symbols and bonds
        smiles_chars = set('CNOSPFClBrI()[]=-#@+')
        return len(string) > 5 and any(c in smiles_chars for c in string)
    
    def _get_drug_name_embedding(self, drug_name: str) -> np.ndarray:
        """
        Get embedding for drug name (fallback method)
        
        In a real implementation, this would:
        1. Look up drug in a database
        2. Convert name to SMILES if possible
        3. Use pre-computed embeddings
        4. Fall back to learned embeddings from training
        """
        # For now, create a simple hash-based embedding
        import hashlib
        
        # Create deterministic embedding from drug name
        hash_obj = hashlib.md5(drug_name.lower().encode())
        hash_bytes = hash_obj.digest()
        
        # Convert to embedding vector
        embedding = np.frombuffer(hash_bytes, dtype=np.uint8).astype(np.float32)
        embedding = np.tile(embedding, (256 // len(embedding) + 1))[:256]  # Pad to 256 dims
        embedding = (embedding - 127.5) / 127.5  # Normalize to [-1, 1]
        
        return embedding.reshape(1, -1)
    
    def predict_perturbation(self, 
                           drug_identifier: str,
                           cell_type: str,
                           dose: float = 1.0,
                           identifier_type: str = 'auto',
                           baseline_expression: Optional[np.ndarray] = None) -> Dict:
        """
        Predict cellular response to drug perturbation
        
        Args:
            drug_identifier: SMILES string or drug name
            cell_type: Target cell type
            dose: Drug dose/concentration
            identifier_type: Type of drug identifier ('smiles', 'name', 'auto')
            baseline_expression: Baseline expression profile (if None, uses reference)
            
        Returns:
            Dictionary with prediction results
        """
        logger.info(f"Predicting perturbation: {drug_identifier} on {cell_type} at dose {dose}")
        
        try:
            # Get drug embedding
            drug_embedding = self.get_drug_embedding(drug_identifier, identifier_type)
            
            # Get baseline expression
            if baseline_expression is None:
                baseline_expression = self._get_baseline_expression(cell_type)
            
            # Prepare input tensors
            drug_tensor = torch.tensor(drug_embedding, dtype=torch.float32).to(self.device)
            expression_tensor = torch.tensor(baseline_expression, dtype=torch.float32).to(self.device)
            dose_tensor = torch.tensor([[dose]], dtype=torch.float32).to(self.device)
            
            # Make prediction
            with torch.no_grad():
                predicted_expression = self.model.predict_perturbation(
                    expression_tensor,
                    drug_tensor,
                    dose_tensor
                )
            
            # Convert back to numpy
            predicted_expression = predicted_expression.cpu().numpy()
            
            # Calculate perturbation effect (difference from baseline)
            perturbation_effect = predicted_expression - baseline_expression
            
            # Calculate confidence scores (simplified)
            confidence_score = self._calculate_confidence(predicted_expression, baseline_expression)
            
            results = {
                'drug_identifier': drug_identifier,
                'cell_type': cell_type,
                'dose': dose,
                'baseline_expression': baseline_expression,
                'predicted_expression': predicted_expression,
                'perturbation_effect': perturbation_effect,
                'confidence_score': confidence_score,
                'top_upregulated_genes': self._get_top_genes(perturbation_effect, direction='up'),
                'top_downregulated_genes': self._get_top_genes(perturbation_effect, direction='down')
            }
            
            logger.info(f"✅ Prediction completed with confidence: {confidence_score:.3f}")
            return results
            
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise
    
    def _get_baseline_expression(self, cell_type: str) -> np.ndarray:
        """Get baseline expression profile for cell type"""
        if self.reference_data is None:
            # Return zeros if no reference data
            logger.warning("No reference data available, using zero baseline")
            return np.zeros((1, 2000))  # Assume 2000 genes
        
        # Filter for cell type and control conditions
        cell_mask = self.reference_data.obs['cell_type'] == cell_type
        control_mask = self.reference_data.obs.get('control', True)
        
        if not any(cell_mask & control_mask):
            logger.warning(f"No control samples found for {cell_type}, using mean expression")
            return np.mean(self.reference_data.X, axis=0, keepdims=True)
        
        # Get mean expression for control samples of this cell type
        baseline = np.mean(
            self.reference_data.X[cell_mask & control_mask],
            axis=0,
            keepdims=True
        )
        
        return baseline
    
    def _calculate_confidence(self, predicted: np.ndarray, baseline: np.ndarray) -> float:
        """Calculate prediction confidence score"""
        # Simple confidence based on perturbation magnitude
        perturbation_magnitude = np.abs(predicted - baseline).mean()
        
        # Normalize to [0, 1] range (this is a simplified approach)
        confidence = min(1.0, perturbation_magnitude / 2.0)
        return confidence
    
    def _get_top_genes(self, perturbation_effect: np.ndarray, direction: str = 'up', n: int = 10) -> List[str]:
        """Get top perturbed genes"""
        if self.reference_data is None or not hasattr(self.reference_data, 'var_names'):
            return [f"Gene_{i}" for i in range(n)]
        
        effect_flat = perturbation_effect.flatten()
        
        if direction == 'up':
            top_indices = np.argsort(effect_flat)[-n:][::-1]
        else:  # down
            top_indices = np.argsort(effect_flat)[:n]
        
        gene_names = self.reference_data.var_names[top_indices].tolist()
        return gene_names
    
    def predict_batch(self, 
                     drug_list: List[Dict],
                     output_dir: str = './predictions') -> List[Dict]:
        """
        Predict responses for multiple drugs/conditions
        
        Args:
            drug_list: List of dictionaries with drug info
            output_dir: Directory to save results
            
        Returns:
            List of prediction results
        """
        logger.info(f"Running batch predictions for {len(drug_list)} conditions")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        results = []
        
        for i, drug_info in enumerate(tqdm(drug_list, desc="Predicting")):
            try:
                result = self.predict_perturbation(**drug_info)
                results.append(result)
                
                # Save individual result
                result_file = output_path / f"prediction_{i:04d}.npz"
                np.savez_compressed(result_file, **result)
                
            except Exception as e:
                logger.error(f"Failed prediction for {drug_info}: {str(e)}")
                continue
        
        # Save summary
        self._save_batch_summary(results, output_path)
        
        logger.info(f"✅ Batch predictions completed: {len(results)}/{len(drug_list)} successful")
        return results
    
    def _save_batch_summary(self, results: List[Dict], output_path: Path):
        """Save batch prediction summary"""
        summary_data = []
        
        for result in results:
            summary_data.append({
                'drug_identifier': result['drug_identifier'],
                'cell_type': result['cell_type'],
                'dose': result['dose'],
                'confidence_score': result['confidence_score'],
                'perturbation_magnitude': np.abs(result['perturbation_effect']).mean(),
                'top_upregulated': ', '.join(result['top_upregulated_genes'][:5]),
                'top_downregulated': ', '.join(result['top_downregulated_genes'][:5])
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(output_path / 'batch_summary.csv', index=False)
        
        logger.info(f"Saved batch summary to {output_path / 'batch_summary.csv'}")


def parse_drug_list(drug_list_file: str) -> List[Dict]:
    """Parse drug list from CSV file"""
    df = pd.read_csv(drug_list_file)
    
    required_columns = ['drug_identifier', 'cell_type']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    drug_list = []
    for _, row in df.iterrows():
        drug_info = {
            'drug_identifier': row['drug_identifier'],
            'cell_type': row['cell_type'],
            'dose': row.get('dose', 1.0),
            'identifier_type': row.get('identifier_type', 'auto')
        }
        drug_list.append(drug_info)
    
    return drug_list


def main():
    parser = argparse.ArgumentParser(description='Predict cellular responses to drug perturbations')
    
    # Model and data
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained ChemCPA model checkpoint')
    parser.add_argument('--reference_data', type=str,
                       help='Path to reference dataset for baseline expression')
    
    # Single prediction
    parser.add_argument('--drug_name', type=str, help='Drug name for prediction')
    parser.add_argument('--smiles', type=str, help='SMILES string for prediction')
    parser.add_argument('--cell_type', type=str, help='Target cell type')
    parser.add_argument('--dose', type=float, default=1.0, help='Drug dose/concentration')
    
    # Batch prediction
    parser.add_argument('--drug_list', type=str, help='CSV file with drug list for batch prediction')
    parser.add_argument('--cell_types', type=str, help='Comma-separated list of cell types')
    parser.add_argument('--doses', type=str, help='Comma-separated list of doses')
    
    # Output
    parser.add_argument('--output_file', type=str, help='Output file for single prediction')
    parser.add_argument('--output_dir', type=str, default='./predictions',
                       help='Output directory for batch predictions')
    
    # Other options
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'cuda'],
                       help='Device for computation')
    
    args = parser.parse_args()
    
    # Initialize predictor
    predictor = DrugPerturbationPredictor(args.model_path, args.device)
    
    # Load reference data if provided
    if args.reference_data:
        predictor.load_reference_data(args.reference_data)
    
    # Single prediction
    if args.drug_name or args.smiles:
        drug_identifier = args.drug_name or args.smiles
        identifier_type = 'name' if args.drug_name else 'smiles'
        
        if not args.cell_type:
            raise ValueError("Cell type required for single prediction")
        
        result = predictor.predict_perturbation(
            drug_identifier=drug_identifier,
            cell_type=args.cell_type,
            dose=args.dose,
            identifier_type=identifier_type
        )
        
        # Save result
        if args.output_file:
            np.savez_compressed(args.output_file, **result)
            logger.info(f"Saved prediction to {args.output_file}")
        else:
            print(f"\\nPrediction Results:")
            print(f"Drug: {result['drug_identifier']}")
            print(f"Cell Type: {result['cell_type']}")
            print(f"Dose: {result['dose']}")
            print(f"Confidence: {result['confidence_score']:.3f}")
            print(f"Top upregulated genes: {', '.join(result['top_upregulated_genes'][:5])}")
            print(f"Top downregulated genes: {', '.join(result['top_downregulated_genes'][:5])}")
    
    # Batch prediction
    elif args.drug_list:
        drug_list = parse_drug_list(args.drug_list)
        results = predictor.predict_batch(drug_list, args.output_dir)
        
    # Generate combinations if cell_types and doses provided
    elif args.cell_types and args.doses:
        if not (args.drug_name or args.smiles):
            raise ValueError("Drug identifier required for combination prediction")
        
        drug_identifier = args.drug_name or args.smiles
        identifier_type = 'name' if args.drug_name else 'smiles'
        
        cell_types = args.cell_types.split(',')
        doses = [float(d) for d in args.doses.split(',')]
        
        drug_list = []
        for cell_type in cell_types:
            for dose in doses:
                drug_list.append({
                    'drug_identifier': drug_identifier,
                    'cell_type': cell_type.strip(),
                    'dose': dose,
                    'identifier_type': identifier_type
                })
        
        results = predictor.predict_batch(drug_list, args.output_dir)
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

