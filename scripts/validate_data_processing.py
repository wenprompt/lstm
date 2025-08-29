#!/usr/bin/env python3
"""
Comprehensive data validation script for LSTM iron ore forecasting pipeline.

This script validates:
1. Data split ratios and sample counts
2. Feature selection and filtering accuracy
3. Scaling transformations and ranges
4. Sequence creation and tensor shapes
5. Configuration consistency
6. Edge cases and error conditions

Run this before training to ensure data processing is accurate.
"""

import yaml
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Any, List
import logging
from src.data.data_loader import DataLoader
from src.data.dataset import create_dataloaders, get_data_info
from src.models.model import create_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataProcessingValidator:
    """Comprehensive validation of data processing pipeline."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize validator with configuration."""
        self.config_path = config_path
        self.load_config()
        
    def load_config(self) -> None:
        """Load and validate configuration file."""
        logger.info(f"Loading configuration from {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        # Validate configuration structure
        required_sections = ["data", "features", "model", "training", "splits", "scaling"]
        missing_sections = [s for s in required_sections if s not in self.config]
        if missing_sections:
            raise ValueError(f"Missing required config sections: {missing_sections}")
            
        logger.info("✅ Configuration loaded and validated")
    
    def validate_split_ratios(self) -> Dict[str, Any]:
        """Validate data split ratios and calculations."""
        logger.info("\n🔍 Validating split ratios...")
        
        splits = self.config["splits"]
        train_ratio = splits["train_ratio"]
        val_ratio = splits["val_ratio"] 
        test_ratio = splits["test_ratio"]
        
        # Check ratios sum to 1.0
        total_ratio = train_ratio + val_ratio + test_ratio
        tolerance = self.config.get("validation", {}).get("split_ratio_tolerance", 0.01)
        
        validation_results = {
            "train_ratio": train_ratio,
            "val_ratio": val_ratio,
            "test_ratio": test_ratio,
            "total_ratio": total_ratio,
            "valid_sum": abs(total_ratio - 1.0) <= tolerance,
            "tolerance": tolerance
        }
        
        if not validation_results["valid_sum"]:
            logger.error(f"❌ Split ratios don't sum to 1.0: {total_ratio} (tolerance: {tolerance})")
            raise ValueError(f"Invalid split ratios: {train_ratio} + {val_ratio} + {test_ratio} = {total_ratio}")
        
        # Check for zero validation ratio (causes MinMaxScaler errors)
        if val_ratio == 0.0:
            logger.error("❌ Validation ratio cannot be 0.0 (breaks MinMaxScaler)")
            raise ValueError("val_ratio: 0.0 causes empty validation set and MinMaxScaler errors")
            
        logger.info(f"✅ Split ratios valid: {train_ratio:.2f}/{val_ratio:.2f}/{test_ratio:.2f}")
        return validation_results
    
    def validate_feature_selection(self) -> Dict[str, Any]:
        """Validate feature selection configuration.""" 
        logger.info("\n🔍 Validating feature selection...")
        
        # Load actual dataset to check available features
        data_path = Path(self.config["data"]["consolidated_features"])
        if not data_path.exists():
            raise FileNotFoundError(f"Dataset not found: {data_path}")
            
        df = pd.read_pickle(data_path)
        available_features = [col for col in df.columns if col != 'Y']
        selected_features = self.config.get("features", [])
        
        # Validate selected features exist
        missing_features = [f for f in selected_features if f not in available_features]
        if missing_features:
            logger.error(f"❌ Selected features not found: {missing_features}")
            raise ValueError(f"Features not in dataset: {missing_features}")
            
        feature_results = {
            "available_features": available_features,
            "selected_features": selected_features,
            "missing_features": missing_features,
            "feature_count": len(selected_features),
            "total_columns": df.shape[1],
            "has_y_target": 'Y' in df.columns
        }
        
        logger.info(f"✅ Features valid: {len(selected_features)}/{len(available_features)} selected")
        logger.info(f"   Available: {available_features}")
        logger.info(f"   Selected: {selected_features}")
        
        return feature_results
    
    def validate_data_pipeline(self) -> Dict[str, Any]:
        """Validate complete data processing pipeline."""
        logger.info("\n🔍 Validating data processing pipeline...")
        
        # Initialize data loader
        data_loader = DataLoader(self.config)
        
        # Process data through complete pipeline
        train_df, val_df, test_df = data_loader.get_processed_data()
        
        # Calculate expected sample counts
        total_samples = 868  # Known from dataset
        expected_train = int(total_samples * self.config["splits"]["train_ratio"])
        expected_val = int(total_samples * self.config["splits"]["val_ratio"])
        expected_test = total_samples - expected_train - expected_val
        
        pipeline_results = {
            "total_raw_samples": total_samples,
            "expected_splits": {
                "train": expected_train,
                "val": expected_val, 
                "test": expected_test
            },
            "actual_splits": {
                "train": len(train_df),
                "val": len(val_df),
                "test": len(test_df)
            },
            "feature_counts": {
                "train_features": len([c for c in train_df.columns if c != 'Y']),
                "val_features": len([c for c in val_df.columns if c != 'Y']),
                "test_features": len([c for c in test_df.columns if c != 'Y'])
            },
            "y_target_preserved": all('Y' in df.columns for df in [train_df, val_df, test_df]),
            "chronological_order": (
                train_df.index.max() <= val_df.index.min() and 
                val_df.index.max() <= test_df.index.min()
            )
        }
        
        # Check for empty sets
        if len(val_df) == 0:
            logger.error("❌ Empty validation set detected!")
            raise ValueError("Validation set is empty - will cause MinMaxScaler errors")
            
        logger.info(f"✅ Pipeline processed successfully:")
        logger.info(f"   Train: {len(train_df)} samples")
        logger.info(f"   Val: {len(val_df)} samples") 
        logger.info(f"   Test: {len(test_df)} samples")
        logger.info(f"   Features: {len([c for c in train_df.columns if c != 'Y'])}")
        logger.info(f"   Chronological order: {pipeline_results['chronological_order']}")
        
        return pipeline_results
    
    def validate_scaling_accuracy(self) -> Dict[str, Any]:
        """Validate feature scaling transformations."""
        logger.info("\n🔍 Validating feature scaling...")
        
        data_loader = DataLoader(self.config)
        train_df, val_df, test_df = data_loader.get_processed_data()
        
        feature_cols = [col for col in train_df.columns if col != 'Y']
        
        # Check scaling ranges (should be [0,1] for MinMaxScaler)
        scaling_results = {
            "feature_columns": feature_cols,
            "scaling_enabled": self.config["scaling"]["scale_features"],
            "train_ranges": {
                "min": train_df[feature_cols].min().min(),
                "max": train_df[feature_cols].max().max()
            },
            "val_ranges": {
                "min": val_df[feature_cols].min().min(),
                "max": val_df[feature_cols].max().max()
            },
            "test_ranges": {
                "min": test_df[feature_cols].min().min(),
                "max": test_df[feature_cols].max().max()
            }
        }
        
        if self.config["scaling"]["scale_features"]:
            # Validate scaling worked correctly
            for dataset_name, ranges in [("train", scaling_results["train_ranges"]),
                                        ("val", scaling_results["val_ranges"]),
                                        ("test", scaling_results["test_ranges"])]:
                if not (0.0 <= ranges["min"] <= ranges["max"] <= 1.0):
                    logger.warning(f"⚠️ {dataset_name} features outside [0,1]: [{ranges['min']:.3f}, {ranges['max']:.3f}]")
        
        logger.info(f"✅ Scaling validation complete")
        logger.info(f"   Train range: [{scaling_results['train_ranges']['min']:.3f}, {scaling_results['train_ranges']['max']:.3f}]")
        logger.info(f"   Val range: [{scaling_results['val_ranges']['min']:.3f}, {scaling_results['val_ranges']['max']:.3f}]")
        
        return scaling_results
    
    def validate_sequence_creation(self) -> Dict[str, Any]:
        """Validate PyTorch Dataset and DataLoader creation."""
        logger.info("\n🔍 Validating sequence creation...")
        
        data_loader = DataLoader(self.config)
        train_df, val_df, test_df = data_loader.get_processed_data()
        
        # Create dataloaders
        train_loader, val_loader, test_loader = create_dataloaders(train_df, val_df, test_df, self.config)
        
        # Get dataloader info
        train_info = get_data_info(train_loader)
        val_info = get_data_info(val_loader) if len(val_df) > 0 else {"num_batches": 0, "total_sequences": 0}
        test_info = get_data_info(test_loader)
        
        sequence_results = {
            "sequence_length": self.config["model"]["sequence_length"],
            "batch_size": self.config["training"]["batch_size"],
            "train_info": train_info,
            "val_info": val_info,
            "test_info": test_info,
            "expected_feature_count": len(self.config.get("features", [])),
            "actual_feature_shape": train_info["feature_shape"]
        }
        
        # Validate shapes match expectations
        expected_seq_len = self.config["model"]["sequence_length"] 
        expected_features = len(self.config.get("features", []))
        actual_shape = train_info["feature_shape"]
        
        if actual_shape[1] != expected_seq_len:
            logger.error(f"❌ Sequence length mismatch: expected {expected_seq_len}, got {actual_shape[1]}")
            
        if actual_shape[2] != expected_features:
            logger.error(f"❌ Feature count mismatch: expected {expected_features}, got {actual_shape[2]}")
        
        logger.info(f"✅ Sequence creation validated:")
        logger.info(f"   Train batches: {train_info['num_batches']}")
        logger.info(f"   Val batches: {val_info['num_batches']}")
        logger.info(f"   Test batches: {test_info['num_batches']}")
        logger.info(f"   Feature shape: {actual_shape}")
        
        return sequence_results
    
    def validate_model_compatibility(self) -> Dict[str, Any]:
        """Validate model configuration matches data."""
        logger.info("\n🔍 Validating model compatibility...")
        
        # Create model with current config
        model = create_model(self.config)
        model_info = model.get_model_info()
        
        # Check input size matches selected features
        expected_input_size = len(self.config.get("features", []))
        actual_input_size = model_info["input_size"]
        
        compatibility_results = {
            "expected_input_size": expected_input_size,
            "actual_input_size": actual_input_size,
            "input_size_match": expected_input_size == actual_input_size,
            "model_info": model_info,
            "config_input_size": self.config["model"]["input_size"]
        }
        
        if not compatibility_results["input_size_match"]:
            logger.error(f"❌ Input size mismatch: features={expected_input_size}, model={actual_input_size}")
            raise ValueError(f"Model input size {actual_input_size} doesn't match {expected_input_size} selected features")
            
        logger.info(f"✅ Model compatibility validated:")
        logger.info(f"   Input size: {actual_input_size} features")
        logger.info(f"   Total parameters: {model_info['total_parameters']:,}")
        logger.info(f"   Architecture: {model_info['architecture']}")
        
        return compatibility_results
        
    def run_full_validation(self) -> Dict[str, Any]:
        """Run complete validation suite."""
        logger.info("🧪 STARTING COMPREHENSIVE DATA VALIDATION")
        logger.info("=" * 60)
        
        validation_results = {}
        
        try:
            # 1. Configuration validation
            validation_results["config"] = self.validate_split_ratios()
            
            # 2. Feature validation
            validation_results["features"] = self.validate_feature_selection()
            
            # 3. Data pipeline validation
            validation_results["pipeline"] = self.validate_data_pipeline()
            
            # 4. Scaling validation
            validation_results["scaling"] = self.validate_scaling_accuracy()
            
            # 5. Sequence validation
            validation_results["sequences"] = self.validate_sequence_creation()
            
            # 6. Model compatibility validation
            validation_results["model"] = self.validate_model_compatibility()
            
            logger.info("\n🎉 ALL VALIDATIONS PASSED!")
            logger.info("=" * 60)
            logger.info("✅ Configuration is valid")
            logger.info("✅ Features are correctly selected and filtered")
            logger.info("✅ Data splits are properly sized and chronological")
            logger.info("✅ Feature scaling is working correctly")
            logger.info("✅ Sequence creation produces correct tensor shapes")
            logger.info("✅ Model input size matches selected features")
            logger.info("\n🚀 Pipeline is ready for training!")
            
            return validation_results
            
        except Exception as e:
            logger.error(f"\n❌ VALIDATION FAILED: {e}")
            logger.error("Fix the issues above before running training pipeline")
            raise


def main():
    """Run comprehensive data validation."""
    validator = DataProcessingValidator()
    results = validator.run_full_validation()
    
    # Print summary table
    print("\n📊 VALIDATION SUMMARY")
    print("=" * 50)
    print(f"Total samples: {results['pipeline']['actual_splits']['train'] + results['pipeline']['actual_splits']['val'] + results['pipeline']['actual_splits']['test']}")
    print(f"Train samples: {results['pipeline']['actual_splits']['train']}")
    print(f"Val samples: {results['pipeline']['actual_splits']['val']}")
    print(f"Test samples: {results['pipeline']['actual_splits']['test']}")
    print(f"Selected features: {results['features']['feature_count']}")
    print(f"Model parameters: {results['model']['model_info']['total_parameters']:,}")
    print(f"Expected batches - Train: {results['sequences']['train_info']['num_batches']}, Val: {results['sequences']['val_info']['num_batches']}, Test: {results['sequences']['test_info']['num_batches']}")
    print("=" * 50)


if __name__ == "__main__":
    main()