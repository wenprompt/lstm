#!/usr/bin/env python3
"""
Main orchestration script for LSTM iron ore price forecasting.

This script executes the complete machine learning pipeline:
1. Data loading and preprocessing
2. Feature scaling and train/val/test splitting  
3. PyTorch Dataset and DataLoader creation
4. Model creation and training
5. Model evaluation and visualization
6. Results saving and reporting

Usage:
    uv run python main.py

Configuration is loaded from config.yaml and all results are saved to results/ directory.
"""

import torch
import yaml
import logging
from pathlib import Path
from typing import Dict, Any
import sys

from src.data.data_loader import DataLoader as IronOreDataLoader
from src.data.dataset import create_dataloaders, get_data_info
from src.models.model import create_model, get_model_summary
from src.training.train import create_trainer
from src.evaluation.evaluate import evaluate_model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('results/training.log')
    ]
)

logger = logging.getLogger(__name__)


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is invalid
    """
    logger.info(f"Loading configuration from {config_path}")
    
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        logger.info("Configuration loaded successfully")
        logger.info(f"Model: {config['model']['hidden_size']} hidden units, "
                   f"{config['model']['number_of_layers']} layers")
        logger.info(f"Training: {config['training']['epochs']} max epochs, "
                   f"batch size {config['training']['batch_size']}")
        
        return config
        
    except yaml.YAMLError as e:
        logger.error(f"Error parsing configuration file: {e}")
        raise


def setup_directories() -> None:
    """
    Create necessary directories for results and logs.
    """
    logger.info("Setting up output directories...")
    
    # Create main results directory
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    # Create subdirectories
    (results_dir / "models").mkdir(exist_ok=True)
    (results_dir / "plots").mkdir(exist_ok=True)
    (results_dir / "logs").mkdir(exist_ok=True)
    
    logger.info(f"Output directories created: {results_dir.absolute()}")


def load_and_preprocess_data(config: Dict[str, Any]) -> tuple:
    """
    Load data and create train/validation/test splits.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    logger.info("="*60)
    logger.info("STEP 1: DATA LOADING AND PREPROCESSING")
    logger.info("="*60)
    
    # Initialize data loader
    data_loader = IronOreDataLoader(config)
    
    # Load and process data
    train_df, val_df, test_df = data_loader.get_processed_data()
    
    # Log dataset information
    logger.info(f"Training set: {len(train_df)} samples")
    logger.info(f"Validation set: {len(val_df)} samples") 
    logger.info(f"Test set: {len(test_df)} samples")
    logger.info(f"Total samples: {len(train_df) + len(val_df) + len(test_df)}")
    logger.info(f"Features: {len([col for col in train_df.columns if col != 'Y'])}")
    
    return train_df, val_df, test_df


def create_datasets(train_df, val_df, test_df, config: Dict[str, Any]) -> tuple:
    """
    Create PyTorch DataLoaders from processed DataFrames.
    
    Args:
        train_df: Training DataFrame
        val_df: Validation DataFrame  
        test_df: Test DataFrame
        config: Configuration dictionary
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    logger.info("="*60)
    logger.info("STEP 2: PYTORCH DATASET CREATION")
    logger.info("="*60)
    
    # Create DataLoaders
    train_loader, val_loader, test_loader = create_dataloaders(
        train_df, val_df, test_df, config
    )
    
    # Log DataLoader information
    train_info = get_data_info(train_loader)
    val_info = get_data_info(val_loader)
    test_info = get_data_info(test_loader)
    
    logger.info("DataLoader Information:")
    logger.info(f"  Train: {train_info['num_batches']} batches, "
               f"feature shape: {train_info['feature_shape']}")
    logger.info(f"  Val: {val_info['num_batches']} batches, "
               f"feature shape: {val_info['feature_shape']}")
    logger.info(f"  Test: {test_info['num_batches']} batches, "
               f"feature shape: {test_info['feature_shape']}")
    
    return train_loader, val_loader, test_loader


def create_and_train_model(train_loader, val_loader, config: Dict[str, Any]) -> tuple:
    """
    Create LSTM model and execute training pipeline.
    
    Args:
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        config: Configuration dictionary
        
    Returns:
        Tuple of (trained_model, training_results)
    """
    logger.info("="*60)
    logger.info("STEP 3: MODEL CREATION AND TRAINING")
    logger.info("="*60)
    
    # Create model
    model = create_model(config)
    
    # Log model information
    model_summary = get_model_summary(model)
    logger.info("Model Architecture:")
    for line in model_summary.split('\n'):
        logger.info(line)
    
    # Create trainer
    trainer = create_trainer(model, config)
    
    # Execute training
    logger.info("Starting training pipeline...")
    training_results = trainer.train(train_loader, val_loader)
    
    # Log training summary
    training_summary = trainer.get_training_summary()
    logger.info("Training Results:")
    for line in training_summary.split('\n'):
        logger.info(line)
    
    return model, training_results


def evaluate_trained_model(model, test_loader, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluate trained model on test set.
    
    Args:
        model: Trained LSTM model
        test_loader: Test DataLoader
        config: Configuration dictionary
        
    Returns:
        Evaluation results dictionary
    """
    logger.info("="*60)
    logger.info("STEP 4: MODEL EVALUATION") 
    logger.info("="*60)
    
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() and 
                         config["device"]["use_cuda"] else "cpu")
    
    # Run evaluation
    evaluation_results = evaluate_model(
        model=model,
        test_loader=test_loader, 
        device=device,
        config=config,
        save_dir=Path("results")
    )
    
    logger.info("Evaluation completed successfully")
    logger.info(f"Results saved to: {evaluation_results['results_path']}")
    logger.info("Plots saved to: results/plots/")
    
    return evaluation_results


def save_final_results(training_results: Dict[str, Any], 
                      evaluation_results: Dict[str, Any],
                      config: Dict[str, Any]) -> None:
    """
    Save consolidated final results.
    
    Args:
        training_results: Results from training
        evaluation_results: Results from evaluation
        config: Configuration used
    """
    logger.info("="*60)
    logger.info("STEP 5: SAVING FINAL RESULTS")
    logger.info("="*60)
    
    # Consolidate results
    final_results = {
        'configuration': config,
        'training': {
            'final_epoch': training_results['final_epoch'],
            'total_time_minutes': training_results['total_time'] / 60,
            'best_val_loss': training_results['best_val_loss'],
            'final_train_loss': training_results['final_train_loss'],
            'final_val_loss': training_results['final_val_loss'],
            'early_stopped': training_results['early_stopped']
        },
        'evaluation': {
            'test_metrics': evaluation_results['metrics'],
            'plot_files': evaluation_results['plot_paths']
        }
    }
    
    # Save to JSON
    import json
    results_file = Path("results/final_results.json")
    with open(results_file, 'w') as f:
        json.dump(final_results, f, indent=2)
    
    logger.info(f"Final results saved to: {results_file}")
    
    # Print final summary
    logger.info("="*60)
    logger.info("PIPELINE EXECUTION COMPLETED SUCCESSFULLY")
    logger.info("="*60)
    logger.info(f"Best validation loss: {training_results['best_val_loss']:.6f}")
    logger.info(f"Test RMSE: {evaluation_results['metrics']['rmse']:.6f}")
    logger.info(f"Test MAE: {evaluation_results['metrics']['mae']:.6f}")
    logger.info(f"Directional accuracy: {evaluation_results['metrics']['directional_accuracy']:.2f}%")
    logger.info(f"Training time: {training_results['total_time']/60:.1f} minutes")
    logger.info(f"Model saved at: {training_results['model_path']}")
    
    # Final layman summary
    direction_success = evaluation_results['metrics']['directional_accuracy']
    overall_quality = "EXCELLENT" if direction_success >= 60 else "GOOD" if direction_success >= 55 else "DECENT" if direction_success >= 50 else "NEEDS_WORK"
    logger.info(f"🎯 FINAL RESULT: {overall_quality} iron ore price forecasting model - predicts price direction correctly {direction_success:.1f}% of the time")
    logger.info("="*60)


def main() -> None:
    """
    Main execution function for the LSTM forecasting pipeline.
    """
    try:
        logger.info("Starting LSTM Iron Ore Forecasting Pipeline")
        logger.info("="*60)
        
        # Load configuration
        config = load_config()
        
        # Setup directories
        setup_directories()
        
        # Step 1: Load and preprocess data
        train_df, val_df, test_df = load_and_preprocess_data(config)
        
        # Step 2: Create PyTorch datasets
        train_loader, val_loader, test_loader = create_datasets(
            train_df, val_df, test_df, config
        )
        
        # Step 3: Create and train model
        model, training_results = create_and_train_model(
            train_loader, val_loader, config
        )
        
        # Step 4: Evaluate model
        evaluation_results = evaluate_trained_model(
            model, test_loader, config
        )
        
        # Step 5: Save final results
        save_final_results(training_results, evaluation_results, config)
        
        logger.info("Pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}")
        logger.exception("Full traceback:")
        sys.exit(1)


if __name__ == "__main__":
    main()