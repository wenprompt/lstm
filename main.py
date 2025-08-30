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
import time
import platform
import psutil
from datetime import datetime

from src.data.data_loader import DataLoader as IronOreDataLoader
from src.data.dataset import create_dataloaders, get_data_info
from src.models.model import create_model, get_model_summary
from src.training.train import create_trainer
from src.evaluation.evaluate import evaluate_model

# Ensure results and logs directories exist for logging
Path("results/logs/training").mkdir(parents=True, exist_ok=True)

# Configure enhanced logging with detailed formatting
log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logging.basicConfig(
    level=logging.INFO,
    format=log_format,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(
            f"results/logs/training/training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        ),
    ],
)

logger = logging.getLogger(__name__)


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file with enhanced logging.

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
        logger.error(f"Configuration file not found: {config_path}")
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    try:
        with open(config_file, "r") as f:
            config = yaml.safe_load(f)

        # Detailed configuration logging
        logger.info("Configuration loaded successfully")
        logger.info("Configuration Summary:")
        logger.info("  Model Architecture:")
        logger.info(f"    Hidden size: {config['model']['hidden_size']}")
        logger.info(f"    Layers: {config['model']['number_of_layers']}")
        logger.info(f"    Sequence length: {config['model']['sequence_length']}")
        logger.info(f"    Bidirectional: {config['model']['bidirectional']}")
        logger.info(f"    Dropout rate: {config['model']['dropout_rate']}")

        logger.info("  Training Parameters:")
        logger.info(f"    Max epochs: {config['training']['epochs']}")
        logger.info(f"    Batch size: {config['training']['batch_size']}")
        logger.info(f"    Learning rate: {config['training']['learning_rate']}")
        logger.info(
            f"    Early stopping patience: {config['training']['early_stopping_patience']}"
        )

        logger.info("  Data Configuration:")
        logger.info(f"    Train ratio: {config['splits']['train_ratio']}")
        logger.info(f"    Val ratio: {config['splits']['val_ratio']}")
        logger.info(f"    Test ratio: {config['splits']['test_ratio']}")

        # Feature selection logging
        feature_count = len(config.get("features", []))
        logger.info(f"    Selected features: {feature_count}")

        return config

    except yaml.YAMLError as e:
        logger.error(f"Error parsing configuration file: {e}")
        raise


def setup_directories() -> Path:
    """
    Create necessary directories for results and logs with enhanced logging.

    Returns:
        Path to results directory
    """
    logger.info("Setting up output directories...")

    # Create main results directory
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    # Create main subdirectories
    main_subdirs = ["models", "plots"]
    for subdir in main_subdirs:
        (results_dir / subdir).mkdir(exist_ok=True)

    # Create logs structure with subdirectories
    logs_dir = results_dir / "logs"
    logs_dir.mkdir(exist_ok=True)
    log_subdirs = ["training", "evaluation", "hypertuning"]
    for log_subdir in log_subdirs:
        (logs_dir / log_subdir).mkdir(exist_ok=True)

    # Check disk space
    disk_usage = psutil.disk_usage(str(results_dir))
    free_gb = disk_usage.free / (1024**3)

    logger.info("Output directories structure:")
    logger.info(f"  Base directory: {results_dir.absolute()}")
    for subdir in main_subdirs:
        subdir_path = results_dir / subdir
        logger.info(f"    {subdir}/: {subdir_path.absolute()}")
    logger.info(f"    logs/: {logs_dir.absolute()}")
    for log_subdir in log_subdirs:
        log_subdir_path = logs_dir / log_subdir
        logger.info(f"      logs/{log_subdir}/: {log_subdir_path.absolute()}")

    logger.info(f"Disk space available: {free_gb:.1f} GB")
    if free_gb < 1.0:
        logger.warning(
            "Low disk space detected! Consider freeing up space before training."
        )

    return results_dir


def load_and_preprocess_data(config: Dict[str, Any]) -> tuple:
    """
    Load data and create train/validation/test splits.

    Args:
        config: Configuration dictionary

    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    logger.info("=" * 60)
    logger.info("STEP 1: DATA LOADING AND PREPROCESSING")
    logger.info("=" * 60)

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
    logger.info("=" * 60)
    logger.info("STEP 2: PYTORCH DATASET CREATION")
    logger.info("=" * 60)

    # Create DataLoaders
    train_loader, val_loader, test_loader = create_dataloaders(
        train_df, val_df, test_df, config
    )

    # Log DataLoader information
    train_info = get_data_info(train_loader)
    val_info = get_data_info(val_loader)
    test_info = get_data_info(test_loader)

    logger.info("DataLoader Information:")
    logger.info(
        f"  Train: {train_info['num_batches']} batches, "
        f"feature shape: {train_info['feature_shape']}"
    )

    # Handle empty validation dataloader (when val_ratio=0)
    if val_info["feature_shape"] is not None:
        logger.info(
            f"  Val: {val_info['num_batches']} batches, "
            f"feature shape: {val_info['feature_shape']}"
        )
    else:
        logger.info("  Val: 0 batches (validation disabled)")

    logger.info(
        f"  Test: {test_info['num_batches']} batches, "
        f"feature shape: {test_info['feature_shape']}"
    )

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
    logger.info("=" * 60)
    logger.info("STEP 3: MODEL CREATION AND TRAINING")
    logger.info("=" * 60)

    # Create model
    model = create_model(config)

    # Log model information
    model_summary = get_model_summary(model)
    logger.info("Model Architecture:")
    for line in model_summary.split("\n"):
        logger.info(line)

    # Create trainer
    trainer = create_trainer(model, config)

    # Execute training
    logger.info("Starting training pipeline...")
    training_results = trainer.train(train_loader, val_loader)

    # Log training summary
    training_summary = trainer.get_training_summary()
    logger.info("Training Results:")
    for line in training_summary.split("\n"):
        logger.info(line)

    return model, training_results


def evaluate_trained_model(
    model, test_loader, config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Evaluate trained model on test set.

    Args:
        model: Trained LSTM model
        test_loader: Test DataLoader
        config: Configuration dictionary

    Returns:
        Evaluation results dictionary
    """
    logger.info("=" * 60)
    logger.info("STEP 4: MODEL EVALUATION")
    logger.info("=" * 60)

    # Determine device
    device = torch.device(
        "cuda" if torch.cuda.is_available() and config["device"]["use_cuda"] else "cpu"
    )

    # Run evaluation
    evaluation_results = evaluate_model(
        model=model,
        test_loader=test_loader,
        device=device,
        config=config,
        save_dir=Path("results"),
    )

    logger.info("Evaluation completed successfully")
    logger.info("Plots saved to: results/plots/")

    return evaluation_results


def save_final_results(
    training_results: Dict[str, Any],
    evaluation_results: Dict[str, Any],
    config: Dict[str, Any],
) -> None:
    """
    Save consolidated final results.

    Args:
        training_results: Results from training
        evaluation_results: Results from evaluation
        config: Configuration used
    """
    logger.info("=" * 60)
    logger.info("STEP 5: SAVING FINAL RESULTS")
    logger.info("=" * 60)

    # Consolidate results
    final_results = {
        "configuration": config,
        "training": {
            "final_epoch": training_results["final_epoch"],
            "total_time_minutes": training_results["total_time"] / 60,
            "best_val_loss": training_results["best_val_loss"],
            "final_train_loss": training_results["final_train_loss"],
            "final_val_loss": training_results["final_val_loss"],
            "early_stopped": training_results["early_stopped"],
        },
        "evaluation": {
            "test_metrics": evaluation_results["metrics"],
            "test_samples": evaluation_results["test_samples"],
            "prediction_statistics": evaluation_results["prediction_statistics"],
            "actual_statistics": evaluation_results["actual_statistics"],
            "model_type": evaluation_results["model_type"],
            "plot_files": evaluation_results["plot_paths"],
        },
    }

    # Save to JSON
    import json

    results_file = Path("results/final_results.json")
    with open(results_file, "w") as f:
        json.dump(final_results, f, indent=2)

    logger.info(f"Final results saved to: {results_file}")

    # Print final summary
    logger.info("=" * 60)
    logger.info("PIPELINE EXECUTION COMPLETED SUCCESSFULLY")
    logger.info("=" * 60)
    
    # Check if validation was enabled by comparing val_ratio
    val_ratio = config.get("splits", {}).get("val_ratio", 0.0)
    has_validation = val_ratio > 0.0
    
    # Log validation results appropriately
    if has_validation:
        logger.info(f"Best validation loss: {training_results['best_val_loss']:.6f}")
    else:
        logger.info(f"Best train loss: {training_results['best_val_loss']:.6f} (no validation)")
    
    logger.info(f"Test RMSE: {evaluation_results['metrics']['rmse']:.6f}")
    logger.info(f"Test MAE: {evaluation_results['metrics']['mae']:.6f}")
    logger.info(
        f"Directional accuracy: {evaluation_results['metrics']['directional_accuracy']:.2f}%"
    )
    logger.info(f"Training time: {training_results['total_time'] / 60:.1f} minutes")
    logger.info(f"Model saved at: {training_results['model_path']}")

    # Final layman summary using the documented rating scale
    direction_success = evaluation_results["metrics"]["directional_accuracy"]

    # Get thresholds from config, falling back to documented defaults
    thresholds = config.get("thresholds", {})
    excellent_thresh = thresholds.get("excellent_accuracy", 60)
    good_thresh = thresholds.get("good_accuracy", 55)
    fair_thresh = thresholds.get("fair_accuracy", 50)

    # Determine quality rating based on the correct scale
    if direction_success >= excellent_thresh:
        overall_quality = "EXCELLENT"
    elif direction_success >= good_thresh:
        overall_quality = "GOOD"
    elif direction_success >= fair_thresh:
        overall_quality = "FAIR"
    else:
        overall_quality = "POOR"

    logger.info(
        f"🎯 FINAL RESULT: {overall_quality} iron ore price forecasting model - predicts price direction correctly {direction_success:.1f}% of the time"
    )
    logger.info("=" * 60)


def log_system_information() -> None:
    """
    Log comprehensive system information for reproducibility.
    """
    logger.info("=" * 80)
    logger.info("LSTM IRON ORE PRICE FORECASTING PIPELINE")
    logger.info("=" * 80)

    # System information
    logger.info("System Information:")
    logger.info(f"  Platform: {platform.platform()}")
    logger.info(f"  Python version: {platform.python_version()}")
    logger.info(f"  PyTorch version: {torch.__version__}")

    # Hardware information
    logger.info("Hardware Information:")
    logger.info(
        f"  CPU: {psutil.cpu_count(logical=False)} cores ({psutil.cpu_count()} logical)"
    )
    logger.info(f"  RAM: {psutil.virtual_memory().total / (1024**3):.1f} GB")

    if torch.cuda.is_available():
        logger.info(f"  GPU: {torch.cuda.get_device_name(0)}")
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        logger.info(f"  GPU Memory: {gpu_memory:.1f} GB")  # type: ignore
        logger.info(f"  CUDA Version: {torch.version.cuda}")  # type: ignore
    else:
        logger.info("  GPU: None available (using CPU)")

    # Timestamp
    logger.info(f"  Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 80 + "\n")


def main() -> None:
    """
    Main execution function for the LSTM forecasting pipeline with enhanced logging.
    """
    start_time = time.time()

    # Log system information
    log_system_information()

    try:
        # Load configuration
        step_start = time.time()
        config = load_config()
        config_time = time.time() - step_start
        logger.info(f"Configuration loaded in {config_time:.2f} seconds\n")

        # Setup directories
        step_start = time.time()
        results_dir = setup_directories()
        setup_time = time.time() - step_start
        logger.info(f"Directory setup completed in {setup_time:.2f} seconds\n")

        # Step 1: Load and preprocess data
        step_start = time.time()
        train_df, val_df, test_df = load_and_preprocess_data(config)
        step_time = time.time() - step_start
        logger.info(
            f"\n⏱️  Step 1 (Data Loading & Preprocessing) completed in {step_time:.1f} seconds\n"
        )

        # Step 2: Create PyTorch datasets
        step_start = time.time()
        train_loader, val_loader, test_loader = create_datasets(
            train_df, val_df, test_df, config
        )
        step_time = time.time() - step_start
        logger.info(
            f"\n⏱️  Step 2 (Dataset Creation) completed in {step_time:.1f} seconds\n"
        )

        # Step 3: Create and train model
        step_start = time.time()
        model, training_results = create_and_train_model(
            train_loader, val_loader, config
        )
        step_time = time.time() - step_start
        logger.info(
            f"\n⏱️  Step 3 (Model Training) completed in {step_time:.1f} seconds\n"
        )

        # Step 4: Evaluate model
        step_start = time.time()
        evaluation_results = evaluate_trained_model(model, test_loader, config)
        step_time = time.time() - step_start
        logger.info(
            f"\n⏱️  Step 4 (Model Evaluation) completed in {step_time:.1f} seconds\n"
        )

        # Step 5: Save final results
        step_start = time.time()
        save_final_results(training_results, evaluation_results, config)
        step_time = time.time() - step_start
        logger.info(
            f"\n⏱️  Step 5 (Results Saving) completed in {step_time:.1f} seconds\n"
        )

        # Calculate total pipeline time and efficiency metrics
        total_time = time.time() - start_time

        # Final pipeline summary with enhanced metrics
        logger.info("=" * 80)
        logger.info("ENHANCED PIPELINE EXECUTION SUMMARY")
        logger.info("=" * 80)
        logger.info(
            f"Total execution time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)"
        )

        # Performance breakdown
        training_time = training_results.get("total_time", 0)
        training_pct = (training_time / total_time) * 100 if total_time > 0 else 0
        logger.info(
            f"Training time: {training_time/60:.1f} min ({training_pct:.1f}% of total)"
        )

        # Resource utilization summary
        if torch.cuda.is_available():
            peak_gpu_memory = torch.cuda.max_memory_allocated(0) / (1024**3)
            logger.info(f"Peak GPU memory usage: {peak_gpu_memory:.2f} GB")

        logger.info(f"Results saved to: {results_dir.absolute()}")
        logger.info("=" * 80)
        logger.info("✅ LSTM IRON ORE FORECASTING PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info("=" * 80)

        # Memory cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    except Exception as e:
        total_time = time.time() - start_time
        logger.error("=" * 80)
        logger.error("❌ PIPELINE EXECUTION FAILED")
        logger.error("=" * 80)
        logger.error(f"Failure occurred after {total_time:.1f} seconds")
        logger.error(f"Error: {str(e)}")
        logger.exception("Detailed traceback:")
        logger.error("=" * 80)
        sys.exit(1)


if __name__ == "__main__":
    main()
