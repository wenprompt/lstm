#!/usr/bin/env python3
"""
Hyperparameter tuning script for the LSTM model.

This script performs a grid search over a defined hyperparameter space.
For each combination of hyperparameters, it runs a short training session
and records the best validation loss.

The results are saved to a CSV file for later analysis.

Usage:
    uv run python -m src.tuning.tune_hyperparameters
"""

import yaml
import logging
import itertools
import pandas as pd
import copy
from pathlib import Path
from typing import Dict, Any, List, Iterator
import sys
import json

# Import core components from the src directory
from src.data.data_loader import DataLoader as IronOreDataLoader
from src.data.dataset import create_dataloaders
from src.models.model import create_model
from src.training.train import create_trainer

# --- Configuration ---

# Define the hyperparameter search space
# Add or remove values to expand or shrink the search
SEARCH_SPACE = {
    "learning_rate": [0.001, 0.0005],
    "gradient_clip_norm": [0.5, 0.3, 0.1],
    "hidden_size": [64, 96, 128],
    "number_of_layers": [1, 2],
    "dropout_rate": [0.2, 0.35, 0.5],
}

# Note: epochs and early_stopping_patience will be read from config.yaml
# This allows consistent configuration across main training and hyperparameter tuning

# Base configuration file
BASE_CONFIG_PATH = "config.yaml"

# Output file for results
RESULTS_CSV_PATH = Path("results/tuning_results.csv")

# --- Logging Setup ---

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stdout,  # Log to console
)
logger = logging.getLogger(__name__)

# --- Helper Functions ---


def generate_hyperparameter_combinations() -> Iterator[Dict[str, Any]]:
    """Generate all hyperparameter combinations from the search space."""
    if not SEARCH_SPACE:
        raise ValueError("SEARCH_SPACE is empty; define at least one hyperparameter.")
    keys, values = zip(*SEARCH_SPACE.items(), strict=True)
    for v in itertools.product(*values):
        yield dict(zip(keys, v, strict=True))


def update_config(
    base_config: Dict[str, Any], params: Dict[str, Any]
) -> Dict[str, Any]:
    """Update the base config with a new set of hyperparameters."""
    config = copy.deepcopy(base_config)

    # Validate required nested sections
    for section in ("training", "model"):
        if section not in config or not isinstance(config[section], dict):
            raise KeyError(f"Missing or invalid '{section}' section in base_config")

    # Update nested dictionary values
    if "learning_rate" in params:
        config["training"]["learning_rate"] = params["learning_rate"]
    if "gradient_clip_norm" in params:
        config["training"]["gradient_clip_norm"] = params["gradient_clip_norm"]
    if "hidden_size" in params:
        config["model"]["hidden_size"] = params["hidden_size"]
    if "number_of_layers" in params:
        config["model"]["number_of_layers"] = params["number_of_layers"]
    if "dropout_rate" in params:
        config["model"]["dropout_rate"] = params["dropout_rate"]

    # Use epochs and early_stopping_patience from config.yaml
    # No override needed - respects the configuration values

    return config


def run_trial(config: Dict[str, Any], test_y_actual: List[float]) -> Dict[str, Any]:
    """
    Run a single training and validation trial with comprehensive metric evaluation.

    Args:
        config: A single, complete configuration dictionary for the trial.

    Returns:
        Dictionary containing all evaluation metrics for the trial.
    """
    try:
        # 1. Load Data
        data_loader = IronOreDataLoader(config)
        train_df, val_df, test_df = data_loader.get_processed_data()

        # 2. Create Datasets
        train_loader, val_loader, _ = create_dataloaders(
            train_df, val_df, test_df, config
        )

        # 3. Create and Train Model
        model = create_model(config)
        trainer = create_trainer(model, config)
        training_results = trainer.train(train_loader, val_loader)

        # 4. Evaluate on validation/test set for comprehensive metrics
        from src.evaluation.evaluate import ModelEvaluator
        import torch

        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            and config.get("device", {}).get("use_cuda", True)
            else "cpu"
        )
        evaluator = ModelEvaluator(model, device, config)

        # Use validation set if available, otherwise use test set for evaluation
        has_validation = len(val_loader.dataset) > 0  # type: ignore
        eval_loader = val_loader if has_validation else _  # test_loader from line 110

        # We need to recreate test_loader since we used _ above
        if not has_validation:
            _, _, eval_loader = create_dataloaders(train_df, val_df, test_df, config)

        # Generate predictions on evaluation set
        _, predictions = evaluator.predict(eval_loader)
        metrics = evaluator.calculate_metrics()

        # Return comprehensive metrics (use consistent naming regardless of eval set)
        eval_set_name = "validation" if has_validation else "test"
        trial_metrics = {
            "best_val_loss": training_results.get("best_val_loss", float("inf")),
            "val_rmse": metrics.get("rmse", float("inf")),
            "val_mae": metrics.get("mae", float("inf")),
            "val_directional_accuracy": metrics.get("directional_accuracy", 0.0),
            "val_r_squared": metrics.get("r_squared", -float("inf")),
            "val_smape": metrics.get("smape", float("inf")),
            "final_epoch": training_results.get("final_epoch", 0),
            "training_time": training_results.get("training_time", 0.0),
            "eval_set_used": eval_set_name,  # Track which set was used for evaluation
            "predictions": json.dumps(predictions.tolist()),  # Add predictions to results
            "test_y_actual": json.dumps(test_y_actual),  # Add actual Y values to results
        }

        logger.info(
            f"Trial completed - Best Val Loss: {trial_metrics['best_val_loss']:.4f} "
            f"({eval_set_name} eval), "
            f"Directional Acc: {trial_metrics['val_directional_accuracy']:.1f}%, "
            f"RMSE: {trial_metrics['val_rmse']:.4f}, R²: {trial_metrics['val_r_squared']:.3f}"
        )

        return trial_metrics

    except Exception:
        logger.exception("Trial failed")
        return {
            "best_val_loss": float("inf"),
            "val_rmse": float("inf"),
            "val_mae": float("inf"),
            "val_directional_accuracy": 0.0,
            "val_r_squared": -float("inf"),
            "val_smape": float("inf"),
            "final_epoch": 0,
            "training_time": 0.0,
            "eval_set_used": "unknown",
        }
    finally:
        try:
            # Best-effort cleanup between trials
            del trainer, model
            import torch as _torch  # local import to avoid top-level dependency
            if _torch.cuda.is_available():
                _torch.cuda.empty_cache()
        except Exception:
            logger.debug("Cleanup after trial failed; continuing.")


# --- Main Execution ---


def main():
    """Main function to run the hyperparameter tuning grid search."""
    logger.info("🚀 Starting hyperparameter tuning script...")

    # Ensure results directory exists
    RESULTS_CSV_PATH.parent.mkdir(exist_ok=True)

    # Load base configuration
    try:
        with open(BASE_CONFIG_PATH, "r") as f:
            base_config = yaml.safe_load(f)
    except FileNotFoundError:
        logger.error(f"Base configuration file not found at {BASE_CONFIG_PATH}")
        return
    except yaml.YAMLError as e:
        logger.error(f"Error parsing base configuration file: {e}")
        return

    # Load data once for all trials
    data_loader = IronOreDataLoader(base_config)
    _, _, test_df = data_loader.get_processed_data()
    if test_df.empty:
        logger.error("Test set is empty. Cannot perform tuning.")
        return
    test_y_actual = test_df['Y'].tolist() # Get actual Y values once

    # Generate all combinations
    param_combinations = list(generate_hyperparameter_combinations())
    total_trials = len(param_combinations)
    logger.info(f"Generated {total_trials} hyperparameter combinations to test.")

    # Store results
    results_data: List[Dict[str, Any]] = []

    # Run all trials
    for i, params in enumerate(param_combinations):
        logger.info("=" * 60)
        logger.info(f"TRIAL {i + 1}/{total_trials}")
        logger.info(f"Parameters: {params}")
        logger.info("=" * 60)

        # Create a specific config for this trial
        trial_config = update_config(base_config, params)

        # Run the trial, passing the actual Y values
        trial_metrics = run_trial(trial_config, test_y_actual)
        logger.info(f"TRIAL {i + 1}/{total_trials} COMPLETED.")

        # Record results
        trial_result = params.copy()
        trial_result.update(trial_metrics)
        results_data.append(trial_result)

    # Process and save results
    logger.info("=" * 60)
    logger.info("🎉 Hyperparameter tuning complete!")

    if not results_data:
        logger.warning("No trials were completed successfully.")
        return

    # Create DataFrame with all metrics
    results_df = pd.DataFrame(results_data)

    # Display results for multiple metrics
    logger.info("=" * 80)
    logger.info("📊 HYPERPARAMETER TUNING RESULTS SUMMARY")
    logger.info("=" * 80)

    # Top 5 by Directional Accuracy (most important for forecasting)
    logger.info("\n🎯 TOP 5 BY DIRECTIONAL ACCURACY (Most Important for Trading):")
    top_by_accuracy = results_df.nlargest(5, "val_directional_accuracy")
    print(
        top_by_accuracy[
            [
                "learning_rate",
                "hidden_size",
                "dropout_rate",
                "val_directional_accuracy",
                "val_rmse",
                "val_r_squared",
            ]
        ].to_string(index=False)
    )

    # Top 5 by RMSE (lowest error)
    logger.info("\n📉 TOP 5 BY RMSE (Lowest Prediction Error):")
    top_by_rmse = results_df.nsmallest(5, "val_rmse")
    print(
        top_by_rmse[
            [
                "learning_rate",
                "hidden_size",
                "dropout_rate",
                "val_rmse",
                "val_directional_accuracy",
                "val_r_squared",
            ]
        ].to_string(index=False)
    )

    # Top 5 by R² (best explained variance)
    logger.info("\n🔄 TOP 5 BY R² (Best Explained Variance):")
    top_by_r2 = results_df.nlargest(5, "val_r_squared")
    print(
        top_by_r2[
            [
                "learning_rate",
                "hidden_size",
                "dropout_rate",
                "val_r_squared",
                "val_directional_accuracy",
                "val_rmse",
            ]
        ].to_string(index=False)
    )

    # Overall ranking (composite score)
    logger.info("\n🏆 TOP 5 BY COMPOSITE SCORE (Balanced Performance):")
    # Normalize metrics and create composite score
    # Higher directional accuracy = better, Lower RMSE = better, Higher R² = better

    # Add epsilon to prevent division by zero when all values are the same
    eps = 1e-8

    accuracy_range = (
        results_df["val_directional_accuracy"].max()
        - results_df["val_directional_accuracy"].min()
    )
    if accuracy_range < eps:
        results_df["normalized_accuracy"] = 1.0  # If all same, give max score
    else:
        results_df["normalized_accuracy"] = (
            results_df["val_directional_accuracy"]
            - results_df["val_directional_accuracy"].min()
        ) / accuracy_range

    rmse_range = results_df["val_rmse"].max() - results_df["val_rmse"].min()
    if rmse_range < eps:
        results_df["normalized_rmse"] = 1.0  # If all same, give max score
    else:
        results_df["normalized_rmse"] = 1 - (
            (results_df["val_rmse"] - results_df["val_rmse"].min()) / rmse_range
        )

    r2_range = results_df["val_r_squared"].max() - results_df["val_r_squared"].min()
    if r2_range < eps:
        results_df["normalized_r2"] = 1.0  # If all same, give max score
    else:
        results_df["normalized_r2"] = (
            results_df["val_r_squared"] - results_df["val_r_squared"].min()
        ) / r2_range

    # Weighted composite score (directional accuracy most important)
    results_df["composite_score"] = (
        0.5 * results_df["normalized_accuracy"]
        + 0.3 * results_df["normalized_rmse"]
        + 0.2 * results_df["normalized_r2"]
    )

    top_composite = results_df.nlargest(5, "composite_score")
    print(
        top_composite[
            [
                "learning_rate",
                "hidden_size",
                "dropout_rate",
                "composite_score",
                "val_directional_accuracy",
                "val_rmse",
                "val_r_squared",
            ]
        ].to_string(index=False)
    )

    # Sort by composite score for CSV output
    results_df = results_df.sort_values(by="composite_score", ascending=False)

    # Save to CSV
    results_df.to_csv(RESULTS_CSV_PATH, index=False)
    logger.info(f"Full tuning results saved to {RESULTS_CSV_PATH}")


if __name__ == "__main__":
    main()
