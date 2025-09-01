#!/usr/bin/env python3
"""
Model results exporter for LSTM iron ore price forecasting.

This module exports detailed model results as a DataFrame containing:
- Date: Test sample dates
- Raw 65% M+1 Price: Unadjusted iron ore futures prices for trading
- Predicted ln(returns): Model predictions (log returns)
- Actual ln(returns): Target Y values (actual log returns)

Used after model evaluation to generate detailed analysis-ready datasets.
"""

import torch
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple
import logging
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class ModelResultsExporter:
    """Export detailed model results with dates and price information."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize model results exporter.

        Args:
            config: Configuration dictionary from config.yaml
        """
        self.config = config

    def get_test_predictions(
        self, model: torch.nn.Module, test_loader: DataLoader, device: torch.device
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate predictions on test set.

        Args:
            model: Trained LSTM model
            test_loader: Test data loader
            device: Computing device (CPU/CUDA)

        Returns:
            Tuple of (predictions, actual_values)
        """
        logger.info("Generating test set predictions for export...")

        model.eval()
        predictions = []
        actuals = []

        with torch.no_grad():
            for sequences, targets in test_loader:
                # Move to device
                sequences = sequences.to(device).float()
                targets = targets.to(device).float()

                # Get predictions
                batch_predictions = model(sequences)

                # Store results
                predictions.extend(batch_predictions.cpu().numpy().flatten())
                actuals.extend(targets.cpu().numpy().flatten())

        logger.info(f"Generated {len(predictions)} test predictions")
        return np.array(predictions), np.array(actuals)

    def load_original_prices(self) -> pd.DataFrame:
        """
        Load raw 65% M+1 prices from continuous futures data.

        Returns:
            DataFrame with dates and raw prices
        """
        logger.info("Loading raw 65% M+1 prices from continuous futures...")

        # Load continuous futures data (contains both raw and adjusted prices)
        continuous_path = Path(self.config["data"]["continuous_futures"])
        df = pd.read_pickle(continuous_path)

        # Extract raw price and Y target from consolidated features for alignment
        consolidated_path = Path(self.config["data"]["consolidated_features"])
        consolidated_df = pd.read_pickle(consolidated_path)

        # Use raw price from continuous futures, Y target from consolidated features
        price_df = pd.DataFrame(
            {"raw_price_65_m1": df["raw_price_65_m1"], "Y": consolidated_df["Y"]}
        )
        price_df.index.name = "date"

        logger.info(f"Loaded {len(price_df)} raw price observations")
        return price_df

    def get_test_date_range(self, total_samples: int) -> Tuple[int, int]:
        """
        Calculate test set indices based on chronological splits.

        Args:
            total_samples: Total number of samples in dataset

        Returns:
            Tuple of (test_start_idx, test_end_idx)
        """
        train_ratio = self.config["splits"]["train_ratio"]
        val_ratio = self.config["splits"]["val_ratio"]

        # Calculate split points
        train_size = int(total_samples * train_ratio)
        val_size = int(total_samples * val_ratio)
        test_start = train_size + val_size
        test_end = total_samples

        logger.info(f"Test set spans indices {test_start} to {test_end}")
        return test_start, test_end

    def create_test_results_dataframe(
        self, model: torch.nn.Module, test_loader: DataLoader, device: torch.device
    ) -> pd.DataFrame:
        """
        Create comprehensive model results DataFrame.

        Args:
            model: Trained LSTM model
            test_loader: Test data loader
            device: Computing device

        Returns:
            DataFrame with columns: date, raw_65_m1_price, predicted_ln_returns, actual_ln_returns
        """
        logger.info("Creating comprehensive model results DataFrame...")

        # Get predictions and actuals
        predictions, actuals = self.get_test_predictions(model, test_loader, device)

        # Load original price data
        price_df = self.load_original_prices()

        # Get test date range
        total_samples = len(price_df)
        test_start, test_end = self.get_test_date_range(total_samples)

        # Extract test period data
        test_price_data = price_df.iloc[test_start:test_end].copy()

        # Account for sequence creation reducing sample count
        # Sequence creation removes (sequence_length - 1) samples from the beginning

        # Align test data with predictions (which are offset by sequence creation)
        if len(test_price_data) > len(predictions):
            # Take the last N samples to match predictions length
            test_price_data = test_price_data.iloc[-len(predictions) :].copy()
        elif len(test_price_data) < len(predictions):
            # Truncate predictions to match available data
            predictions = predictions[: len(test_price_data)]
            actuals = actuals[: len(test_price_data)]

        # Create results DataFrame
        results_df = pd.DataFrame(
            {
                "date": test_price_data.index,
                "raw_65_m1_price": test_price_data["raw_price_65_m1"].values,
                "predicted_ln_returns": predictions,
                "actual_ln_returns": actuals,
            }
        )

        # Reset index to make date a regular column
        results_df.reset_index(drop=True, inplace=True)

        # Validate data alignment
        logger.info("Validating model results alignment...")
        logger.info(f"  Test samples: {len(results_df)}")
        logger.info(
            f"  Date range: {results_df['date'].min()} to {results_df['date'].max()}"
        )
        logger.info(
            f"  Raw price range: {results_df['raw_65_m1_price'].min():.2f} to {results_df['raw_65_m1_price'].max():.2f}"
        )
        logger.info(
            f"  Predicted returns range: {results_df['predicted_ln_returns'].min():.4f} to {results_df['predicted_ln_returns'].max():.4f}"
        )
        logger.info(
            f"  Actual returns range: {results_df['actual_ln_returns'].min():.4f} to {results_df['actual_ln_returns'].max():.4f}"
        )

        return results_df

    def export_model_results(
        self,
        model: torch.nn.Module,
        test_loader: DataLoader,
        device: torch.device,
        save_dir: Path = Path("results"),
    ) -> Dict[str, Any]:
        """
        Export comprehensive model results to CSV and return metadata.

        Args:
            model: Trained LSTM model
            test_loader: Test data loader
            device: Computing device
            save_dir: Directory to save results

        Returns:
            Dictionary with export metadata and file paths
        """
        logger.info("Exporting detailed model results...")

        # Create model results DataFrame
        results_df = self.create_test_results_dataframe(model, test_loader, device)

        # Create model_results subdirectory
        model_results_dir = save_dir / "model_results"
        model_results_dir.mkdir(parents=True, exist_ok=True)

        # Save to CSV
        csv_path = model_results_dir / "model_results.csv"
        results_df.to_csv(csv_path, index=False)

        # Save to pickle for fast loading
        pkl_path = model_results_dir / "model_results.pkl"
        results_df.to_pickle(pkl_path)

        # Calculate summary statistics for export metadata
        export_metadata = {
            "export_files": {"csv": str(csv_path), "pickle": str(pkl_path)},
            "sample_count": len(results_df),
            "date_range": {
                "start": str(results_df["date"].min()),
                "end": str(results_df["date"].max()),
            },
            "price_statistics": {
                "min": float(results_df["raw_65_m1_price"].min()),
                "max": float(results_df["raw_65_m1_price"].max()),
                "mean": float(results_df["raw_65_m1_price"].mean()),
                "std": float(results_df["raw_65_m1_price"].std()),
            },
            "prediction_statistics": {
                "min": float(results_df["predicted_ln_returns"].min()),
                "max": float(results_df["predicted_ln_returns"].max()),
                "mean": float(results_df["predicted_ln_returns"].mean()),
                "std": float(results_df["predicted_ln_returns"].std()),
            },
            "actual_returns_statistics": {
                "min": float(results_df["actual_ln_returns"].min()),
                "max": float(results_df["actual_ln_returns"].max()),
                "mean": float(results_df["actual_ln_returns"].mean()),
                "std": float(results_df["actual_ln_returns"].std()),
            },
        }

        logger.info("Model results exported successfully:")
        logger.info(f"  CSV: {csv_path}")
        logger.info(f"  Pickle: {pkl_path}")
        logger.info(f"  Samples: {export_metadata['sample_count']}")
        date_range_info = export_metadata["date_range"]
        if isinstance(date_range_info, dict):
            logger.info(
                f"  Date range: {date_range_info['start']} to {date_range_info['end']}"
            )
        else:
            logger.info(f"  Date range: {date_range_info}")

        # Display sample of results
        logger.info("Sample results (first 5 rows):")
        logger.info(f"  {results_df.head().to_string(index=False)}")

        return export_metadata


def export_model_results(
    model: torch.nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    config: Dict[str, Any],
    save_dir: Path = Path("results"),
) -> Dict[str, Any]:
    """
    Factory function to export model results.

    Args:
        model: Trained LSTM model
        test_loader: Test data loader
        device: Computing device
        config: Configuration dictionary
        save_dir: Directory to save results

    Returns:
        Dictionary with export metadata
    """
    exporter = ModelResultsExporter(config)
    return exporter.export_model_results(model, test_loader, device, save_dir)
