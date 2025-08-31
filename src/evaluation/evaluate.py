#!/usr/bin/env python3
"""
Evaluation and visualization module for LSTM iron ore forecasting.

This module implements:
- Performance metrics: RMSE, MAE, Directional Accuracy
- Visualization of actual vs predicted returns
- Model evaluation pipeline with test data
- Results saving and reporting

Based on verified scikit-learn metrics:
- mean_squared_error and mean_absolute_error from sklearn.metrics
- Custom directional accuracy calculation for forecasting evaluation
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error  # type: ignore
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Dict, Any, List, Tuple
import logging
import json
from datetime import datetime

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Comprehensive evaluation class for LSTM forecasting model.

    Provides metrics calculation, visualization, and result analysis
    for time series forecasting performance assessment.
    """

    def __init__(self, model: nn.Module, device: torch.device, config: Dict[str, Any]):
        """
        Initialize model evaluator.

        Args:
            model: Trained LSTM model to evaluate
            device: Device to run evaluation on (CPU or CUDA)
            config: Configuration dictionary for thresholds
        """
        self.model = model
        self.device = device
        self.config = config
        self.model.to(self.device)

        # Evaluation results storage
        self.predictions: np.ndarray = np.array([])
        self.actuals: np.ndarray = np.array([])
        self.evaluation_metrics: Dict[str, float] = {}

        logger.info(f"Evaluator initialized with device: {self.device}")

    def predict(self, test_loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate predictions on test dataset.

        Args:
            test_loader: DataLoader containing test sequences

        Returns:
            Tuple of (predictions, actual_values) as numpy arrays
        """
        logger.info("Generating predictions on test data...")

        self.model.eval()  # Set model to evaluation mode
        predictions_list: List[float] = []
        actuals_list: List[float] = []

        with torch.no_grad():  # Disable gradient computation for inference
            for sequences, targets in test_loader:
                # Move data to device
                sequences = sequences.to(self.device)
                targets = targets.to(self.device)

                # Forward pass
                batch_predictions = self.model(sequences)

                # Convert to numpy and store
                predictions_list.extend(batch_predictions.cpu().numpy().flatten())
                actuals_list.extend(targets.cpu().numpy().flatten())

        # Convert to numpy arrays
        self.predictions = np.array(predictions_list)
        self.actuals = np.array(actuals_list)

        logger.info(f"Generated {len(self.predictions)} predictions")
        logger.info(
            f"Prediction range: [{self.predictions.min():.4f}, {self.predictions.max():.4f}]"
        )
        logger.info(
            f"Actual range: [{self.actuals.min():.4f}, {self.actuals.max():.4f}]"
        )

        return self.predictions, self.actuals

    def calculate_metrics(self) -> Dict[str, float]:
        """
        Calculate comprehensive evaluation metrics.

        Returns:
            Dictionary containing all evaluation metrics
        """
        if len(self.predictions) == 0 or len(self.actuals) == 0:
            raise ValueError("No predictions available. Run predict() first.")

        logger.info("Calculating evaluation metrics...")

        rmse = np.sqrt(mean_squared_error(self.actuals, self.predictions))
        mae = mean_absolute_error(self.actuals, self.predictions)
        directional_accuracy = self._calculate_directional_accuracy()
        ss_res = np.sum((self.actuals - self.predictions) ** 2)
        ss_tot = np.sum((self.actuals - np.mean(self.actuals)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
        mse = mean_squared_error(self.actuals, self.predictions)
        mape = self._calculate_mape()
        smape = self._calculate_smape()

        self.evaluation_metrics = {
            "rmse": float(rmse),
            "mae": float(mae),
            "mse": float(mse),
            "directional_accuracy": float(directional_accuracy),
            "r_squared": float(r_squared),
            "mape": float(mape),
            "smape": float(smape),
        }

        logger.info("Evaluation metrics calculated:")
        logger.info(f"  RMSE: {rmse:.6f}")
        logger.info(f"  MAE: {mae:.6f}")
        logger.info(f"  Directional Accuracy: {directional_accuracy:.2f}%")
        logger.info(f"  R²: {r_squared:.4f}")
        logger.info(f"  MAPE (robust): {mape:.2f}%")
        logger.info(f"  SMAPE: {smape:.2f}%")

        thresholds = self.config.get("thresholds", {})
        excellent_thresh = thresholds.get("excellent_accuracy", 60)
        good_thresh = thresholds.get("good_accuracy", 55)
        fair_thresh = thresholds.get("fair_accuracy", 50)
        strong_corr = thresholds.get("strong_correlation", 0.5)
        mod_corr = thresholds.get("moderate_correlation", 0.2)

        accuracy_rating = (
            "EXCELLENT"
            if directional_accuracy >= excellent_thresh
            else "GOOD"
            if directional_accuracy >= good_thresh
            else "FAIR"
            if directional_accuracy >= fair_thresh
            else "POOR"
        )
        error_rating = "LOW" if rmse < 1.0 else "MODERATE" if rmse < 2.0 else "HIGH"
        correlation_rating = (
            "STRONG"
            if r_squared > strong_corr
            else "MODERATE"
            if r_squared > mod_corr
            else "WEAK"
        )
        logger.info(
            f"📈 Model Quality: {accuracy_rating} direction prediction ({directional_accuracy:.1f}%), {error_rating} error rate, {correlation_rating} correlation"
        )

        return self.evaluation_metrics

    def _calculate_directional_accuracy(self) -> float:
        """
        Calculate directional accuracy for forecasting evaluation.

        Returns:
            Directional accuracy as percentage (0-100)
        """
        if len(self.predictions) <= 1:
            return 0.0

        pred_directions = np.sign(self.predictions)
        actual_directions = np.sign(self.actuals)

        correct_directions = np.sum(pred_directions == actual_directions)
        total_predictions = len(self.predictions)

        directional_accuracy = (correct_directions / total_predictions) * 100
        logger.debug(
            f"Directional accuracy: {correct_directions}/{total_predictions} correct"
        )
        return directional_accuracy

    def _calculate_mape(self, eps: float = 1e-6) -> float:
        """
        Calculate robust Mean Absolute Percentage Error.

        Args:
            eps: Epsilon value to clamp the denominator and avoid division by zero.

        Returns:
            MAPE as a percentage.
        """
        denom = np.maximum(np.abs(self.actuals), eps)
        return float(np.mean(np.abs((self.actuals - self.predictions) / denom)) * 100)

    def _calculate_smape(self, eps: float = 1e-6) -> float:
        """
        Calculate Symmetric Mean Absolute Percentage Error.

        Args:
            eps: Epsilon value to avoid division by zero.

        Returns:
            SMAPE as a percentage.
        """
        denom = np.maximum((np.abs(self.actuals) + np.abs(self.predictions)) / 2.0, eps)
        return float(np.mean(np.abs(self.predictions - self.actuals) / denom) * 100)

    def create_visualizations(
        self, save_dir: Path = Path("results/plots")
    ) -> Dict[str, str]:
        """
        Create comprehensive, well-labeled visualizations of model performance.

        Args:
            save_dir: Directory to save plot files

        Returns:
            Dictionary mapping plot names to file paths
        """
        logger.info("Creating evaluation visualizations with enhanced labeling...")

        save_dir.mkdir(parents=True, exist_ok=True)
        saved_plots = {}

        plt.style.use("seaborn-v0_8-whitegrid")
        palette: List[Any] = sns.color_palette("husl", 8)

        saved_plots["scatter"] = self._create_scatter_plot(save_dir, palette)
        saved_plots["timeseries"] = self._create_timeseries_plot(save_dir, palette)
        saved_plots["residuals"] = self._create_residuals_plot(save_dir, palette)
        saved_plots["distributions"] = self._create_distribution_plot(save_dir, palette)

        logger.info(f"Created {len(saved_plots)} visualization plots")
        return saved_plots

    def _create_scatter_plot(self, save_dir: Path, palette: List[Any]) -> str:
        """Create an enhanced actual vs. predicted scatter plot."""
        fig, ax = plt.subplots(figsize=(10, 8))

        ax.scatter(self.actuals, self.predictions, alpha=0.7, s=30, c=[palette[0]])

        min_val = min(self.actuals.min(), self.predictions.min())
        max_val = max(self.actuals.max(), self.predictions.max())
        ax.plot(
            [min_val, max_val],
            [min_val, max_val],
            color=palette[1],
            linestyle="--",
            lw=2,
            label="Perfect Prediction",
        )

        ax.set_xlabel("Actual Log Returns (%)", fontsize=12, fontweight="bold")
        ax.set_ylabel("Predicted Log Returns (%)", fontsize=12, fontweight="bold")
        ax.set_title(
            "Prediction Accuracy: Actual vs. Predicted Returns",
            fontsize=16,
            fontweight="bold",
            pad=20,
        )
        fig.suptitle(
            "Points closer to the red dashed line indicate higher prediction accuracy.",
            fontsize=10,
            y=0.92,
        )

        ax.legend()
        ax.grid(True, which="both", linestyle="--", linewidth=0.5)

        metrics_text = (
            f"RMSE: {self.evaluation_metrics['rmse']:.4f}\n"
            f"MAE: {self.evaluation_metrics['mae']:.4f}\n"
            f"R²: {self.evaluation_metrics['r_squared']:.4f}"
        )
        ax.text(
            0.05,
            0.95,
            metrics_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

        plt.tight_layout(rect=(0, 0, 1, 0.96))
        plot_path = save_dir / "actual_vs_predicted_scatter.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        return str(plot_path)

    def _create_timeseries_plot(self, save_dir: Path, palette: List[Any]) -> str:
        """Create an enhanced time series plot with error visualization."""
        fig, ax = plt.subplots(figsize=(15, 8))

        time_indices = range(len(self.actuals))

        ax.plot(
            time_indices,
            self.actuals,
            color=palette[0],
            label="Actual Returns",
            linewidth=2,
            alpha=0.8,
        )
        ax.plot(
            time_indices,
            self.predictions,
            color=palette[1],
            label="Predicted Returns",
            linewidth=2,
            linestyle="--",
        )

        ax.fill_between(
            time_indices,
            self.actuals,
            self.predictions,
            color=palette[2],
            alpha=0.2,
            label="Prediction Error",
        )

        ax.set_xlabel("Test Sample Index", fontsize=12, fontweight="bold")
        ax.set_ylabel("Log Returns (%)", fontsize=12, fontweight="bold")
        ax.set_title(
            "Prediction Over Time: Actual vs. Predicted Returns",
            fontsize=16,
            fontweight="bold",
            pad=20,
        )
        fig.suptitle(
            "Compares model predictions against actuals over the test set. Shaded area shows error magnitude.",
            fontsize=10,
            y=0.92,
        )

        ax.legend(loc="upper left")
        ax.grid(True, which="both", linestyle="--", linewidth=0.5)
        ax.axhline(0, color="black", linewidth=0.5, linestyle="-")

        plt.tight_layout(rect=(0, 0, 1, 0.96))
        plot_path = save_dir / "timeseries_comparison.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        return str(plot_path)

    def _create_residuals_plot(self, save_dir: Path, palette: List[Any]) -> str:
        """Create an enhanced 4-in-1 residuals analysis plot for model diagnostics."""
        residuals = self.actuals - self.predictions
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(
            "Model Diagnostics: Residuals Analysis", fontsize=18, fontweight="bold"
        )

        # 1. Residuals vs Predicted
        ax1 = axes[0, 0]
        sns.scatterplot(
            x=self.predictions, y=residuals, ax=ax1, color=palette[0], alpha=0.7
        )
        ax1.axhline(y=0, color=palette[1], linestyle="--")
        ax1.set_title(
            "1. Residuals vs. Predicted Values", fontsize=12, fontweight="bold"
        )
        ax1.set_xlabel("Predicted Values")
        ax1.set_ylabel("Residuals (Actual - Predicted)")
        ax1.text(
            0.95,
            0.01,
            "Ideal: Random scatter around y=0",
            ha="right",
            va="bottom",
            transform=ax1.transAxes,
            fontsize=9,
            style="italic",
        )

        # 2. Residuals Distribution
        ax2 = axes[0, 1]
        sns.histplot(residuals, kde=True, ax=ax2, color=palette[2], bins=30)
        ax2.axvline(
            residuals.mean(),
            color=palette[1],
            linestyle="--",
            label=f"Mean: {residuals.mean():.3f}",
        )
        ax2.set_title("2. Distribution of Residuals", fontsize=12, fontweight="bold")
        ax2.set_xlabel("Residual Value")
        ax2.set_ylabel("Frequency")
        ax2.legend()
        ax2.text(
            0.95,
            0.01,
            "Ideal: Normal distribution centered at 0",
            ha="right",
            va="bottom",
            transform=ax2.transAxes,
            fontsize=9,
            style="italic",
        )

        # 3. Q-Q plot
        ax3 = axes[1, 0]
        from scipy import stats  # type: ignore

        stats.probplot(residuals, dist="norm", plot=ax3)
        ax3.get_lines()[0].set_markerfacecolor(palette[0])
        ax3.get_lines()[0].set_markeredgecolor(palette[0])
        ax3.get_lines()[1].set_color(palette[1])
        ax3.set_title(
            "3. Normality Check: Q-Q Plot of Residuals", fontsize=12, fontweight="bold"
        )
        ax3.text(
            0.95,
            0.01,
            "Ideal: Points fall along the red line",
            ha="right",
            va="bottom",
            transform=ax3.transAxes,
            fontsize=9,
            style="italic",
        )

        # 4. Residuals over time
        ax4 = axes[1, 1]
        ax4.plot(residuals, color=palette[3], lw=1.5)
        ax4.axhline(y=0, color=palette[1], linestyle="--")
        ax4.set_title(
            "4. Residuals vs. Observation Order", fontsize=12, fontweight="bold"
        )
        ax4.set_xlabel("Test Sample Index")
        ax4.set_ylabel("Residuals")
        ax4.text(
            0.95,
            0.01,
            "Ideal: No clear patterns or trends",
            ha="right",
            va="bottom",
            transform=ax4.transAxes,
            fontsize=9,
            style="italic",
        )

        plt.tight_layout(rect=(0, 0.03, 1, 0.95))
        plot_path = save_dir / "residuals_analysis.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        return str(plot_path)

    def _create_distribution_plot(self, save_dir: Path, palette: List[Any]) -> str:
        """Create an enhanced distribution comparison plot using KDE and Box plots."""
        fig, axes = plt.subplots(1, 2, figsize=(15, 7))
        fig.suptitle(
            "Statistical Comparison: Actual vs. Predicted Distributions",
            fontsize=16,
            fontweight="bold",
        )

        # 1. Density Plot (KDE)
        ax1 = axes[0]
        sns.kdeplot(self.actuals, ax=ax1, color=palette[0], label="Actual", fill=True)
        sns.kdeplot(
            self.predictions, ax=ax1, color=palette[1], label="Predicted", fill=True
        )
        ax1.axvline(
            self.actuals.mean(),
            color=palette[0],
            linestyle="--",
            lw=1.5,
            label=f"Actual Mean: {self.actuals.mean():.3f}",
        )
        ax1.axvline(
            self.predictions.mean(),
            color=palette[1],
            linestyle="--",
            lw=1.5,
            label=f"Predicted Mean: {self.predictions.mean():.3f}",
        )
        ax1.set_title(
            "Density Plot of Actual vs. Predicted Values",
            fontsize=12,
            fontweight="bold",
        )
        ax1.set_xlabel("Log Returns (%)")
        ax1.set_ylabel("Density")
        ax1.legend()
        ax1.text(
            0.95,
            0.01,
            "Ideal: Distributions are very similar",
            ha="right",
            va="bottom",
            transform=ax1.transAxes,
            fontsize=9,
            style="italic",
        )

        # 2. Box Plot
        ax2 = axes[1]
        box_data = [self.actuals, self.predictions]
        sns.boxplot(data=box_data, ax=ax2, palette=[palette[0], palette[1]])
        ax2.set_xticklabels(["Actual", "Predicted"])
        ax2.set_title(
            "Box Plot of Actual vs. Predicted Values", fontsize=12, fontweight="bold"
        )
        ax2.set_ylabel("Log Returns (%)")
        ax2.text(
            0.95,
            0.01,
            "Ideal: Boxes have similar median, size, and whisker length",
            ha="right",
            va="bottom",
            transform=ax2.transAxes,
            fontsize=9,
            style="italic",
        )

        plt.tight_layout(rect=(0, 0.03, 1, 0.95))
        plot_path = save_dir / "distribution_comparison.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        return str(plot_path)

    def save_results(
        self,
        save_dir: Path = Path("results"),
        additional_info: Dict[str, Any] | None = None,
    ) -> str:
        """
        Save evaluation results to JSON file.

        Args:
            save_dir: Directory to save results
            additional_info: Additional information to include in results

        Returns:
            Path to saved results file
        """
        logger.info("Saving evaluation results...")

        # Create save directory
        save_dir.mkdir(parents=True, exist_ok=True)

        # Prepare results dictionary
        results = {
            "evaluation_timestamp": datetime.now().isoformat(),
            "test_samples": len(self.predictions),
            "metrics": self.evaluation_metrics,
            "prediction_statistics": {
                "mean_prediction": float(np.mean(self.predictions)),
                "std_prediction": float(np.std(self.predictions)),
                "min_prediction": float(np.min(self.predictions)),
                "max_prediction": float(np.max(self.predictions)),
            },
            "actual_statistics": {
                "mean_actual": float(np.mean(self.actuals)),
                "std_actual": float(np.std(self.actuals)),
                "min_actual": float(np.min(self.actuals)),
                "max_actual": float(np.max(self.actuals)),
            },
        }

        # Add additional information
        results.update(additional_info or {})

        # Save to JSON file
        results_path = save_dir / "evaluation_results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

        logger.info(f"Results saved to: {results_path}")

        return str(results_path)

    def get_evaluation_summary(self) -> str:
        """
        Generate a formatted summary of evaluation results.

        Returns:
            Formatted string with evaluation summary
        """
        if not self.evaluation_metrics:
            return "No evaluation results available. Run evaluation first."

        summary = f"""
{"=" * 60}
LSTM MODEL EVALUATION SUMMARY
{"=" * 60}
Test Samples: {len(self.predictions):,}
Evaluation Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

PERFORMANCE METRICS:
├─ Root Mean Squared Error (RMSE): {self.evaluation_metrics["rmse"]:.6f}
├─ Mean Absolute Error (MAE): {self.evaluation_metrics["mae"]:.6f}
├─ Mean Squared Error (MSE): {self.evaluation_metrics["mse"]:.6f}
├─ Directional Accuracy: {self.evaluation_metrics["directional_accuracy"]:.2f}%
├─ R-squared (R²): {self.evaluation_metrics["r_squared"]:.4f}
├─ MAPE (robust): {self.evaluation_metrics["mape"]:.2f}%
└─ SMAPE: {self.evaluation_metrics["smape"]:.2f}%

PREDICTION STATISTICS:
├─ Mean: {np.mean(self.predictions):.4f}
├─ Std Dev: {np.std(self.predictions):.4f}
├─ Min: {np.min(self.predictions):.4f}
└─ Max: {np.max(self.predictions):.4f}

ACTUAL STATISTICS:
├─ Mean: {np.mean(self.actuals):.4f}
├─ Std Dev: {np.std(self.actuals):.4f}
├─ Min: {np.min(self.actuals):.4f}
└─ Max: {np.max(self.actuals):.4f}
{"=" * 60}
        """

        return summary.strip()


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    config: Dict[str, Any],
    save_dir: Path = Path("results"),
) -> Dict[str, Any]:
    """
    Complete model evaluation pipeline.

    Args:
        model: Trained LSTM model to evaluate
        test_loader: DataLoader with test data
        device: Device for computation
        save_dir: Directory to save results
        config: Configuration dictionary for thresholds

    Returns:
        Dictionary containing evaluation results and file paths
    """
    logger.info("Starting complete model evaluation...")

    # Create evaluator
    evaluator = ModelEvaluator(model, device, config)

    # Generate predictions
    predictions, actuals = evaluator.predict(test_loader)

    # Calculate metrics
    metrics = evaluator.calculate_metrics()

    # Create visualizations
    plot_paths = evaluator.create_visualizations(save_dir / "plots")

    # Print summary
    summary = evaluator.get_evaluation_summary()
    print(summary)

    logger.info("Model evaluation completed successfully")

    # Return comprehensive evaluation data (no separate JSON file created)
    return {
        "metrics": metrics,
        "predictions": predictions,
        "actuals": actuals,
        "plot_paths": plot_paths,
        "summary": summary,
        # Include detailed statistics for final_results.json
        "test_samples": len(predictions),
        "prediction_statistics": {
            "mean_prediction": float(np.mean(predictions)),
            "std_prediction": float(np.std(predictions)),
            "min_prediction": float(np.min(predictions)),
            "max_prediction": float(np.max(predictions)),
        },
        "actual_statistics": {
            "mean_actual": float(np.mean(actuals)),
            "std_actual": float(np.std(actuals)),
            "min_actual": float(np.min(actuals)),
            "max_actual": float(np.max(actuals)),
        },
        "model_type": "bidirectional_lstm",
    }
