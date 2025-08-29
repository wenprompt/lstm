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
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error  # type: ignore
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
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
        logger.info(f"Prediction range: [{self.predictions.min():.4f}, {self.predictions.max():.4f}]")
        logger.info(f"Actual range: [{self.actuals.min():.4f}, {self.actuals.max():.4f}]")
        
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
        
        # Root Mean Squared Error
        rmse = np.sqrt(mean_squared_error(self.actuals, self.predictions))
        
        # Mean Absolute Error
        mae = mean_absolute_error(self.actuals, self.predictions)
        
        # Directional Accuracy (percentage of correctly predicted directions)
        directional_accuracy = self._calculate_directional_accuracy()
        
        # Additional regression metrics
        # R-squared (coefficient of determination)
        ss_res = np.sum((self.actuals - self.predictions) ** 2)
        ss_tot = np.sum((self.actuals - np.mean(self.actuals)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
        
        # Mean Squared Error (for completeness)
        mse = mean_squared_error(self.actuals, self.predictions)
        
        # Mean Absolute Percentage Error (MAPE)
        mape = self._calculate_mape()
        
        self.evaluation_metrics = {
            'rmse': float(rmse),
            'mae': float(mae),
            'mse': float(mse),
            'directional_accuracy': float(directional_accuracy),
            'r_squared': float(r_squared),
            'mape': float(mape)
        }
        
        logger.info("Evaluation metrics calculated:")
        logger.info(f"  RMSE: {rmse:.6f}")
        logger.info(f"  MAE: {mae:.6f}")
        logger.info(f"  Directional Accuracy: {directional_accuracy:.2f}%")
        logger.info(f"  R²: {r_squared:.4f}")
        logger.info(f"  MAPE: {mape:.2f}%")
        
        # Layman performance summary with configurable thresholds
        thresholds = self.config.get("thresholds", {})
        excellent_thresh = thresholds.get("excellent_accuracy", 60)
        good_thresh = thresholds.get("good_accuracy", 55) 
        fair_thresh = thresholds.get("fair_accuracy", 50)
        strong_corr = thresholds.get("strong_correlation", 0.5)
        mod_corr = thresholds.get("moderate_correlation", 0.2)
        
        accuracy_rating = "EXCELLENT" if directional_accuracy >= excellent_thresh else "GOOD" if directional_accuracy >= good_thresh else "FAIR" if directional_accuracy >= fair_thresh else "POOR"
        error_rating = "LOW" if rmse < 1.0 else "MODERATE" if rmse < 2.0 else "HIGH"
        correlation_rating = "STRONG" if r_squared > strong_corr else "MODERATE" if r_squared > mod_corr else "WEAK"
        logger.info(f"📈 Model Quality: {accuracy_rating} direction prediction ({directional_accuracy:.1f}%), {error_rating} error rate, {correlation_rating} correlation")
        
        return self.evaluation_metrics
    
    def _calculate_directional_accuracy(self) -> float:
        """
        Calculate directional accuracy for forecasting evaluation.
        
        Measures the percentage of predictions that correctly predict
        the direction of price movement (up/down) compared to actual.
        
        Returns:
            Directional accuracy as percentage (0-100)
        """
        if len(self.predictions) <= 1:
            return 0.0
        
        # Calculate direction changes for predictions and actuals
        # Positive = up, Negative = down, Zero = no change
        pred_directions = np.sign(self.predictions)
        actual_directions = np.sign(self.actuals)
        
        # Count correct directional predictions
        correct_directions = np.sum(pred_directions == actual_directions)
        total_predictions = len(self.predictions)
        
        directional_accuracy = (correct_directions / total_predictions) * 100
        
        logger.debug(f"Directional accuracy: {correct_directions}/{total_predictions} correct")
        
        return directional_accuracy
    
    def _calculate_mape(self) -> float:
        """
        Calculate Mean Absolute Percentage Error.
        
        Returns:
            MAPE as percentage
        """
        # Avoid division by zero for actual values close to zero
        non_zero_mask = np.abs(self.actuals) > 1e-6
        
        if not np.any(non_zero_mask):
            logger.warning("All actual values are near zero, MAPE calculation unreliable")
            return float('inf')
        
        filtered_actuals = self.actuals[non_zero_mask]
        filtered_predictions = self.predictions[non_zero_mask]
        
        mape = np.mean(np.abs((filtered_actuals - filtered_predictions) / filtered_actuals)) * 100
        
        return mape
    
    def create_visualizations(self, save_dir: Path = Path("results/plots")) -> Dict[str, str]:
        """
        Create comprehensive visualizations of model performance.
        
        Args:
            save_dir: Directory to save plot files
            
        Returns:
            Dictionary mapping plot names to file paths
        """
        logger.info("Creating evaluation visualizations...")
        
        # Create save directory
        save_dir.mkdir(parents=True, exist_ok=True)
        
        saved_plots = {}
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # 1. Actual vs Predicted scatter plot
        saved_plots['scatter'] = self._create_scatter_plot(save_dir)
        
        # 2. Time series plot of predictions vs actuals
        saved_plots['timeseries'] = self._create_timeseries_plot(save_dir)
        
        # 3. Residuals analysis
        saved_plots['residuals'] = self._create_residuals_plot(save_dir)
        
        # 4. Distribution comparison
        saved_plots['distributions'] = self._create_distribution_plot(save_dir)
        
        logger.info(f"Created {len(saved_plots)} visualization plots")
        
        return saved_plots
    
    def _create_scatter_plot(self, save_dir: Path) -> str:
        """Create actual vs predicted scatter plot."""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Scatter plot
        ax.scatter(self.actuals, self.predictions, alpha=0.6, s=20)
        
        # Perfect prediction line (y=x)
        min_val = min(self.actuals.min(), self.predictions.min())
        max_val = max(self.actuals.max(), self.predictions.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
        
        # Labels and title
        ax.set_xlabel('Actual Log Returns (%)', fontsize=12)
        ax.set_ylabel('Predicted Log Returns (%)', fontsize=12)
        ax.set_title('LSTM Model: Actual vs Predicted Log Returns', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add metrics text
        metrics_text = f"RMSE: {self.evaluation_metrics['rmse']:.4f}\n"
        metrics_text += f"MAE: {self.evaluation_metrics['mae']:.4f}\n"
        metrics_text += f"R²: {self.evaluation_metrics['r_squared']:.4f}"
        ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        # Save plot
        plot_path = save_dir / "actual_vs_predicted_scatter.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(plot_path)
    
    def _create_timeseries_plot(self, save_dir: Path) -> str:
        """Create time series plot of predictions vs actuals."""
        fig, ax = plt.subplots(figsize=(15, 8))
        
        # Time series indices (since we don't have actual dates in test loader)
        time_indices = range(len(self.actuals))
        
        # Plot actual and predicted values
        ax.plot(time_indices, self.actuals, label='Actual', linewidth=1.5, alpha=0.8)
        ax.plot(time_indices, self.predictions, label='Predicted', linewidth=1.5, alpha=0.8)
        
        # Labels and title
        ax.set_xlabel('Test Sample Index', fontsize=12)
        ax.set_ylabel('Log Returns (%)', fontsize=12)
        ax.set_title('LSTM Model: Time Series Comparison (Test Set)', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = save_dir / "timeseries_comparison.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(plot_path)
    
    def _create_residuals_plot(self, save_dir: Path) -> str:
        """Create residuals analysis plots."""
        residuals = self.actuals - self.predictions
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Residuals vs Predicted
        ax1.scatter(self.predictions, residuals, alpha=0.6, s=20)
        ax1.axhline(y=0, color='r', linestyle='--')
        ax1.set_xlabel('Predicted Values')
        ax1.set_ylabel('Residuals')
        ax1.set_title('Residuals vs Predicted')
        ax1.grid(True, alpha=0.3)
        
        # 2. Residuals histogram
        ax2.hist(residuals, bins=30, alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Residuals')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Residuals Distribution')
        ax2.grid(True, alpha=0.3)
        
        # 3. Q-Q plot (normal distribution check)
        from scipy import stats  # type: ignore
        stats.probplot(residuals, dist="norm", plot=ax3)
        ax3.set_title('Q-Q Plot (Normal Distribution Check)')
        ax3.grid(True, alpha=0.3)
        
        # 4. Residuals over time
        ax4.plot(residuals, linewidth=1)
        ax4.axhline(y=0, color='r', linestyle='--')
        ax4.set_xlabel('Test Sample Index')
        ax4.set_ylabel('Residuals')
        ax4.set_title('Residuals Over Time')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = save_dir / "residuals_analysis.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(plot_path)
    
    def _create_distribution_plot(self, save_dir: Path) -> str:
        """Create distribution comparison plot."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 1. Histogram comparison
        ax1.hist(self.actuals, bins=30, alpha=0.7, label='Actual', edgecolor='black')
        ax1.hist(self.predictions, bins=30, alpha=0.7, label='Predicted', edgecolor='black')
        ax1.set_xlabel('Log Returns (%)')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Distribution Comparison')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Box plot comparison
        box_data = [self.actuals, self.predictions]
        ax2.boxplot(box_data, labels=['Actual', 'Predicted'])
        ax2.set_ylabel('Log Returns (%)')
        ax2.set_title('Box Plot Comparison')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = save_dir / "distribution_comparison.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(plot_path)
    
    def save_results(self, save_dir: Path = Path("results"), 
                     additional_info: Dict[str, Any] = {}) -> str:
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
            'evaluation_timestamp': datetime.now().isoformat(),
            'test_samples': len(self.predictions),
            'metrics': self.evaluation_metrics,
            'prediction_statistics': {
                'mean_prediction': float(np.mean(self.predictions)),
                'std_prediction': float(np.std(self.predictions)),
                'min_prediction': float(np.min(self.predictions)),
                'max_prediction': float(np.max(self.predictions))
            },
            'actual_statistics': {
                'mean_actual': float(np.mean(self.actuals)),
                'std_actual': float(np.std(self.actuals)),
                'min_actual': float(np.min(self.actuals)),
                'max_actual': float(np.max(self.actuals))
            }
        }
        
        # Add additional information
        results.update(additional_info)
        
        # Save to JSON file
        results_path = save_dir / "evaluation_results.json"
        with open(results_path, 'w') as f:
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
{'='*60}
LSTM MODEL EVALUATION SUMMARY
{'='*60}
Test Samples: {len(self.predictions):,}
Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

PERFORMANCE METRICS:
├─ Root Mean Squared Error (RMSE): {self.evaluation_metrics['rmse']:.6f}
├─ Mean Absolute Error (MAE): {self.evaluation_metrics['mae']:.6f}
├─ Mean Squared Error (MSE): {self.evaluation_metrics['mse']:.6f}
├─ Directional Accuracy: {self.evaluation_metrics['directional_accuracy']:.2f}%
├─ R-squared (R²): {self.evaluation_metrics['r_squared']:.4f}
└─ Mean Absolute Percentage Error (MAPE): {self.evaluation_metrics['mape']:.2f}%

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
{'='*60}
        """
        
        return summary.strip()


def evaluate_model(model: nn.Module, test_loader: DataLoader, 
                  device: torch.device, config: Dict[str, Any], 
                  save_dir: Path = Path("results")) -> Dict[str, Any]:
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
    
    # Save results
    results_path = evaluator.save_results(save_dir, {
        'plot_paths': plot_paths,
        'model_type': 'bidirectional_lstm'
    })
    
    # Print summary
    summary = evaluator.get_evaluation_summary()
    print(summary)
    
    logger.info("Model evaluation completed successfully")
    
    return {
        'metrics': metrics,
        'predictions': predictions,
        'actuals': actuals,
        'plot_paths': plot_paths,
        'results_path': results_path,
        'summary': summary
    }