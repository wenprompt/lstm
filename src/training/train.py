#!/usr/bin/env python3
"""
Training module for LSTM iron ore price forecasting.

This module implements:
- Training loop with early stopping and validation monitoring
- Adam optimizer with gradient clipping
- MSE loss function for regression
- Model checkpointing (save best weights)
- Progress tracking and logging

Based on verified PyTorch documentation:
- torch.optim.Adam for optimization
- torch.nn.MSELoss for regression loss
- torch.nn.utils.clip_grad_norm_ for gradient clipping
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Dict, Any, List
import logging
import time
import psutil
import gc

logger = logging.getLogger(__name__)


class EarlyStopping:
    """
    Early stopping to stop training when validation loss doesn't improve.

    Monitors validation loss and stops training if no improvement is observed
    for a specified number of epochs (patience). Saves the best model weights.
    """

    def __init__(
        self,
        patience: int = 15,
        min_delta: float = 0.0,
        restore_best_weights: bool = True,
    ):
        """
        Initialize early stopping monitor.

        Args:
            patience: Number of epochs with no improvement to wait before stopping
            min_delta: Minimum change in monitored metric to qualify as improvement
            restore_best_weights: Whether to restore best weights when stopping
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights

        self.best_loss = float("inf")
        self.counter = 0
        self.best_weights: Dict[str, Any] | None = None
        self.early_stop = False

        logger.info(
            f"Early stopping initialized: patience={patience}, min_delta={min_delta}"
        )

    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        """
        Check if training should stop based on validation loss.

        Args:
            val_loss: Current validation loss
            model: Model to potentially save weights from

        Returns:
            True if training should stop, False otherwise
        """
        # Check if validation loss improved
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0

            # Save best model weights
            if self.restore_best_weights:
                self.best_weights = {
                    k: v.cpu().clone() for k, v in model.state_dict().items()
                }

            logger.debug(f"Validation loss improved to {val_loss:.6f}")

        else:
            self.counter += 1
            logger.debug(
                f"No improvement in validation loss: {self.counter}/{self.patience}"
            )

            if self.counter >= self.patience:
                self.early_stop = True
                logger.info(
                    f"Early stopping triggered after {self.counter} epochs without improvement"
                )

                # Restore best weights if enabled
                if self.restore_best_weights and self.best_weights is not None:
                    device = next(model.parameters()).device
                    model.load_state_dict(
                        {k: v.to(device) for k, v in self.best_weights.items()}
                    )
                    logger.info(
                        f"Restored best weights with validation loss: {self.best_loss:.6f}"
                    )

        return self.early_stop


class LSTMTrainer:
    """
    Complete training pipeline for LSTM iron ore forecasting model.

    Handles model training, validation, early stopping, gradient clipping,
    and progress monitoring with comprehensive logging.
    """

    def __init__(self, model: nn.Module, config: Dict[str, Any]):
        """
        Initialize LSTM trainer with model and configuration.

        Args:
            model: LSTM model to train
            config: Configuration dictionary from config.yaml
        """
        self.model = model
        self.config = config

        # Training configuration
        self.learning_rate = config["training"]["learning_rate"]  # 0.001
        self.epochs = config["training"]["epochs"]  # 200
        self.gradient_clip_norm = config["training"]["gradient_clip_norm"]  # 1.0
        self.early_stopping_patience = config["training"]["early_stopping_patience"]

        # Device configuration
        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available() and config["device"]["use_cuda"]
            else "cpu"
        )
        self.model.to(self.device)

        # Loss function: MSE for regression (verified from PyTorch docs)
        self.criterion = nn.MSELoss()

        # Adam optimizer (verified from PyTorch docs)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=self.early_stopping_patience, restore_best_weights=True
        )

        # Training history
        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
            "epoch_times": [],
            "learning_rates": [],
        }

        # System information logging
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info(f"  GPU: {gpu_name} ({gpu_memory:.1f}GB)")
        else:
            cpu_count = psutil.cpu_count(logical=False)
            ram_gb = psutil.virtual_memory().total / 1024**3
            logger.info(f"  CPU: {cpu_count} cores, RAM: {ram_gb:.1f}GB")

        logger.info("Trainer initialized:")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Learning rate: {self.learning_rate}")
        logger.info(f"  Max epochs: {self.epochs}")
        logger.info(f"  Gradient clip norm: {self.gradient_clip_norm}")
        logger.info(f"  Early stopping patience: {self.early_stopping_patience}")

        # Model parameter count logging
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        logger.info(
            f"  Model parameters: {total_params:,} total, {trainable_params:,} trainable"
        )

    def train_epoch(self, train_loader: DataLoader) -> float:
        """
        Train model for one epoch.

        Args:
            train_loader: DataLoader for training data

        Returns:
            Average training loss for the epoch
        """
        self.model.train()  # Set model to training mode
        total_loss = 0.0
        num_batches = 0
        batch_losses = []
        gradient_norms = []

        for batch_idx, (sequences, targets) in enumerate(train_loader):
            # Move data to device and ensure float tensors
            sequences = sequences.to(self.device).float()  # (batch, seq_len, features)
            targets = targets.to(self.device).float()  # (batch,) or (batch,1)
            if targets.ndim == 1:
                targets = targets.unsqueeze(1)

            # Zero gradients (verified PyTorch pattern)
            self.optimizer.zero_grad()

            # Forward pass
            predictions = self.model(sequences)  # Shape: (batch, 1)

            # Calculate loss
            loss = self.criterion(predictions, targets)

            # Backward pass
            loss.backward()

            # Gradient clipping to prevent exploding gradients (verified PyTorch API)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.gradient_clip_norm
            )
            gradient_norms.append(grad_norm.item())

            # Update parameters
            self.optimizer.step()

            # Accumulate loss and batch statistics
            batch_loss = loss.item()
            total_loss += batch_loss
            batch_losses.append(batch_loss)
            num_batches += 1

            # Log progress at regular intervals (every 5 batches or 25% intervals)
            log_interval = max(
                1, len(train_loader) // 4
            )  # Log 4 times per epoch minimum
            if batch_idx % log_interval == 0 or batch_idx == len(train_loader) - 1:
                logger.info(
                    f"    Batch {batch_idx + 1:3d}/{len(train_loader):3d} "
                    f"| Loss: {loss.item():.6f} "
                    f"| Avg Loss: {total_loss / num_batches:.6f} "
                    f"| Progress: {100 * (batch_idx + 1) / len(train_loader):5.1f}%"
                )

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

        # Log detailed training statistics
        if batch_losses:
            min_loss = min(batch_losses)
            max_loss = max(batch_losses)
            avg_grad_norm = (
                sum(gradient_norms) / len(gradient_norms) if gradient_norms else 0.0
            )
            max_grad_norm = max(gradient_norms) if gradient_norms else 0.0

            logger.info("    Training epoch stats:")
            logger.info(f"      Loss range: {min_loss:.6f} to {max_loss:.6f}")
            logger.info(
                f"      Gradient norms: avg={avg_grad_norm:.4f}, max={max_grad_norm:.4f}"
            )

            # Memory usage logging
            if torch.cuda.is_available():
                gpu_memory_used = torch.cuda.memory_allocated(0) / 1024**3
                gpu_memory_cached = torch.cuda.memory_reserved(0) / 1024**3
                logger.info(
                    f"      GPU memory: {gpu_memory_used:.2f}GB used, {gpu_memory_cached:.2f}GB cached"
                )
            else:
                ram_usage = psutil.virtual_memory().percent
                logger.info(f"      RAM usage: {ram_usage:.1f}%")

        return avg_loss

    def validate_epoch(self, val_loader: DataLoader) -> float:
        """
        Validate model for one epoch.

        Args:
            val_loader: DataLoader for validation data

        Returns:
            Average validation loss for the epoch
        """
        self.model.eval()  # Set model to evaluation mode
        total_loss = 0.0
        num_batches = 0
        batch_losses = []
        predictions_stats = []

        # Log validation batch count for clarity
        total_val_batches = len(val_loader)
        logger.info(f"    Validating on {total_val_batches} batches...")

        # Disable gradient computation for validation
        with torch.no_grad():
            for batch_idx, (sequences, targets) in enumerate(val_loader):
                # Move data to device and ensure float tensors
                sequences = sequences.to(self.device).float()
                targets = targets.to(self.device).float()
                if targets.ndim == 1:
                    targets = targets.unsqueeze(1)

                # Forward pass
                predictions = self.model(sequences)

                # Calculate loss
                loss = self.criterion(predictions, targets)
                batch_loss = loss.item()

                # Collect prediction statistics
                with torch.no_grad():
                    pred_mean = predictions.mean().item()
                    pred_std = (
                        predictions.std().item() if predictions.numel() > 1 else 0.0
                    )
                    target_mean = targets.mean().item()
                    predictions_stats.append(
                        {
                            "pred_mean": pred_mean,
                            "pred_std": pred_std,
                            "target_mean": target_mean,
                        }
                    )

                # Accumulate loss
                total_loss += batch_loss
                batch_losses.append(batch_loss)
                num_batches += 1

                # Log validation progress (less frequent than training)
                if batch_idx == 0 or batch_idx == total_val_batches - 1:
                    logger.info(
                        f"    Val Batch {batch_idx + 1:2d}/{total_val_batches:2d} | Loss: {loss.item():.6f}"
                    )

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

        # Log detailed validation statistics
        if batch_losses and predictions_stats:
            min_loss = min(batch_losses)
            max_loss = max(batch_losses)

            # Prediction statistics across all batches
            all_pred_means = [s["pred_mean"] for s in predictions_stats]
            all_target_means = [s["target_mean"] for s in predictions_stats]
            avg_pred_mean = sum(all_pred_means) / len(all_pred_means)
            avg_target_mean = sum(all_target_means) / len(all_target_means)

            logger.info(
                f"    Validation completed: {num_batches} batches, avg loss: {avg_loss:.6f}"
            )
            logger.info("    Validation stats:")
            logger.info(f"      Loss range: {min_loss:.6f} to {max_loss:.6f}")
            logger.info(
                f"      Predictions avg: {avg_pred_mean:.6f}, targets avg: {avg_target_mean:.6f}"
            )
        else:
            logger.info(
                f"    Validation completed: {num_batches} batches, avg loss: {avg_loss:.6f}"
            )

        return avg_loss

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        save_dir: Path = Path("results/models"),
    ) -> Dict[str, Any]:
        """
        Complete training loop with early stopping and progress monitoring.

        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data (can be empty if val_ratio=0)
            save_dir: Directory to save model checkpoints

        Returns:
            Dictionary containing training history and results
        """
        logger.info("Starting LSTM model training...")
        logger.info(f"Training samples: {len(train_loader.dataset)}")  # type: ignore

        # Check if validation is enabled
        has_validation = len(val_loader.dataset) > 0  # type: ignore
        if has_validation:
            logger.info(f"Validation samples: {len(val_loader.dataset)}")  # type: ignore
        else:
            logger.info("Validation samples: 0 (validation disabled)")

        logger.info(f"Batch size: {train_loader.batch_size}")

        # Create save directory
        save_dir.mkdir(parents=True, exist_ok=True)

        # Training metrics
        best_model_path = save_dir / "best_model.pt"
        best_val_loss = float("inf")
        start_time = time.time()
        train_loss: float = float("nan")
        val_loss: float = float("nan")

        for epoch in range(self.epochs):
            epoch_start_time = time.time()

            logger.info(f"Epoch {epoch + 1}/{self.epochs} - Training Phase")

            # Training phase
            train_loss = self.train_epoch(train_loader)

            # Validation phase (only if validation data available)
            if has_validation:
                logger.info(f"Epoch {epoch + 1}/{self.epochs} - Validation Phase")
                val_loss = self.validate_epoch(val_loader)
            else:
                # Use training loss as proxy when no validation set
                val_loss = train_loss

            # Record epoch time
            epoch_time = time.time() - epoch_start_time

            # Store history
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["epoch_times"].append(epoch_time)
            self.history["learning_rates"].append(self.optimizer.param_groups[0]["lr"])

            # Calculate improvement indicators
            is_best = val_loss < best_val_loss
            improvement = (
                f"↓{best_val_loss - val_loss:+.6f}"
                if is_best
                else f"↑{val_loss - best_val_loss:+.6f}"
            )

            # Calculate training progress statistics
            progress_percent = ((epoch + 1) / self.epochs) * 100
            epochs_remaining = self.epochs - (epoch + 1)

            # Loss trend analysis (if we have history)
            if len(self.history["train_loss"]) >= 2:
                train_trend = train_loss - self.history["train_loss"][-2]
                train_trend_str = f"({train_trend:+.6f})"
            else:
                train_trend_str = "(first epoch)"

            # Log comprehensive epoch results
            logger.info(
                f"Epoch {epoch + 1}/{self.epochs} Complete [{progress_percent:.1f}% progress]:"
            )
            logger.info(f"  Train Loss: {train_loss:.6f} {train_trend_str}")
            if has_validation:
                logger.info(f"  Val Loss:   {val_loss:.6f} ({improvement})")
                logger.info(f"  Best Val:   {min(self.history['val_loss']):.6f}")
            else:
                logger.info("  Val Loss:   N/A (validation disabled)")
                logger.info("  Best Val:   N/A (validation disabled)")
            logger.info(
                f"  Time:       {epoch_time:.2f}s (est. {epochs_remaining * epoch_time:.0f}s remaining)"
            )
            logger.info(f"  LR:         {self.optimizer.param_groups[0]['lr']:.6f}")

            # Memory cleanup and logging after each epoch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gpu_memory_used = torch.cuda.memory_allocated(0) / 1024**3
                logger.info(f"  GPU Mem:    {gpu_memory_used:.2f}GB")
            else:
                gc.collect()  # Force garbage collection on CPU
                ram_usage = psutil.virtual_memory().percent
                logger.info(f"  RAM Usage:  {ram_usage:.1f}%")

            # Layman summary
            thresholds = self.config.get("thresholds", {})
            plateau_tol = thresholds.get("plateau_tolerance", 0.001)
            fast_time = thresholds.get("fast_epoch_time", 10)
            medium_time = thresholds.get("medium_epoch_time", 30)

            if has_validation:
                performance = (
                    "IMPROVING"
                    if is_best
                    else (
                        "PLATEAU"
                        if abs(val_loss - best_val_loss) < plateau_tol
                        else "DECLINING"
                    )
                )
            else:
                # Without validation, assess based on training loss trend
                if len(self.history["train_loss"]) > 1:
                    train_improving = (
                        train_loss < self.history["train_loss"][-2] - plateau_tol
                    )
                    performance = "IMPROVING" if train_improving else "PLATEAU"
                else:
                    performance = "TRAINING"

            speed = (
                "FAST"
                if epoch_time < fast_time
                else "MEDIUM" if epoch_time < medium_time else "SLOW"
            )
            validation_status = (
                "with validation monitoring"
                if has_validation
                else "without validation (using training loss)"
            )
            logger.info(
                f"  📊 Summary: Model performance is {performance}, training speed is {speed} ({validation_status})"
            )

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "config": self.config,
                    },
                    best_model_path,
                )
                logger.debug(f"Best model saved: {best_model_path}")

            # Check early stopping (use val_loss which is either validation loss or training loss)
            if self.early_stopping(val_loss, self.model):
                if has_validation:
                    logger.info(
                        f"Early stopping at epoch {epoch + 1} (validation loss stopped improving)"
                    )
                else:
                    logger.info(
                        f"Early stopping at epoch {epoch + 1} (training loss stopped improving)"
                    )
                break

        # Training completed
        total_time = time.time() - start_time
        final_epoch = len(self.history["train_loss"])

        logger.info("Training completed!")
        logger.info(f"Total epochs: {final_epoch}")
        logger.info(f"Total time: {total_time / 60:.2f} minutes")
        
        # Fixed validation logging to show N/A when validation is disabled
        if has_validation:
            logger.info(f"Best validation loss: {best_val_loss:.6f}")
            logger.info(f"Final val loss: {val_loss:.6f}")
        else:
            logger.info(f"Best train loss: {best_val_loss:.6f} (no validation)")
            logger.info("Final val loss: N/A (validation disabled)")
        
        logger.info(f"Final train loss: {train_loss:.6f}")

        # Comprehensive training completion analysis
        early_stopped_msg = (
            "stopped early (model stopped improving)"
            if self.early_stopping.early_stop
            else "completed all epochs"
        )

        # Time analysis
        thresholds = self.config.get("thresholds", {})
        quick_time = thresholds.get("quick_training_time", 300)
        moderate_time = thresholds.get("moderate_training_time", 1800)
        time_assessment = (
            "QUICK"
            if total_time < quick_time
            else "MODERATE" if total_time < moderate_time else "LENGTHY"
        )

        # Training efficiency metrics
        avg_epoch_time = total_time / final_epoch if final_epoch > 0 else 0
        epochs_per_minute = final_epoch / (total_time / 60) if total_time > 0 else 0

        # Loss improvement analysis
        if len(self.history["train_loss"]) >= 2:
            initial_train_loss = self.history["train_loss"][0]
            final_train_loss = self.history["train_loss"][-1]
            train_improvement = (
                (initial_train_loss - final_train_loss) / initial_train_loss
            ) * 100

            if has_validation and len(self.history["val_loss"]) >= 2:
                initial_val_loss = self.history["val_loss"][0]
                final_val_loss = self.history["val_loss"][-1]
                val_improvement = (
                    (initial_val_loss - final_val_loss) / initial_val_loss
                ) * 100
                logger.info(
                    f"  Loss improvement: train {train_improvement:+.1f}%, val {val_improvement:+.1f}%"
                )
            else:
                logger.info(f"  Loss improvement: train {train_improvement:+.1f}%")

        logger.info(
            f"  Training efficiency: {epochs_per_minute:.1f} epochs/min, {avg_epoch_time:.1f}s/epoch avg"
        )

        logger.info(
            f"🎯 Training Summary: {early_stopped_msg}, took {time_assessment.lower()} time ({total_time / 60:.1f} min)"
        )

        # Save final model and training history
        final_model_path = save_dir / "final_model.pt"
        torch.save(
            {
                "epoch": final_epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "training_history": self.history,
                "config": self.config,
            },
            final_model_path,
        )

        # Return training results
        return {
            "final_epoch": final_epoch,
            "total_time": total_time,
            "best_val_loss": best_val_loss,
            "final_train_loss": train_loss,
            "final_val_loss": val_loss,
            "training_history": self.history,
            "model_path": str(best_model_path),
            "early_stopped": self.early_stopping.early_stop,
        }

    def get_training_summary(self) -> str:
        """
        Generate a formatted summary of training results.

        Returns:
            Formatted string with training summary
        """
        if not self.history["train_loss"]:
            return "No training history available"

        final_epoch = len(self.history["train_loss"])
        best_val_loss = min(self.history["val_loss"])
        best_epoch = self.history["val_loss"].index(best_val_loss) + 1
        total_time = sum(self.history["epoch_times"])
        
        # Check if validation was enabled (all val_loss should be different from train_loss if validation was used)
        has_validation = not all(
            abs(train - val) < 1e-10 
            for train, val in zip(self.history["train_loss"], self.history["val_loss"])
        )

        if has_validation:
            val_loss_display = f"{best_val_loss:.6f} (Epoch {best_epoch})"
            final_val_loss_display = f"{self.history['val_loss'][-1]:.6f}"
        else:
            val_loss_display = "N/A (validation disabled)"
            final_val_loss_display = "N/A (validation disabled)"

        summary = f"""
{"=" * 50}
LSTM TRAINING SUMMARY
{"=" * 50}
Total Epochs: {final_epoch}
Best Val Loss: {val_loss_display}
Final Train Loss: {self.history["train_loss"][-1]:.6f}
Final Val Loss: {final_val_loss_display}
Training Time: {total_time / 60:.2f} minutes
Avg Time/Epoch: {total_time / final_epoch:.2f} seconds
Early Stopped: {self.early_stopping.early_stop}
Device: {self.device}
{"=" * 50}
        """

        return summary.strip()


def create_trainer(model: nn.Module, config: Dict[str, Any]) -> LSTMTrainer:
    """
    Factory function to create LSTM trainer.

    Args:
        model: LSTM model to train
        config: Configuration dictionary

    Returns:
        Initialized LSTMTrainer ready for training
    """
    logger.info("Creating LSTM trainer...")
    trainer = LSTMTrainer(model, config)
    logger.info("Trainer created successfully")
    return trainer


def load_model_checkpoint(
    model: nn.Module, checkpoint_path: str, device: torch.device
) -> Dict[str, Any]:
    """
    Load model and optimizer state from checkpoint.

    Args:
        model: Model to load weights into
        checkpoint_path: Path to checkpoint file
        device: Device to load tensors to

    Returns:
        Dictionary containing checkpoint information
    """
    logger.info(f"Loading checkpoint from {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Load model state
    model.load_state_dict(checkpoint["model_state_dict"])

    logger.info("Checkpoint loaded successfully:")
    logger.info(f"  Epoch: {checkpoint['epoch']}")
    logger.info(f"  Train Loss: {checkpoint['train_loss']:.6f}")
    logger.info(f"  Val Loss: {checkpoint['val_loss']:.6f}")

    return checkpoint
