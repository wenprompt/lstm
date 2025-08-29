#!/usr/bin/env python3
"""
LSTM model architecture for iron ore price forecasting.

This module implements a bidirectional LSTM model using verified PyTorch API:
- torch.nn.LSTM with bidirectional=True parameter
- Configurable architecture via config.yaml
- Layer normalization and dropout for regularization
- Single output prediction for next-day log returns

Based on verified PyTorch documentation from /pytorch/pytorch.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Tuple, Optional
import logging
import copy

logger = logging.getLogger(__name__)


class IronOreLSTM(nn.Module):
    """
    Bidirectional LSTM model for iron ore price forecasting.
    
    Architecture:
    - Input: sequences of 12 features over sequence_length timesteps
    - Bidirectional LSTM layers with configurable hidden_size and num_layers
    - Optional layer normalization and dropout for regularization
    - Fully connected output layer for single prediction
    
    Per PyTorch LSTM documentation:
    - bidirectional=True doubles the hidden_size output
    - Input shape: (batch, seq_len, input_size) with batch_first=True
    - Output shape: (batch, seq_len, hidden_size * 2) for bidirectional
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize LSTM model with configuration parameters.
        
        Args:
            config: Configuration dictionary containing model hyperparameters
        """
        super(IronOreLSTM, self).__init__()
        
        # Extract model configuration parameters
        self.input_size = config["model"]["input_size"]  # 12 features
        self.hidden_size = config["model"]["hidden_size"]  # e.g., 96
        self.num_layers = config["model"]["number_of_layers"]  # e.g., 2
        self.dropout_rate = config["model"]["dropout_rate"]  # e.g., 0.35
        self.bidirectional = config["model"]["bidirectional"]  # True
        self.use_layer_norm = config["model"]["layer_norm"]  # True
        
        logger.info("Initializing LSTM model:")
        logger.info(f"  Input size: {self.input_size}")
        logger.info(f"  Hidden size: {self.hidden_size}")
        logger.info(f"  Number of layers: {self.num_layers}")
        logger.info(f"  Bidirectional: {self.bidirectional}")
        logger.info(f"  Dropout rate: {self.dropout_rate}")
        logger.info(f"  Layer normalization: {self.use_layer_norm}")
        
        # Main LSTM layer - using verified PyTorch API
        self.lstm = nn.LSTM(
            input_size=self.input_size,          # 12 features
            hidden_size=self.hidden_size,        # Hidden units per direction
            num_layers=self.num_layers,          # Number of stacked LSTM layers  
            batch_first=True,                    # Input shape: (batch, seq, feature)
            dropout=self.dropout_rate if self.num_layers > 1 else 0,  # Dropout between layers
            bidirectional=self.bidirectional     # Process sequences forward and backward
        )
        
        # Calculate LSTM output size (doubles for bidirectional)
        lstm_output_size = self.hidden_size * 2 if self.bidirectional else self.hidden_size
        
        # Optional layer normalization for training stability
        self.layer_norm: Optional[nn.LayerNorm] = None
        if self.use_layer_norm:
            self.layer_norm = nn.LayerNorm(lstm_output_size)
            
        # Additional dropout layer after LSTM (separate from inter-layer dropout)
        self.dropout = nn.Dropout(self.dropout_rate)
        
        # Fully connected layer for final prediction
        # Maps from LSTM output to single value (next-day log return prediction)
        self.fc = nn.Linear(lstm_output_size, 1)
        
        # Initialize weights using Xavier/Glorot initialization for better convergence
        self._init_weights()
        
        logger.info("Model architecture:")
        logger.info(f"  LSTM output size: {lstm_output_size}")
        logger.info("  Final output: single prediction")
        
    def _init_weights(self) -> None:
        """
        Initialize model weights using Xavier/Glorot initialization.
        
        This helps with gradient flow and training stability.
        LSTM weights are initialized by default, so we focus on the FC layer.
        """
        # Initialize fully connected layer weights
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)
        
        logger.info("Model weights initialized using Xavier uniform distribution")
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the LSTM model.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)
               - batch_size: Number of sequences in batch
               - sequence_length: Number of timesteps (e.g., 20)
               - input_size: Number of features per timestep (12)
               
        Returns:
            predictions: Tensor of shape (batch_size, 1) containing predictions
                        for next-day percentage log returns
        """
        batch_size, seq_len, features = x.shape
        
        # Validate input dimensions
        if features != self.input_size:
            raise ValueError(f"Expected {self.input_size} input features, got {features}")
            
        # Pass through LSTM layers
        # lstm_out shape: (batch_size, seq_len, hidden_size * num_directions)
        # hidden states are initialized to zeros automatically
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Use the last timestep output for prediction
        # Shape: (batch_size, hidden_size * num_directions)
        last_output = lstm_out[:, -1, :]
        
        # Apply layer normalization if configured
        if self.layer_norm is not None:
            last_output = self.layer_norm(last_output)
            
        # Apply dropout for regularization
        last_output = self.dropout(last_output)
        
        # Generate final prediction through fully connected layer
        # Shape: (batch_size, 1)
        predictions = self.fc(last_output)
        
        return predictions
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get comprehensive model information for logging and debugging.
        
        Returns:
            Dictionary containing model architecture details and parameter counts
        """
        # Count trainable parameters
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        # Calculate parameter breakdown
        lstm_params = sum(p.numel() for p in self.lstm.parameters())
        fc_params = sum(p.numel() for p in self.fc.parameters())
        
        model_info = {
            "architecture": "Bidirectional LSTM" if self.bidirectional else "Unidirectional LSTM",
            "input_size": self.input_size,
            "hidden_size": self.hidden_size, 
            "num_layers": self.num_layers,
            "bidirectional": self.bidirectional,
            "dropout_rate": self.dropout_rate,
            "layer_norm": self.use_layer_norm,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "lstm_parameters": lstm_params,
            "fc_parameters": fc_params,
            "parameter_breakdown": {
                "lstm": lstm_params,
                "layer_norm": sum(p.numel() for p in self.layer_norm.parameters()) if self.layer_norm else 0,
                "fully_connected": fc_params
            }
        }
        
        return model_info
        
    def init_hidden(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Initialize hidden and cell states for LSTM.
        
        This method is useful for inference or when you need explicit control
        over the initial hidden states.
        
        Args:
            batch_size: Size of the batch
            device: Device to create tensors on (CPU or CUDA)
            
        Returns:
            Tuple of (hidden_state, cell_state) tensors
            Shape: (num_layers * num_directions, batch_size, hidden_size)
        """
        num_directions = 2 if self.bidirectional else 1
        dtype = next(self.parameters()).dtype

        hidden = torch.zeros(
            self.num_layers * num_directions,
            batch_size,
            self.hidden_size,
            device=device,
            dtype=dtype,
        )

        cell = torch.zeros(
            self.num_layers * num_directions,
            batch_size,
            self.hidden_size,
            device=device,
            dtype=dtype,
        )
        
        return hidden, cell


def create_model(config: Dict[str, Any]) -> IronOreLSTM:
    """
    Factory function to create LSTM model with configuration.
    
    Args:
        config: Configuration dictionary from config.yaml
        
    Returns:
        Initialized IronOreLSTM model ready for training
    """
    logger.info("Creating LSTM model from configuration...")

    # Work on a copy to avoid mutating input config
    model_config = copy.deepcopy(config)

    # Calculate dynamic input size based on selected features
    selected_features = model_config.get("features", [])
    if selected_features:
        dynamic_input_size = len(selected_features)
        model_config["model"]["input_size"] = dynamic_input_size
        logger.info(f"Dynamic input size calculated: {dynamic_input_size} features")
    else:
        logger.warning("No features specified in config, using default input_size from config")

    model = IronOreLSTM(model_config)
    
    # Log model information
    model_info = model.get_model_info()
    logger.info("Model created successfully:")
    logger.info(f"  Architecture: {model_info['architecture']}")
    logger.info(f"  Input features: {model_info['input_size']}")
    logger.info(f"  Total parameters: {model_info['total_parameters']:,}")
    logger.info(f"  Trainable parameters: {model_info['trainable_parameters']:,}")
    
    # Model complexity summary
    thresholds = model_config.get("thresholds", {})
    simple_threshold = thresholds.get("simple_model_params", 10000)
    moderate_threshold = thresholds.get("moderate_model_params", 100000)
    complexity = "SIMPLE" if model_info['total_parameters'] < simple_threshold else "MODERATE" if model_info['total_parameters'] < moderate_threshold else "COMPLEX"
    logger.info(f"🧠 Neural Network: {complexity} model with {model_info['total_parameters']/1000:.0f}K parameters, designed to learn iron ore price patterns")
    
    return model


def get_model_summary(model: IronOreLSTM) -> str:
    """
    Generate a formatted summary of the model architecture.
    
    Args:
        model: LSTM model instance
        
    Returns:
        Formatted string containing model summary
    """
    info = model.get_model_info()
    
    summary = f"""
{'='*50}
LSTM MODEL SUMMARY
{'='*50}
Architecture: {info['architecture']}
Input Features: {info['input_size']}
Hidden Size: {info['hidden_size']}
Layers: {info['num_layers']}
Dropout: {info['dropout_rate']:.2f}
Layer Norm: {info['layer_norm']}

Parameter Count:
- Total: {info['total_parameters']:,}
- Trainable: {info['trainable_parameters']:,}
- LSTM: {info['lstm_parameters']:,}
- FC: {info['fc_parameters']:,}
{'='*50}
    """
    
    return summary.strip()