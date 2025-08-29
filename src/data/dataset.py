#!/usr/bin/env python3
"""
PyTorch Dataset implementation for LSTM time series forecasting.

This module creates PyTorch Dataset and DataLoader classes for:
- Converting pandas DataFrames to PyTorch tensors
- Creating sliding window sequences for LSTM input
- Proper batching and data loading for training

Based on verified PyTorch official documentation:
- torch.utils.data.Dataset: map-style dataset with __getitem__ and __len__
- torch.utils.data.DataLoader: wraps Dataset for batch loading
"""

import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Dict, Any
import logging

logger = logging.getLogger(__name__)


class LSTMTimeSeriesDataset(Dataset):
    """
    PyTorch Dataset for LSTM time series forecasting.
    
    Creates sliding window sequences from time series data.
    Each sequence contains sequence_length timesteps of features (X)
    and the corresponding target value (Y) for forecasting.
    
    Implements map-style Dataset per PyTorch documentation:
    - __getitem__: supports key-based data retrieval 
    - __len__: defines the dataset size
    """
    
    def __init__(self, dataframe: pd.DataFrame, sequence_length: int):
        """
        Initialize LSTM time series dataset.
        
        Args:
            dataframe: DataFrame with features and Y target column
            sequence_length: Number of timesteps in each sequence (from config)
        """
        self.data = dataframe
        self.sequence_length = sequence_length
        
        # Separate features (X) from target (Y)
        # All columns except 'Y' are features
        self.feature_columns = [col for col in dataframe.columns if col != 'Y']
        
        # Convert to numpy arrays for faster indexing
        self.features = dataframe[self.feature_columns].values.astype(np.float32)
        self.targets = dataframe['Y'].values.astype(np.float32)
        
        # Calculate number of valid sequences
        # We need sequence_length points for features + 1 point for target
        self.num_samples = len(dataframe) - sequence_length
        
        logger.info(f"Dataset initialized: {self.num_samples} sequences")
        logger.info(f"Feature shape: {self.features.shape}")
        logger.info(f"Target shape: {self.targets.shape}")
        logger.info(f"Sequence length: {sequence_length}")
        
    def __len__(self) -> int:
        """
        Return the total number of sequences in dataset.
        
        Required method for PyTorch Dataset as per official documentation.
        
        Returns:
            Number of valid sequences that can be created
        """
        return self.num_samples
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single sequence and its target.
        
        Required method for PyTorch Dataset as per official documentation.
        Creates sliding window: uses features from [idx:idx+sequence_length]
        to predict target at [idx+sequence_length].
        
        Args:
            idx: Index of the sequence to retrieve
            
        Returns:
            Tuple of (sequence_features, target_value)
            - sequence_features: (sequence_length, num_features) 
            - target_value: scalar target for prediction
        """
        if idx < 0 or idx >= self.num_samples:
            raise IndexError(f"Index {idx} out of range [0, {self.num_samples})")
            
        # Get sequence of features: [idx:idx+sequence_length]
        # Shape: (sequence_length, num_features)
        sequence_features = self.features[idx:idx + self.sequence_length]
        
        # Get target value at the end of sequence: [idx+sequence_length] 
        # This is the value we want to predict given the sequence
        target_value = self.targets[idx + self.sequence_length]
        
        # Convert to PyTorch tensors
        sequence_tensor = torch.from_numpy(sequence_features)
        target_tensor = torch.tensor(target_value)
        
        return sequence_tensor, target_tensor


def create_dataloaders(train_df: pd.DataFrame, val_df: pd.DataFrame, 
                      test_df: pd.DataFrame, config: Dict[str, Any]) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create PyTorch DataLoaders for train/validation/test datasets.
    
    Uses verified PyTorch DataLoader parameters from official documentation:
    - batch_size: number of samples per batch
    - shuffle: reshuffles data at every epoch (True for train, False for val/test)
    - num_workers: subprocesses for data loading (0 for main process only)
    - drop_last: drops last incomplete batch if True
    
    Args:
        train_df: Training dataset DataFrame
        val_df: Validation dataset DataFrame  
        test_df: Test dataset DataFrame
        config: Configuration dictionary
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    logger.info("Creating PyTorch DataLoaders...")
    
    # Get parameters from config
    sequence_length = config["model"]["sequence_length"]
    batch_size = config["training"]["batch_size"]
    num_workers = config.get("dataloader", {}).get("num_workers", 0)
    
    # Create Dataset instances for each split
    train_dataset = LSTMTimeSeriesDataset(train_df, sequence_length)
    val_dataset = LSTMTimeSeriesDataset(val_df, sequence_length)  
    test_dataset = LSTMTimeSeriesDataset(test_df, sequence_length)
    
    # Create DataLoaders with appropriate settings
    # Train: shuffle=True to randomize batches each epoch
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,  # Randomize training batches
        num_workers=num_workers,  # Configurable worker processes
        drop_last=False  # Keep all samples
    )
    
    # Validation: shuffle=False to maintain consistent evaluation
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,  # Keep order for consistent validation
        num_workers=num_workers,
        drop_last=False
    )
    
    # Test: shuffle=False to maintain temporal order for analysis
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,  # Keep chronological order for testing
        num_workers=num_workers,
        drop_last=False
    )
    
    logger.info(f"Train DataLoader: {len(train_loader)} batches ({len(train_dataset)} sequences)")
    logger.info(f"Val DataLoader: {len(val_loader)} batches ({len(val_dataset)} sequences)")
    logger.info(f"Test DataLoader: {len(test_loader)} batches ({len(test_dataset)} sequences)")
    
    return train_loader, val_loader, test_loader


def get_data_info(dataloader: DataLoader) -> Dict[str, Any]:
    """
    Get information about a DataLoader for debugging/validation.
    
    Args:
        dataloader: PyTorch DataLoader to inspect
        
    Returns:
        Dictionary with DataLoader statistics and sample shapes
    """
    # Get first batch to inspect shapes
    first_batch = next(iter(dataloader))
    features, targets = first_batch
    
    info = {
        "num_batches": len(dataloader),
        "batch_size": dataloader.batch_size,
        "total_sequences": len(dataloader.dataset),  # type: ignore
        "feature_shape": list(features.shape),  # [batch_size, seq_length, num_features]
        "target_shape": list(targets.shape),    # [batch_size]
        "feature_dtype": str(features.dtype),
        "target_dtype": str(targets.dtype)
    }
    
    return info