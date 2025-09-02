#!/usr/bin/env python3
"""
Data loading and preprocessing module for LSTM iron ore forecasting.

This module handles:
- Loading consolidated features dataset (12 features + Y target)
- Chronological train/validation/test splits
- Feature scaling (MinMaxScaler for features, no scaling for target)
- Data validation and preprocessing
"""

import pandas as pd
from sklearn.preprocessing import MinMaxScaler  # type: ignore
from pathlib import Path
from typing import Tuple, Dict, Any
import logging

logger = logging.getLogger(__name__)


class DataLoader:
    """
    Load and preprocess data for LSTM model training.

    Handles consolidated features dataset with proper chronological splitting
    and feature scaling per PRD requirements.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize DataLoader with configuration.

        Args:
            config: Configuration dictionary containing data settings
        """
        self.config = config
        self.scaler: MinMaxScaler = MinMaxScaler()
        self.data: pd.DataFrame = pd.DataFrame()

    def load_consolidated_data(self) -> None:
        """
        Load consolidated features dataset from pickle file.

        Expected format: 868 observations × 13 columns (12 features + Y target)
        Date range: March 30, 2022 - August 11, 2025 (last row removed due to missing Y)
        """
        logger.info("Loading consolidated features dataset...")

        # Load data from pickle file (faster than CSV)
        data_path = Path(self.config["data"]["consolidated_features"])
        if not data_path.exists():
            raise FileNotFoundError(
                f"Consolidated features file not found: {data_path}"
            )

        self.data = pd.read_pickle(data_path)

        # Validate data structure
        expected_cols = 13  # 12 features + Y target
        if self.data.shape[1] != expected_cols:
            logger.warning(
                f"Expected {expected_cols} columns, got {self.data.shape[1]}"
            )

        # Verify Y target column exists and check for missing values
        if "Y" not in self.data.columns:
            raise KeyError("Target column 'Y' not found in consolidated dataset")
        missing_y = self.data["Y"].isnull().sum()
        if missing_y > 0:
            logger.warning(f"Found {missing_y} missing Y values")

        logger.info(
            f"Data loaded successfully: {self.data.shape} (observations × features)"
        )
        logger.info(
            f"Date range: {self.data.index.min().date()} to {self.data.index.max().date()}"
        )
        logger.info(
            f"Time span: {(self.data.index.max() - self.data.index.min()).days} days"
        )

        # Log data quality statistics
        total_values = self.data.size
        missing_values = self.data.isnull().sum().sum()
        missing_percentage = (missing_values / total_values) * 100

        logger.info("Data quality overview:")
        logger.info(f"  Total values: {total_values:,}")
        logger.info(f"  Missing values: {missing_values:,} ({missing_percentage:.2f}%)")

        # Feature-wise missing value analysis
        missing_by_feature = self.data.isnull().sum()
        if missing_values > 0:
            logger.info("  Missing values by feature:")
            for feature, count in missing_by_feature.items():
                if count > 0:
                    pct = (count / len(self.data)) * 100
                    logger.info(f"    {feature}: {count} ({pct:.1f}%)")
        else:
            logger.info("  ✓ No missing values detected")

        # Basic statistical overview of Y target
        y_series = self.data["Y"]
        logger.info("Target variable (Y) statistics:")
        logger.info(f"  Mean: {y_series.mean():.6f}, Std: {y_series.std():.6f}")
        logger.info(f"  Range: [{y_series.min():.6f}, {y_series.max():.6f}]")

    def create_chronological_splits(
        self,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Create chronological train/validation/test splits.

        Supports val_ratio=0 for train/test splits only.

        Returns:
            Tuple of (train_df, val_df, test_df). val_df will be empty if val_ratio=0.
        """
        logger.info("Creating chronological data splits...")

        # Get split ratios from config
        train_ratio = self.config["splits"]["train_ratio"]
        val_ratio = self.config["splits"]["val_ratio"]
        test_ratio = self.config["splits"]["test_ratio"]

        # Validate ratios sum to 1.0
        total_ratio = train_ratio + val_ratio + test_ratio
        tolerance = self.config.get("validation", {}).get("split_ratio_tolerance", 0.01)
        if abs(total_ratio - 1.0) > tolerance:
            raise ValueError(f"Split ratios must sum to 1.0, got {total_ratio}")

        # Calculate split indices (chronological order)
        n_samples = len(self.data)
        train_end = int(n_samples * train_ratio)
        val_end = int(n_samples * (train_ratio + val_ratio))

        # Create splits maintaining chronological order
        train_df = self.data.iloc[:train_end].copy()
        val_df = self.data.iloc[train_end:val_end].copy()
        test_df = self.data.iloc[val_end:].copy()

        # Calculate split percentages (actual vs configured)
        actual_train_pct = len(train_df) / n_samples * 100
        actual_val_pct = len(val_df) / n_samples * 100
        actual_test_pct = len(test_df) / n_samples * 100

        logger.info("Chronological data splits created:")
        logger.info(
            f"  Train: {len(train_df):3d} samples ({actual_train_pct:5.1f}%) "
            f"[{train_df.index.min().date()} to {train_df.index.max().date()}]"
        )

        if val_ratio == 0:
            logger.info("  Val:   0 samples (  0.0%) [validation disabled]")
        else:
            logger.info(
                f"  Val:   {len(val_df):3d} samples ({actual_val_pct:5.1f}%) "
                f"[{val_df.index.min().date()} to {val_df.index.max().date()}]"
            )

        logger.info(
            f"  Test:  {len(test_df):3d} samples ({actual_test_pct:5.1f}%) "
            f"[{test_df.index.min().date()} to {test_df.index.max().date()}]"
        )

        # Log target variable statistics for each split
        logger.info("Target variable (Y) statistics by split:")
        logger.info(
            f"  Train Y: mean={train_df['Y'].mean():.6f}, std={train_df['Y'].std():.6f}"
        )
        if len(val_df) > 0:
            logger.info(
                f"  Val Y:   mean={val_df['Y'].mean():.6f}, std={val_df['Y'].std():.6f}"
            )
        logger.info(
            f"  Test Y:  mean={test_df['Y'].mean():.6f}, std={test_df['Y'].std():.6f}"
        )

        # Data split summary
        total_samples = len(train_df) + len(val_df) + len(test_df)
        train_years = (train_df.index.max() - train_df.index.min()).days / 365.25

        if val_ratio == 0:
            logger.info(
                f"📊 Data Split: Using {train_years:.1f} years for training, {total_samples} total daily observations (validation disabled)"
            )
        else:
            logger.info(
                f"📊 Data Split: Using {train_years:.1f} years for training, {total_samples} total daily observations"
            )

        return train_df, val_df, test_df

    def filter_selected_features(
        self, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Filter datasets to include only selected features from config plus Y target.

        Args:
            train_df: Training dataset
            val_df: Validation dataset
            test_df: Test dataset

        Returns:
            Tuple of filtered (train_df, val_df, test_df) with selected features only
        """
        logger.info("Applying configurable feature selection...")

        # Get selected features from config
        selected_features = self.config.get("features", [])
        if not selected_features:
            logger.warning(
                "No features specified in config, using all available features"
            )
            selected_features = [col for col in train_df.columns if col != "Y"]

        # Always include Y target variable
        columns_to_keep = selected_features + ["Y"]

        # Validate selected features exist in data
        available_features = [col for col in train_df.columns if col != "Y"]
        missing_features = [f for f in selected_features if f not in available_features]
        if missing_features:
            raise ValueError(
                f"Selected features not found in dataset: {missing_features}"
            )

        # Filter datasets to selected features
        train_filtered = train_df[columns_to_keep].copy()
        val_filtered = val_df[columns_to_keep].copy()
        test_filtered = test_df[columns_to_keep].copy()

        logger.info("Feature selection applied:")
        logger.info(
            f"  Available features: {len(available_features)} ({available_features})"
        )
        logger.info(
            f"  Selected features: {len(selected_features)} ({selected_features})"
        )
        logger.info(
            f"  Filtered dataset shape: {train_filtered.shape[1] - 1} features + Y target"
        )

        # Feature selection summary for layman
        feature_reduction = len(available_features) - len(selected_features)
        if feature_reduction > 0:
            logger.info(
                f"🎯 Feature Focus: Using {len(selected_features)}/{len(available_features)} features "
                f"(removed {feature_reduction} features to focus model on most relevant signals)"
            )
        else:
            logger.info(
                f"🎯 Feature Focus: Using all {len(selected_features)} available features for comprehensive analysis"
            )

        # Log top 5 rows of filtered features for inspection
        logger.info("📊 Feature Data Preview (top 5 rows):")
        feature_cols = [col for col in train_filtered.columns if col != "Y"]
        try:
            preview_data = train_filtered.head(5)
            logger.info(f"  Features: {feature_cols}")
            for idx, row in preview_data.iterrows():
                feature_values = [f"{row[col]:.4f}" for col in feature_cols]
                target_value = row["Y"]
                logger.info(
                    f"  Row {idx}: [{', '.join(feature_values)}] → Y={target_value:.6f}"
                )
        except Exception as e:
            logger.warning(f"Could not preview feature data: {e}")

        return train_filtered, val_filtered, test_filtered

    def scale_features(
        self, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Apply feature scaling to X features only (not Y target).

        Per PRD: Apply MinMaxScaler to input features, leave Y unscaled
        Fit scaler on training data only to prevent data leakage.

        Args:
            train_df: Training dataset
            val_df: Validation dataset
            test_df: Test dataset

        Returns:
            Tuple of scaled (train_df, val_df, test_df)
        """
        logger.info("Applying feature scaling...")

        # Identify feature columns (all except Y target)
        feature_cols = [col for col in train_df.columns if col != "Y"]
        logger.info(f"Scaling {len(feature_cols)} features: {feature_cols}")

        # Fit scaler on training features only (prevent data leakage)
        self.scaler.fit(train_df[feature_cols])

        # Transform all datasets using training-fitted scaler
        train_df_scaled = train_df.copy()
        val_df_scaled = val_df.copy()
        test_df_scaled = test_df.copy()

        train_df_scaled[feature_cols] = self.scaler.transform(train_df[feature_cols])

        # Only transform validation set if it has samples
        if len(val_df) > 0:
            val_df_scaled[feature_cols] = self.scaler.transform(val_df[feature_cols])

        test_df_scaled[feature_cols] = self.scaler.transform(test_df[feature_cols])

        logger.info("Feature scaling completed")
        # Detailed scaling statistics
        original_min = train_df[feature_cols].min().min()
        original_max = train_df[feature_cols].max().max()
        scaled_min = train_df_scaled[feature_cols].min().min()
        scaled_max = train_df_scaled[feature_cols].max().max()

        logger.info("Feature scaling transformation completed:")
        logger.info(f"  Original range: [{original_min:.6f}, {original_max:.6f}]")
        logger.info(f"  Scaled range:   [{scaled_min:.3f}, {scaled_max:.3f}]")

        # Log scaling parameters for reproducibility
        logger.info("Scaling parameters (MinMaxScaler):")
        scaler_min = self.scaler.data_min_
        scaler_scale = self.scaler.scale_
        logger.info(f"  Data min: [{scaler_min.min():.6f}, {scaler_min.max():.6f}]")
        logger.info(
            f"  Scale factors: [{scaler_scale.min():.6f}, {scaler_scale.max():.6f}]"
        )

        # Verify Y target is unscaled
        y_unchanged = train_df_scaled["Y"].equals(train_df["Y"]) and test_df_scaled[
            "Y"
        ].equals(test_df["Y"])
        logger.info(
            f"  Target (Y) scaling: {'unchanged (correct)' if y_unchanged else 'MODIFIED (ERROR)'}"
        )

        # Feature scaling summary
        logger.info(
            f"🔧 Features Prepared: All {len(feature_cols)} features normalized to 0-1 range for optimal neural network training"
        )

        return train_df_scaled, val_df_scaled, test_df_scaled

    def get_processed_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Complete data loading and preprocessing pipeline.

        Returns:
            Tuple of processed (train_df, val_df, test_df) with selected features and scaling
        """
        # Load consolidated dataset
        self.load_consolidated_data()

        # Create chronological splits
        train_df, val_df, test_df = self.create_chronological_splits()

        # Apply configurable feature selection
        train_df, val_df, test_df = self.filter_selected_features(
            train_df, val_df, test_df
        )

        # Apply feature scaling (if enabled in config)
        if self.config["scaling"]["scale_features"]:
            train_df, val_df, test_df = self.scale_features(train_df, val_df, test_df)

        # Final pipeline summary
        final_feature_count = len([col for col in train_df.columns if col != "Y"])
        logger.info("\n" + "=" * 60)
        logger.info("DATA PREPROCESSING PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)
        logger.info("Final dataset specifications:")
        logger.info(
            f"  Features: {final_feature_count} (selected from available features)"
        )
        logger.info(
            f"  Samples: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}"
        )
        logger.info(
            f"  Scaling: {'enabled' if self.config['scaling']['scale_features'] else 'disabled'}"
        )
        logger.info("  Data quality: ready for LSTM training")
        logger.info("=" * 60 + "\n")

        return train_df, val_df, test_df
