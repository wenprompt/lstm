#!/usr/bin/env python3
"""
Build consolidated feature dataset combining all 12 features.

This script combines:
- 2 processed continuous futures series (65% M+1 DSP, 62% M+1 Close)
- 6 daily features from group.csv
- 2 daily index features from Raw_65and62_Index.csv
- 2 weekly features (IOCJ Inventory, IOCJ Weekly Shipment) with forward-fill

Uses the M+1 futures timeline as the master date index to ensure alignment
with the target variable (65% M+1 DSP log returns).
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ConsolidatedFeaturesBuilder:
    """Build consolidated dataset with all 12 features."""

    def __init__(
        self,
        raw_dir: Path = Path("data/raw"),
        processed_dir: Path = Path("data/processed"),
        output_dir: Path = Path("data/processed"),
    ):
        """Initialize with data directory paths."""
        self.raw_dir = raw_dir
        self.processed_dir = processed_dir
        self.output_dir = output_dir
        self.master_timeline: Optional[pd.DatetimeIndex] = None

    def load_futures_data(self) -> pd.DataFrame:
        """Load processed continuous futures data and set master timeline."""
        logger.info("Loading processed continuous futures data...")

        futures_df = pd.read_csv(
            self.processed_dir / "continuous_futures_m1.csv",
            index_col="date",
            parse_dates=True,
        )

        # Set master timeline from futures data
        if isinstance(futures_df.index, pd.DatetimeIndex):
            self.master_timeline = futures_df.index
            logger.info(
                f"Master timeline set: {len(self.master_timeline)} dates from {self.master_timeline.min()} to {self.master_timeline.max()}"
            )
        else:
            raise ValueError("Futures data must have DatetimeIndex")

        return futures_df

    def load_group_features(self) -> pd.DataFrame:
        """Load 6 daily features from group.csv."""
        logger.info("Loading group features...")

        group_df = pd.read_csv(
            self.raw_dir / "group.csv",
            encoding="utf-8-sig",  # Handle BOM
        )

        # Parse DD/MM/YYYY format
        group_df["Date"] = pd.to_datetime(
            group_df["Date"], format="%d/%m/%Y", dayfirst=True
        )
        group_df.set_index("Date", inplace=True)

        logger.info(f"Group features loaded: {group_df.shape}")
        return group_df

    def load_index_features(self) -> pd.DataFrame:
        """Load 2 daily index features from Raw_65and62_Index.csv."""
        logger.info("Loading index features...")

        index_df = pd.read_csv(
            self.raw_dir / "Raw_65and62_Index.csv", encoding="utf-8-sig"
        )

        # Parse YYYY-MM-DD format (different from other files)
        index_df["Date"] = pd.to_datetime(index_df["Date"], format="%Y-%m-%d")
        index_df.set_index("Date", inplace=True)

        logger.info(f"Index features loaded: {index_df.shape}")
        return index_df

    def load_weekly_features(self) -> pd.DataFrame:
        """Load and forward-fill 2 weekly features."""
        logger.info("Loading weekly features...")

        # Load IOCJ Inventory
        inventory_df = pd.read_csv(
            self.raw_dir / "IOCJ Inventory.csv", encoding="utf-8-sig"
        )
        inventory_df["Date"] = pd.to_datetime(
            inventory_df["Date"], format="%d/%m/%Y", dayfirst=True
        )
        inventory_df.set_index("Date", inplace=True)

        # Load IOCJ Weekly Shipment
        shipment_df = pd.read_csv(
            self.raw_dir / "IOCJ Weekly Shipment.csv", encoding="utf-8-sig"
        )
        shipment_df["Date"] = pd.to_datetime(
            shipment_df["Date"], format="%d/%m/%Y", dayfirst=True, errors="coerce"
        )

        # Remove rows with invalid dates (NaT)
        shipment_df = shipment_df.dropna(subset=["Date"])
        shipment_df.set_index("Date", inplace=True)

        # Remove duplicate dates if any
        if shipment_df.index.has_duplicates:
            logger.warning(
                "Found duplicate dates in shipment data, keeping first occurrence"
            )
            shipment_df = shipment_df[~shipment_df.index.duplicated(keep="first")]

        # Combine weekly features - handle potential duplicate dates
        weekly_df = pd.concat([inventory_df, shipment_df], axis=1, join="outer")

        # Remove duplicate dates if any
        if weekly_df.index.has_duplicates:
            logger.warning(
                "Found duplicate dates in weekly data, keeping first occurrence"
            )
            weekly_df = weekly_df[~weekly_df.index.duplicated(keep="first")]

        # Clean column names (remove trailing spaces)
        weekly_df.columns = weekly_df.columns.str.strip()

        logger.info(f"Weekly features loaded: {weekly_df.shape}")
        return weekly_df

    def apply_forward_fill(self, weekly_df: pd.DataFrame) -> pd.DataFrame:
        """Apply forward-fill to weekly data using master timeline."""
        if self.master_timeline is None:
            raise ValueError("Master timeline must be set before applying forward-fill")

        logger.info("Applying forward-fill to weekly features...")

        # Create extended timeline for forward-filling
        extended_timeline = pd.date_range(
            start=weekly_df.index.min(), end=self.master_timeline.max(), freq="D"
        )

        # Reindex to daily frequency and forward-fill
        weekly_daily = weekly_df.reindex(extended_timeline)
        weekly_daily = weekly_daily.ffill()

        # Filter to master timeline
        weekly_aligned = weekly_daily.reindex(self.master_timeline)

        # Count forward-filled values
        original_points = len(weekly_df.dropna())
        filled_points = len(weekly_aligned.dropna())
        logger.info(
            f"Forward-fill applied: {original_points} original → {filled_points} daily observations"
        )

        return weekly_aligned

    def align_all_features(
        self,
        futures_df: pd.DataFrame,
        group_df: pd.DataFrame,
        index_df: pd.DataFrame,
        weekly_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Align all feature datasets to master timeline."""
        logger.info("Aligning all features to master timeline...")

        if self.master_timeline is None:
            raise ValueError("Master timeline must be set")

        # Align each dataset to master timeline
        futures_aligned = futures_df.reindex(self.master_timeline)
        group_aligned = group_df.reindex(self.master_timeline)
        index_aligned = index_df.reindex(self.master_timeline)

        # Combine all features
        consolidated = pd.concat(
            [
                futures_aligned,
                group_aligned,
                index_aligned,
                weekly_df,  # Already aligned in apply_forward_fill
            ],
            axis=1,
        )

        # Clean up columns - remove empty/unnamed columns
        consolidated = consolidated.loc[
            :, ~consolidated.columns.str.contains("^Unnamed")
        ]

        # Remove contract month tracking columns and raw price columns (not features for ML)
        feature_columns = [
            col
            for col in consolidated.columns
            if not col.startswith("contract_month") and not col.startswith("raw_price")
        ]
        consolidated = consolidated[feature_columns]

        # Forward-fill ALL features to handle missing values
        logger.info("Applying forward-fill to all features...")
        before_fill = consolidated.isnull().sum()
        consolidated = consolidated.ffill()
        after_fill = consolidated.isnull().sum()

        logger.info("Forward-fill results:")
        for col in consolidated.columns:
            filled = before_fill[col] - after_fill[col]
            if filled > 0:
                logger.info(f"  {col}: filled {filled} missing values")

        # Calculate target variable Y: next-day percentage log return of 65% M+1 DSP
        logger.info("Calculating target variable Y (next-day log return)...")
        price_65 = consolidated["price_65_m1"]
        y_values = (price_65.shift(-1) / price_65).apply(
            lambda x: np.log(x) * 100 if pd.notna(x) and x > 0 else pd.NA
        )
        consolidated["Y"] = y_values

        logger.info(
            f"Y values calculated: {y_values.notna().sum()} valid out of {len(y_values)} total"
        )

        # Remove last row (has features but no Y value for training)
        consolidated = consolidated.dropna(subset=["Y"])
        logger.info(f"Removed rows with missing Y: final shape {consolidated.shape}")

        logger.info(f"Final columns: {list(consolidated.columns)}")
        return consolidated

    def validate_features(self, df: pd.DataFrame) -> None:
        """Validate consolidated features dataset."""
        logger.info("Validating consolidated features...")

        expected_features = 13  # 12 features + 1 target variable Y
        actual_features = len(df.columns)

        if actual_features != expected_features:
            logger.warning(
                f"Expected {expected_features} columns (12 features + Y), got {actual_features}"
            )

        # Check date range
        logger.info(f"Date range: {df.index.min()} to {df.index.max()}")
        logger.info(f"Total observations: {len(df)}")

        # Check missing data by column
        missing_summary = df.isnull().sum()
        logger.info(f"Missing values per feature:\n{missing_summary}")

        # Weekly features should have fewer missing values due to forward-fill
        weekly_cols = ["IOCJ Inventory", "IOCJ Weekly shipment"]
        for col in weekly_cols:
            if col in df.columns:
                missing_count = df[col].isnull().sum()
                logger.info(f"{col} missing after forward-fill: {missing_count}")

    def save_results(self, df: pd.DataFrame) -> None:
        """Save consolidated features to both CSV and pickle."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Save as CSV
        csv_path = self.output_dir / "consolidated_features_y.csv"
        df.to_csv(csv_path)

        # Save as pickle
        pkl_path = self.output_dir / "consolidated_features_y.pkl"
        df.to_pickle(pkl_path)

        logger.info(f"Saved consolidated features to {csv_path} and {pkl_path}")

        # Print concise summary
        print(f"\n✅ Dataset saved: {df.shape[0]} observations × {df.shape[1]} columns")
        print(f"📁 Files: {csv_path.name} & {pkl_path.name}")
        print(f"🎯 Y values: {df['Y'].notna().sum()} valid")


def main():
    """Main function to build consolidated features dataset."""
    try:
        # Initialize builder
        builder = ConsolidatedFeaturesBuilder()

        # Load futures data and set master timeline
        futures_df = builder.load_futures_data()

        # Load all feature datasets
        group_df = builder.load_group_features()
        index_df = builder.load_index_features()
        weekly_df = builder.load_weekly_features()

        # Apply forward-fill to weekly data
        weekly_filled = builder.apply_forward_fill(weekly_df)

        # Align all features to master timeline
        consolidated_df = builder.align_all_features(
            futures_df, group_df, index_df, weekly_filled
        )

        # Validate and save results
        builder.validate_features(consolidated_df)
        builder.save_results(consolidated_df)

        print("\n✅ Consolidated features construction completed successfully!")

    except Exception as e:
        logger.error(f"Error building consolidated features: {e}")
        raise


if __name__ == "__main__":
    main()
