#!/usr/bin/env python3
"""
Build continuous M+1 futures series using backward cumulative adjustment method.

This script processes raw futures contract data to create continuous price series
for 65% M+1 DSP and 62% M+1 Close, implementing the backward cumulative adjustment
process where the most recent contract serves as the anchor (never adjusted) and
all historical contracts are adjusted working backwards through time.
"""

import pickle
import pandas as pd
from pathlib import Path
from datetime import timedelta
from typing import Dict, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ContinuousFuturesBuilder:
    """Build continuous M+1 futures series using backward cumulative adjustment."""

    def __init__(self, data_dir: Path = Path("data/raw")):
        """Initialize with data directory path."""
        self.data_dir = data_dir
        self.contracts_65: Optional[Dict] = None
        self.contracts_62: Optional[Dict] = None

    def load_raw_data(self) -> None:
        """Load raw futures contract data from pickle files."""
        logger.info("Loading raw futures contract data...")

        # Load 65% M+1 DSP contracts
        with open(self.data_dir / "Raw_M65F_DSP.pkl", "rb") as f:
            self.contracts_65 = pickle.load(f)

        # Load 62% FEF Close contracts
        with open(self.data_dir / "Raw_FEF_Close.pkl", "rb") as f:
            self.contracts_62 = pickle.load(f)

        logger.info(f"Loaded 65% contracts: {len(self.contracts_65) if self.contracts_65 else 0} periods")
        logger.info(f"Loaded 62% contracts: {len(self.contracts_62) if self.contracts_62 else 0} periods")

    def get_m1_contract_for_month(self, year: int, month: int) -> Tuple[int, int]:
        """
        Get M+1 contract (M+1 expiry) for given calendar month.

        Args:
            year: Calendar year
            month: Calendar month (1-12)

        Returns:
            Tuple of (contract_year, contract_month) for M+1 contract
        """
        # M+1 logic: month M uses contract expiring in M+1
        contract_month = month + 1
        contract_year = year

        # Handle year rollover
        if contract_month > 12:
            contract_month -= 12
            contract_year += 1

        return contract_year, contract_month

    def get_last_trading_day_of_month(
        self, contract_data: pd.Series, year: int, month: int
    ) -> Optional[pd.Timestamp]:
        """
        Find the last trading day of a given month in contract data.

        Args:
            contract_data: Pandas Series with DatetimeIndex
            year: Year
            month: Month

        Returns:
            Last trading day of the month, or None if not found
        """
        # Get all dates in the specified month
        month_start = pd.Timestamp(year, month, 1)
        if month == 12:
            month_end = pd.Timestamp(year + 1, 1, 1) - timedelta(days=1)
        else:
            month_end = pd.Timestamp(year, month + 1, 1) - timedelta(days=1)

        # Filter contract data to this month
        month_data = contract_data[
            (contract_data.index >= month_start) & (contract_data.index <= month_end)
        ]

        # Find last non-NaN trading day
        non_nan_data = month_data.dropna()
        if len(non_nan_data) > 0:
            return non_nan_data.index[-1]
        return None

    def build_continuous_series(
        self, contracts: Dict, series_name: str
    ) -> pd.DataFrame:
        """
        Build continuous M+1 series using backward cumulative adjustment.

        Args:
            contracts: Dictionary of contract data {Period: Series}
            series_name: Name for the price series

        Returns:
            DataFrame with continuous series and contract tracking
        """
        logger.info(f"Building continuous series for {series_name}...")

        # Sort contract periods
        periods = sorted(contracts.keys())

        # Create date range for M+1 mapping
        start_date = pd.Timestamp("2020-01-01")
        end_date = pd.Timestamp("2025-12-31")
        date_range = pd.date_range(start_date, end_date, freq="D")

        # Build M+1 contract mapping for each date
        m1_mapping = {}
        for date in date_range:
            year, month = date.year, date.month
            contract_year, contract_month = self.get_m1_contract_for_month(year, month)
            contract_period = pd.Period(
                f"{contract_year}-{contract_month:02d}", freq="M"
            )
            m1_mapping[date] = contract_period

        # Start backward cumulative adjustment from most recent contract
        adjusted_contracts = {}

        # Find the most recent contract that has data (anchor - never adjusted)
        latest_period = periods[-1]
        adjusted_contracts[latest_period] = contracts[latest_period].copy()
        logger.info(f"Using {latest_period} as anchor contract (never adjusted)")

        # Work backwards through all previous contracts
        for i in range(len(periods) - 2, -1, -1):
            current_period = periods[i]
            next_period = periods[i + 1]

            # Get the already adjusted next contract
            adjusted_next_contract = adjusted_contracts[next_period]
            current_contract = contracts[current_period].copy()

            # Find rollover day (last trading day of current period)
            year, month = current_period.year, current_period.month
            rollover_day = self.get_last_trading_day_of_month(
                current_contract, year, month
            )

            if rollover_day is None:
                logger.warning(
                    f"No rollover day found for {current_period}, skipping adjustment"
                )
                adjusted_contracts[current_period] = current_contract
                continue

            # Calculate adjustment ratio using adjusted next contract
            current_price = current_contract.get(rollover_day)
            next_price = adjusted_next_contract.get(rollover_day)

            if pd.isna(current_price) or pd.isna(next_price):
                logger.warning(
                    f"Missing prices on rollover day {rollover_day} for {current_period}"
                )
                adjusted_contracts[current_period] = current_contract
                continue

            # Backward cumulative adjustment ratio
            adjustment_ratio = next_price / current_price

            # Apply adjustment to entire current contract history
            adjusted_contracts[current_period] = current_contract * adjustment_ratio

            logger.info(
                f"Adjusted {current_period}: ratio={adjustment_ratio:.6f} on {rollover_day}"
            )

        # Build continuous daily series using M+1 mapping
        continuous_data = []

        for date, m1_period in m1_mapping.items():
            if m1_period in adjusted_contracts:
                contract_data = adjusted_contracts[m1_period]
                price = contract_data.get(date)

                if not pd.isna(price):
                    continuous_data.append(
                        {
                            "date": date,
                            f"price_{series_name}": price,
                            f"contract_month_{series_name}": str(m1_period),
                        }
                    )

        # Convert to DataFrame
        df = pd.DataFrame(continuous_data)
        if len(df) > 0:
            df.set_index("date", inplace=True)
            df.sort_index(inplace=True)

        logger.info(
            f"Built continuous series for {series_name}: {len(df)} daily observations"
        )
        return df

    def build_combined_series(self) -> pd.DataFrame:
        """Build combined continuous series for both 65% and 62% contracts."""
        if self.contracts_65 is None or self.contracts_62 is None:
            raise ValueError("Raw data must be loaded first. Call load_raw_data().")

        # Build individual series
        series_65 = self.build_continuous_series(self.contracts_65, "65_m1")
        series_62 = self.build_continuous_series(self.contracts_62, "62_m1")

        # Combine series on common dates
        combined = pd.concat([series_65, series_62], axis=1, join="inner")

        # Filter to March 30, 2022 onwards per PRD requirement
        start_date = pd.Timestamp("2022-03-30")
        combined = combined[combined.index >= start_date]

        logger.info(
            f"Combined series: {len(combined)} daily observations from {start_date}"
        )

        return combined

    def save_results(
        self, df: pd.DataFrame, output_dir: Path = Path("data/processed")
    ) -> None:
        """Save continuous futures series to both CSV and pickle."""
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save as CSV for inspection
        csv_path = output_dir / "continuous_futures_m1.csv"
        df.to_csv(csv_path)

        # Save as pickle for faster loading
        pkl_path = output_dir / "continuous_futures_m1.pkl"
        df.to_pickle(pkl_path)

        logger.info(f"Saved continuous futures to {csv_path} and {pkl_path}")

        # Print summary statistics
        print("\n" + "=" * 60)
        print("CONTINUOUS FUTURES SERIES SUMMARY")
        print("=" * 60)
        print(f"Date range: {df.index.min()} to {df.index.max()}")
        print(f"Total observations: {len(df)}")
        print(f"Columns: {list(df.columns)}")
        print("\nFirst 5 rows:")
        print(df.head())
        print("\nLast 5 rows:")
        print(df.tail())
        print("\nData info:")
        df.info()


def main():
    """Main function to build continuous futures series."""
    try:
        # Initialize builder
        builder = ContinuousFuturesBuilder()

        # Load raw data
        builder.load_raw_data()

        # Build continuous series
        continuous_df = builder.build_combined_series()

        # Save results
        builder.save_results(continuous_df)

        print("\n✅ Continuous futures construction completed successfully!")

    except Exception as e:
        logger.error(f"Error building continuous futures: {e}")
        raise


if __name__ == "__main__":
    main()
