#!/usr/bin/env python3
"""
Test script to check backward cumulative adjustment ratios using actual data.

This script examines real futures contract prices to verify:
1. Whether ratios are positive and reasonable
2. The direction of adjustment (should ratios be > 1 or < 1?)
3. Specific example: last day of July 2025, check Aug-25 vs Sep-25 prices
"""

import pickle
import pandas as pd
from datetime import timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_contracts():
    """Load both contract datasets."""
    # Load 65% M+1 DSP contracts
    with open("data/raw/Raw_M65F_DSP.pkl", "rb") as f:
        contracts_65 = pickle.load(f)
        
    # Load 62% FEF Close contracts  
    with open("data/raw/Raw_FEF_Close.pkl", "rb") as f:
        contracts_62 = pickle.load(f)
        
    return contracts_65, contracts_62


def get_last_trading_day_of_month(contract_data, year, month):
    """Find the last trading day of a given month in contract data."""
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


def check_ratio_example(contracts, dataset_name):
    """Check specific example: last day of July 2025, Aug-25 vs Sep-25 contracts."""
    print(f"\n=== {dataset_name} Analysis ===")
    
    # Get available periods
    periods = sorted(contracts.keys())
    print(f"Available periods: {periods}")
    
    # Look for Aug-25 and Sep-25 contracts
    aug_2025 = None
    sep_2025 = None
    
    for period in periods:
        if period.year == 2025 and period.month == 8:
            aug_2025 = period
        elif period.year == 2025 and period.month == 9:
            sep_2025 = period
            
    if aug_2025 is None or sep_2025 is None:
        print("Missing Aug-25 or Sep-25 contracts")
        print(f"Aug-25: {aug_2025}, Sep-25: {sep_2025}")
        return
        
    print(f"Found contracts: {aug_2025}, {sep_2025}")
    
    # Get contract data
    aug_contract = contracts[aug_2025]
    sep_contract = contracts[sep_2025]
    
    # Find last trading day of July 2025
    last_day_july = get_last_trading_day_of_month(aug_contract, 2025, 7)
    
    if last_day_july is None:
        print("No trading day found for last day of July 2025")
        return
        
    print(f"Last trading day of July 2025: {last_day_july}")
    
    # Get prices on that day
    aug_price = aug_contract.get(last_day_july)
    sep_price = sep_contract.get(last_day_july)
    
    print(f"Aug-25 contract price on {last_day_july}: {aug_price}")
    print(f"Sep-25 contract price on {last_day_july}: {sep_price}")
    
    if pd.notna(aug_price) and pd.notna(sep_price):
        ratio = sep_price / aug_price
        print(f"Ratio (Sep-25 / Aug-25): {ratio:.6f}")
        print(f"Ratio > 1? {ratio > 1}")
        
        # This ratio would be used to adjust Aug-25 contract
        # Aug-25 adjusted = Aug-25 original * ratio
        print("If we adjust Aug-25 by this ratio:")
        print(f"  Aug-25 adjusted price on {last_day_july}: {aug_price * ratio:.4f}")
        print(f"  Sep-25 reference price on {last_day_july}: {sep_price:.4f}")
        print(f"  Difference after adjustment: {abs(aug_price * ratio - sep_price):.8f}")
        
    else:
        print(f"Missing price data: Aug={aug_price}, Sep={sep_price}")


def check_multiple_rollover_examples(contracts, dataset_name):
    """Check multiple rollover examples to see typical ratio patterns."""
    print(f"\n=== {dataset_name} Rollover Pattern Analysis ===")
    
    periods = sorted(contracts.keys())
    
    # Check several consecutive months
    examples = []
    for i in range(len(periods) - 1):
        older_period = periods[i]
        newer_period = periods[i + 1]
        
        # Skip if not consecutive months
        if newer_period.month != (older_period.month % 12) + 1:
            continue
            
        # Find rollover day (last day of month before older period expires)
        rollover_month = older_period.month - 1
        rollover_year = older_period.year
        if rollover_month == 0:
            rollover_month = 12
            rollover_year -= 1
            
        rollover_day = get_last_trading_day_of_month(
            contracts[older_period], rollover_year, rollover_month
        )
        
        if rollover_day is None:
            continue
            
        # Get prices
        older_price = contracts[older_period].get(rollover_day)
        newer_price = contracts[newer_period].get(rollover_day)
        
        if pd.notna(older_price) and pd.notna(newer_price):
            ratio = newer_price / older_price
            examples.append({
                'rollover_day': rollover_day,
                'older_contract': older_period,
                'newer_contract': newer_period,
                'older_price': older_price,
                'newer_price': newer_price,
                'ratio': ratio
            })
    
    # Show first 10 examples
    print("Sample rollover ratios (newer/older):")
    print("Date        | Older -> Newer | Older Price | Newer Price | Ratio")
    print("-" * 70)
    
    for example in examples[:10]:
        print(f"{example['rollover_day'].strftime('%Y-%m-%d')} | "
              f"{example['older_contract']} -> {example['newer_contract']} | "
              f"{example['older_price']:8.2f} | {example['newer_price']:8.2f} | "
              f"{example['ratio']:8.6f}")
              
    # Statistics
    ratios = [ex['ratio'] for ex in examples]
    if ratios:
        print("\nRatio Statistics:")
        print(f"  Count: {len(ratios)}")
        print(f"  Min: {min(ratios):.6f}")
        print(f"  Max: {max(ratios):.6f}")
        print(f"  Mean: {sum(ratios)/len(ratios):.6f}")
        print(f"  Ratios > 1: {sum(1 for r in ratios if r > 1)} ({100*sum(1 for r in ratios if r > 1)/len(ratios):.1f}%)")


def main():
    """Main function to test ratio calculations."""
    try:
        # Load contract data
        print("Loading contract data...")
        contracts_65, contracts_62 = load_contracts()
        
        # Check specific July 2025 example for both datasets
        check_ratio_example(contracts_65, "65% M+1 DSP")
        check_ratio_example(contracts_62, "62% FEF Close")
        
        # Check broader patterns
        check_multiple_rollover_examples(contracts_65, "65% M+1 DSP")
        check_multiple_rollover_examples(contracts_62, "62% FEF Close")
        
        print("\n✅ Ratio analysis completed!")
        
    except Exception as e:
        logger.error(f"Error in ratio analysis: {e}")
        raise


if __name__ == "__main__":
    main()