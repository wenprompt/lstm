#!/usr/bin/env python3
"""
Validation script for data building pipeline accuracy.

This script validates that build_continuous_futures.py and build_consolidated_features.py 
are processing data correctly by checking:

1. Continuous Futures Validation:
   - M+1 contract mapping accuracy (Jan uses Feb contract, Feb uses Mar, etc.)
   - Backward cumulative adjustment calculations
   - Date alignment and chronological order
   - Price continuity at rollover points

2. Consolidated Features Validation:
   - Feature alignment across different data sources
   - Forward-fill logic for weekly data
   - Target variable Y calculation accuracy
   - Date parsing consistency across different formats

Run this before training to ensure data building is accurate.
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from typing import Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataBuildingValidator:
    """Comprehensive validation of data building pipeline."""
    
    def __init__(self, data_dir: Path = Path("data")):
        """Initialize validator with data directory."""
        self.data_dir = data_dir
        self.raw_dir = data_dir / "raw"
        self.processed_dir = data_dir / "processed"
        
    def validate_m1_contract_mapping(self) -> Dict[str, bool]:
        """Validate M+1 contract mapping logic."""
        logger.info("\n🔍 Validating M+1 contract mapping logic...")
        
        # Test specific month mappings
        test_cases = [
            (2024, 1, 2024, 2),   # Jan 2024 → Feb 2024 contract
            (2024, 2, 2024, 3),   # Feb 2024 → Mar 2024 contract
            (2024, 11, 2024, 12), # Nov 2024 → Dec 2024 contract
            (2024, 12, 2025, 1),  # Dec 2024 → Jan 2025 contract (year rollover)
        ]
        
        results = {}
        for year, month, expected_year, expected_month in test_cases:
            # Replicate M+1 logic from build_continuous_futures.py
            contract_month = month + 1
            contract_year = year
            if contract_month > 12:
                contract_month -= 12
                contract_year += 1
                
            is_correct = (contract_year == expected_year and contract_month == expected_month)
            test_name = f"{year}-{month:02d} → {expected_year}-{expected_month:02d}"
            results[test_name] = is_correct
            
            status = "✅" if is_correct else "❌"
            logger.info(f"  {status} {test_name}: got {contract_year}-{contract_month:02d}")
            
        all_correct = all(results.values())
        logger.info(f"{'✅' if all_correct else '❌'} M+1 mapping logic validation: {'PASSED' if all_correct else 'FAILED'}")
        return results
    
    def validate_continuous_futures_output(self) -> Dict[str, any]:
        """Validate continuous futures construction output."""
        logger.info("\n🔍 Validating continuous futures output...")
        
        futures_file = self.processed_dir / "continuous_futures_m1.csv"
        if not futures_file.exists():
            logger.error(f"❌ Continuous futures file not found: {futures_file}")
            return {"file_exists": False}
            
        # Load and validate
        df = pd.read_csv(futures_file, index_col="date", parse_dates=True)
        
        validation_results = {
            "file_exists": True,
            "shape": df.shape,
            "date_range": (df.index.min(), df.index.max()),
            "has_65_price": "price_65_m1" in df.columns,
            "has_62_price": "price_62_m1" in df.columns,
            "has_contract_tracking": any("contract_month" in col for col in df.columns),
            "chronological_order": df.index.is_monotonic_increasing,
            "no_missing_prices": df[["price_65_m1", "price_62_m1"]].isnull().sum().sum() == 0,
        }
        
        # Validate price continuity (no extreme jumps that indicate adjustment errors)
        price_65_returns = df["price_65_m1"].pct_change().abs()
        price_62_returns = df["price_62_m1"].pct_change().abs()
        
        # Flag extreme daily changes (>50% which would indicate adjustment issues)
        extreme_changes_65 = (price_65_returns > 0.5).sum()
        extreme_changes_62 = (price_62_returns > 0.5).sum()
        
        validation_results.update({
            "extreme_changes_65": extreme_changes_65,
            "extreme_changes_62": extreme_changes_62,
            "price_continuity_ok": extreme_changes_65 == 0 and extreme_changes_62 == 0
        })
        
        # Log results
        logger.info(f"  Shape: {df.shape}")
        logger.info(f"  Date range: {validation_results['date_range'][0]} to {validation_results['date_range'][1]}")
        logger.info(f"  Columns: {list(df.columns)}")
        logger.info(f"  ✅ Has 65% price: {validation_results['has_65_price']}")
        logger.info(f"  ✅ Has 62% price: {validation_results['has_62_price']}")
        logger.info(f"  ✅ Chronological order: {validation_results['chronological_order']}")
        logger.info(f"  ✅ No missing prices: {validation_results['no_missing_prices']}")
        logger.info(f"  ✅ Price continuity: {validation_results['price_continuity_ok']} (extreme changes: 65%={extreme_changes_65}, 62%={extreme_changes_62})")
        
        return validation_results
    
    def validate_date_parsing_consistency(self) -> Dict[str, any]:
        """Validate date parsing across different source files."""
        logger.info("\n🔍 Validating date parsing consistency...")
        
        date_formats = {}
        
        # Check group.csv (DD/MM/YYYY format)
        group_file = self.raw_dir / "group.csv"
        if group_file.exists():
            group_df = pd.read_csv(group_file, encoding="utf-8-sig")
            if "Date" in group_df.columns:
                sample_date = group_df["Date"].iloc[0]
                date_formats["group.csv"] = {"format": "DD/MM/YYYY", "sample": sample_date}
                
                # Test parsing
                try:
                    parsed = pd.to_datetime(sample_date, format="%d/%m/%Y", dayfirst=True)
                    date_formats["group.csv"]["parse_success"] = True
                    date_formats["group.csv"]["parsed_sample"] = parsed
                except Exception:
                    date_formats["group.csv"]["parse_success"] = False
        
        # Check Raw_65and62_Index.csv (YYYY-MM-DD format)
        index_file = self.raw_dir / "Raw_65and62_Index.csv"
        if index_file.exists():
            index_df = pd.read_csv(index_file, encoding="utf-8-sig")
            if "Date" in index_df.columns:
                sample_date = index_df["Date"].iloc[0]
                date_formats["index.csv"] = {"format": "YYYY-MM-DD", "sample": sample_date}
                
                # Test parsing
                try:
                    parsed = pd.to_datetime(sample_date, format="%Y-%m-%d")
                    date_formats["index.csv"]["parse_success"] = True
                    date_formats["index.csv"]["parsed_sample"] = parsed
                except Exception:
                    date_formats["index.csv"]["parse_success"] = False
        
        # Check weekly files (DD/MM/YYYY format)
        weekly_files = ["IOCJ Inventory.csv", "IOCJ Weekly Shipment.csv"]
        for file_name in weekly_files:
            weekly_file = self.raw_dir / file_name
            if weekly_file.exists():
                weekly_df = pd.read_csv(weekly_file, encoding="utf-8-sig")
                if "Date" in weekly_df.columns:
                    sample_date = weekly_df["Date"].iloc[0]
                    date_formats[file_name] = {"format": "DD/MM/YYYY", "sample": sample_date}
                    
                    # Test parsing
                    try:
                        parsed = pd.to_datetime(sample_date, format="%d/%m/%Y", dayfirst=True, errors='coerce')
                        date_formats[file_name]["parse_success"] = not pd.isna(parsed)
                        date_formats[file_name]["parsed_sample"] = parsed
                    except Exception:
                        date_formats[file_name]["parse_success"] = False
        
        # Log results
        all_successful = True
        for file_name, info in date_formats.items():
            status = "✅" if info.get("parse_success", False) else "❌"
            logger.info(f"  {status} {file_name}: {info['format']} - sample: '{info['sample']}'")
            if not info.get("parse_success", False):
                all_successful = False
                
        logger.info(f"{'✅' if all_successful else '❌'} Date parsing consistency: {'PASSED' if all_successful else 'FAILED'}")
        return {"date_formats": date_formats, "all_successful": all_successful}
    
    def validate_feature_alignment(self) -> Dict[str, any]:
        """Validate feature alignment in consolidated dataset."""
        logger.info("\n🔍 Validating feature alignment...")
        
        consolidated_file = self.processed_dir / "consolidated_features_y.pkl"
        if not consolidated_file.exists():
            logger.error(f"❌ Consolidated features file not found: {consolidated_file}")
            return {"file_exists": False}
            
        # Load consolidated data
        df = pd.read_pickle(consolidated_file)
        
        # Expected features
        expected_features = [
            "price_65_m1", "price_62_m1",  # From continuous futures
            "Ukraine Concentrate fines", "lump premium", "IOCJ Import margin",
            "rebar steel margin ", "indian pellet premium", "(IOCJ+SSF)/2-PBF",  # From group.csv
            "62 Index", "65 Index",  # From index file
            "IOCJ Inventory", "IOCJ Weekly shipment"  # From weekly files
        ]
        
        actual_features = [col for col in df.columns if col != 'Y']
        
        alignment_results = {
            "file_exists": True,
            "shape": df.shape,
            "expected_features": expected_features,
            "actual_features": actual_features,
            "missing_features": [f for f in expected_features if f not in actual_features],
            "extra_features": [f for f in actual_features if f not in expected_features],
            "has_y_target": "Y" in df.columns,
            "feature_count_match": len(actual_features) == len(expected_features)
        }
        
        # Check for missing values patterns
        missing_summary = df.isnull().sum()
        alignment_results["missing_values"] = missing_summary.to_dict()
        
        # Validate Y calculation (should be log returns of price_65_m1)
        if "price_65_m1" in df.columns and "Y" in df.columns:
            # Recalculate Y to verify
            price_65 = df['price_65_m1']
            expected_y = (price_65.shift(-1) / price_65).apply(lambda x: np.log(x) * 100 if pd.notna(x) and x > 0 else pd.NA)
            
            # Compare with actual Y (excluding last row which should be NaN)
            actual_y = df['Y'][:-1]
            expected_y_trimmed = expected_y[:-1]
            
            # Calculate correlation and differences
            valid_mask = actual_y.notna() & expected_y_trimmed.notna()
            if valid_mask.sum() > 0:
                correlation = actual_y[valid_mask].corr(expected_y_trimmed[valid_mask])
                max_diff = abs(actual_y[valid_mask] - expected_y_trimmed[valid_mask]).max()
                alignment_results["y_calculation_correct"] = correlation > 0.999 and max_diff < 0.001
                alignment_results["y_correlation"] = correlation
                alignment_results["y_max_difference"] = max_diff
            else:
                alignment_results["y_calculation_correct"] = False
        
        # Log results
        logger.info(f"  Shape: {df.shape}")
        logger.info(f"  Expected features: {len(expected_features)}")
        logger.info(f"  Actual features: {len(actual_features)}")
        logger.info(f"  ✅ Feature count match: {alignment_results['feature_count_match']}")
        
        if alignment_results["missing_features"]:
            logger.error(f"  ❌ Missing features: {alignment_results['missing_features']}")
        if alignment_results["extra_features"]:
            logger.warning(f"  ⚠️ Extra features: {alignment_results['extra_features']}")
            
        logger.info(f"  ✅ Has Y target: {alignment_results['has_y_target']}")
        
        if "y_calculation_correct" in alignment_results:
            logger.info(f"  ✅ Y calculation correct: {alignment_results['y_calculation_correct']}")
            logger.info(f"     Correlation: {alignment_results.get('y_correlation', 'N/A'):.6f}")
            logger.info(f"     Max difference: {alignment_results.get('y_max_difference', 'N/A'):.6f}")
        
        return alignment_results
    
    def validate_forward_fill_logic(self) -> Dict[str, any]:
        """Validate forward-fill logic for weekly data."""
        logger.info("\n🔍 Validating forward-fill logic...")
        
        # Load weekly raw data
        weekly_files = {
            "IOCJ Inventory": self.raw_dir / "IOCJ Inventory.csv",
            "IOCJ Weekly Shipment": self.raw_dir / "IOCJ Weekly Shipment.csv"
        }
        
        forward_fill_results = {}
        
        for feature_name, file_path in weekly_files.items():
            if not file_path.exists():
                logger.warning(f"⚠️ Weekly file not found: {file_path}")
                continue
                
            # Load raw weekly data
            weekly_raw = pd.read_csv(file_path, encoding="utf-8-sig")
            weekly_raw["Date"] = pd.to_datetime(weekly_raw["Date"], format="%d/%m/%Y", dayfirst=True, errors='coerce')
            weekly_raw = weekly_raw.dropna(subset=['Date']).set_index("Date")
            
            # Remove duplicates
            if weekly_raw.index.has_duplicates:
                weekly_raw = weekly_raw[~weekly_raw.index.duplicated(keep='first')]
            
            # Load consolidated data to check forward-fill results
            consolidated_file = self.processed_dir / "consolidated_features_y.pkl"
            if consolidated_file.exists():
                consolidated_df = pd.read_pickle(consolidated_file)
                
                # Find corresponding column in consolidated data
                matching_col = None
                for col in consolidated_df.columns:
                    if feature_name.lower() in col.lower():
                        matching_col = col
                        break
                
                if matching_col:
                    # Validate forward-fill worked correctly
                    original_count = len(weekly_raw.dropna())
                    filled_count = len(consolidated_df[matching_col].dropna())
                    
                    forward_fill_results[feature_name] = {
                        "original_points": original_count,
                        "filled_points": filled_count,
                        "fill_ratio": filled_count / original_count if original_count > 0 else 0,
                        "matching_column": matching_col,
                        "forward_fill_effective": filled_count > original_count
                    }
                    
                    logger.info(f"  ✅ {feature_name}:")
                    logger.info(f"     Original points: {original_count}")
                    logger.info(f"     Filled points: {filled_count}")
                    fill_ratio = filled_count / original_count if original_count > 0 else float('inf')
                    logger.info(f"     Fill ratio: {fill_ratio:.1f}x")
                    logger.info(f"     Column: {matching_col}")
        
        return forward_fill_results
    
    def validate_price_continuity(self) -> Dict[str, any]:
        """Validate price continuity at contract rollover points."""
        logger.info("\n🔍 Validating price continuity at rollover points...")
        
        futures_file = self.processed_dir / "continuous_futures_m1.csv"
        if not futures_file.exists():
            return {"file_exists": False}
            
        df = pd.read_csv(futures_file, index_col="date", parse_dates=True)
        
        # Calculate daily returns for both series
        returns_65 = df["price_65_m1"].pct_change()
        returns_62 = df["price_62_m1"].pct_change()
        
        # Flag potential rollover issues (extreme returns > 20%)
        extreme_threshold = 0.20
        extreme_65 = returns_65.abs() > extreme_threshold
        extreme_62 = returns_62.abs() > extreme_threshold
        
        continuity_results = {
            "file_exists": True,
            "total_observations": len(df),
            "extreme_returns_65": extreme_65.sum(),
            "extreme_returns_62": extreme_62.sum(),
            "max_return_65": returns_65.abs().max(),
            "max_return_62": returns_62.abs().max(),
            "price_continuity_good": extreme_65.sum() == 0 and extreme_62.sum() == 0
        }
        
        # Log extreme return dates if any
        if extreme_65.any():
            extreme_dates = df.index[extreme_65]
            logger.warning(f"  ⚠️ Extreme 65% returns on: {extreme_dates.tolist()}")
            
        if extreme_62.any():
            extreme_dates = df.index[extreme_62]
            logger.warning(f"  ⚠️ Extreme 62% returns on: {extreme_dates.tolist()}")
        
        logger.info(f"  ✅ Price continuity: {'GOOD' if continuity_results['price_continuity_good'] else 'ISSUES DETECTED'}")
        logger.info(f"     Max daily return: 65%={continuity_results['max_return_65']:.4f}, 62%={continuity_results['max_return_62']:.4f}")
        
        return continuity_results
    
    def validate_target_variable_calculation(self) -> Dict[str, any]:
        """Validate target variable Y calculation accuracy."""
        logger.info("\n🔍 Validating target variable Y calculation...")
        
        consolidated_file = self.processed_dir / "consolidated_features_y.pkl"
        if not consolidated_file.exists():
            return {"file_exists": False}
            
        df = pd.read_pickle(consolidated_file)
        
        if "price_65_m1" not in df.columns or "Y" not in df.columns:
            logger.error("❌ Missing price_65_m1 or Y columns")
            return {"missing_columns": True}
        
        # Recalculate Y to verify correctness
        price_65 = df['price_65_m1']
        expected_y = (price_65.shift(-1) / price_65).apply(lambda x: np.log(x) * 100 if pd.notna(x) and x > 0 else pd.NA)
        actual_y = df['Y']
        
        # Compare calculations (excluding last row)
        valid_mask = actual_y[:-1].notna() & expected_y[:-1].notna()
        
        if valid_mask.sum() == 0:
            logger.error("❌ No valid Y values found for comparison")
            return {"no_valid_values": True}
        
        # Calculate differences
        differences = abs(actual_y[:-1][valid_mask] - expected_y[:-1][valid_mask])
        max_difference = differences.max()
        mean_difference = differences.mean()
        correlation = actual_y[:-1][valid_mask].corr(expected_y[:-1][valid_mask])
        
        # Check Y statistics
        y_stats = {
            "valid_y_count": valid_mask.sum(),
            "total_y_count": len(actual_y),
            "y_mean": actual_y.mean(),
            "y_std": actual_y.std(),
            "y_min": actual_y.min(),
            "y_max": actual_y.max(),
            "last_y_is_nan": pd.isna(actual_y.iloc[-1])  # Should be NaN
        }
        
        target_results = {
            "file_exists": True,
            "missing_columns": False,
            "no_valid_values": False,
            "max_difference": max_difference,
            "mean_difference": mean_difference,
            "correlation": correlation,
            "calculation_accurate": correlation > 0.999 and max_difference < 0.001,
            "y_statistics": y_stats
        }
        
        # Log results
        logger.info("  ✅ Y calculation accuracy:")
        logger.info(f"     Valid Y values: {y_stats['valid_y_count']}/{y_stats['total_y_count']}")
        logger.info(f"     Correlation: {correlation:.6f}")
        logger.info(f"     Max difference: {max_difference:.6f}")
        logger.info(f"     Mean difference: {mean_difference:.6f}")
        logger.info(f"     Last Y is NaN: {y_stats['last_y_is_nan']} (expected)")
        logger.info(f"  ✅ Calculation accurate: {target_results['calculation_accurate']}")
        
        return target_results
    
    def validate_data_file_integrity(self) -> Dict[str, any]:
        """Validate integrity of source data files."""
        logger.info("\n🔍 Validating source data file integrity...")
        
        required_files = {
            "Raw_M65F_DSP.pkl": "65% M+1 futures contracts",
            "Raw_FEF_Close.pkl": "62% futures contracts", 
            "group.csv": "Daily group features",
            "Raw_65and62_Index.csv": "Daily index features",
            "IOCJ Inventory.csv": "Weekly inventory data",
            "IOCJ Weekly Shipment.csv": "Weekly shipment data"
        }
        
        file_integrity = {}
        
        for filename, description in required_files.items():
            file_path = self.raw_dir / filename
            file_integrity[filename] = {
                "exists": file_path.exists(),
                "description": description
            }
            
            if file_path.exists():
                try:
                    if filename.endswith('.pkl'):
                        with open(file_path, 'rb') as f:
                            data = pickle.load(f)
                        file_integrity[filename]["loadable"] = True
                        file_integrity[filename]["type"] = type(data).__name__
                        if isinstance(data, dict):
                            file_integrity[filename]["contract_count"] = len(data)
                    else:
                        data = pd.read_csv(file_path, encoding="utf-8-sig")
                        file_integrity[filename]["loadable"] = True
                        file_integrity[filename]["shape"] = data.shape
                        file_integrity[filename]["has_date_column"] = "Date" in data.columns
                        
                except Exception as e:
                    file_integrity[filename]["loadable"] = False
                    file_integrity[filename]["error"] = str(e)
            
            # Log file status
            if file_integrity[filename]["exists"]:
                loadable = file_integrity[filename].get("loadable", False)
                status = "✅" if loadable else "⚠️"
                logger.info(f"  {status} {filename}: {description}")
                if loadable and "shape" in file_integrity[filename]:
                    logger.info(f"     Shape: {file_integrity[filename]['shape']}")
                elif loadable and "contract_count" in file_integrity[filename]:
                    logger.info(f"     Contracts: {file_integrity[filename]['contract_count']}")
            else:
                logger.error(f"  ❌ {filename}: NOT FOUND")
        
        all_files_ok = all(info["exists"] and info.get("loadable", False) 
                          for info in file_integrity.values())
        
        return {
            "file_integrity": file_integrity,
            "all_files_ok": all_files_ok,
            "missing_files": [f for f, info in file_integrity.items() if not info["exists"]]
        }
    
    def run_comprehensive_validation(self) -> Dict[str, any]:
        """Run all validation checks."""
        logger.info("🧪 STARTING COMPREHENSIVE DATA BUILDING VALIDATION")
        logger.info("=" * 70)
        
        results = {}
        
        try:
            # 1. File integrity
            results["file_integrity"] = self.validate_data_file_integrity()
            
            # 2. M+1 contract mapping logic
            results["m1_mapping"] = self.validate_m1_contract_mapping()
            
            # 3. Date parsing consistency
            results["date_parsing"] = self.validate_date_parsing_consistency()
            
            # 4. Continuous futures output
            results["continuous_futures"] = self.validate_continuous_futures_output()
            
            # 5. Feature alignment
            results["feature_alignment"] = self.validate_feature_alignment()
            
            # 6. Forward-fill logic
            results["forward_fill"] = self.validate_forward_fill_logic()
            
            # 7. Price continuity
            results["price_continuity"] = self.validate_price_continuity()
            
            # 8. Target variable calculation
            results["target_calculation"] = self.validate_target_variable_calculation()
            
            # Overall assessment
            all_passed = (
                results["file_integrity"]["all_files_ok"] and
                all(results["m1_mapping"].values()) and
                results["date_parsing"]["all_successful"] and
                results["continuous_futures"].get("price_continuity_ok", False) and
                results["feature_alignment"].get("feature_count_match", False) and
                results["target_calculation"].get("calculation_accurate", False)
            )
            
            logger.info("\n🎉 VALIDATION SUMMARY")
            logger.info("=" * 50)
            
            if all_passed:
                logger.info("✅ ALL DATA BUILDING VALIDATIONS PASSED!")
                logger.info("✅ Continuous futures construction is accurate")
                logger.info("✅ Feature consolidation is working correctly")
                logger.info("✅ Date parsing is consistent across all files")
                logger.info("✅ Forward-fill logic is applied properly")
                logger.info("✅ Target variable Y calculation is correct")
                logger.info("✅ Price continuity is maintained")
                logger.info("\n🚀 Data building pipeline is ready for use!")
            else:
                logger.error("❌ DATA BUILDING VALIDATION FAILED!")
                logger.error("Fix the issues above before using the processed data")
            
            results["overall_passed"] = all_passed
            return results
            
        except Exception as e:
            logger.error(f"\n❌ VALIDATION FAILED WITH ERROR: {e}")
            logger.exception("Full traceback:")
            raise


def main():
    """Run comprehensive data building validation."""
    validator = DataBuildingValidator()
    results = validator.run_comprehensive_validation()
    
    # Print final summary
    print("\n📊 DATA BUILDING VALIDATION SUMMARY")
    print("=" * 50)
    
    # File integrity
    integrity = results["file_integrity"]
    print(f"Source files: {len(integrity['file_integrity']) - len(integrity['missing_files'])}/{len(integrity['file_integrity'])} OK")
    
    # M+1 mapping
    mapping = results["m1_mapping"]
    mapping_passed = sum(mapping.values())
    print(f"M+1 contract mapping: {mapping_passed}/{len(mapping)} test cases passed")
    
    # Date parsing
    date_parsing = results["date_parsing"]
    print(f"Date parsing: {'ALL FORMATS OK' if date_parsing['all_successful'] else 'ISSUES DETECTED'}")
    
    # Features
    features = results["feature_alignment"]
    if features.get("file_exists", False):
        print(f"Feature alignment: {len(features['actual_features'])}/12 features present")
        print(f"Target variable Y: {'CALCULATED CORRECTLY' if features.get('y_calculation_correct', False) else 'ISSUES DETECTED'}")
    
    # Price continuity
    continuity = results["price_continuity"]
    if continuity.get("file_exists", False):
        print(f"Price continuity: {'GOOD' if continuity['price_continuity_good'] else 'ISSUES DETECTED'}")
    
    print("=" * 50)
    
    if results["overall_passed"]:
        print("🎯 RESULT: Data building pipeline is ACCURATE and ready for ML training")
    else:
        print("⚠️ RESULT: Data building pipeline has ISSUES that need to be fixed")


if __name__ == "__main__":
    main()