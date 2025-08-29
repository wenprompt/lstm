#!/usr/bin/env python3
"""
Script to inspect the structure and contents of pickle files.
"""

import pickle
import pandas as pd
from pathlib import Path

def inspect_pickle_file(file_path: str) -> None:
    """Inspect a pickle file and print its structure."""
    print(f"\n{'='*60}")
    print(f"INSPECTING: {file_path}")
    print(f"{'='*60}")
    
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        print(f"Type: {type(data)}")
        
        if isinstance(data, pd.DataFrame):
            print(f"Shape: {data.shape}")
            print(f"Columns: {list(data.columns)}")
            print(f"Index type: {type(data.index)}")
            print(f"Date range: {data.index.min()} to {data.index.max()}")
            print("\nFirst 5 rows:")
            print(data.head())
            print("\nLast 5 rows:")
            print(data.tail())
            print("\nData types:")
            print(data.dtypes)
            print("\nBasic statistics:")
            print(data.describe())
            
        elif isinstance(data, dict):
            print(f"Dictionary with {len(data)} keys: {list(data.keys())}")
            for key, value in data.items():
                print(f"\n  Key '{key}':")
                print(f"    Type: {type(value)}")
                if isinstance(value, pd.DataFrame):
                    print(f"    Shape: {value.shape}")
                    print(f"    Columns: {list(value.columns)}")
                    if hasattr(value.index, 'min') and hasattr(value.index, 'max'):
                        print(f"    Date range: {value.index.min()} to {value.index.max()}")
                elif isinstance(value, (list, tuple)):
                    print(f"    Length: {len(value)}")
                    if len(value) > 0:
                        print(f"    First item: {value[0]}")
                        print(f"    Last item: {value[-1]}")
                else:
                    print(f"    Value: {value}")
                    
        elif isinstance(data, (list, tuple)):
            print(f"Length: {len(data)}")
            if len(data) > 0:
                print(f"First item type: {type(data[0])}")
                print(f"First item: {data[0]}")
                if len(data) > 1:
                    print(f"Last item: {data[-1]}")
                    
        else:
            print(f"Content: {data}")
            
    except Exception as e:
        print(f"ERROR loading {file_path}: {e}")

def main():
    """Main function to inspect both pickle files."""
    data_dir = Path("/home/wenhaowang/projects/m65lstm/data")
    
    pickle_files = [
        "Raw_M65F_DSP.pkl",
        "Raw_FEF_Close.pkl"
    ]
    
    for file_name in pickle_files:
        file_path = data_dir / file_name
        if file_path.exists():
            inspect_pickle_file(str(file_path))
        else:
            print(f"\nFile not found: {file_path}")

if __name__ == "__main__":
    main()