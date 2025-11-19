#!/usr/bin/env python3
"""
Synthetic Data Generation Script for Heart Disease Dataset

This script generates synthetic tabular data using CTGAN/SDV to augment the 
cleaned heart disease dataset from 1,888 rows to exactly 8,768 rows.

Requirements:
    - Install PyTorch CPU first:
      pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    - Then install synthesis requirements:
      pip install -r requirements-synthesis.txt

Usage:
    python scripts/generate_synthetic_compatible.py
"""

import os
import sys
import warnings
from pathlib import Path
import pandas as pd
import numpy as np

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Constants
TARGET_TOTAL_ROWS = 8768
CLEANED_DATA_PATH = 'data/cleaned_merged_heart_dataset.csv'
OUTPUT_PATH = 'data/synthetic_augmented_heart_dataset.csv'
RANDOM_SEED = 42

# Set random seeds for reproducibility
np.random.seed(RANDOM_SEED)


def check_dependencies():
    """Check if required packages are available and fallback appropriately."""
    print("Checking dependencies...")
    
    # Check for torch
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__} found")
    except ImportError:
        print("✗ PyTorch not found!")
        print("  Install with: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu")
        sys.exit(1)
    
    # Try SDV first, fallback to CTGAN
    use_sdv = False
    try:
        from sdv.single_table import CTGANSynthesizer
        from sdv.metadata import SingleTableMetadata
        print("✓ SDV found - will use SDV's CTGANSynthesizer")
        use_sdv = True
    except ImportError:
        try:
            from ctgan import CTGAN
            print("✓ CTGAN found - will use standalone CTGAN")
            print("  (SDV not available, using fallback)")
        except ImportError:
            print("✗ Neither SDV nor CTGAN found!")
            print("  Install with: pip install -r requirements-synthesis.txt")
            sys.exit(1)
    
    return use_sdv


def load_and_validate_data(filepath):
    """Load and validate the cleaned dataset."""
    print(f"\nLoading data from {filepath}...")
    
    if not os.path.exists(filepath):
        print(f"✗ Error: File not found: {filepath}")
        sys.exit(1)
    
    df = pd.read_csv(filepath)
    print(f"✓ Loaded {len(df)} rows, {len(df.columns)} columns")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Data types:\n{df.dtypes.to_dict()}")
    
    # Check for target column
    target_candidates = ['target', 'HeartDisease', 'heart_disease', 'label']
    target_col = None
    for col in target_candidates:
        if col in df.columns:
            target_col = col
            break
    
    if target_col:
        print(f"\n  Target column: '{target_col}'")
        print(f"  Class distribution: {df[target_col].value_counts().to_dict()}")
    
    return df, target_col


def generate_synthetic_data_sdv(original_df, n_samples, target_col=None):
    """Generate synthetic data using SDV's CTGANSynthesizer."""
    from sdv.single_table import CTGANSynthesizer
    from sdv.metadata import SingleTableMetadata
    
    print(f"\n{'='*60}")
    print("Generating synthetic data using SDV CTGANSynthesizer...")
    print(f"{'='*60}")
    
    # Create metadata
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(original_df)
    
    # Print detected metadata
    print("\nDetected metadata:")
    print(f"  Columns: {list(metadata.columns.keys())}")
    
    # Create and configure synthesizer
    synthesizer = CTGANSynthesizer(
        metadata,
        epochs=300,
        verbose=True,
        cuda=False  # Use CPU
    )
    
    print("\nTraining CTGAN model on cleaned dataset...")
    print(f"  Training samples: {len(original_df)}")
    print(f"  Features: {len(original_df.columns)}")
    
    # Fit the model
    synthesizer.fit(original_df)
    
    print(f"\nGenerating {n_samples} synthetic samples...")
    synthetic_df = synthesizer.sample(num_rows=n_samples)
    
    print(f"✓ Generated {len(synthetic_df)} synthetic rows")
    
    return synthetic_df


def generate_synthetic_data_ctgan(original_df, n_samples, target_col=None):
    """Generate synthetic data using standalone CTGAN."""
    from ctgan import CTGAN
    
    print(f"\n{'='*60}")
    print("Generating synthetic data using standalone CTGAN...")
    print(f"{'='*60}")
    
    # Identify discrete columns
    discrete_columns = []
    for col in original_df.columns:
        if original_df[col].dtype in ['int64', 'int32']:
            # Check if it's truly discrete (limited unique values)
            n_unique = original_df[col].nunique()
            if n_unique <= 20:  # Heuristic for discrete columns
                discrete_columns.append(col)
    
    print(f"\nDetected discrete columns: {discrete_columns}")
    
    # Create and configure CTGAN
    ctgan = CTGAN(
        epochs=300,
        verbose=True,
        cuda=False  # Use CPU
    )
    
    print("\nTraining CTGAN model on cleaned dataset...")
    print(f"  Training samples: {len(original_df)}")
    print(f"  Features: {len(original_df.columns)}")
    
    # Fit the model
    ctgan.fit(original_df, discrete_columns)
    
    print(f"\nGenerating {n_samples} synthetic samples...")
    synthetic_df = ctgan.sample(n_samples)
    
    print(f"✓ Generated {len(synthetic_df)} synthetic rows")
    
    return synthetic_df


def ensure_compatibility(synthetic_df, original_df):
    """Ensure synthetic data is fully compatible with original data."""
    print("\nEnsuring compatibility with original dataset...")
    
    # 1. Ensure exact column order
    synthetic_df = synthetic_df[original_df.columns]
    print("✓ Column order matched")
    
    # 2. Cast dtypes to match original
    for col in original_df.columns:
        original_dtype = original_df[col].dtype
        try:
            if original_dtype == 'int64' or original_dtype == 'int32':
                synthetic_df[col] = synthetic_df[col].round().astype(original_dtype)
            elif original_dtype == 'float64' or original_dtype == 'float32':
                synthetic_df[col] = synthetic_df[col].astype(original_dtype)
            else:
                synthetic_df[col] = synthetic_df[col].astype(original_dtype)
        except Exception as e:
            print(f"  Warning: Could not cast {col} to {original_dtype}: {e}")
    
    print("✓ Data types matched")
    
    # 3. Clip values to reasonable ranges based on original data
    for col in original_df.columns:
        if original_df[col].dtype in ['int64', 'int32', 'float64', 'float32']:
            min_val = original_df[col].min()
            max_val = original_df[col].max()
            synthetic_df[col] = synthetic_df[col].clip(min_val, max_val)
    
    print("✓ Values clipped to original ranges")
    
    return synthetic_df


def concatenate_and_deduplicate(original_df, synthetic_df):
    """Concatenate original and synthetic data, then remove duplicates."""
    print("\nCombining datasets...")
    
    initial_synthetic_count = len(synthetic_df)
    print(f"  Original data: {len(original_df)} rows")
    print(f"  Synthetic data: {initial_synthetic_count} rows")
    
    # Concatenate
    combined_df = pd.concat([original_df, synthetic_df], ignore_index=True)
    print(f"  Combined (before deduplication): {len(combined_df)} rows")
    
    # Remove exact duplicates
    combined_df_dedup = combined_df.drop_duplicates()
    duplicates_removed = len(combined_df) - len(combined_df_dedup)
    
    print(f"✓ Removed {duplicates_removed} duplicate rows")
    print(f"  Final row count: {len(combined_df_dedup)} rows")
    
    return combined_df_dedup, duplicates_removed


def save_augmented_dataset(df, output_path):
    """Save the augmented dataset to CSV."""
    print(f"\nSaving augmented dataset to {output_path}...")
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    df.to_csv(output_path, index=False)
    print(f"✓ Saved {len(df)} rows to {output_path}")
    
    # Verify file was created
    if os.path.exists(output_path):
        file_size = os.path.getsize(output_path) / 1024  # KB
        print(f"  File size: {file_size:.2f} KB")
    else:
        print("✗ Error: File was not created!")


def main():
    """Main execution function."""
    print("="*60)
    print("SYNTHETIC DATA GENERATION FOR HEART DISEASE DATASET")
    print("="*60)
    
    # Check dependencies
    use_sdv = check_dependencies()
    
    # Load original data
    original_df, target_col = load_and_validate_data(CLEANED_DATA_PATH)
    
    # Calculate number of synthetic samples needed
    n_synthetic = TARGET_TOTAL_ROWS - len(original_df)
    print(f"\nTarget total rows: {TARGET_TOTAL_ROWS}")
    print(f"Current rows: {len(original_df)}")
    print(f"Synthetic rows to generate: {n_synthetic}")
    
    if n_synthetic <= 0:
        print("\n✗ Dataset already has target number of rows or more!")
        sys.exit(0)
    
    # Generate synthetic data
    if use_sdv:
        synthetic_df = generate_synthetic_data_sdv(original_df, n_synthetic, target_col)
    else:
        synthetic_df = generate_synthetic_data_ctgan(original_df, n_synthetic, target_col)
    
    # Ensure compatibility
    synthetic_df = ensure_compatibility(synthetic_df, original_df)
    
    # Concatenate and deduplicate
    augmented_df, duplicates = concatenate_and_deduplicate(original_df, synthetic_df)
    
    # Save result
    save_augmented_dataset(augmented_df, OUTPUT_PATH)
    
    # Final summary
    print("\n" + "="*60)
    print("GENERATION SUMMARY")
    print("="*60)
    print(f"Original rows: {len(original_df)}")
    print(f"Synthetic rows generated: {n_synthetic}")
    print(f"Duplicates removed: {duplicates}")
    print(f"Final augmented dataset: {len(augmented_df)} rows")
    print(f"Target achieved: {len(augmented_df) == TARGET_TOTAL_ROWS}")
    print(f"Output file: {OUTPUT_PATH}")
    print("="*60)
    
    if target_col:
        print("\nClass distribution in augmented dataset:")
        print(augmented_df[target_col].value_counts().to_dict())


if __name__ == "__main__":
    main()
