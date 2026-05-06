import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np


def download_dataset_from_kaggle(output_dir='bronze'):
    """
    Download the VLR VALORANT dataset from Kaggle using kagglehub.
    Skips download if the dataset already exists locally.
    
    Install dependencies first:
    pip install kagglehub[pandas-datasets]
    
    Args:
        output_dir: Directory to save the dataset (default: 'bronze')
    """
    # Check if dataset already exists
    if os.path.exists(output_dir) and os.path.isdir(output_dir):
        print(f"Dataset already exists at '{output_dir}/'")
        print(f"To use it, run: python VLR.py {output_dir}")
        return None
    
    try:
        import kagglehub
        from kagglehub import KaggleDatasetAdapter
        
        print(f'Downloading VLR VALORANT dataset to {output_dir}...')
        
        # Load the latest version of the dataset
        df = kagglehub.load_dataset(
            KaggleDatasetAdapter.PANDAS,
            "pvcodes/vlr-gg-valorant-match-analytics-dataset",
            "",  # file_path - leave empty to get all files
        )
        
        print(f"Dataset downloaded successfully!")
        print(f"First 5 records:")
        print(df.head())
        
        return df
    except ImportError:
        print("kagglehub not installed. Install it with:")
        print("pip install kagglehub[pandas-datasets]")
        return None


def find_acs_column(columns):
    candidates = [
        'average_combat_score',
        'ACS',
        'acs',
        'Average ACS',
        'Avg ACS',
        'average_acs',
        'avg_acs',
        'avg acs',
        'average acs',
    ]
    for candidate in candidates:
        if candidate in columns:
            return candidate
    return None


def load_dataset_optimized(path):
    """Load and aggregate ACS data efficiently from large number of CSV files, deduplicating by player."""
    if not os.path.exists(path):
        raise FileNotFoundError(f'Dataset not found: {path}')
    
    if os.path.isdir(path):
        csv_files = sorted(list(Path(path).rglob('data.csv')))
        if not csv_files:
            raise FileNotFoundError(f'No data.csv files found in {path}')
        
        print(f'Found {len(csv_files)} CSV files to process...')
        print('Extracting ACS values (deduplicating by player_id)...')
        
        # Use a dictionary to deduplicate by player_id
        player_acs = {}  # {player_id: average_combat_score}
        acs_col = None
        
        # Process files in optimized batches
        batch_size = 500
        for batch_idx in range(0, len(csv_files), batch_size):
            batch = csv_files[batch_idx:batch_idx+batch_size]
            
            for csv_file in batch:
                try:
                    # Read player_id and average_combat_score
                    df = pd.read_csv(csv_file, usecols=['player_id', 'average_combat_score'])
                    if acs_col is None:
                        acs_col = 'average_combat_score'
                    
                    # Process each row, keeping only one record per player_id
                    for idx, row in df.iterrows():
                        player_id = row['player_id']
                        acs_val = row['average_combat_score']
                        
                        # Only add if valid ACS value and not already recorded
                        if pd.notna(acs_val):
                            try:
                                acs_val = float(acs_val)
                                if player_id not in player_acs:
                                    player_acs[player_id] = acs_val
                            except (ValueError, TypeError):
                                pass
                except (FileNotFoundError, KeyError, ValueError):
                    # Column doesn't exist or file is corrupted, skip
                    pass
                except Exception as e:
                    pass
            
            # Print progress
            processed = min(batch_idx + batch_size, len(csv_files))
            print(f'  Processed {processed}/{len(csv_files)} files... ({len(player_acs)} unique players)')
        
        if not player_acs:
            raise ValueError('No valid ACS data could be found.')
        
        # Convert to DataFrame for consistency
        all_acs_values = list(player_acs.values())
        df = pd.DataFrame({acs_col: all_acs_values})
        return df, acs_col
    else:
        # Single CSV file
        df = pd.read_csv(path)
        acs_col = find_acs_column(df.columns)
        if acs_col is None:
            raise ValueError('Could not find an ACS column.')
        return df, acs_col


def load_dataset(path):
    return load_dataset_optimized(path)


def plot_acs_histogram(df, acs_col, output_path=None):
    values = df[acs_col].dropna().astype(float)
    average_acs = values.mean()
    median_acs = values.median()
    min_acs = values.min()
    max_acs = values.max()
    std_acs = values.std()
    
    print(f'\n=== ACS Statistics for Professional VALORANT Players ===')
    print(f'Total players analyzed: {len(values)}')
    print(f'Average ACS: {average_acs:.2f}')
    print(f'Median ACS: {median_acs:.2f}')
    print(f'Min ACS: {min_acs:.2f}')
    print(f'Max ACS: {max_acs:.2f}')
    print(f'Std Dev: {std_acs:.2f}')
    print('=' * 60 + '\n')

    sns.set(style='whitegrid')
    plt.figure(figsize=(10, 6))
    sns.histplot(values, bins=30, kde=True, color='#b87333')
    plt.title('Histogram of Average Combat Score for Professional VALORANT Players')
    plt.xlabel('Average ACS')
    plt.ylabel('Number of Players')
    plt.axvline(average_acs, color='red', linestyle='--', linewidth=2, label=f'Mean: {average_acs:.2f}')
    plt.axvline(median_acs, color='green', linestyle='--', linewidth=2, label=f'Median: {median_acs:.2f}')
    plt.legend()
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path)
        print(f'Histogram saved to {output_path}')
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Generate a histogram of average ACS from the VLR bronze dataset.')
    parser.add_argument(
        'dataset',
        nargs='?', 
        default='bronze',
        help='Path to the VLR bronze dataset directory or CSV file (default: bronze)'
    )
    parser.add_argument(
        '--output',
        help='Optional path to save the histogram image',
        default=None,
    )
    parser.add_argument(
        '--download',
        help='Download the VLR dataset from Kaggle (requires kagglehub installed)',
        action='store_true',
    )
    args = parser.parse_args()

    if args.download:
        download_dataset_from_kaggle()
        return

    df, acs_col = load_dataset(args.dataset)
    plot_acs_histogram(df, acs_col, args.output)


if __name__ == '__main__':
    main()
