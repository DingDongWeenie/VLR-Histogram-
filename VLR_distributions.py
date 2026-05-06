import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np


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


def generate_synthetic_distributions(n_samples=500):
    """
    Generate synthetic distributions based on professional VALORANT player metrics.
    
    Returns:
        dict: Five distributions representing different probability shapes
    """
    np.random.seed(42)
    
    # 1. EXPONENTIAL - Time to First Elimination (seconds)
    # Most eliminations happen early, exponentially fewer as time increases
    exponential = np.random.exponential(scale=8, size=n_samples)
    exponential = np.clip(exponential, 0, 45)
    
    # 2. NORMAL - Average Combat Score (ACS)
    # Most players cluster around the mean (~180 ACS)
    normal = np.random.normal(loc=180, scale=25, size=n_samples)
    normal = np.clip(normal, 50, 350)
    
    # 3. POISSON - Kills Per Round
    # Count-based data with random occurrences per round
    poisson = np.random.poisson(lam=1.8, size=n_samples)
    
    # 4. BINOMIAL - Headshot Rate (out of 10 shots)
    # Binary outcomes: hit or miss, representing shot accuracy
    binomial = np.random.binomial(n=10, p=0.65, size=n_samples)
    
    # 5. TRIANGULAR - Ability Usage Frequency (%)
    # Peak usage mid-range, lower at extremes
    triangular = np.random.triangular(left=10, mode=65, right=100, size=n_samples)
    
    return {
        'exponential': exponential,
        'normal': normal,
        'poisson': poisson,
        'binomial': binomial,
        'triangular': triangular
    }


def plot_four_distributions(output_path=None):
    """Create a grid of distribution charts."""
    
    print('Generating synthetic professional VALORANT player distributions...')
    distributions = generate_synthetic_distributions(n_samples=500)
    
    sns.set(style='whitegrid')
    fig, axes = plt.subplots(3, 2, figsize=(16, 14))
    fig.suptitle('Probability Distributions in Professional VALORANT Player Metrics', fontsize=14, fontweight='bold', y=0.995)
    
    # Flatten axes for easier iteration
    axes_flat = axes.flatten()
    
    # Color palette for each distribution type
    colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#95e1d3', '#f38181']
    
    # Define all distributions with their metadata
    dist_configs = [
        {
            'name': 'exponential',
            'title': 'Exponential - Time to First Elimination (seconds)',
            'xlabel': 'Seconds',
            'ylabel': 'Number of Players',
            'description': 'Most eliminations occur early,\nexponentially fewer over time'
        },
        {
            'name': 'normal',
            'title': 'Normal - Average Combat Score (ACS)',
            'xlabel': 'ACS',
            'ylabel': 'Number of Players',
            'description': 'Most players cluster around\nthe mean performance'
        },
        {
            'name': 'poisson',
            'title': 'Poisson - Kills Per Round (Individual Player)',
            'xlabel': 'Number of Kills',
            'ylabel': 'Number of Players',
            'description': 'Individual player kills per round\n(1 player, not team aggregate)'
        },
        {
            'name': 'binomial',
            'title': 'Binomial - Headshot Rate (out of 10 shots)',
            'xlabel': 'Headshots',
            'ylabel': 'Number of Players',
            'description': 'Binary outcomes: success/failure\nfrom repeated trials'
        },
        {
            'name': 'triangular',
            'title': 'Triangular - Ability Usage Frequency (%)',
            'xlabel': 'Frequency (%)',
            'ylabel': 'Number of Players',
            'description': 'Peak usage mid-range,\nlower at extremes'
        },
    ]
    
    for idx, config in enumerate(dist_configs):
        ax = axes_flat[idx]
        values = distributions[config['name']]
        mean_val = values.mean()
        median_val = np.median(values)
        
        sns.histplot(values, bins=25, kde=True, color=colors[idx], ax=ax)
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.1f}')
        ax.axvline(median_val, color='green', linestyle='--', linewidth=2, label=f'Median: {median_val:.1f}')
        ax.set_title(config['title'], fontweight='bold', fontsize=11)
        ax.set_xlabel(config['xlabel'], fontsize=10)
        ax.set_ylabel(config['ylabel'], fontsize=10)
        ax.legend(fontsize=9)
        ax.tick_params(labelsize=9)
        ax.text(0.98, 0.97, config['description'], 
                transform=ax.transAxes, fontsize=8, verticalalignment='top', 
                horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Hide the last subplot if we have an odd number of distributions
    if len(dist_configs) % 2 != 0:
        axes_flat[-1].set_visible(False)
    
    plt.tight_layout(pad=3.0, h_pad=3.0, w_pad=2.5)
    
    # Print statistics
    print('\n=== Distribution Statistics ===')
    print(f'Sample size per distribution: {len(distributions["normal"])}')
    
    for dist_name, values in distributions.items():
        print(f'\n{dist_name.upper()}:')
        print(f'  Mean: {values.mean():.2f}')
        print(f'  Median: {np.median(values):.2f}')
        print(f'  Std Dev: {values.std():.2f}')
        print(f'  Skewness: {pd.Series(values).skew():.3f}')
        print(f'  Kurtosis: {pd.Series(values).kurtosis():.3f}')
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f'\nChart saved to {output_path}')
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Generate 4 distribution shapes from professional VALORANT player metrics.')
    parser.add_argument(
        '--output',
        help='Optional path to save the chart image',
        default=None,
    )
    args = parser.parse_args()
    
    plot_four_distributions(args.output)


if __name__ == '__main__':
    main()
