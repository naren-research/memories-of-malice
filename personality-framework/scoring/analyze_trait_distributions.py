"""
Analyze OCEAN Personality Trait Score Distributions

This script generates histograms for each OCEAN personality trait showing
the distribution of scores across the dataset with configurable bin width.
Additionally, it computes per-trait z-scores using the formula:
z_i,t = (s_i,t - μ_t) / σ_t
where s_i,t is the score for sample i on trait t, μ_t is the mean score
for trait t, and σ_t is the standard deviation for trait t.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import numpy as np
from pathlib import Path
from scipy import stats
import seaborn as sns
import json
import sys
import os

# CONFIGURATION PARAMETERS
BIN_WIDTH = 0.001  # Reduced bin width as requested
OUTPUT_DPI = 300    # Resolution for saved plots
ALPHA_TRANSPARENCY = 0.7  # Transparency for histogram bars

# Font sizes (increased by 50%)
FONT_SIZE_TITLE = 42
FONT_SIZE_AXIS_LABEL = 39
FONT_SIZE_LEGEND = 39
FONT_SIZE_TICKS = 33

# Enhanced color palette for better visual separation
TRAIT_COLORS = {
    'openness': '#9b5de5',
    'conscientiousness': '#f15bb5',
    'extraversion': '#fee440',
    'agreeableness': '#00bbf9',
    'neuroticism': '#00f5d4'
}

# Set style for better aesthetics with custom styling
plt.style.use('default')  # Start with default style
plt.rcParams['figure.facecolor'] = 'white'  # White figure background
plt.rcParams['axes.facecolor'] = 'white'   # White axes background
plt.rcParams['grid.color'] = 'black'       # Black grid lines
plt.rcParams['axes.edgecolor'] = 'black'   # Black axes
plt.rcParams['xtick.color'] = 'black'      # Black x-ticks
plt.rcParams['ytick.color'] = 'black'      # Black y-ticks
plt.rcParams['text.color'] = 'black'       # Black text
plt.rcParams['font.size'] = FONT_SIZE_TICKS # Default font size

# Match IEEE paper font (Times/serif) - IEEE uses Times for text
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif']
plt.rcParams['mathtext.fontset'] = 'dejavuserif'  # For math symbols

def get_trait_scores(df, traits):
    """Extract scores for each trait from the dataframe."""
    trait_scores = {}
    for trait in traits:
        scores = []
        # Check if we have the dictionary column or if it's already expanded (cached)
        # The cached parquet might save dicts as structs or similar, but let's assume
        # we are reading the processed parquet which has 'dialogue_scores' as a dict
        # or we might need to re-extract.
        # Actually, for the cached file 'ocean_scores_with_z_scores.parquet', 
        # 'dialogue_scores' is preserved as a struct/dict column.
        
        # Vectorized extraction is faster if possible, but iteration is safe
        for _, row in df.iterrows():
            score_data = row['dialogue_scores']
            if score_data and trait in score_data:
                scores.append(score_data[trait])
        trait_scores[trait] = scores
    return trait_scores

def process_data(input_file: Path, output_dir: Path, force_reprocess: bool = False):
    """
    Load data, compute z-scores, and return the dataframe and statistics.
    Uses caching to avoid re-computation.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cached_parquet = output_dir / 'ocean_scores_with_z_scores.parquet'
    cached_stats = output_dir / 'trait_z_scores.json'
    
    if not force_reprocess and cached_parquet.exists() and cached_stats.exists():
        print(f"✓ Found cached data at {cached_parquet}")
        print("  Loading cached data (use --force or menu option to reprocess)...")
        df = pd.read_parquet(cached_parquet)
        with open(cached_stats, 'r') as f:
            stats_data = json.load(f)
        trait_statistics = stats_data['trait_statistics']
        return df, trait_statistics

    print(f"Reading {input_file}...")
    df = pd.read_parquet(input_file)
    
    print(f"Total samples: {len(df)}")
    
    # OCEAN traits
    traits = ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']
    
    # Extract scores for each trait
    print("\nExtracting trait scores...")
    # We need to compute statistics on TRAIN split only
    if 'split' not in df.columns:
        raise ValueError("'split' column not found. Cannot compute train-only statistics.")
    
    train_df = df[df['split'] == 'train']
    
    trait_statistics = {}
    
    for trait in traits:
        train_scores = []
        for _, row in train_df.iterrows():
            score_data = row['dialogue_scores']
            if trait in score_data:
                train_scores.append(score_data[trait])
        
        train_scores = np.array(train_scores)
        mean_score = np.mean(train_scores)
        std_score = np.std(train_scores)
        
        # Guard against σ≈0
        if std_score < 1e-6:
            std_score = 1.0
        
        trait_statistics[trait] = {
            'mean': float(mean_score),
            'std': float(std_score),
            'train_count': len(train_scores)
        }
        print(f"  {trait}: μ={mean_score:.4f}, σ={std_score:.4f}")

    # Save z-score statistics
    z_scores_output = {
        'trait_statistics': trait_statistics,
        'z_score_formula': 'z_i,t = (s_i,t - μ_t) / σ_t',
        'metadata': {'total_samples': len(df)}
    }
    
    with open(cached_stats, 'w') as f:
        json.dump(z_scores_output, f, indent=2)
    
    # Compute z-scores
    print("\nComputing z-scores...")
    z_scores_data = []
    
    for idx, row in df.iterrows():
        sample_data = row.to_dict()
        sample_data['dialogue_z_scores'] = {}
        
        has_valid_scores = True
        has_extreme_z = False
        
        for trait in traits:
            if trait in row['dialogue_scores']:
                score = row['dialogue_scores'][trait]
                mean_score = trait_statistics[trait]['mean']
                std_score = trait_statistics[trait]['std']
                
                z_score = (score - mean_score) / std_score
                
                if abs(z_score) > 4.0:
                    has_extreme_z = True
                
                sample_data['dialogue_z_scores'][trait] = np.float32(z_score)
            else:
                has_valid_scores = False
        
        if has_valid_scores and not has_extreme_z:
            z_scores_data.append(sample_data)
            
    z_scores_df = pd.DataFrame(z_scores_data)
    
    # Save cached parquet
    z_scores_df.to_parquet(cached_parquet, index=False)
    print(f"  ✓ Saved processed data to {cached_parquet}")
    
    return z_scores_df, trait_statistics

def setup_plot_style(ax, title, xlabel, ylabel):
    """Apply consistent styling to axes."""
    # ax.set_title(title, fontsize=FONT_SIZE_TITLE, color='black', pad=20)  # Title removed
    ax.set_xlabel(xlabel, fontsize=FONT_SIZE_AXIS_LABEL, color='black')
    ax.set_ylabel(ylabel, fontsize=FONT_SIZE_AXIS_LABEL, color='black')
    
    ax.tick_params(axis='both', which='major', labelsize=FONT_SIZE_TICKS, colors='black')
    
    # Black spines
    for spine in ax.spines.values():
        spine.set_color('black')

def plot_individual_histograms(trait_scores, output_dir):
    print("\nGenerating individual histograms...")
    traits = list(trait_scores.keys())
    trait_labels = {t: t.capitalize() for t in traits}
    
    bins = np.arange(0, 1 + BIN_WIDTH, BIN_WIDTH)
    
    for trait in traits:
        scores = trait_scores[trait]
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        
        plt.figure(figsize=(14, 9))
        ax = plt.gca()
        
        # Plot histogram (Counts, not density)
        n, bins_edges, patches = plt.hist(
            scores, 
            bins=bins, 
            edgecolor='none', 
            alpha=0.6,
            color=TRAIT_COLORS[trait],
            label='Distribution',
            density=False
        )
        
        # Add stats lines
        plt.axvline(mean_score, color='darkred', linestyle='--', linewidth=2, label=f'Mean: {mean_score:.3f}')
        plt.axvspan(mean_score - std_score, mean_score + std_score, alpha=0.2, color='darkred', label=f'±1 Std Dev')

        setup_plot_style(ax, 
                        f'{trait_labels[trait]} Score Distribution', 
                        'Trait Score', 
                        'Number of Samples')
        
        plt.legend(loc='upper right', fontsize=FONT_SIZE_LEGEND, framealpha=0.9, fancybox=False, edgecolor='black')
        plt.tight_layout()
        plt.savefig(output_dir / f'{trait}_distribution.png', dpi=OUTPUT_DPI, bbox_inches='tight')
        plt.savefig(output_dir / f'{trait}_distribution.pgf', bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved {trait}_distribution.png")

def plot_overlay(trait_scores, trait_statistics, output_dir):
    print("\nGenerating overlay plot...")
    traits = list(trait_scores.keys())
    # Use single-letter abbreviations
    trait_labels = {
        'openness': 'O',
        'conscientiousness': 'C',
        'extraversion': 'E',
        'agreeableness': 'A',
        'neuroticism': 'N'
    }
    
    plt.figure(figsize=(16, 10)) # Slightly larger for big fonts
    ax = plt.gca()
    
    bins = np.arange(0, 1 + BIN_WIDTH, BIN_WIDTH)
    
    for trait in traits:
        scores = trait_scores[trait]
        stats = trait_statistics[trait]
        
        # Histogram with counts
        # To make curves smooth with counts, we can still use the smoothing trick 
        # but scaled to counts.
        
        n, bins_edges = np.histogram(scores, bins=bins)
        bin_centers = (bins_edges[:-1] + bins_edges[1:]) / 2
        
        # Gaussian smoothing
        from scipy.ndimage import gaussian_filter1d
        # With 0.001 bin width, sigma=3 provides optimal smoothness
        smoothed_n = gaussian_filter1d(n.astype(float), sigma=3) 
        
        label = f'{trait_labels[trait]} ($\mu$={stats["mean"]:.3f})'
        
        plt.plot(bin_centers, smoothed_n, 
                 color=TRAIT_COLORS[trait], linewidth=3.0, 
                 alpha=0.9)
        
        plt.fill_between(bin_centers, smoothed_n, alpha=0.2, color=TRAIT_COLORS[trait])

    setup_plot_style(ax, 'OCEAN Traits in JIC Dataset', 'Trait Score', 'Number of Samples')
    
    # Format Y-axis with 'k' notation (e.g., 2k, 4k, 6k)
    def format_thousands(x, p):
        if x >= 1000:
            return f'{int(x/1000)}k'
        else:
            return f'{int(x)}'
    ax.yaxis.set_major_formatter(plt.FuncFormatter(format_thousands))
    
    # Disable grid
    ax.grid(False)
    
    # Create custom legend with color blocks
    legend_handles = []
    for trait in traits:
        label = f'{trait_labels[trait]} ($\mu$={trait_statistics[trait]["mean"]:.3f})'
        c = TRAIT_COLORS[trait]
        patch = mpatches.Patch(facecolor=mcolors.to_rgba(c, 0.2), 
                             edgecolor=mcolors.to_rgba(c, 0.9), 
                             linewidth=2.0,
                             label=label)
        legend_handles.append(patch)

    plt.legend(handles=legend_handles, loc='upper left', fontsize=FONT_SIZE_LEGEND, framealpha=0.95, fancybox=False, edgecolor='black')
    plt.tight_layout()
    
    output_file = output_dir / 'all_traits_overlay.png'
    plt.savefig(output_file, dpi=OUTPUT_DPI, bbox_inches='tight')
    plt.savefig(output_file.with_suffix('.pgf'), bbox_inches='tight')
    print(f"  ✓ Saved {output_file}")
    plt.close()

def plot_z_scores(df, output_dir):
    print("\nGenerating Z-Score plots...")
    traits = ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']
    # Use single-letter abbreviations
    trait_labels = {
        'openness': 'O',
        'conscientiousness': 'C',
        'extraversion': 'E',
        'agreeableness': 'A',
        'neuroticism': 'N'
    }
    z_bins = np.arange(-4.5, 4.5 + 0.1, 0.1)
    
    # Combined Z-Score Overlay
    plt.figure(figsize=(16, 10))
    ax = plt.gca()
    
    for trait in traits:
        # Extract z-scores safely
        z_scores = []
        for _, row in df.iterrows():
            if trait in row['dialogue_z_scores']:
                z_scores.append(row['dialogue_z_scores'][trait])
        
        if not z_scores: continue
            
        n, bins_edges = np.histogram(z_scores, bins=z_bins)
        bin_centers = (bins_edges[:-1] + bins_edges[1:]) / 2
        
        from scipy.ndimage import gaussian_filter1d
        smoothed_n = gaussian_filter1d(n.astype(float), sigma=2)
        
        # Calculate mean for label
        mean_z = np.mean(z_scores)
        
        plt.plot(bin_centers, smoothed_n, color=TRAIT_COLORS[trait], linewidth=3.0, alpha=0.9)
        plt.fill_between(bin_centers, smoothed_n, alpha=0.2, color=TRAIT_COLORS[trait])
        
        # Add to handles
        label = f'{trait_labels[trait]} ($\\mu$={mean_z:.3f})'
        # Manually track for legend to ensure order if needed, but here we can just rebuild it
    
    setup_plot_style(ax, 'OCEAN Personality Traits (Z-Scores) in JIC Dataset', 'Z-Score', 'Number of Samples')
    
    # Format Y-axis with 'k' notation (e.g., 2k, 4k, 6k)
    def format_thousands(x, p):
        if x >= 1000:
            return f'{int(x/1000)}k'
        else:
            return f'{int(x)}'
    ax.yaxis.set_major_formatter(plt.FuncFormatter(format_thousands))
    
    # Disable grid
    ax.grid(False)
    
    # Create custom legend with color blocks
    legend_handles = []
    for trait in traits:
        # Re-calculate mean for label (efficient enough)
        z_scores = [row['dialogue_z_scores'][trait] for _, row in df.iterrows() if trait in row['dialogue_z_scores']]
        if not z_scores: continue
        mean_z = np.mean(z_scores)
        label = f'{trait_labels[trait]} ($\\mu$={mean_z:.3f})'
        c = TRAIT_COLORS[trait]
        patch = mpatches.Patch(facecolor=mcolors.to_rgba(c, 0.2), 
                             edgecolor=mcolors.to_rgba(c, 0.9), 
                             linewidth=2.0,
                             label=label)
        legend_handles.append(patch)

    plt.legend(handles=legend_handles, loc='upper left', fontsize=FONT_SIZE_LEGEND, framealpha=0.95, fancybox=False, edgecolor='black')
    plt.tight_layout()
    
    plt.savefig(output_dir / 'all_traits_z_score_overlay.png', dpi=OUTPUT_DPI, bbox_inches='tight')
    plt.savefig(output_dir / 'all_traits_z_score_overlay.pgf', bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved all_traits_z_score_overlay.png")

def main():
    script_dir = Path(__file__).parent
    input_file = script_dir / "ocean_scores_with_uuid.parquet"
    if not input_file.exists():
        input_file = script_dir / "ocean_scores.parquet"
    output_dir = script_dir / "trait_distribution_analysis"
    
    if not input_file.exists():
        print(f"❌ Error: {input_file} not found!")
        sys.exit(1)

    # Interactive Menu
    print("\n" + "="*50)
    print("OCEAN Trait Analysis & Plotting Tool")
    print("="*50)
    print("1. Generate ALL plots")
    print("2. Generate Both Overlay Plots (OCEAN Traits + Z-Scores)")
    print("3. Generate Individual Histograms")
    print("4. Generate Z-Score Plots Only")
    print("5. Force Reprocess Data (Ignore Cache)")
    print("0. Exit")
    
    choice = input("\nEnter choice (0-5): ").strip()
    
    if choice == '0':
        sys.exit(0)
    
    force_reprocess = (choice == '5')
    if force_reprocess:
        print("\nWill reprocess data. Which plots to generate after?")
        print("1. All")
        print("2. Overlay Only")
        sub_choice = input("Enter choice (1-2): ").strip()
        choice = sub_choice if sub_choice in ['1', '2'] else '1'

    # Load Data
    df, trait_stats = process_data(input_file, output_dir, force_reprocess)
    
    # Extract scores for plotting
    traits = ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']
    trait_scores = get_trait_scores(df, traits)

    if choice == '1':
        plot_overlay(trait_scores, trait_stats, output_dir)
        plot_individual_histograms(trait_scores, output_dir)
        plot_z_scores(df, output_dir)
    elif choice == '2':
        plot_overlay(trait_scores, trait_stats, output_dir)
        plot_z_scores(df, output_dir)
    elif choice == '3':
        plot_individual_histograms(trait_scores, output_dir)
    elif choice == '4':
        plot_z_scores(df, output_dir)
    
    print("\n✅ Done!")

if __name__ == "__main__":
    main()
