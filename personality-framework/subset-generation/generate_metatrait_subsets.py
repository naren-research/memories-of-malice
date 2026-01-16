"""
Generate Personality Metatrait-based Subsets

This script computes two higher-order metatraits from OCEAN z-scores:
- Stability (Alpha) = (Agreeableness + Conscientiousness + Emotional Stability) / 3
  where Emotional Stability = -Neuroticism
- Plasticity (Beta) = (Extraversion + Openness) / 2

Creates four subsets based on high/low values of each metatrait and generates
distribution plots for analysis.

References:
- Digman, J. M. (1997). Higher-order factors of the Big Five.
- DeYoung, C. G. (2006). Higher-order factors of the Big Five predict conformity.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from pathlib import Path
from scipy import stats
from scipy.ndimage import gaussian_filter1d
import json

# CONFIGURATION
SUBSET_SIZE = 2941  # Number of samples per subset
OUTPUT_DPI = 300
BIN_WIDTH_Z = 0.1  # Bin width for z-score histograms

# Enhanced color palette
TRAIT_COLORS = {
    'openness': '#9b5de5',
    'conscientiousness': '#f15bb5',
    'extraversion': '#fee440',
    'agreeableness': '#00bbf9',
    'neuroticism': '#00f5d4'
}

METATRAIT_COLORS = {
    'stability': '#1a1a1a',
    'plasticity': '#ff6f00'
}

# Set plotting style
plt.style.use('default')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['grid.color'] = 'black'
plt.rcParams['axes.edgecolor'] = 'black'
plt.rcParams['xtick.color'] = 'black'
plt.rcParams['ytick.color'] = 'black'
plt.rcParams['text.color'] = 'black'

# Font Sizes (increased by 50%)
FONT_SIZE_TITLE = 42
FONT_SIZE_LABEL = 39
FONT_SIZE_TICKS = 33
FONT_SIZE_LEGEND = 39

plt.rcParams['font.size'] = FONT_SIZE_TICKS

# Match IEEE paper font (Times/serif)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif']
plt.rcParams['mathtext.fontset'] = 'dejavuserif'


def compute_metatraits(df):
    """
    Compute Stability and Plasticity metatrait z-scores.
    
    Args:
        df: DataFrame with dialogue_z_scores column containing OCEAN z-scores
        
    Returns:
        DataFrame with added 'stability_z' and 'plasticity_z' columns
    """
    print("Computing metatrait scores...")
    
    stability_scores = []
    plasticity_scores = []
    
    for idx, row in df.iterrows():
        z_scores = row['dialogue_z_scores']
        
        # Stability = (A + C - N) / 3
        # Using -N because Emotional Stability is inverse of Neuroticism
        stability = (
            z_scores['agreeableness'] + 
            z_scores['conscientiousness'] + 
            (-z_scores['neuroticism'])
        ) / 3.0
        
        # Plasticity = (E + O) / 2
        plasticity = (
            z_scores['extraversion'] + 
            z_scores['openness']
        ) / 2.0
        
        stability_scores.append(stability)
        plasticity_scores.append(plasticity)
    
    df['stability_z'] = stability_scores
    df['plasticity_z'] = plasticity_scores
    
    print(f"  Stability:  μ={np.mean(stability_scores):.4f}, σ={np.std(stability_scores):.4f}, "
          f"min={np.min(stability_scores):.4f}, max={np.max(stability_scores):.4f}")
    print(f"  Plasticity: μ={np.mean(plasticity_scores):.4f}, σ={np.std(plasticity_scores):.4f}, "
          f"min={np.min(plasticity_scores):.4f}, max={np.max(plasticity_scores):.4f}")
    
    return df


def create_subsets(df, subset_size, output_dir):
    """
    Create eleven subsets based on metatrait patterns.
    
    Args:
        df: DataFrame with stability_z and plasticity_z
        subset_size: Number of samples per subset
        output_dir: Directory to save subset files
        
    Returns:
        Dictionary mapping subset names to DataFrames
    """
    print(f"\nCreating subsets of size {subset_size}...")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    subsets = {}
    
    # 1. Sort by stability (descending for high, ascending for low)
    df_sorted_stability = df.sort_values('stability_z', ascending=False)
    subsets['stability_high'] = df_sorted_stability.head(subset_size).copy()
    subsets['stability_low'] = df_sorted_stability.tail(subset_size).copy()
    
    # 2. Sort by plasticity (descending for high, ascending for low)
    df_sorted_plasticity = df.sort_values('plasticity_z', ascending=False)
    subsets['plasticity_high'] = df_sorted_plasticity.head(subset_size).copy()
    subsets['plasticity_low'] = df_sorted_plasticity.tail(subset_size).copy()
    
    # 3. Neutral stability (closest to 0)
    df_with_stability_dist = df.copy()
    df_with_stability_dist['stability_abs_dist'] = df_with_stability_dist['stability_z'].abs()
    df_sorted_neutral_stability = df_with_stability_dist.sort_values('stability_abs_dist')
    subsets['stability_neutral'] = df_sorted_neutral_stability.head(subset_size).copy()
    
    # 4. Neutral plasticity (closest to 0)
    df_with_plasticity_dist = df.copy()
    df_with_plasticity_dist['plasticity_abs_dist'] = df_with_plasticity_dist['plasticity_z'].abs()
    df_sorted_neutral_plasticity = df_with_plasticity_dist.sort_values('plasticity_abs_dist')
    subsets['plasticity_neutral'] = df_sorted_neutral_plasticity.head(subset_size).copy()
    
    # 5. Neutral both (both closest to 0 - minimize combined distance)
    df_with_both_dist = df.copy()
    df_with_both_dist['combined_abs_dist'] = (df_with_both_dist['stability_z'].abs() + 
                                               df_with_both_dist['plasticity_z'].abs())
    df_sorted_neutral_both = df_with_both_dist.sort_values('combined_abs_dist')
    subsets['stability_neutral_plasticity_neutral'] = df_sorted_neutral_both.head(subset_size).copy()
    
    # 6. High stability + high plasticity (combined metatrait sum highest)
    df_with_sum = df.copy()
    df_with_sum['metatrait_sum'] = df_with_sum['stability_z'] + df_with_sum['plasticity_z']
    df_sorted_sum = df_with_sum.sort_values('metatrait_sum', ascending=False)
    subsets['stability_high_plasticity_high'] = df_sorted_sum.head(subset_size).copy()
    
    # 7. Low stability + low plasticity (both Stability < -1.0 AND Plasticity < -1.0, then sort by sum)
    df_low_quadrant = df[(df['stability_z'] < -1.0) & (df['plasticity_z'] < -1.0)].copy()
    df_low_quadrant['metatrait_sum'] = df_low_quadrant['stability_z'] + df_low_quadrant['plasticity_z']
    df_low_quadrant_sorted = df_low_quadrant.sort_values('metatrait_sum', ascending=False)
    subsets['stability_low_plasticity_low'] = df_low_quadrant_sorted.head(subset_size).copy()
    if len(df_low_quadrant) < subset_size:
        print(f"    ⚠️  Warning: Only {len(df_low_quadrant):,} samples meet criteria (wanted {subset_size:,})")
    
    # 8. High stability + low plasticity (stable but rigid)
    df_high_stab_low_plast = df[(df['stability_z'] > 1.0) & (df['plasticity_z'] < -1.0)].copy()
    df_high_stab_low_plast['stab_plast_diff'] = df_high_stab_low_plast['stability_z'] - df_high_stab_low_plast['plasticity_z']
    df_high_stab_low_plast_sorted = df_high_stab_low_plast.sort_values('stab_plast_diff', ascending=False)
    subsets['stability_high_plasticity_low'] = df_high_stab_low_plast_sorted.head(subset_size).copy()
    if len(df_high_stab_low_plast) < subset_size:
        print(f"    ⚠️  Warning: Only {len(df_high_stab_low_plast):,} samples meet criteria (wanted {subset_size:,})")
    
    # 9. Low stability + high plasticity (open but unstable)
    df_low_stab_high_plast = df[(df['stability_z'] < -1.0) & (df['plasticity_z'] > 1.0)].copy()
    df_low_stab_high_plast['plast_stab_diff'] = df_low_stab_high_plast['plasticity_z'] - df_low_stab_high_plast['stability_z']
    df_low_stab_high_plast_sorted = df_low_stab_high_plast.sort_values('plast_stab_diff', ascending=False)
    subsets['stability_low_plasticity_high'] = df_low_stab_high_plast_sorted.head(subset_size).copy()
    if len(df_low_stab_high_plast) < subset_size:
        print(f"    ⚠️  Warning: Only {len(df_low_stab_high_plast):,} samples meet criteria (wanted {subset_size:,})")
    
    # Save subsets and print statistics
    for subset_name, subset_df in subsets.items():
        # Save as parquet
        output_file = output_dir / f'{subset_name}.parquet'
        subset_df.to_parquet(output_file, index=False)
        
        # Print statistics
        stability_mean = subset_df['stability_z'].mean()
        stability_std = subset_df['stability_z'].std()
        plasticity_mean = subset_df['plasticity_z'].mean()
        plasticity_std = subset_df['plasticity_z'].std()
        
        print(f"\n  {subset_name}:")
        print(f"    Samples: {len(subset_df):,}")
        print(f"    Stability:  $\mu$={stability_mean:.4f}, $\sigma$={stability_std:.4f}")
        print(f"    Plasticity: $\mu$={plasticity_mean:.4f}, $\sigma$={plasticity_std:.4f}")
        print(f"    Saved to: {output_file}")
    
    # Save metadata
    metadata = {
        'subset_size': subset_size,
        'total_samples': len(df),
        'subsets': {
            name: {
                'file': f'{name}.parquet',
                'n_samples': len(subset_df),
                'stability': {
                    'mean': float(subset_df['stability_z'].mean()),
                    'std': float(subset_df['stability_z'].std()),
                    'min': float(subset_df['stability_z'].min()),
                    'max': float(subset_df['stability_z'].max())
                },
                'plasticity': {
                    'mean': float(subset_df['plasticity_z'].mean()),
                    'std': float(subset_df['plasticity_z'].std()),
                    'min': float(subset_df['plasticity_z'].min()),
                    'max': float(subset_df['plasticity_z'].max())
                }
            }
            for name, subset_df in subsets.items()
        },
        'metatrait_formulas': {
            'stability': '(agreeableness_z + conscientiousness_z - neuroticism_z) / 3',
            'plasticity': '(extraversion_z + openness_z) / 2'
        }
    }
    
    metadata_file = output_dir / 'subset_metadata.json'
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"\n  ✓ Metadata saved to: {metadata_file}")
    
    return subsets


def plot_metatrait_distribution(subset_df, subset_name, metatrait, output_dir):
    """
    Plot distribution for a single metatrait.
    
    Args:
        subset_df: DataFrame for the subset
        subset_name: Name of the subset (e.g., 'stability_high')
        metatrait: 'stability' or 'plasticity'
        output_dir: Directory to save plots
    """
    scores = subset_df[f'{metatrait}_z'].values
    
    # Statistics
    mean_score = np.mean(scores)
    median_score = np.median(scores)
    std_score = np.std(scores)
    min_score = np.min(scores)
    max_score = np.max(scores)
    
    # Color
    color = METATRAIT_COLORS[metatrait]
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Bins
    bins = np.arange(-4.5, 4.5 + BIN_WIDTH_Z, BIN_WIDTH_Z)
    
    # Histogram
    n, bins_edges, patches = plt.hist(
        scores,
        bins=bins,
        edgecolor='white',
        linewidth=0.5,
        alpha=0.6,
        color=color,
        label=f'{metatrait.capitalize()} Distribution'
    )
    
    # Add gradient effect
    for i, patch in enumerate(patches):
        patch.set_alpha(0.7 if i % 2 == 0 else 0.5)
    
    # KDE overlay
    try:
        kde = stats.gaussian_kde(scores, bw_method='silverman')
        x_range = np.linspace(min_score, max_score, 200)
        kde_values = kde(x_range) * len(scores) * BIN_WIDTH_Z
        plt.plot(x_range, kde_values, color='darkred', linewidth=1.5,
                alpha=0.7, linestyle='--', label='KDE Smooth')
    except:
        pass
    
    # Statistical lines
    plt.axvline(mean_score, color='darkred', linestyle='--', linewidth=2,
               alpha=0.8, label=f'Mean: {mean_score:.3f}')
    plt.axvline(median_score, color='darkgreen', linestyle='--', linewidth=2,
               alpha=0.8, label=f'Median: {median_score:.3f}')
    plt.axvline(min_score, color='darkorange', linestyle=':', linewidth=2,
               alpha=0.7, label=f'Min: {min_score:.3f}')
    plt.axvline(max_score, color='darkviolet', linestyle=':', linewidth=2,
               alpha=0.7, label=f'Max: {max_score:.3f}')
    
    # Standard deviation bands
    plt.axvspan(mean_score - std_score, mean_score + std_score,
                alpha=0.25, color='darkred', label=f'±1 Std Dev: {std_score:.3f}')
    plt.axvspan(mean_score - 2*std_score, mean_score + 2*std_score,
                alpha=0.1, color='darkred')
    
    # Reference lines
    for ref_z in [-2, -1, 0, 1, 2]:
        plt.axvline(ref_z, color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
    
    # Labels
    plt.xlabel('Z-Score', fontsize=FONT_SIZE_LABEL, color='black')
    plt.ylabel('Number of Samples', fontsize=FONT_SIZE_LABEL, color='black')
    plt.title(f'{metatrait.capitalize()} Distribution - {subset_name.replace("_", " ").title()}',
              fontsize=FONT_SIZE_TITLE, color='black', pad=20)
    
    # Grid and legend
    plt.grid(False)
    # Grid and legend
    plt.grid(False)
    plt.legend(loc='upper right', fontsize=FONT_SIZE_LEGEND, framealpha=0.9,
              fancybox=False, edgecolor='black')

    # Format Y-axis with 'k' notation
    def format_thousands(x, p):
        if x >= 1000:
            return f'{int(x/1000)}k'
        else:
            return f'{int(x)}'
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(format_thousands))
    
    plt.tight_layout()
    
    # Save
    output_file = output_dir / f'{subset_name}_{metatrait}_distribution.png'
    plt.savefig(output_file, dpi=OUTPUT_DPI, bbox_inches='tight')
    plt.savefig(output_file.with_suffix('.pgf'), bbox_inches='tight')
    plt.close()


def plot_metatrait_overlay(subset_df, subset_name, output_dir):
    """
    Plot overlay of both metatraits (Stability + Plasticity).
    
    Args:
        subset_df: DataFrame for the subset
        subset_name: Name of the subset
        output_dir: Directory to save plots
    """
    plt.figure(figsize=(14, 9))
    
    bins = np.arange(-4.5, 4.5 + BIN_WIDTH_Z, BIN_WIDTH_Z)
    
    for metatrait in ['stability', 'plasticity']:
        scores = subset_df[f'{metatrait}_z'].values
        color = METATRAIT_COLORS[metatrait]
        
        # Calculate histogram
        n, bins_edges = np.histogram(scores, bins=bins)
        bin_centers = (bins_edges[:-1] + bins_edges[1:]) / 2
        
        # Smooth
        smoothed_n = gaussian_filter1d(n.astype(float), sigma=2)
        
        # Plot
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        
        # Use single-letter abbreviation
        metatrait_abbrev = 'S' if metatrait == 'stability' else 'P'
        plt.fill_between(bin_centers, smoothed_n, alpha=0.2, color=color)
        plt.plot(bin_centers, smoothed_n, color=color, linewidth=3.0, alpha=0.9)
        
        # Mean line
        plt.axvline(mean_score, color=color, linestyle='--', linewidth=1.5, alpha=0.6)
    
    # Reference lines removed per user request
    # for ref_z in [-2, -1, 0, 1, 2]:
    #     if ref_z == 0:
    #         plt.axvline(ref_z, color='black', linestyle='-', linewidth=1, alpha=0.5,
    #                    label='Reference Lines')
    #     else:
    #         plt.axvline(ref_z, color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
    
    # Labels
    plt.xlabel('Z-Score', fontsize=FONT_SIZE_LABEL, color='black')
    plt.ylabel('Number of Samples', fontsize=FONT_SIZE_LABEL, color='black')
    # plt.title(f'Metatrait Overlay - {subset_name.replace("_", " ").title()}',
    #           fontsize=FONT_SIZE_TITLE, color='black', pad=20)  # Title removed
    
    # Grid and legend
    plt.grid(False)
    
    # Determine legend location and transparency based on subset
    if subset_name in ['stability_low', 'plasticity_low', 'stability_low_plasticity_low']:
        legend_loc = 'upper right'
        legend_alpha = 0.95
    elif subset_name == 'stability_high_plasticity_low':
        legend_loc = 'lower center'
        legend_alpha = 0.7
    else:
        legend_loc = 'upper left'
        legend_alpha = 0.95
    

    
    # Create custom legend with color blocks
    legend_handles = []
    for metatrait in ['stability', 'plasticity']:
        scores = subset_df[f'{metatrait}_z'].values
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        metatrait_abbrev = 'S' if metatrait == 'stability' else 'P'
        label = f'{metatrait_abbrev} ($\mu$={mean_score:.2f}, $\sigma$={std_score:.2f})'
        c = METATRAIT_COLORS[metatrait]
        patch = mpatches.Patch(facecolor=mcolors.to_rgba(c, 0.2), 
                             edgecolor=mcolors.to_rgba(c, 0.9), 
                             linewidth=2.0,
                             label=label)
        legend_handles.append(patch)

    plt.legend(handles=legend_handles, loc=legend_loc, fontsize=FONT_SIZE_LEGEND * 0.75, framealpha=legend_alpha,
              fancybox=False, edgecolor='black')

    # Format Y-axis with 'k' notation
    def format_thousands(x, p):
        if x >= 1000:
            return f'{int(x/1000)}k'
        else:
            return f'{int(x)}'
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(format_thousands))
    
    # Set black axes
    ax = plt.gca()
    ax.spines['bottom'].set_color('black')
    ax.spines['top'].set_color('black')
    ax.spines['left'].set_color('black')
    ax.spines['right'].set_color('black')
    ax.tick_params(colors='black')
    
    plt.tight_layout()
    
    # Save
    output_file = output_dir / f'{subset_name}_metatrait_overlay.png'
    plt.savefig(output_file, dpi=OUTPUT_DPI, bbox_inches='tight')
    plt.savefig(output_file.with_suffix('.pgf'), bbox_inches='tight')
    plt.close()


def plot_ocean_overlay(subset_df, subset_name, output_dir):
    """
    Plot overlay of all 5 OCEAN traits.
    
    Args:
        subset_df: DataFrame for the subset
        subset_name: Name of the subset
        output_dir: Directory to save plots
    """
    plt.figure(figsize=(14, 9))
    
    bins = np.arange(-4.5, 4.5 + BIN_WIDTH_Z, BIN_WIDTH_Z)
    traits = ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']
    # Use single-letter abbreviations
    trait_labels = {
        'openness': 'O',
        'conscientiousness': 'C',
        'extraversion': 'E',
        'agreeableness': 'A',
        'neuroticism': 'N'
    }
    
    for trait in traits:
        # Extract scores
        scores = [row['dialogue_z_scores'][trait] for _, row in subset_df.iterrows()
                 if trait in row['dialogue_z_scores']]
        
        color = TRAIT_COLORS[trait]
        
        # Calculate histogram
        n, bins_edges = np.histogram(scores, bins=bins)
        bin_centers = (bins_edges[:-1] + bins_edges[1:]) / 2
        
        # Smooth
        smoothed_n = gaussian_filter1d(n.astype(float), sigma=1.5)
        
        # Plot
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        
        plt.fill_between(bin_centers, smoothed_n, alpha=0.2, color=color)
        plt.plot(bin_centers, smoothed_n, color=color, linewidth=3.0,
                label=f'{trait_labels[trait]} ($\mu$={mean_score:.2f}, $\sigma$={std_score:.2f})',
                alpha=0.9)
    
    # Reference lines
    for ref_z in [-2, -1, 0, 1, 2]:
        if ref_z == 0:
            plt.axvline(ref_z, color='black', linestyle='-', linewidth=1, alpha=0.5)
        else:
            plt.axvline(ref_z, color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
    
    # Labels
    plt.xlabel('Z-Score', fontsize=FONT_SIZE_LABEL, color='black')
    plt.ylabel('Number of Samples', fontsize=FONT_SIZE_LABEL, color='black')
    plt.title(f'OCEAN Trait Overlay - {subset_name.replace("_", " ").title()}',
              fontsize=FONT_SIZE_TITLE, color='black', pad=20)
    
    # Grid and legend
    plt.grid(False)
    # Grid and legend
    plt.grid(False)
    plt.legend(loc='upper left', fontsize=FONT_SIZE_LEGEND, framealpha=0.95,
              fancybox=False, edgecolor='black')

    # Format Y-axis with 'k' notation
    def format_thousands(x, p):
        if x >= 1000:
            return f'{int(x/1000)}k'
        else:
            return f'{int(x)}'
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(format_thousands))
    
    # Set black axes
    ax = plt.gca()
    ax.spines['bottom'].set_color('black')
    ax.spines['top'].set_color('black')
    ax.spines['left'].set_color('black')
    ax.spines['right'].set_color('black')
    ax.tick_params(colors='black')
    
    plt.tight_layout()
    
    # Save
    output_file = output_dir / f'{subset_name}_ocean_overlay.png'
    plt.savefig(output_file, dpi=OUTPUT_DPI, bbox_inches='tight')
    plt.savefig(output_file.with_suffix('.pgf'), bbox_inches='tight')
    plt.close()


def generate_all_plots(subsets, output_dir):
    """
    Generate all plots for all subsets.
    
    Args:
        subsets: Dictionary mapping subset names to DataFrames
        output_dir: Directory to save plots
    """
    print("\nGenerating plots...")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    total_plots = len(subsets) * 4  # 2 individual + 1 metatrait overlay + 1 ocean overlay
    plot_count = 0
    
    for subset_name, subset_df in subsets.items():
        print(f"\n  {subset_name}:")
        
        # Individual metatrait distributions
        for metatrait in ['stability', 'plasticity']:
            plot_metatrait_distribution(subset_df, subset_name, metatrait, output_dir)
            plot_count += 1
            print(f"    ✓ {metatrait.capitalize()} distribution [{plot_count}/{total_plots}]")
        
        # Metatrait overlay (Stability + Plasticity)
        plot_metatrait_overlay(subset_df, subset_name, output_dir)
        plot_count += 1
        print(f"    ✓ Metatrait overlay [{plot_count}/{total_plots}]")
        
        # OCEAN overlay
        plot_ocean_overlay(subset_df, subset_name, output_dir)
        plot_count += 1
        print(f"    ✓ OCEAN overlay [{plot_count}/{total_plots}]")
    
    print(f"\n  ✓ All {plot_count} plots generated successfully!")


def plot_full_dataset_metatrait_overlay(df, output_dir):
    """
    Plot metatrait overlay for the entire dataset (not subsets).
    
    Args:
        df: DataFrame with stability_z and plasticity_z columns
        output_dir: Directory to save plot
    """
    print("\nGenerating full dataset metatrait overlay...")
    
    plt.figure(figsize=(16, 10))  # Match the size from analyze_trait_distributions
    ax = plt.gca()
    
    bins = np.arange(-4.5, 4.5 + BIN_WIDTH_Z, BIN_WIDTH_Z)
    
    for metatrait in ['stability', 'plasticity']:
        scores = df[f'{metatrait}_z'].values
        color = METATRAIT_COLORS[metatrait]
        
        # Calculate histogram
        n, bins_edges = np.histogram(scores, bins=bins)
        bin_centers = (bins_edges[:-1] + bins_edges[1:]) / 2
        
        # Smooth
        smoothed_n = gaussian_filter1d(n.astype(float), sigma=2)
        
        # Plot
        mean_score = np.mean(scores)
        
        # Use single-letter abbreviation
        metatrait_abbrev = 'S' if metatrait == 'stability' else 'P'
        plt.plot(bin_centers, smoothed_n, color=color, linewidth=3.0, 
                alpha=0.9)
        
        plt.fill_between(bin_centers, smoothed_n, alpha=0.2, color=color)
    
    # Reference lines removed per user request
    # for ref_z in [-2, -1, 0, 1, 2]:
    #     plt.axvline(ref_z, color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
    
    # Setup plot style - font sizes increased by 50%
    # ax.set_title('Metatraits in JIC Dataset', fontsize=42, color='black', pad=20)  # Title removed
    ax.set_xlabel('Z-Score', fontsize=39, color='black')
    ax.set_ylabel('Number of Samples', fontsize=39, color='black')
    
    ax.tick_params(axis='both', which='major', labelsize=33, colors='black')
    
    # Black spines
    for spine in ax.spines.values():
        spine.set_color('black')
    
    # Disable grid
    ax.grid(False)
    
    # Legend with font size increased by 50%
    # Create custom legend with color blocks
    legend_handles = []
    for metatrait in ['stability', 'plasticity']:
        scores = df[f'{metatrait}_z'].values
        mean_score = np.mean(scores)
        metatrait_abbrev = 'S' if metatrait == 'stability' else 'P'
        label = f'{metatrait_abbrev} ($\mu$={mean_score:.3f})'
        c = METATRAIT_COLORS[metatrait]
        patch = mpatches.Patch(facecolor=mcolors.to_rgba(c, 0.2), 
                             edgecolor=mcolors.to_rgba(c, 0.9), 
                             linewidth=2.0,
                             label=label)
        legend_handles.append(patch)

    plt.legend(handles=legend_handles, loc='upper left', fontsize=39, framealpha=0.95,
              fancybox=False, edgecolor='black')

    # Format Y-axis with 'k' notation
    def format_thousands(x, p):
        if x >= 1000:
            return f'{int(x/1000)}k'
        else:
            return f'{int(x)}'
    ax.yaxis.set_major_formatter(plt.FuncFormatter(format_thousands))
    
    plt.tight_layout()
    
    # Save
    output_file = output_dir / 'full_dataset_metatrait_overlay.png'
    plt.savefig(output_file, dpi=OUTPUT_DPI, bbox_inches='tight')
    plt.savefig(output_file.with_suffix('.pgf'), bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved {output_file}")



def load_existing_subsets(subsets_dir):
    """Load existing subset parquet files."""
    print(f"\nLoading existing subsets from: {subsets_dir}")
    subsets = {}
    
    # List of expected subsets
    expected_subsets = [
        'stability_high', 'stability_low', 
        'plasticity_high', 'plasticity_low',
        'stability_neutral', 'plasticity_neutral',
        'stability_neutral_plasticity_neutral',
        'stability_high_plasticity_high',
        'stability_low_plasticity_low',
        'stability_high_plasticity_low',
        'stability_low_plasticity_high'
    ]
    
    for name in expected_subsets:
        file_path = subsets_dir / f'{name}.parquet'
        if file_path.exists():
            try:
                subsets[name] = pd.read_parquet(file_path)
                print(f"  ✓ Loaded {name} ({len(subsets[name]):,} samples)")
            except Exception as e:
                print(f"  ❌ Error loading {name}: {e}")
        else:
            print(f"  ⚠️  Missing {name}")
            
    return subsets

def main():
    """Main execution function."""
    print("="*80)
    print("METATRAIT-BASED SUBSET GENERATION & PLOTTING")
    print("="*80)
    
    # Paths
    script_dir = Path(__file__).parent.parent
    input_file = script_dir.parent.parent / 'personality-trait-enhancing' / 'trait-scoring' / 'trait_distribution_analysis' / 'ocean_scores_with_z_scores.parquet'
    subsets_dir = Path('./subsets')
    plots_dir = subsets_dir / 'plots'  # Plots inside subsets directory
    
    while True:
        print("\n" + "="*80)
        print("MENU")
        print("="*80)
        print("1. Generate Datasets & ALL Plots (Full Process)")
        print("2. Generate ALL Plots Only (Load existing subsets)")
        print("3. Generate Full Dataset Metatrait Overlay Only (Figure 3)")
        print("4. Generate Subset Metatrait Overlays Only")
        print("5. Generate Subset OCEAN Overlays Only")
        print("6. Generate Individual Metatrait Distributions Only")
        print("7. Generate Plots for Specific Subset")
        print("0. Exit")
        
        choice = input("\nEnter choice (0-7): ").strip()
        
        if choice == '0':
            print("Exiting...")
            break
            
        elif choice == '1':
            # Full process: Generate datasets and all plots
            if not input_file.exists():
                print(f"❌ Error: Input file not found: {input_file}")
                print("Please run analyze_trait_distributions.py first to generate z-scores.")
                continue
            
            print(f"\nLoading data from: {input_file}")
            df = pd.read_parquet(input_file)
            print(f"  Loaded {len(df):,} samples")
            
            required_cols = ['uuid', 'dialogue_z_scores']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                print(f"❌ Error: Missing required columns: {missing_cols}")
                continue
            
            df = compute_metatraits(df)
            
            if len(df) < SUBSET_SIZE * 11:
                print(f"⚠️  Warning: Dataset has {len(df):,} samples, but need {SUBSET_SIZE * 11:,} for 11 non-overlapping subsets")
                print(f"   Subsets may overlap. Consider reducing SUBSET_SIZE or using more data.")
            
            subsets = create_subsets(df, SUBSET_SIZE, subsets_dir)
            plot_full_dataset_metatrait_overlay(df, plots_dir)
            generate_all_plots(subsets, plots_dir)
            
            print("\n" + "="*80)
            print("SUMMARY")
            print("="*80)
            print(f"  Total samples processed: {len(df):,}")
            print(f"  Subsets created: {len(subsets)}")
            print(f"  Samples per subset: {SUBSET_SIZE:,}")
            print(f"  Subsets directory: {subsets_dir}")
            print(f"  Plots directory: {plots_dir}")
            print(f"\n  Metatrait formulas:")
            print(f"    Stability  = (A + C - N) / 3")
            print(f"    Plasticity = (E + O) / 2")
            print("="*80)
            print("✅ Generation complete!")
            break
            
        elif choice == '2':
            # Generate all plots from existing subsets
            if not subsets_dir.exists():
                print(f"❌ Error: Subsets directory not found: {subsets_dir}")
                continue
            
            # Load full dataset for full dataset overlay
            if not input_file.exists():
                print(f"❌ Error: Input file not found: {input_file}")
                continue
            
            print(f"\nLoading data from: {input_file}")
            df = pd.read_parquet(input_file)
            df = compute_metatraits(df)
            
            subsets = load_existing_subsets(subsets_dir)
            if not subsets:
                print("❌ No subsets loaded. Cannot generate plots.")
                continue
            
            plot_full_dataset_metatrait_overlay(df, plots_dir)
            generate_all_plots(subsets, plots_dir)
            print("\n✅ All plots generated!")
            break
            
        elif choice == '3':
            # Generate full dataset metatrait overlay only
            if not input_file.exists():
                print(f"❌ Error: Input file not found: {input_file}")
                continue
            
            print(f"\nLoading data from: {input_file}")
            df = pd.read_parquet(input_file)
            df = compute_metatraits(df)
            
            plot_full_dataset_metatrait_overlay(df, plots_dir)
            print("\n✅ Full dataset metatrait overlay generated!")
            break
            
        elif choice == '4':
            # Generate subset metatrait overlays only
            if not subsets_dir.exists():
                print(f"❌ Error: Subsets directory not found: {subsets_dir}")
                continue
            
            subsets = load_existing_subsets(subsets_dir)
            if not subsets:
                print("❌ No subsets loaded.")
                continue
            
            print("\nGenerating subset metatrait overlays...")
            for subset_name, subset_df in subsets.items():
                plot_metatrait_overlay(subset_df, subset_name, plots_dir)
                print(f"  ✓ {subset_name}")
            print("\n✅ Subset metatrait overlays generated!")
            break
            
        elif choice == '5':
            # Generate subset OCEAN overlays only
            if not subsets_dir.exists():
                print(f"❌ Error: Subsets directory not found: {subsets_dir}")
                continue
            
            subsets = load_existing_subsets(subsets_dir)
            if not subsets:
                print("❌ No subsets loaded.")
                continue
            
            print("\nGenerating subset OCEAN overlays...")
            for subset_name, subset_df in subsets.items():
                plot_ocean_overlay(subset_df, subset_name, plots_dir)
                print(f"  ✓ {subset_name}")
            print("\n✅ Subset OCEAN overlays generated!")
            break
            
        elif choice == '6':
            # Generate individual metatrait distributions only
            if not subsets_dir.exists():
                print(f"❌ Error: Subsets directory not found: {subsets_dir}")
                continue
            
            subsets = load_existing_subsets(subsets_dir)
            if not subsets:
                print("❌ No subsets loaded.")
                continue
            
            print("\nGenerating individual metatrait distributions...")
            for subset_name, subset_df in subsets.items():
                for metatrait in ['stability', 'plasticity']:
                    plot_metatrait_distribution(subset_df, subset_name, metatrait, plots_dir)
                print(f"  ✓ {subset_name}")
            print("\n✅ Individual metatrait distributions generated!")
            break
            
        elif choice == '7':
            # Generate plots for a specific subset
            if not subsets_dir.exists():
                print(f"❌ Error: Subsets directory not found: {subsets_dir}")
                continue
            
            subsets = load_existing_subsets(subsets_dir)
            if not subsets:
                print("❌ No subsets loaded.")
                continue
            
            print("\nAvailable subsets:")
            subset_list = list(subsets.keys())
            for i, name in enumerate(subset_list, 1):
                print(f"  {i}. {name}")
            
            subset_choice = input(f"\nEnter subset number (1-{len(subset_list)}): ").strip()
            try:
                subset_idx = int(subset_choice) - 1
                if 0 <= subset_idx < len(subset_list):
                    subset_name = subset_list[subset_idx]
                    subset_df = subsets[subset_name]
                    
                    print(f"\nGenerating plots for {subset_name}...")
                    plot_metatrait_distribution(subset_df, subset_name, 'stability', plots_dir)
                    plot_metatrait_distribution(subset_df, subset_name, 'plasticity', plots_dir)
                    plot_metatrait_overlay(subset_df, subset_name, plots_dir)
                    plot_ocean_overlay(subset_df, subset_name, plots_dir)
                    print(f"\n✅ All plots for {subset_name} generated!")
                else:
                    print("❌ Invalid subset number.")
            except ValueError:
                print("❌ Invalid input.")
            break
            
        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    main()
