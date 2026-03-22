"""
BrainFit Replication Analysis - Complete Analysis Pipeline
Based on Manning et al. (2022) Scientific Reports methodology

Run this after data preprocessing and quality control.
Assumes you have:
- behavioral_summary.pkl: behavioral scores per participant
- fitness_summary.pkl: aggregated fitness features per participant
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
from statsmodels.stats.multitest import multipletests
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data_copy"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

N_BOOTSTRAP_ITERATIONS = 10000
ALPHA = 0.05
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)

# Set plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ============================================================================
# STEP 4: BOOTSTRAP CORRELATION ANALYSIS
# ============================================================================

def bootstrap_correlation_analysis(
    df,
    fitness_features,
    behavioral_features,
    n_iterations=10000,
    alpha=0.05,
    random_state=42
):
    """
    Paper's bootstrap correlation method (Manning et al., 2022, page 5):
    
    For each iteration:
    1. Sample N participants WITH REPLACEMENT
    2. Compute correlations for all feature pairs (pairwise-complete)
    3. Track sign consistency across iterations
    4. Report correlations significant in >97.5% of iterations (two-tailed p<0.05)
    
    Uses Fisher z-transformation for averaging correlations (paper's method).
    
    Parameters:
    -----------
    df : pd.DataFrame
        Merged dataset with both fitness and behavioral features
    fitness_features : list
        List of fitness feature column names
    behavioral_features : list
        List of behavioral feature column names
    n_iterations : int
        Number of bootstrap iterations (paper uses 10,000)
    alpha : float
        Significance threshold (default 0.05 for two-tailed test)
    random_state : int
        Random seed for reproducibility
        
    Returns:
    --------
    pd.DataFrame
        Results with columns: fitness_feature, behavioral_feature, correlation,
        p_value, sign, positive_proportion, n_bootstrap_samples, ci_lower, ci_upper
    """
    
    np.random.seed(random_state)
    n_participants = len(df)
    
    print("\n" + "="*80)
    print("BOOTSTRAP CORRELATION ANALYSIS")
    print("="*80)
    print(f"\nParameters:")
    print(f"  Iterations: {n_iterations:,}")
    print(f"  Participants: {n_participants}")
    print(f"  Fitness features: {len(fitness_features)}")
    print(f"  Behavioral features: {len(behavioral_features)}")
    print(f"  Total correlation tests: {len(fitness_features) * len(behavioral_features):,}")
    print(f"  Significance criterion: {100 * (1 - alpha/2):.1f}% sign consistency (two-tailed p<{alpha})")
    
    # Store correlations for each pair across iterations
    correlation_distributions = defaultdict(list)
    valid_sample_counts = defaultdict(int)
    
    print("\nRunning bootstrap iterations...")
    for iteration in tqdm(range(n_iterations), desc="Bootstrap sampling"):
        # Sample with replacement
        sampled_indices = np.random.choice(
            df.index, 
            size=n_participants, 
            replace=True
        )
        sampled_df = df.loc[sampled_indices].reset_index(drop=True)
        
        # Compute all pairwise correlations
        for fitness_feat in fitness_features:
            for behav_feat in behavioral_features:
                # Get pairwise-complete data
                pair_data = sampled_df[[fitness_feat, behav_feat]].dropna()
                
                # Need at least 3 observations for meaningful correlation
                if len(pair_data) < 3:
                    continue
                
                # Compute Pearson correlation (use numpy directly to avoid pandas/numpy compat bug)
                x = pair_data[fitness_feat].to_numpy(dtype=float)
                y = pair_data[behav_feat].to_numpy(dtype=float)
                if x.std() == 0 or y.std() == 0:
                    continue
                r = np.corrcoef(x, y)[0, 1]
                
                if not np.isnan(r):
                    correlation_distributions[(fitness_feat, behav_feat)].append(r)
                    valid_sample_counts[(fitness_feat, behav_feat)] += 1
    
    # Analyze distributions and identify significant correlations
    print("\nAnalyzing correlation distributions...")
    results = []
    
    # Significance threshold (paper's method)
    threshold = 1 - alpha / 2  # 0.975 for alpha=0.05
    
    for (fitness_feat, behav_feat), correlations in tqdm(
        correlation_distributions.items(), 
        desc="Computing statistics"
    ):
        correlations = np.array(correlations)
        n_samples = len(correlations)
        
        # Skip if too few valid samples (require at least 50% of iterations)
        if n_samples < n_iterations * 0.5:
            continue
        
        # Compute proportion of positive correlations
        positive_proportion = (correlations > 0).mean()
        
        # Check for significance using paper's criterion
        is_significant_positive = positive_proportion >= threshold
        is_significant_negative = positive_proportion <= (1 - threshold)
        
        if is_significant_positive or is_significant_negative:
            # Use Fisher z-transformation for averaging (paper's equation, page 5)
            # E[r] = tanh(mean(arctanh(r)))
            z_values = np.arctanh(np.clip(correlations, -0.999, 0.999))
            mean_z = z_values.mean()
            mean_r = np.tanh(mean_z)
            
            # Compute p-value
            if is_significant_positive:
                p_value = 2 * (1 - positive_proportion)
                sign = 'positive'
            else:
                p_value = 2 * positive_proportion
                sign = 'negative'
            
            # Compute confidence intervals
            ci_lower = np.percentile(correlations, 2.5)
            ci_upper = np.percentile(correlations, 97.5)
            
            results.append({
                'fitness_feature': fitness_feat,
                'behavioral_feature': behav_feat,
                'correlation': mean_r,
                'p_value': p_value,
                'sign': sign,
                'positive_proportion': positive_proportion,
                'n_bootstrap_samples': n_samples,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'ci_width': ci_upper - ci_lower
            })
    
    # Create results dataframe
    results_df = pd.DataFrame(results)
    
    if len(results_df) > 0:
        results_df = results_df.sort_values('p_value').reset_index(drop=True)
    
    # Print summary
    print("\n" + "="*80)
    print("BOOTSTRAP RESULTS SUMMARY")
    print("="*80)
    print(f"\nSignificant correlations found: {len(results_df)}")
    
    if len(results_df) > 0:
        print(f"  Positive correlations: {(results_df['sign'] == 'positive').sum()}")
        print(f"  Negative correlations: {(results_df['sign'] == 'negative').sum()}")
        
        print(f"\nCorrelation strength statistics:")
        print(f"  Mean |r|: {results_df['correlation'].abs().mean():.3f}")
        print(f"  Median |r|: {results_df['correlation'].abs().median():.3f}")
        print(f"  Max |r|: {results_df['correlation'].abs().max():.3f}")
        
        print(f"\nTop 5 strongest positive correlations:")
        top_positive = results_df[results_df['sign'] == 'positive'].nlargest(5, 'correlation')
        for _, row in top_positive.iterrows():
            print(f"  {row['fitness_feature'][:30]:30s} ↔ {row['behavioral_feature']:25s}  r={row['correlation']:6.3f}  p={row['p_value']:.6f}")
        
        if (results_df['sign'] == 'negative').any():
            print(f"\nTop 5 strongest negative correlations:")
            top_negative = results_df[results_df['sign'] == 'negative'].nsmallest(5, 'correlation')
            for _, row in top_negative.iterrows():
                print(f"  {row['fitness_feature'][:30]:30s} ↔ {row['behavioral_feature']:25s}  r={row['correlation']:6.3f}  p={row['p_value']:.6f}")
    else:
        print("\n  No significant correlations found.")
        print("  This could indicate:")
        print("  - Insufficient sample size")
        print("  - High data sparsity")
        print("  - True absence of associations")
    
    return results_df


# ============================================================================
# STEP 5: MULTIPLE COMPARISON CORRECTION
# ============================================================================

def apply_multiple_comparison_correction(correlation_results, alpha=0.05, method='fdr_bh'):
    """
    Apply multiple comparison correction to control false discovery rate.
    
    Parameters:
    -----------
    correlation_results : pd.DataFrame
        Results from bootstrap_correlation_analysis
    alpha : float
        Family-wise error rate threshold
    method : str
        Correction method: 'fdr_bh' (Benjamini-Hochberg), 'bonferroni', 'holm'
        
    Returns:
    --------
    pd.DataFrame
        Results with additional columns: p_value_corrected, significant_corrected
    """
    
    print("\n" + "="*80)
    print("MULTIPLE COMPARISON CORRECTION")
    print("="*80)
    print(f"\nMethod: {method}")
    print(f"Alpha: {alpha}")
    
    if len(correlation_results) == 0:
        print("\nNo correlations to correct (empty results)")
        return correlation_results
    
    correlation_results = correlation_results.copy()
    
    # Apply correction
    reject, pvals_corrected, alphacSidak, alphacBonf = multipletests(
        correlation_results['p_value'],
        alpha=alpha,
        method=method
    )
    
    # Add corrected results
    correlation_results['p_value_corrected'] = pvals_corrected
    correlation_results['significant_corrected'] = reject
    
    # Summary statistics
    n_total = len(correlation_results)
    n_sig_uncorrected = (correlation_results['p_value'] < alpha).sum()
    n_sig_corrected = reject.sum()
    
    print(f"\nResults:")
    print(f"  Total tests: {n_total}")
    print(f"  Significant (uncorrected p < {alpha}): {n_sig_uncorrected} ({n_sig_uncorrected/n_total*100:.1f}%)")
    print(f"  Significant (corrected p < {alpha}): {n_sig_corrected} ({n_sig_corrected/n_total*100:.1f}%)")
    print(f"  Rejection rate: {(n_total - n_sig_corrected)/n_total*100:.1f}%")
    
    if method == 'bonferroni':
        print(f"  Bonferroni-corrected threshold: {alphacBonf:.6f}")
    
    if n_sig_corrected > 0:
        print(f"\nCorrected significant correlations by sign:")
        sig_df = correlation_results[correlation_results['significant_corrected']]
        print(f"  Positive: {(sig_df['sign'] == 'positive').sum()}")
        print(f"  Negative: {(sig_df['sign'] == 'negative').sum()}")
        
        print(f"\nTop 10 corrected significant correlations:")
        top_sig = sig_df.nlargest(10, 'correlation', keep='all')[
            ['fitness_feature', 'behavioral_feature', 'correlation', 'p_value', 'p_value_corrected']
        ]
        print(top_sig.to_string(index=False))
    
    return correlation_results


# ============================================================================
# STEP 6: VISUALIZATIONS
# ============================================================================

def create_correlation_heatmap(correlation_results, output_path=None, title='Fitness-Behavior Correlations'):
    """
    Create publication-quality correlation heatmap (paper Figure 5 style).
    
    Parameters:
    -----------
    correlation_results : pd.DataFrame
        Corrected correlation results
    output_path : Path or str, optional
        Where to save the figure (default: RESULTS_DIR/correlation_heatmap.png)
    title : str
        Plot title
    """
    
    print("\n" + "="*80)
    print("CREATING CORRELATION HEATMAP")
    print("="*80)
    
    # Filter to significant correlations (corrected)
    sig_corr = correlation_results[correlation_results['significant_corrected']].copy()
    
    if len(sig_corr) == 0:
        print("\nNo significant correlations to visualize")
        print("Skipping heatmap creation")
        return
    
    print(f"\nVisualizing {len(sig_corr)} significant correlations")
    
    # Pivot to matrix format
    pivot = sig_corr.pivot_table(
        index='fitness_feature',
        columns='behavioral_feature',
        values='correlation',
        aggfunc='first'
    )
    
    # Determine figure size based on matrix dimensions
    n_fitness = len(pivot.index)
    n_behavioral = len(pivot.columns)
    figsize = (max(10, n_behavioral * 0.8), max(8, n_fitness * 0.4))
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap
    sns.heatmap(
        pivot,
        annot=True,
        fmt='.2f',
        cmap='RdBu_r',
        center=0,
        vmin=-0.7,
        vmax=0.7,
        cbar_kws={'label': 'Correlation Coefficient (r)', 'shrink': 0.8},
        linewidths=0.5,
        linecolor='gray',
        ax=ax,
        square=False,
        annot_kws={'size': 9}
    )
    
    # Styling
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Behavioral/Memory Measures', fontsize=13, fontweight='bold')
    ax.set_ylabel('Fitness Measures', fontsize=13, fontweight='bold')
    
    # Rotate labels
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    
    plt.tight_layout()
    
    # Save
    if output_path is None:
        output_path = RESULTS_DIR / 'correlation_heatmap.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved: {output_path}")
    
    plt.show()


def create_scatter_matrix(df, correlation_results, top_n=12, output_path=None):
    """
    Create scatter plots for top N correlations.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Merged dataset
    correlation_results : pd.DataFrame
        Corrected correlation results
    top_n : int
        Number of top correlations to plot
    output_path : Path or str, optional
        Where to save the figure
    """
    
    print("\n" + "="*80)
    print("CREATING SCATTER PLOT MATRIX")
    print("="*80)
    
    sig_corr = correlation_results[correlation_results['significant_corrected']].copy()
    
    if len(sig_corr) == 0:
        print("\nNo significant correlations to plot")
        return
    
    # Get top N by absolute correlation
    sig_corr['abs_corr'] = sig_corr['correlation'].abs()
    top_corr = sig_corr.nlargest(min(top_n, len(sig_corr)), 'abs_corr')
    
    print(f"\nCreating scatter plots for top {len(top_corr)} correlations")
    
    # Create grid
    n_plots = len(top_corr)
    n_cols = 3
    n_rows = int(np.ceil(n_plots / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    axes = axes.flatten() if n_plots > 1 else [axes]
    
    for idx, (_, row) in enumerate(top_corr.iterrows()):
        ax = axes[idx]
        
        fitness_feat = row['fitness_feature']
        behav_feat = row['behavioral_feature']
        r = row['correlation']
        p = row['p_value_corrected']
        ci_lower = row['ci_lower']
        ci_upper = row['ci_upper']
        
        # Get pairwise-complete data
        plot_df = df[[fitness_feat, behav_feat]].dropna()
        
        if len(plot_df) < 2:
            ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center')
            ax.axis('off')
            continue
        
        # Scatter plot
        ax.scatter(
            plot_df[fitness_feat],
            plot_df[behav_feat],
            alpha=0.6,
            s=60,
            edgecolors='black',
            linewidth=0.5
        )
        
        # Regression line
        z = np.polyfit(plot_df[fitness_feat], plot_df[behav_feat], 1)
        p_fit = np.poly1d(z)
        x_line = np.linspace(plot_df[fitness_feat].min(), plot_df[fitness_feat].max(), 100)
        try:
            ax.plot(x_line, p_fit(x_line), "r-", alpha=0.8, linewidth=2.5)
            ax.fill_between(
                x_line,
                p_fit(x_line) - 0.1,
                p_fit(x_line) + 0.1,
                alpha=0.1,
                color='red'
            )
        except Exception:
            ax.text(
                0.5,
                0.95,
                "Regression line skipped",
                transform=ax.transAxes,
                ha='center',
                va='top',
                fontsize=8,
                color='darkred'
            )
        
        # Labels and title
        ax.set_xlabel(fitness_feat, fontsize=10, fontweight='bold')
        ax.set_ylabel(behav_feat, fontsize=10, fontweight='bold')
        ax.set_title(
            f'r = {r:.3f} [{ci_lower:.3f}, {ci_upper:.3f}]\np = {p:.4f} (n={len(plot_df)})', 
            fontsize=11, 
            fontweight='bold'
        )
        ax.grid(alpha=0.3, linestyle='--')
    
    # Hide extra subplots
    for idx in range(n_plots, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Top Significant Fitness-Behavior Correlations', 
                 fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    
    # Save
    if output_path is None:
        output_path = RESULTS_DIR / 'top_correlations_scatter.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved: {output_path}")
    
    plt.show()


def create_correlation_distribution_plot(correlation_results, output_path=None):
    """
    Plot distribution of correlation coefficients.
    """
    
    print("\n" + "="*80)
    print("CREATING CORRELATION DISTRIBUTION PLOT")
    print("="*80)
    
    if len(correlation_results) == 0:
        print("\nNo correlations to plot")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: All correlations
    ax1 = axes[0]
    ax1.hist(
        correlation_results['correlation'], 
        bins=30, 
        edgecolor='black', 
        alpha=0.7,
        color='steelblue'
    )
    ax1.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero')
    ax1.set_xlabel('Correlation Coefficient (r)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax1.set_title('Distribution of All Bootstrap Correlations', fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3, axis='y')
    
    # Plot 2: Significant vs non-significant
    ax2 = axes[1]
    
    sig_corr = correlation_results[correlation_results['significant_corrected']]
    nonsig_corr = correlation_results[~correlation_results['significant_corrected']]
    
    ax2.hist(
        nonsig_corr['correlation'], 
        bins=30, 
        alpha=0.5, 
        label=f'Non-significant (n={len(nonsig_corr)})',
        color='gray',
        edgecolor='black'
    )
    ax2.hist(
        sig_corr['correlation'], 
        bins=30, 
        alpha=0.7, 
        label=f'Significant (n={len(sig_corr)})',
        color='orange',
        edgecolor='black'
    )
    ax2.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero')
    ax2.set_xlabel('Correlation Coefficient (r)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax2.set_title('Significant vs Non-Significant Correlations', fontsize=13, fontweight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save
    if output_path is None:
        output_path = RESULTS_DIR / 'correlation_distribution.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved: {output_path}")
    
    plt.show()


def create_task_specificity_plot(correlation_results, output_path=None):
    """
    Analyze and visualize task-specific associations (paper's key finding).
    
    Shows which fitness metrics associate with which specific tasks.
    """
    
    print("\n" + "="*80)
    print("CREATING TASK SPECIFICITY ANALYSIS")
    print("="*80)
    
    sig_corr = correlation_results[correlation_results['significant_corrected']].copy()
    
    if len(sig_corr) == 0:
        print("\nNo significant correlations for task specificity analysis")
        return
    
    # Count associations per fitness feature
    fitness_task_counts = sig_corr.groupby('fitness_feature').agg({
        'behavioral_feature': lambda x: list(x),
        'correlation': ['count', 'mean']
    }).reset_index()
    
    fitness_task_counts.columns = ['fitness_feature', 'associated_tasks', 'n_tasks', 'mean_r']
    fitness_task_counts = fitness_task_counts.sort_values('n_tasks', ascending=False)
    
    # Count associations per behavioral feature
    task_fitness_counts = sig_corr.groupby('behavioral_feature').agg({
        'fitness_feature': lambda x: list(x),
        'correlation': ['count', 'mean']
    }).reset_index()
    
    task_fitness_counts.columns = ['behavioral_feature', 'associated_fitness', 'n_fitness', 'mean_r']
    task_fitness_counts = task_fitness_counts.sort_values('n_fitness', ascending=False)
    
    print(f"\nTask Specificity Summary:")
    print(f"\nFitness features by number of task associations:")
    for _, row in fitness_task_counts.head(10).iterrows():
        tasks_str = ', '.join(row['associated_tasks'][:3])
        if len(row['associated_tasks']) > 3:
            tasks_str += f", ... ({len(row['associated_tasks'])} total)"
        print(f"  {row['fitness_feature'][:40]:40s}: {row['n_tasks']} tasks ({tasks_str})")
    
    print(f"\nBehavioral tasks by number of fitness associations:")
    for _, row in task_fitness_counts.iterrows():
        fitness_str = ', '.join([f[:20] for f in row['associated_fitness'][:3]])
        if len(row['associated_fitness']) > 3:
            fitness_str += f", ... ({len(row['associated_fitness'])} total)"
        print(f"  {row['behavioral_feature']:30s}: {row['n_fitness']} fitness features ({fitness_str})")
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Number of task associations per fitness feature
    ax1 = axes[0]
    top_fitness = fitness_task_counts.head(15)
    ax1.barh(range(len(top_fitness)), top_fitness['n_tasks'], color='steelblue', edgecolor='black')
    ax1.set_yticks(range(len(top_fitness)))
    ax1.set_yticklabels(top_fitness['fitness_feature'], fontsize=9)
    ax1.set_xlabel('Number of Associated Tasks', fontsize=11, fontweight='bold')
    ax1.set_title('Fitness Features: Task Association Breadth', fontsize=12, fontweight='bold')
    ax1.grid(alpha=0.3, axis='x')
    ax1.invert_yaxis()
    
    # Plot 2: Number of fitness associations per task
    ax2 = axes[1]
    ax2.barh(range(len(task_fitness_counts)), task_fitness_counts['n_fitness'], color='coral', edgecolor='black')
    ax2.set_yticks(range(len(task_fitness_counts)))
    ax2.set_yticklabels(task_fitness_counts['behavioral_feature'], fontsize=10)
    ax2.set_xlabel('Number of Associated Fitness Features', fontsize=11, fontweight='bold')
    ax2.set_title('Behavioral Tasks: Fitness Association Breadth', fontsize=12, fontweight='bold')
    ax2.grid(alpha=0.3, axis='x')
    ax2.invert_yaxis()
    
    plt.tight_layout()
    
    # Save
    if output_path is None:
        output_path = RESULTS_DIR / 'task_specificity.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved: {output_path}")
    
    plt.show()
    
    return fitness_task_counts, task_fitness_counts


# ============================================================================
# STEP 7: COMPREHENSIVE REPORTING
# ============================================================================

def generate_comprehensive_report(
    merged_df,
    correlation_results,
    behavioral_df,
    fitness_df,
    output_path=None
):
    """
    Generate publication-ready analysis report.
    
    Parameters:
    -----------
    merged_df : pd.DataFrame
        Merged analysis dataset
    correlation_results : pd.DataFrame
        Corrected correlation results
    behavioral_df : pd.DataFrame
        Original behavioral data
    fitness_df : pd.DataFrame
        Original fitness data
    output_path : Path or str, optional
        Where to save report
    """
    
    print("\n" + "="*80)
    print("GENERATING COMPREHENSIVE REPORT")
    print("="*80)
    
    sig_corr = correlation_results[correlation_results['significant_corrected']].copy()
    
    report = f"""
{'='*80}
BRAINFIT REPLICATION ANALYSIS - COMPREHENSIVE REPORT
{'='*80}

Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
Based on: Manning et al. (2022) Scientific Reports
         "Fitness tracking reveals task-specific associations between 
          memory, mental health, and physical activity"

{'='*80}
DATASET CHARACTERISTICS
{'='*80}

Participants:
  Total: {len(merged_df)}
  With behavioral data: {len(behavioral_df)}
  With fitness data: {len(fitness_df)}
  
Behavioral Features:
  Total: {len(behavioral_df.columns)}
  Features: {', '.join(behavioral_df.columns.tolist())}
  
Fitness Features:
  Total: {len(fitness_df.columns)}
  Completeness range: {fitness_df.notna().mean().min()*100:.1f}% - {fitness_df.notna().mean().max()*100:.1f}%
  
Data Quality:
  Mean behavioral completeness: {behavioral_df.notna().mean().mean()*100:.1f}%
  Mean fitness completeness: {fitness_df.notna().mean().mean()*100:.1f}%
  
{'='*80}
BOOTSTRAP CORRELATION ANALYSIS
{'='*80}

Parameters:
  Bootstrap iterations: {N_BOOTSTRAP_ITERATIONS:,}
  Significance threshold: p < {ALPHA} (two-tailed)
  Sign consistency criterion: {100*(1-ALPHA/2):.1f}%
  Multiple comparison correction: Benjamini-Hochberg FDR
  
Tests Performed:
  Total feature pairs: {len(correlation_results):,}
  
Results (Uncorrected):
  Significant correlations (p < {ALPHA}): {(correlation_results['p_value'] < ALPHA).sum()}
  
Results (FDR-Corrected):
  Significant correlations: {len(sig_corr)}
  Positive correlations: {(sig_corr['sign'] == 'positive').sum()}
  Negative correlations: {(sig_corr['sign'] == 'negative').sum()}

"""
    
    if len(sig_corr) > 0:
        report += f"""
Correlation Strength (FDR-Significant):
  Mean |r|: {sig_corr['correlation'].abs().mean():.3f}
  Median |r|: {sig_corr['correlation'].abs().median():.3f}
  Range: [{sig_corr['correlation'].min():.3f}, {sig_corr['correlation'].max():.3f}]
  
Confidence Interval Widths:
  Mean CI width: {sig_corr['ci_width'].mean():.3f}
  Median CI width: {sig_corr['ci_width'].median():.3f}

{'='*80}
STRONGEST POSITIVE ASSOCIATIONS (FDR-CORRECTED)
{'='*80}

"""
        
        # Top 10 positive
        sig_positive = sig_corr[sig_corr['sign'] == 'positive'].nlargest(10, 'correlation')
        
        if len(sig_positive) > 0:
            for rank, (_, row) in enumerate(sig_positive.iterrows(), 1):
                report += f"""
{rank}. {row['fitness_feature']} ↔ {row['behavioral_feature']}
   r = {row['correlation']:.3f} (95% CI: [{row['ci_lower']:.3f}, {row['ci_upper']:.3f}])
   p = {row['p_value_corrected']:.6f}
   Bootstrap samples: {row['n_bootstrap_samples']:,}
"""
        
        report += f"""
{'='*80}
STRONGEST NEGATIVE ASSOCIATIONS (FDR-CORRECTED)
{'='*80}

"""
        
        # Top 10 negative
        sig_negative = sig_corr[sig_corr['sign'] == 'negative'].nsmallest(10, 'correlation')
        
        if len(sig_negative) > 0:
            for rank, (_, row) in enumerate(sig_negative.iterrows(), 1):
                report += f"""
{rank}. {row['fitness_feature']} ↔ {row['behavioral_feature']}
   r = {row['correlation']:.3f} (95% CI: [{row['ci_lower']:.3f}, {row['ci_upper']:.3f}])
   p = {row['p_value_corrected']:.6f}
   Bootstrap samples: {row['n_bootstrap_samples']:,}
"""
        
        # Task specificity analysis
        report += f"""
{'='*80}
TASK SPECIFICITY ANALYSIS
{'='*80}

This analysis identifies fitness metrics that show selective associations
with specific behavioral tasks (paper's primary finding).

"""
        
        # Count associations per feature
        fitness_counts = sig_corr.groupby('fitness_feature')['behavioral_feature'].apply(list).to_dict()
        task_counts = sig_corr.groupby('behavioral_feature')['fitness_feature'].apply(list).to_dict()
        
        report += f"""
Fitness Features with Selective Associations:
(Features that correlate with some tasks but not others)

"""
        
        for fitness_feat, tasks in sorted(fitness_counts.items(), key=lambda x: len(x[1]), reverse=True)[:10]:
            report += f"\n{fitness_feat}:\n"
            report += f"  Associated with {len(tasks)} task(s): {', '.join(tasks)}\n"
        
        report += f"""
Behavioral Tasks with Selective Fitness Associations:
(Tasks that correlate with some fitness metrics but not others)

"""
        
        for task, fitness_feats in sorted(task_counts.items(), key=lambda x: len(x[1]), reverse=True):
            report += f"\n{task}:\n"
            report += f"  Associated with {len(fitness_feats)} fitness feature(s)\n"
            for feat in fitness_feats[:5]:
                report += f"    - {feat}\n"
            if len(fitness_feats) > 5:
                report += f"    ... and {len(fitness_feats) - 5} more\n"
    
    else:
        report += "\nNo significant correlations found after FDR correction.\n"
        report += "\nPossible explanations:\n"
        report += "  - Insufficient sample size (N={len(merged_df)})\n"
        report += "  - High data sparsity in fitness features\n"
        report += "  - True absence of strong associations\n"
        report += "  - Overly conservative correction for small samples\n"
    
    report += f"""
{'='*80}
RECOMMENDATIONS FOR INTERPRETATION
{'='*80}

1. Task Specificity: Different fitness metrics associate with different
   cognitive tasks, suggesting differentiated effects (paper's main finding).

2. Effect Sizes: Correlation magnitudes should be interpreted in context
   of measurement noise and individual differences.

3. Causality: These are correlational findings. Experimental manipulation
   would be needed to establish causal relationships.

4. Replication: Cross-validation on held-out test set recommended.

5. Clinical Significance: Statistical significance does not necessarily
   imply practical/clinical significance.

{'='*80}
ANALYSIS COMPLETE
{'='*80}

Output files generated:
  - bootstrap_correlations_fdr.csv: Full correlation results
  - correlation_heatmap.png: Visualization of significant correlations
  - top_correlations_scatter.png: Scatter plots of strongest associations
  - correlation_distribution.png: Distribution of correlation coefficients
  - task_specificity.png: Task-specific association analysis
  - analysis_report.txt: This comprehensive report

For questions or issues, refer to:
Manning et al. (2022) Sci Rep 12:13822
https://doi.org/10.1038/s41598-022-17781-0

"""
    
    print(report)
    
    # Save report
    if output_path is None:
        output_path = RESULTS_DIR / 'analysis_report.txt'
    
    with open(output_path, 'w') as f:
        f.write(report)
    
    print(f"\nReport saved: {output_path}")
    
    return report


def save_all_results(correlation_results, merged_df):
    """
    Save all analysis results to disk.
    """
    
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)
    
    # 1. Full correlation results
    csv_path = RESULTS_DIR / 'bootstrap_correlations_fdr.csv'
    correlation_results.to_csv(csv_path, index=False)
    print(f"\n✓ Saved: {csv_path}")
    
    # 2. Significant correlations only
    sig_path = RESULTS_DIR / 'significant_correlations_only.csv'
    sig_corr = correlation_results[correlation_results['significant_corrected']]
    sig_corr.to_csv(sig_path, index=False)
    print(f"✓ Saved: {sig_path}")
    
    # 3. Merged analysis data
    data_path = RESULTS_DIR / 'merged_analysis_data.csv'
    merged_df.to_csv(data_path, index=True)
    print(f"✓ Saved: {data_path}")
    
    # 4. Summary statistics
    summary_path = RESULTS_DIR / 'summary_statistics.txt'
    with open(summary_path, 'w') as f:
        f.write("SUMMARY STATISTICS\n")
        f.write("="*80 + "\n\n")
        f.write(f"Total correlations tested: {len(correlation_results)}\n")
        f.write(f"Significant (FDR-corrected): {sig_corr.shape[0]}\n")
        f.write(f"Mean |r| (significant): {sig_corr['correlation'].abs().mean():.3f}\n")
        f.write(f"Median |r| (significant): {sig_corr['correlation'].abs().median():.3f}\n")
    print(f"✓ Saved: {summary_path}")
    
    print(f"\nAll results saved to: {RESULTS_DIR}")


# ============================================================================
# MAIN ANALYSIS PIPELINE
# ============================================================================

def run_complete_analysis():
    """
    Execute complete analysis pipeline from Step 4 onwards.
    """
    
    print("\n" + "="*80)
    print("BRAINFIT REPLICATION ANALYSIS - COMPLETE PIPELINE")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Bootstrap iterations: {N_BOOTSTRAP_ITERATIONS:,}")
    print(f"  Significance threshold: {ALPHA}")
    print(f"  Random seed: {RANDOM_SEED}")
    print(f"  Data directory: {DATA_DIR}")
    print(f"  Results directory: {RESULTS_DIR}")
    
    # Load data
    print("\n" + "="*80)
    print("LOADING DATA")
    print("="*80)
    
    behavioral_df = pd.read_pickle(DATA_DIR / "behavioral_summary.pkl")
    fitness_df = pd.read_pickle(DATA_DIR / "fitness_summary.pkl")
    
    print(f"\nBehavioral data: {behavioral_df.shape}")
    print(f"Fitness data: {fitness_df.shape}")
    
    # Merge datasets
    print("\nMerging datasets...")
    merged_df = behavioral_df.join(fitness_df, how='inner')
    print(f"Merged data: {merged_df.shape}")
    print(f"Participants: {len(merged_df)}")
    
    # Separate features
    behavioral_features = behavioral_df.columns.tolist()
    fitness_features = fitness_df.columns.tolist()
    
    print(f"\nBehavioral features: {len(behavioral_features)}")
    print(f"Fitness features: {len(fitness_features)}")
    
    # STEP 4: Bootstrap correlation analysis
    correlation_results = bootstrap_correlation_analysis(
        df=merged_df,
        fitness_features=fitness_features,
        behavioral_features=behavioral_features,
        n_iterations=N_BOOTSTRAP_ITERATIONS,
        alpha=ALPHA,
        random_state=RANDOM_SEED
    )
    
    # STEP 5: Multiple comparison correction
    correlation_results = apply_multiple_comparison_correction(
        correlation_results,
        alpha=ALPHA,
        method='fdr_bh'
    )
    
    # STEP 6: Create visualizations
    create_correlation_heatmap(correlation_results)
    create_scatter_matrix(merged_df, correlation_results, top_n=12)
    create_correlation_distribution_plot(correlation_results)
    create_task_specificity_plot(correlation_results)
    
    # STEP 7: Generate comprehensive report
    report = generate_comprehensive_report(
        merged_df=merged_df,
        correlation_results=correlation_results,
        behavioral_df=behavioral_df,
        fitness_df=fitness_df
    )
    
    # Save all results
    save_all_results(correlation_results, merged_df)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print(f"\nAll outputs saved to: {RESULTS_DIR}")
    print("\nGenerated files:")
    print("  1. bootstrap_correlations_fdr.csv - Full results")
    print("  2. significant_correlations_only.csv - FDR-significant only")
    print("  3. merged_analysis_data.csv - Analysis dataset")
    print("  4. correlation_heatmap.png - Heatmap visualization")
    print("  5. top_correlations_scatter.png - Scatter plots")
    print("  6. correlation_distribution.png - Distribution plot")
    print("  7. task_specificity.png - Task specificity analysis")
    print("  8. analysis_report.txt - Comprehensive report")
    print("  9. summary_statistics.txt - Quick summary")
    
    return correlation_results, merged_df, report


# ============================================================================
# EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Run complete analysis
    correlation_results, merged_df, report = run_complete_analysis()
    
    # Print final summary
    print("\n" + "="*80)
    print("QUICK SUMMARY")
    print("="*80)
    
    sig_count = correlation_results['significant_corrected'].sum()
    total_count = len(correlation_results)
    
    print(f"\n✓ Analysis completed successfully")
    print(f"\n✓ Found {sig_count} significant correlations out of {total_count} tested")
    print(f"  ({sig_count/total_count*100:.1f}% discovery rate after FDR correction)")
    
    if sig_count > 0:
        print(f"\n✓ Strongest correlation:")
        strongest = correlation_results[correlation_results['significant_corrected']].iloc[0]
        print(f"  {strongest['fitness_feature']} ↔ {strongest['behavioral_feature']}")
        print(f"  r = {strongest['correlation']:.3f}, p = {strongest['p_value_corrected']:.6f}")
    
    print(f"\n✓ All results saved to: {RESULTS_DIR}")
    print(f"\n✓ Review 'analysis_report.txt' for detailed findings")