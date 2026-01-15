"""
Feature Diagnostics Visualization Suite
Generates publication-quality figures for diagnostic outputs.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Scientific plotting style
plt.style.use('seaborn-v0_8-paper')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'

# ============================================================================
# CONFIGURATION
# ============================================================================

ROOT_DIR = Path(__file__).resolve().parents[2]
BASE_DIR = ROOT_DIR / "analysis" / "diagnostics"
OUTPUT_DIR = BASE_DIR  # single folder for all outputs
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Thresholds
CORR_THRESHOLD = 0.7
HIGH_CORR_THRESHOLD = 0.9
VIF_THRESHOLD = 10
COND_NUM_THRESHOLD = 30
STABILITY_THRESHOLD = 0.2
VARIANCE_THRESHOLD = 0.01
DOMINATED_THRESHOLD = 0.95

# Color palettes
COLORS = {
    'primary': '#2E86AB',
    'secondary': '#A23B72',
    'accent': '#F18F01',
    'warning': '#C73E1D',
    'neutral': '#6C757D',
    'success': '#06A77D'
}

# ============================================================================
# DATA LOADING
# ============================================================================

def load_diagnostics():
    """Load all diagnostic CSV files."""
    print("Loading diagnostic data...")
    
    files = {
        'pairwise': BASE_DIR / 'pairwise_dependence.csv',
        'theme': BASE_DIR / 'theme_redundancy.csv',
        'temporal': BASE_DIR / 'temporal_stability.csv',
        'interpret': BASE_DIR / 'interpretability_checks.csv'
    }
    
    data = {}
    for key, path in files.items():
        if path.exists():
            data[key] = pd.read_csv(path)
            print(f"  Loaded {key}: {len(data[key])} rows")
        else:
            print(f"  Missing {key}: {path}")
            data[key] = None
    
    # Normalize column names for pairwise and temporal files
    if data.get("pairwise") is not None:
        rename_map = {
            "feat_a": "feature_a",
            "feat_b": "feature_b",
            "spearman": "spearman",
            "kendall": "kendall",
            "mi": "mi"
        }
        data["pairwise"].rename(columns=rename_map, inplace=True)

    if data.get("temporal") is not None:
        temp = data["temporal"]
        for old, new in [
            ("rho_early", "early_rho"),
            ("rho_mid", "mid_rho"),
            ("rho_late", "late_rho")
        ]:
            if old in temp.columns:
                temp.rename(columns={old: new}, inplace=True)

    return data

# ============================================================================
# VISUALIZATION 1: CHORD DIAGRAM (High Correlations)
# ============================================================================

def plot_chord_diagram(df_pairwise, threshold=0.7):
    """Network graph showing high-correlation relationships."""
    if df_pairwise is None or len(df_pairwise) == 0:
        print("Skipping chord diagram: no data")
        return
    
    print("\n[1/9] Generating network graph (high correlations)...")
    
    df_high = df_pairwise[df_pairwise['spearman'].abs() >= threshold].copy()
    if len(df_high) == 0:
        print(f"  No correlations >= {threshold}")
        return
    
    import networkx as nx
    G = nx.Graph()
    for _, row in df_high.iterrows():
        G.add_edge(row['feature_a'], row['feature_b'], weight=abs(row['spearman']))
    
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    fig, ax = plt.subplots(figsize=(14, 10))
    edges = G.edges()
    weights = [G[u][v]['weight'] for u, v in edges]
    
    nx.draw_networkx_edges(
        G, pos, 
        width=[w*3 for w in weights],
        alpha=0.4,
        edge_color=weights,
        edge_cmap=plt.cm.RdYlBu_r,
        edge_vmin=threshold,
        edge_vmax=1.0,
        ax=ax
    )
    node_sizes = [G.degree(node) * 300 for node in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=COLORS['primary'], alpha=0.8, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold', ax=ax)
    
    ax.set_title(f'Feature Correlation Network (|rho| ≥ {threshold})', fontsize=14, fontweight='bold', pad=20)
    ax.axis('off')
    
    sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlBu_r, norm=plt.Normalize(vmin=threshold, vmax=1.0))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label('|Spearman rho|', rotation=270, labelpad=20)
    
    plt.savefig(OUTPUT_DIR / '01_correlation_network.png')
    plt.close()
    print(f"  OK Saved: 01_correlation_network.png ({len(df_high)} edges)")

# ============================================================================
# VISUALIZATION 2: CLUSTERED HEATMAP
# ============================================================================

def plot_correlation_heatmap(df_pairwise):
    """Hierarchical clustered heatmap of pairwise correlations."""
    if df_pairwise is None or len(df_pairwise) == 0:
        print("Skipping heatmap: no data")
        return
    
    print("\n[2/9] Generating clustered correlation heatmap...")
    
    features = sorted(set(df_pairwise['feature_a']) | set(df_pairwise['feature_b']))
    n = len(features)
    
    if n > 100:
        print(f"  Warning: {n} features. Using top 50 by max |rho|")
        max_corr = df_pairwise.groupby('feature_a')['spearman'].apply(lambda x: x.abs().max())
        top_features = max_corr.nlargest(50).index.tolist()
        df_sub = df_pairwise[
            df_pairwise['feature_a'].isin(top_features) &
            df_pairwise['feature_b'].isin(top_features)
        ]
        features = top_features
        n = len(features)
    else:
        df_sub = df_pairwise
    
    corr_matrix = pd.DataFrame(np.eye(n), index=features, columns=features)
    for _, row in df_sub.iterrows():
        if row['feature_a'] in features and row['feature_b'] in features:
            corr_matrix.loc[row['feature_a'], row['feature_b']] = row['spearman']
            corr_matrix.loc[row['feature_b'], row['feature_a']] = row['spearman']
    
    sns.clustermap(
        corr_matrix,
        cmap='RdBu_r',
        center=0,
        vmin=-1, vmax=1,
        linewidths=0,
        cbar_kws={'label': 'Spearman rho', 'shrink': 0.5},
        figsize=(16, 14),
        dendrogram_ratio=0.1,
        method='ward'
    )
    plt.suptitle('Hierarchical Clustering of Feature Correlations', y=0.98, fontsize=14, fontweight='bold')
    plt.savefig(OUTPUT_DIR / '02_correlation_heatmap.png')
    plt.close()
    print(f"  OK Saved: 02_correlation_heatmap.png ({n}×{n} matrix)")

# ============================================================================
# VISUALIZATION 3: VIF LOLLIPOP CHART
# ============================================================================

def plot_vif_lollipop(df_theme):
    """Lollipop chart for VIF values by theme."""
    if df_theme is None or len(df_theme) == 0:
        print("Skipping VIF chart: no data")
        return
    
    print("\n[3/9] Generating VIF lollipop chart...")
    
    df_plot = df_theme.sort_values('max_vif', ascending=True).copy()
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.hlines(y=range(len(df_plot)), xmin=0, xmax=df_plot['max_vif'], color=COLORS['neutral'], alpha=0.4, linewidth=2)
    colors = [COLORS['warning'] if v > VIF_THRESHOLD else COLORS['primary'] for v in df_plot['max_vif']]
    ax.scatter(df_plot['max_vif'], range(len(df_plot)), s=120, c=colors, alpha=0.8, edgecolors='white', linewidth=1.5, zorder=3)
    ax.axvline(VIF_THRESHOLD, color=COLORS['warning'], linestyle='--', linewidth=2, alpha=0.6, label=f'VIF threshold = {VIF_THRESHOLD}')
    ax.set_yticks(range(len(df_plot)))
    ax.set_yticklabels(df_plot['theme'])
    ax.set_xlabel('Maximum VIF', fontweight='bold')
    ax.set_title('Variance Inflation Factors by Feature Theme', fontsize=13, fontweight='bold', pad=15)
    ax.legend(loc='lower right')
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '03_vif_lollipop.png')
    plt.close()
    print(f"  OK Saved: 03_vif_lollipop.png")

# ============================================================================
# VISUALIZATION 4: CONDITION NUMBER
# ============================================================================

def plot_condition_numbers(df_theme):
    """Bar plot for condition numbers across themes."""
    if df_theme is None or len(df_theme) == 0:
        print("Skipping condition number plot: no data")
        return
    
    print("\n[4/9] Generating condition number comparison...")
    
    col = 'condition_number' if 'condition_number' in df_theme.columns else 'cond_num'
    if col not in df_theme.columns:
        print("  Missing condition number column; skipping.")
        return
    df_plot = df_theme.sort_values(col, ascending=False)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(
        range(len(df_plot)),
        df_plot[col],
        color=[COLORS['warning'] if c > COND_NUM_THRESHOLD else COLORS['primary'] for c in df_plot[col]],
        alpha=0.7
    )
    ax.axvline(COND_NUM_THRESHOLD, color=COLORS['warning'], linestyle='--', linewidth=2, alpha=0.6, label=f'Ill-conditioning threshold = {COND_NUM_THRESHOLD}')
    ax.set_yticks(range(len(df_plot)))
    ax.set_yticklabels(df_plot['theme'])
    ax.set_xlabel('Condition Number', fontweight='bold')
    ax.set_title('Matrix Condition Numbers by Theme', fontsize=13, fontweight='bold', pad=15)
    ax.legend(loc='lower right')
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '04_condition_numbers.png')
    plt.close()
    print(f"  OK Saved: 04_condition_numbers.png")

# ============================================================================
# VISUALIZATION 5: PCA VARIANCE
# ============================================================================

def plot_pca_variance(df_theme):
    """Bar chart showing variance explained by PC1/PC2 per theme."""
    if df_theme is None or 'pc1_var' not in df_theme.columns:
        print("Skipping PCA plot: no data")
        return
    
    print("\n[5/9] Generating PCA variance chart...")
    
    df_plot = df_theme.sort_values('pc1_var', ascending=True).copy()
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.barh(range(len(df_plot)), df_plot['pc1_var'] * 100, color=COLORS['primary'], alpha=0.7, label='PC1 variance (%)')
    if 'pc2_var' in df_plot.columns:
        ax.barh(range(len(df_plot)), df_plot['pc2_var'] * 100, left=df_plot['pc1_var'] * 100, color=COLORS['neutral'], alpha=0.5, label='PC2 variance (%)')
    
    ax.set_yticks(range(len(df_plot)))
    ax.set_yticklabels(df_plot['theme'])
    ax.set_xlabel('Variance Explained (%)', fontweight='bold')
    ax.set_xlim(0, 100)
    ax.set_title('PCA Variance: First Two Components by Theme', fontsize=13, fontweight='bold', pad=15)
    ax.legend(loc='lower right')
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '05_pca_variance.png')
    plt.close()
    print(f"  OK Saved: 05_pca_variance.png")

# ============================================================================
# VISUALIZATION 6: TEMPORAL STABILITY RIDGE PLOTS
# ============================================================================

def plot_temporal_stability_ridges(df_temporal):
    """Ridge-style violin plots for correlation distributions across periods."""
    if df_temporal is None or len(df_temporal) == 0:
        print("Skipping temporal stability plot: no data")
        return
    
    print("\n[6/9] Generating temporal stability ridge plots...")
    
    periods = []
    for col in ['early_rho', 'mid_rho', 'late_rho']:
        if col in df_temporal.columns:
            period_name = col.replace('_rho', '').replace('_', ' ').title()
            periods.append({'period': period_name, 'correlations': df_temporal[col].dropna().values})
    
    if not periods:
        print("  No temporal columns found")
        return
    
    fig, axes = plt.subplots(len(periods), 1, figsize=(12, 3 * len(periods)), sharex=True)
    if len(periods) == 1:
        axes = [axes]
    colors = [COLORS['primary'], COLORS['accent'], COLORS['secondary']]
    
    for ax, period_data, color in zip(axes, periods, colors):
        corrs = period_data['correlations']
        parts = ax.violinplot([corrs], positions=[0], vert=False, widths=0.7, showmeans=True, showmedians=True)
        for pc in parts['bodies']:
            pc.set_facecolor(color)
            pc.set_alpha(0.6)
        ax.set_yticks([0])
        ax.set_yticklabels([period_data['period']])
        ax.set_xlim(-1, 1)
        ax.grid(axis='x', alpha=0.3)
        ax.axvline(0, color='black', linestyle='-', linewidth=0.8, alpha=0.3)
        stats_text = f"mean={corrs.mean():.2f}, sd={corrs.std():.2f}, n={len(corrs)}"
        ax.text(0.98, 0.5, stats_text, transform=ax.transAxes, ha='right', va='center', fontsize=9, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    axes[-1].set_xlabel('Spearman rho', fontweight='bold')
    fig.suptitle('Temporal Stability of Pairwise Correlations', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '06_temporal_stability_ridges.png')
    plt.close()
    print(f"  OK Saved: 06_temporal_stability_ridges.png")

# ============================================================================
# VISUALIZATION 7: TEMPORAL STABILITY SLOPEGRAPH
# ============================================================================

def plot_temporal_slopegraph(df_temporal, top_n=20):
    """Slopegraph connecting Early -> Mid -> Late correlations for top pairs."""
    if df_temporal is None or len(df_temporal) == 0:
        print("Skipping slopegraph: no data")
        return
    
    print("\n[7/9] Generating temporal stability slopegraph...")
    
    req_cols = ['early_rho', 'mid_rho', 'late_rho']
    if not all(c in df_temporal.columns for c in req_cols):
        print("  Missing temporal columns")
        return
    
    df_temporal['max_drift'] = df_temporal[req_cols].max(axis=1) - df_temporal[req_cols].min(axis=1)
    df_plot = df_temporal.nlargest(top_n, 'max_drift').copy()
    df_plot['pair'] = df_plot['feature_a'] + ' -> ' + df_plot['feature_b'] if 'feature_a' in df_plot.columns else [f"Pair {i+1}" for i in range(len(df_plot))]
    
    fig, ax = plt.subplots(figsize=(12, 10))
    periods_x = [0, 1, 2]
    period_labels = ['Early', 'Mid', 'Late']
    
    for _, row in df_plot.iterrows():
        values = [row['early_rho'], row['mid_rho'], row['late_rho']]
        drift = row['max_drift']
        if drift > STABILITY_THRESHOLD * 1.5:
            color = COLORS['warning']; alpha = 0.8; linewidth = 2
        elif drift > STABILITY_THRESHOLD:
            color = COLORS['accent']; alpha = 0.6; linewidth = 1.5
        else:
            color = COLORS['primary']; alpha = 0.4; linewidth = 1
        ax.plot(periods_x, values, marker='o', markersize=6, color=color, alpha=alpha, linewidth=linewidth)
    
    ax.set_xticks(periods_x)
    ax.set_xticklabels(period_labels)
    ax.set_ylabel('Spearman rho', fontweight='bold')
    ax.set_ylim(-1, 1)
    ax.axhline(0, color='black', linestyle='-', linewidth=0.8, alpha=0.3)
    ax.grid(axis='y', alpha=0.3)
    ax.set_title(f'Temporal Stability: Top {top_n} Most Variable Feature Pairs', fontsize=13, fontweight='bold', pad=15)
    
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=COLORS['warning'], linewidth=2, label='High drift (>0.3)'),
        Line2D([0], [0], color=COLORS['accent'], linewidth=1.5, label='Moderate drift (0.2-0.3)'),
        Line2D([0], [0], color=COLORS['primary'], linewidth=1, label='Stable (<0.2)')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '07_temporal_slopegraph.png')
    plt.close()
    print(f"  OK Saved: 07_temporal_slopegraph.png")

# ============================================================================
# VISUALIZATION 8: INTERPRETABILITY SCATTER
# ============================================================================

def plot_interpretability_scatter(df_interpret):
    """Scatterplot: Variance vs. Max |rho| with quadrant labels."""
    if df_interpret is None or len(df_interpret) == 0:
        print("Skipping interpretability scatter: no data")
        return
    
    print("\n[8/9] Generating interpretability scatterplot...")
    
    if 'variance' not in df_interpret.columns or 'max_abs_rho' not in df_interpret.columns:
        print("  Missing required columns (variance, max_abs_rho)")
        return
    
    df_plot = df_interpret.copy()
    sizes = df_plot['entropy'].fillna(1) * 50 if 'entropy' in df_plot.columns else 100
    
    def categorize(row):
        if row['variance'] < VARIANCE_THRESHOLD and row['max_abs_rho'] > DOMINATED_THRESHOLD:
            return 'Low-var & Dominated'
        elif row['variance'] < VARIANCE_THRESHOLD:
            return 'Low-variance'
        elif row['max_abs_rho'] > DOMINATED_THRESHOLD:
            return 'Dominated'
        elif row['max_abs_rho'] < 0.5 and row['variance'] > 1.0:
            return 'Unique & High-info'
        else:
            return 'Standard'
    
    df_plot['category'] = df_plot.apply(categorize, axis=1)
    cat_colors = {
        'Low-var & Dominated': COLORS['warning'],
        'Low-variance': COLORS['accent'],
        'Dominated': COLORS['secondary'],
        'Unique & High-info': COLORS['success'],
        'Standard': COLORS['neutral']
    }
    
    fig, ax = plt.subplots(figsize=(12, 10))
    for cat, color in cat_colors.items():
        subset = df_plot[df_plot['category'] == cat]
        if len(subset) > 0:
            ax.scatter(
                subset['variance'],
                subset['max_abs_rho'],
                s=sizes if not hasattr(sizes, '__iter__') else sizes[subset.index],
                c=color,
                alpha=0.6,
                edgecolors='white',
                linewidth=0.5,
                label=f'{cat} (n={len(subset)})'
            )
    
    ax.axhline(DOMINATED_THRESHOLD, color=COLORS['warning'], linestyle='--', linewidth=1.5, alpha=0.5, label=f'Dominated threshold (|rho|={DOMINATED_THRESHOLD})')
    ax.axvline(VARIANCE_THRESHOLD, color=COLORS['accent'], linestyle='--', linewidth=1.5, alpha=0.5, label=f'Low-variance threshold ({VARIANCE_THRESHOLD})')
    
    ax.set_xlabel('Feature Variance', fontweight='bold')
    ax.set_ylabel('Max |rho| with Any Other Feature', fontweight='bold')
    ax.set_xscale('log')
    ax.set_ylim(0, 1.05)
    ax.set_title('Feature Interpretability: Variance vs. Dominance', fontsize=13, fontweight='bold', pad=15)
    ax.legend(loc='upper left', fontsize=8, framealpha=0.9)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '08_interpretability_scatter.png')
    plt.close()
    print(f"  OK Saved: 08_interpretability_scatter.png")

# ============================================================================
# VISUALIZATION 9: VARIANCE DISTRIBUTION BY THEME
# ============================================================================

def plot_variance_by_theme(df_interpret):
    """Violin/box plot for variance distribution across feature themes."""
    if df_interpret is None or 'variance' not in df_interpret.columns:
        print("Skipping variance distribution plot: no data")
        return
    
    print("\n[9/9] Generating variance distribution by theme...")
    
    if 'theme' not in df_interpret.columns:
        df_interpret['theme'] = df_interpret['feature'].str.split('_').str[0]
    
    theme_counts = df_interpret['theme'].value_counts()
    valid_themes = theme_counts[theme_counts >= 3].index
    df_plot = df_interpret[df_interpret['theme'].isin(valid_themes)].copy()
    
    if len(df_plot) == 0:
        print("  Not enough features per theme")
        return
    
    fig, ax = plt.subplots(figsize=(12, 8))
    parts = ax.violinplot(
        [df_plot[df_plot['theme'] == t]['variance'].values for t in valid_themes],
        positions=range(len(valid_themes)),
        widths=0.7,
        showmeans=True,
        showmedians=True
    )
    for pc in parts['bodies']:
        pc.set_facecolor(COLORS['primary'])
        pc.set_alpha(0.6)
    
    ax.boxplot(
        [df_plot[df_plot['theme'] == t]['variance'].values for t in valid_themes],
        positions=range(len(valid_themes)),
        widths=0.3,
        patch_artist=True,
        boxprops=dict(facecolor='white', alpha=0.8),
        medianprops=dict(color=COLORS['warning'], linewidth=2)
    )
    
    ax.axhline(VARIANCE_THRESHOLD, color=COLORS['accent'], linestyle='--', linewidth=2, alpha=0.6, label=f'Low-variance threshold = {VARIANCE_THRESHOLD}')
    ax.set_xticks(range(len(valid_themes)))
    ax.set_xticklabels(valid_themes, rotation=45, ha='right')
    ax.set_ylabel('Feature Variance (log scale)', fontweight='bold')
    ax.set_yscale('log')
    ax.set_title('Feature Variance Distribution by Theme', fontsize=13, fontweight='bold', pad=15)
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '09_variance_by_theme.png')
    plt.close()
    print(f"  OK Saved: 09_variance_by_theme.png")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("=" * 70)
    print("FEATURE DIAGNOSTICS VISUALIZATION SUITE")
    print("=" * 70)
    
    data = load_diagnostics()
    
    plot_chord_diagram(data['pairwise'], threshold=CORR_THRESHOLD)
    plot_correlation_heatmap(data['pairwise'])
    plot_vif_lollipop(data['theme'])
    plot_condition_numbers(data['theme'])
    plot_pca_variance(data['theme'])
    plot_temporal_stability_ridges(data['temporal'])
    plot_temporal_slopegraph(data['temporal'], top_n=20)
    plot_interpretability_scatter(data['interpret'])
    plot_variance_by_theme(data['interpret'])
    
    print("\n" + "=" * 70)
    print("OK ALL VISUALIZATIONS COMPLETE")
    print(f"  Output directory: {OUTPUT_DIR}")
    print("  Generated 9 publication-quality figures")
    print("=" * 70)


if __name__ == "__main__":
    main()
