import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib import ticker
from scipy.stats import probplot, f_oneway

COLOR_PLOT = 'tan'
COLOR_FONT = 'dimgray'

def plot_target_distribution(df, column='price', color=COLOR_PLOT):
    """
    Visualize distribution of the target column
    using histogram, boxplot, log-scale, and Q-Q plot.
    """
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    target = df[column]

    # Histogram
    ax1 = axes[0, 0]
    ax1.hist(target, bins=50, alpha=0.7, edgecolor='dimgray', color=color)
    ax1.axvline(target.mean(), color='red', linestyle='-',
                label=f"Mean: ${target.mean():.0f}")
    ax1.axvline(target.median(), color='orange', linestyle='--',
                label=f"Median: ${target.median():.0f}")
    ax1.set_title("Rent Distribution")
    ax1.set_xlabel("Monthly Rent ($)")
    ax1.set_ylabel("Frequency")
    ax1.xaxis.set_major_formatter(ticker.EngFormatter())
    ax1.legend()

    # Boxplot
    ax2 = axes[0, 1]
    ax2.boxplot(target, orientation='horizontal')
    ax2.set_title("Rent Box Plot")
    ax2.set_ylabel("Monthly Rent ($)")
    ax2.xaxis.set_major_formatter(ticker.EngFormatter())

    # Log-transformed histogram
    ax3 = axes[1, 0]
    log_target = np.log1p(target)
    ax3.hist(log_target, bins=50, alpha=0.7, edgecolor='dimgray', color=color)
    ax3.set_title("Log-Transformed Rent Distribution")
    ax3.set_xlabel("Log(1 + Monthly Rent)")
    ax3.set_ylabel("Frequency")

    # Q-Q plot for normality
    ax4 = axes[1, 1]
    probplot(target, dist="norm", plot=ax4)
    qq_line = ax4.get_lines()[0]
    qq_line.set_markerfacecolor(color)
    qq_line.set_markeredgecolor(color)
    ax4.yaxis.set_major_formatter(ticker.EngFormatter())
    ax4.set_title("Q-Q Plot (Normal Distribution)")

    plt.tight_layout()
    plt.show()

def plot_numerical_hist(df, numerical_cols, n_cols=3):
    """Plots histograms for numerical columns in a grid layout."""
    n_features = len(numerical_cols)
    n_rows = (n_features + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 3*n_rows))

    for i, col in enumerate(numerical_cols):
        row = i // n_cols
        col_idx = i % n_cols
        ax = axes[row, col_idx]
        ax.hist(df[col].dropna(),
                bins=30, alpha=0.7,
                edgecolor=COLOR_FONT,
                color=COLOR_PLOT)
        ax.set_title(f"{col}")
        ax.set_xlabel(col)
        ax.set_ylabel('Frequency')

    # Remove empty subplots
    for i in range(n_features, n_rows * n_cols):
        row = i // n_cols
        col_idx = i % n_cols
        fig.delaxes(axes[row, col_idx])

    plt.tight_layout()
    plt.show()

def plot_numerical_vs_target(
    df,
    numerical_cols,
    target_col='price',
    filter_iqr=True):
    """
    Plots scatterplots of numerical columns
    against the target with optional IQR filtering.
    """
    n_features = len(numerical_cols)
    n_cols = 2
    n_rows = (n_features + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 3 * n_rows))
    axes = axes.flatten()

    for i, col in enumerate(numerical_cols):
        data = df.copy()

        if filter_iqr:
            # Filter only for the X variable (not the target)
            q1, q3 = data[col].quantile([0.25, 0.75])
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            data = data[(data[col] >= lower_bound) & (data[col] <= upper_bound)]

        ax = axes[i]
        ax.scatter(data=data, x=col, y=target_col, alpha=0.5, s=5, color=COLOR_PLOT)
        ax.set_title(f"{target_col.title()} vs {col}" + (" (IQR)" if filter_iqr else ""))
        ax.set_xlabel(col)
        ax.set_xlabel('')
        ax.yaxis.set_major_formatter(ticker.EngFormatter())

        if len(data) > 2:
            corr = data[col].corr(data[target_col])
            ax.text(0.05, 0.95, f"Corr {corr:.3f}",
                    transform=ax.transAxes,
                    va='top')

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

def analyze_categorical_feature(df, col):
    """
    Print unique counts, top values, and ANOVA test results
    for a categorical feature.
    """
    print(f"\n{col.upper()}")
    value_counts = df[col].value_counts()
    print(f"Unique values: {df[col].nunique()}")
    print(f"\nTop 10 values:\n{value_counts.head(10)}")

    rent_stats = (df.groupby(col)['price']
                  .agg(['mean', 'median', 'count'])
                  .sort_values('mean', ascending=False)
                  .round(2))
    print(f"\nAverage rent by {col}:")
    display(rent_stats.head(10))

    groups = [df[df[col] == category]['price']
              for category in df[col].unique() if pd.notna(category)]
    if len(groups) > 1:
        f_stat, p_value = f_oneway(*groups)
        print(f"\nANOVA F-statistic: {f_stat:.2f}, p-value: {p_value:.2e}")

def plot_categorical_distribution(df, col):
    """
    Visualize the frequency and rent distribution
    for the top 10 categories of a column.
    """
    value_counts = df[col].value_counts()
    top_10 = value_counts.head(10)
    df_top = df[df[col].isin(top_10.index)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    plt.suptitle(f"{col.upper()} (Top 10)", fontsize=14)

    # Bar chart
    ax1.bar(top_10.index, top_10.values, color=COLOR_PLOT)
    ax1.set_title(f"Frequency Distribution")
    ax1.set_ylabel("Count")
    # ax1.set_xticks(range(len(top_10)))
    # ax1.set_xticklabels(top_10.index, rotation=60, ha='right')
    ax1.tick_params(axis='x', rotation=60)

    # Boxplot
    sns.boxplot(df_top, x=col, y='price',
                order=top_10.index,
                color=COLOR_PLOT,
                fliersize=3, ax=ax2)
    ax2.set_title(f"Rent Distribution")
    ax2.set_ylabel("Monthly Rent ($)")
    ax2.tick_params(axis='x', rotation=60)
    ax2.yaxis.set_major_formatter(ticker.EngFormatter())

    plt.tight_layout()
    plt.show()

def analyze_neighborhoods(df, col):
    """
    Analyze rent stats for neighborhoods
    and plot average rent for the top 10 with ≥10 listings.
    """
    print(f"\n{col.upper()}")

    stats = (df.groupby(col)['price']
             .agg(['count', 'mean', 'median', 'std'])
             .round(2)
             .sort_values('mean', ascending=False))

    stats = stats[stats['count'] >= 10]
    print("Top neighborhoods (min 10 listings):")
    display(stats.head(15))

    top_10 = stats.head(10)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(top_10.index[::-1,], top_10['mean'][::-1], color=COLOR_PLOT)
    ax.set_title(f"Average Rent by {col.title()} (Top 10, Min 10 Listings)")
    ax.set_xlabel("Average Monthly Rent ($)")
    ax.xaxis.set_major_formatter(ticker.EngFormatter())
    plt.show
