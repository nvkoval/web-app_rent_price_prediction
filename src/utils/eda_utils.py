import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib import ticker
from scipy.stats import probplot, f_oneway

COLOR_PLOT = 'tan'
COLOR_FONT = 'dimgray'

def plot_histogram(ax, data, title, xlabel, color, is_log=False):
    # ax.hist(data, bins=50, alpha=0.7, edgecolor='dimgray', color=color)
    sns.histplot(data, bins=50, alpha=0.7, edgecolor='dimgray', color=color, ax=ax)
    ax.axvline(data.mean(), color='red', linestyle='-',
               label=f"Mean: {data.mean():.0f}")
    ax.axvline(data.median(), color='orange', linestyle='--',
               label=f"Median: {data.median():.0f}")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Frequency")
    if not is_log:
        ax.xaxis.set_major_formatter(ticker.EngFormatter())
    ax.legend()

def plot_boxplot(ax, data, title, ylabel, color, is_log=False):
    # ax.boxplot(data, orientation='horizontal',
    #            patch_artist=True, boxprops={'facecolor': color})
    sns.boxplot(data, orient='h', color=color, ax=ax)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    if not is_log:
        ax.xaxis.set_major_formatter(ticker.EngFormatter())

def plot_probplot(ax, data, title, color, is_log=False):
    probplot(data, dist="norm", plot=ax)
    qq_line = ax.get_lines()[0]
    qq_line.set_markerfacecolor(color)
    qq_line.set_markeredgecolor(color)
    ax.set_title(title)
    if not is_log:
        ax.yaxis.set_major_formatter(ticker.EngFormatter())

def plot_target_distribution(df, column='price', color=COLOR_PLOT):
    """
    Visualize distribution of the target column
    using histogram, boxplot, log-scale, and Q-Q plot.
    """
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    target = df[column]
    log_target = np.log1p(target)

    # Histogram
    plot_histogram(axes[0, 0], target,
                   title="Rent Distribution",
                   xlabel="Monthly Rent ($)",
                   color=color)

    # Boxplot
    plot_boxplot(axes[0, 1], target,
                 title="Rent Box Plot",
                 ylabel="Monthly Rent ($)",
                 color=color)

    # Log-transformed histogram
    plot_histogram(axes[1, 0], log_target,
                   title="Log-Transformed Rent Distribution",
                   xlabel="Log(1 + Monthly Rent)",
                   color=color,
                   is_log=True)

    # Q-Q plot for normality
    plot_probplot(axes[1, 1], target,
                  title="Q-Q Plot (Normal Distribution)",
                  color=color, is_log=False)

    plt.tight_layout()
    plt.show()

def plot_target_logtarget_distribution(df, column='price', color=COLOR_PLOT):
    """
    Visualize distribution of the target column and its log-transformation.
    Shows histogram, boxplot, and Q-Q plots.
    """
    target = df[column]
    log_target = np.log1p(target)

    fig, axes = plt.subplots(3, 2, figsize=(10, 10))

    # Histograms
    plot_histogram(axes[0, 0], target,
                   title="Rent Distribution",
                   xlabel="Monthly Rent ($)",
                   color=color)
    plot_histogram(axes[0, 1], log_target,
                   title="Log-Transformed Rent Distribution",
                   xlabel="Log(1 + Monthly Rent)",
                   color=color,
                   is_log=True)

    # Boxplots
    plot_boxplot(axes[1, 0], target,
                 title="Rent Box Plot",
                 ylabel="Monthly Rent ($)",
                 color=color)
    plot_boxplot(axes[1, 1], log_target,
                 title="Log-Transformed Rent Box Plot",
                 ylabel="Log(1 + Monthly Rent)",
                 color=color,
                 is_log=True)

    # Q-Q plots for normality
    plot_probplot(axes[2, 0], target,
                  title="Q-Q Plot (Normal Distribution)",
                  color=color, is_log=False)
    plot_probplot(axes[2, 1], log_target,
                  title="Q-Q Plot (Log-Transformed)",
                  color=color, is_log=True)

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

    rent_stats = (df.groupby(col, observed=True)['price']
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
