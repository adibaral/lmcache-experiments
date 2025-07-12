import os
import json
import re
from typing import Dict, List
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------------
# Data Loading and Extraction
# ----------------------------

def load_query_files(folder_path: str) -> Dict[str, List[dict]]:
    """
    Load all *_query.json files from the folder and group them by approach.
    """
    data_by_approach = {}
    for filename in os.listdir(folder_path):
        if filename.endswith('_query.json'):
            match = re.match(r'(.+?)_query\.json$', filename)
            if match:
                approach = match.group(1)
                file_path = os.path.join(folder_path, filename)
                with open(file_path, 'r') as f:
                    queries = json.load(f)
                data_by_approach[approach] = queries
    return data_by_approach


def extract_metrics(data_by_approach: Dict[str, List[dict]]) -> pd.DataFrame:
    """
    Extract relevant metrics from all approaches into a DataFrame.
    """
    rows = []
    for approach, queries in data_by_approach.items():
        for q in queries:
            query_text = q.get("query")
            _id = q.get("_id")
            for mode in ["cold", "warm"]:
                if mode in q:
                    metrics = q[mode]
                    rows.append({
                        "id": _id,
                        "approach": approach,
                        "query": query_text,
                        "mode": mode,
                        "tfft": metrics.get("tfft"),
                        "response_time": metrics.get("response_time"),
                        "prompt_tokens": metrics.get("prompt_tokens"),
                        "completion_tokens": metrics.get("completion_tokens"),
                        "total_tokens": metrics.get("total_tokens"),
                        "request_id": metrics.get("request_id"),
                    })
    return pd.DataFrame(rows)


def compute_aggregated_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute average and max metrics grouped by approach and mode.
    """
    agg_df = df.groupby(["approach", "mode"]).agg({
        "tfft": ["mean", "max"],
        "response_time": ["mean", "max"],
        "prompt_tokens": "mean",
        "completion_tokens": "mean",
        "total_tokens": "mean"
    })

    # Flatten column multi-index
    agg_df.columns = [
        f"{metric}_{stat}" if stat else metric
        for metric, stat in agg_df.columns
    ]
    
    return agg_df.reset_index()


def compute_per_query_tfft_diff(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute cold vs warm TTFT ratio for each query across all approaches.
    """
    cold_df = df[df["mode"] == "cold"].copy()
    warm_df = df[df["mode"] == "warm"].copy()

    merged = pd.merge(
        cold_df,
        warm_df,
        on=["id", "approach"],
        suffixes=("_cold", "_warm")
    )

    merged["tfft_diff"] = merged["tfft_cold"] - merged["tfft_warm"]
    merged["tfft_ratio"] = merged["tfft_cold"] / merged["tfft_warm"]

    return merged

# ----------------------------
# Plotting
# ----------------------------

def plot_avg_tfft_by_mode(agg_df: pd.DataFrame):
    plt.figure(figsize=(8, 5))
    sns.barplot(
        data=agg_df,
        x="approach",
        y="tfft_mean",
        hue="mode",
        palette="coolwarm"
    )
    plt.ylabel("Avg TTFT (s)")
    plt.title("Average TTFT by Approach and Mode")
    plt.tight_layout()
    plt.savefig("avg_ttft_by_mode.png", dpi=300)


def plot_response_time_by_mode(agg_df: pd.DataFrame):
    plt.figure(figsize=(8, 5))
    sns.barplot(
        data=agg_df,
        x="approach",
        y="response_time_mean",
        hue="mode",
        palette="viridis"
    )
    plt.ylabel("Avg Response Time (s)")
    plt.title("Average Response Time by Approach and Mode")
    plt.tight_layout()
    plt.savefig("avg_response_time_by_mode.png", dpi=300)


def plot_tfft_vs_tokens(df: pd.DataFrame, token_type: str = "prompt_tokens"):
    plt.figure(figsize=(8, 5))
    sns.scatterplot(
        data=df,
        x=token_type,
        y="tfft",
        hue="mode",
        style="approach",
        alpha=0.7
    )
    plt.title(f"TTFT vs {token_type.replace('_', ' ').title()}")
    plt.xlabel(token_type.replace("_", " ").title())
    plt.ylabel("TTFT (s)")
    plt.tight_layout()
    plt.savefig(f"tfft_vs_{token_type}.png", dpi=300)


def plot_tfft_ratio_distribution(diff_df: pd.DataFrame):
    plt.figure(figsize=(8, 5))
    sns.histplot(diff_df["tfft_ratio"], bins=30, kde=True, color="green")
    plt.title("Distribution of Cold/Warm TTFT Ratio")
    plt.xlabel("Cold TTFT / Warm TTFT")
    plt.ylabel("Query Count")
    plt.tight_layout()
    plt.savefig("tfft_ratio_distribution.png", dpi=300)

def plot_tfft_vs_tokens_with_regression(df: pd.DataFrame, token_type: str = "prompt_tokens"):
    plt.figure(figsize=(8, 5))
    sns.scatterplot(
        data=df,
        x=token_type,
        y="tfft",
        hue="mode",
        style="approach",
        alpha=0.7
    )
    sns.regplot(
        data=df,
        x=token_type,
        y="tfft",
        scatter=False,
        color="black",
        line_kws={"lw":2, "alpha":0.7}
    )
    plt.title(f"TTFT vs {token_type.replace('_', ' ').title()} with Regression Line")
    plt.xlabel(token_type.replace("_", " ").title())
    plt.ylabel("TTFT (s)")
    plt.tight_layout()
    plt.savefig(f"ttft_vs_{token_type}_regression.png", dpi=300)
    plt.close()


def plot_pairwise_relationships(df: pd.DataFrame):
    sns.pairplot(
        df,
        vars=["tfft", "response_time", "prompt_tokens", "completion_tokens"],
        hue="mode",
        kind="scatter",
        plot_kws={"alpha":0.7}
    )
    plt.suptitle("Pairwise Relationships of Metrics", y=1.02)
    plt.savefig("pairplot_metrics.png", dpi=300)
    plt.close()


def plot_tfft_boxplot(df: pd.DataFrame):
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x="approach", y="tfft", hue="mode", palette="Set2")
    plt.title("TTFT Distribution by Approach and Mode")
    plt.ylabel("TTFT (s)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("tfft_boxplot.png", dpi=300)
    plt.close()

def plot_cold_vs_warm_tfft(diff_df: pd.DataFrame):
    plt.figure(figsize=(8, 8))
    sns.scatterplot(
        data=diff_df,
        x="tfft_cold",
        y="tfft_warm",
        hue="approach",
        alpha=0.7,
        s=50
    )
    max_val = max(diff_df["tfft_cold"].max(), diff_df["tfft_warm"].max())
    plt.plot([0, max_val], [0, max_val], 'k--', label='Equal TTFT')
    plt.xlabel("Cold TTFT (s)")
    plt.ylabel("Warm TTFT (s)")
    plt.title("Cold vs Warm TTFT per Query")
    plt.legend()
    plt.tight_layout()
    plt.savefig("cold_vs_warm_tfft.png", dpi=300)
    plt.close()

def plot_tfft_diff_hist(diff_df: pd.DataFrame):
    plt.figure(figsize=(8, 5))
    sns.histplot(diff_df["tfft_diff"], bins=30, kde=True, color="coral")
    plt.title("Distribution of TTFT Differences (Cold - Warm)")
    plt.xlabel("TTFT Difference (s)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig("tfft_diff_hist.png", dpi=300)
    plt.close()

def plot_ttft_improvement_percentage(df: pd.DataFrame):
    # Compute mean TTFT per approach and mode
    means = df.groupby(['approach', 'mode'])['tfft'].mean().unstack()
    means['improvement_pct'] = (means['cold'] - means['warm']) / means['cold'] * 100

    plt.figure(figsize=(8,5))
    sns.barplot(x=means.index, y='improvement_pct', data=means.reset_index(), palette='coolwarm')
    plt.ylabel("TTFT Improvement (%)")
    plt.xlabel("Approach")
    plt.title("Average TTFT Improvement from Cold to Warm by Approach")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("ttft_improvement_percentage.png", dpi=300)
    plt.close()


def plot_paired_tfft(df: pd.DataFrame):
    plt.figure(figsize=(10, 6))
    # Prepare data
    df_sorted = df.sort_values(['approach', 'id', 'mode'])
    for approach in df['approach'].unique():
        subset = df[df['approach'] == approach]
        for qid in subset['id'].unique():
            qdata = subset[subset['id'] == qid]
            if len(qdata) == 2:  # cold and warm
                plt.plot(
                    [0,1],
                    qdata.sort_values('mode')['tfft'],
                    marker='o',
                    label=approach if qid == subset['id'].unique()[0] else "",
                    alpha=0.3
                )
    plt.xticks([0, 1], ['Cold', 'Warm'])
    plt.ylabel('TTFT (s)')
    plt.title('Paired TTFT per Query Across Approaches')
    plt.legend()
    plt.tight_layout()
    plt.savefig("paired_tfft.png", dpi=300)
    plt.close()

def plot_ttft_variance(df: pd.DataFrame):
    var_df = df.groupby(['approach', 'mode'])['tfft'].std().reset_index()
    plt.figure(figsize=(8,5))
    sns.barplot(data=var_df, x='approach', y='tfft', hue='mode', palette='magma')
    plt.ylabel("TTFT Std Deviation (s)")
    plt.title("TTFT Variability by Approach and Mode")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("ttft_variability.png", dpi=300)
    plt.close()

def plot_tfft_vs_total_tokens(df: pd.DataFrame):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=df,
        x='total_tokens',
        y='tfft',
        hue='approach',
        style='mode',
        alpha=0.7,
        palette='tab10',
        s=60
    )
    plt.xlabel("Total Tokens")
    plt.ylabel("TTFT (seconds)")
    plt.title("TTFT vs Total Tokens by Approach and Mode")
    plt.legend(title="Approach / Mode", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig("tfft_vs_total_tokens.png", dpi=300)
    plt.close()



def main(folder_path: str):
    data = load_query_files(folder_path)
    df = extract_metrics(data)
    agg_df = compute_aggregated_metrics(df)
    diff_df = compute_per_query_tfft_diff(df)

    plot_avg_tfft_by_mode(agg_df)
    plot_response_time_by_mode(agg_df)
    plot_tfft_vs_tokens(df, "prompt_tokens")
    plot_tfft_vs_tokens(df, "completion_tokens")
    plot_tfft_ratio_distribution(diff_df)
    plot_tfft_vs_tokens_with_regression(df, "prompt_tokens")
    plot_tfft_vs_tokens_with_regression(df, "completion_tokens")
    plot_pairwise_relationships(df)
    plot_tfft_boxplot(df)
    plot_cold_vs_warm_tfft(diff_df)
    plot_tfft_diff_hist(diff_df)
    plot_ttft_improvement_percentage(df)
    plot_paired_tfft(df)
    plot_ttft_variance(df)
    plot_tfft_vs_total_tokens(df)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark TTFT and response times.")
    parser.add_argument("--folder", type=str, help="Path to folder with *_query.json files")
    args = parser.parse_args()
    main(args.folder)
