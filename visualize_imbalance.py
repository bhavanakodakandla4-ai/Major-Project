import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import Counter
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src.config import (
    PROCESSED_DATA_DIR, RAW_DATA_DIR, PLOTS_DIR, RANDOM_STATE
)

OUTPUT_DIR = os.path.join(PLOTS_DIR, "smote_analysis")
os.makedirs(OUTPUT_DIR, exist_ok=True)

PALETTE = {
    "benign":  "#4CAF50",
    "attack":  "#F44336",
    "bg":      "#0F1117",
    "surface": "#1A1D27",
    "card":    "#252836",
    "text":    "#E0E0E0",
    "subtext": "#9E9E9E",
    "accent":  "#7C4DFF",
    "grid":    "#2A2D3A",
}

plt.rcParams.update({
    "figure.facecolor":  PALETTE["bg"],
    "axes.facecolor":    PALETTE["surface"],
    "axes.edgecolor":    PALETTE["grid"],
    "axes.labelcolor":   PALETTE["text"],
    "axes.titlecolor":   PALETTE["text"],
    "xtick.color":       PALETTE["subtext"],
    "ytick.color":       PALETTE["subtext"],
    "text.color":        PALETTE["text"],
    "grid.color":        PALETTE["grid"],
    "grid.linewidth":    0.6,
    "font.family":       "DejaVu Sans",
    "font.size":         11,
    "axes.titlesize":    14,
    "axes.labelsize":    12,
})


def load_dataset(dataset_name):
    proc = os.path.join(PROCESSED_DATA_DIR, f"{dataset_name}_processed.csv")
    raw  = os.path.join(RAW_DATA_DIR, f"{dataset_name}.csv")

    if os.path.exists(proc):
        print(f"[OK] Loading processed dataset: {proc}")
        df = pd.read_csv(proc, low_memory=False, nrows=300_000)
        label_col = "Label_Binary" if "Label_Binary" in df.columns else "Label"
    elif os.path.exists(raw):
        print(f"[OK] Loading raw dataset: {raw}")
        df = pd.read_csv(raw, low_memory=False, nrows=300_000)
        label_col = "Label" if "Label" in df.columns else None
    else:
        raise FileNotFoundError(f"No dataset found for: {dataset_name}")

    return df, label_col


def extract_labels(df, label_col):
    if label_col == "Label_Binary":
        y = df[label_col].astype(int)
        labels = {0: "Benign", 1: "Attack"}
    else:
        benign_vals = {"Benign", "BENIGN", "benign"}
        y = df[label_col].apply(lambda v: 0 if str(v).strip() in benign_vals else 1)
        labels = {0: "Benign", 1: "Attack"}

    counter = Counter(y)
    return y, counter, labels


def fig1_bar_chart(counter, dataset_name):
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor(PALETTE["bg"])

    classes = ["Benign (0)", "Attack (1)"]
    counts  = [counter[0], counter[1]]
    colors  = [PALETTE["benign"], PALETTE["attack"]]

    bars = ax.bar(classes, counts, color=colors, width=0.5, zorder=3,
                  edgecolor=PALETTE["bg"], linewidth=1.2)

    for bar, count in zip(bars, counts):
        pct = count / sum(counts) * 100
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(counts) * 0.015,
            f"{count:,}\n({pct:.1f}%)",
            ha="center", va="bottom",
            color=PALETTE["text"], fontsize=11, fontweight="bold",
        )

    ax.set_title(f"Class Distribution Before SMOTE\n{dataset_name}", pad=16)
    ax.set_ylabel("Sample Count")
    ax.yaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)

    total = sum(counts)
    ir = max(counts) / max(min(counts), 1)
    fig.text(0.5, 0.01, f"Total samples: {total:,}   |   Imbalance Ratio: {ir:.1f}:1",
             ha="center", color=PALETTE["subtext"], fontsize=10)

    path = os.path.join(OUTPUT_DIR, f"{dataset_name}_1_bar_class_distribution.png")
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


def fig2_pie_chart(counter, dataset_name):
    fig, ax = plt.subplots(figsize=(7, 7))
    fig.patch.set_facecolor(PALETTE["bg"])

    sizes  = [counter[0], counter[1]]
    labels = ["Benign", "Attack"]
    colors = [PALETTE["benign"], PALETTE["attack"]]
    explode = (0.04, 0.04)

    wedges, texts, autotexts = ax.pie(
        sizes, labels=None, colors=colors, explode=explode,
        autopct="%1.1f%%", startangle=140,
        wedgeprops={"edgecolor": PALETTE["bg"], "linewidth": 2},
        textprops={"color": PALETTE["text"]},
        pctdistance=0.75,
    )
    for at in autotexts:
        at.set_fontsize(13)
        at.set_fontweight("bold")
        at.set_color(PALETTE["bg"])

    legend_patches = [
        mpatches.Patch(color=colors[i], label=f"{labels[i]}: {sizes[i]:,}") for i in range(2)
    ]
    ax.legend(handles=legend_patches, loc="lower center",
              bbox_to_anchor=(0.5, -0.08), ncol=2,
              framealpha=0, labelcolor=PALETTE["text"], fontsize=11)

    ax.set_title(f"Class Proportion Before SMOTE\n{dataset_name}", pad=16)

    path = os.path.join(OUTPUT_DIR, f"{dataset_name}_2_pie_class_proportion.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


def fig3_imbalance_ratio_gauge(counter, dataset_name):
    ir = counter[0] / max(counter[1], 1)
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor(PALETTE["bg"])

    ratios   = [1, 5, 10, 20, 50, 100]
    severity = ["1:1\nPerfect", "5:1\nMild", "10:1\nModerate",
                "20:1\nHigh", "50:1\nSevere", "100:1\nExtreme"]
    seg_colors = ["#4CAF50", "#8BC34A", "#FFEB3B", "#FF9800", "#F44336", "#B71C1C"]

    for i in range(len(ratios) - 1):
        ax.barh(0, ratios[i + 1] - ratios[i], left=ratios[i],
                color=seg_colors[i], height=0.6, alpha=0.85)
        ax.text(ratios[i], -0.6, severity[i], ha="left",
                color=PALETTE["subtext"], fontsize=9)

    clamped_ir = min(ir, ratios[-1])
    ax.axvline(clamped_ir, color="white", linewidth=3, linestyle="--", zorder=5)
    ax.text(clamped_ir, 0.45, f"  IR = {ir:.1f}:1",
            color="white", fontsize=13, fontweight="bold", va="center")

    ax.set_xlim(0, 105)
    ax.set_ylim(-1, 1)
    ax.set_xlabel("Imbalance Ratio (Majority : Minority)")
    ax.set_title(f"Imbalance Ratio Gauge — Before SMOTE\n{dataset_name}", pad=16)
    ax.set_yticks([])
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.yaxis.grid(False)

    path = os.path.join(OUTPUT_DIR, f"{dataset_name}_3_imbalance_ratio_gauge.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


def fig4_pca_scatter(df, y, label_col, dataset_name):
    feature_cols = [c for c in df.columns if c not in {label_col, "Label", "Label_Binary"}]
    X = df[feature_cols].copy()
    X = X.replace([np.inf, -np.inf], np.nan).fillna(X.median(numeric_only=True))
    X = X.select_dtypes(include=[np.number])

    n_samples = min(8000, len(X))
    idx = np.random.RandomState(RANDOM_STATE).choice(len(X), n_samples, replace=False)
    X_s = X.iloc[idx]
    y_s = y.iloc[idx]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_s)
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_scaled)

    fig, ax = plt.subplots(figsize=(9, 7))
    fig.patch.set_facecolor(PALETTE["bg"])

    for cls, color, label in [(0, PALETTE["benign"], "Benign"), (1, PALETTE["attack"], "Attack")]:
        mask = y_s.values == cls
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                   c=color, label=f"{label} ({mask.sum():,})",
                   alpha=0.35, s=8, linewidths=0, zorder=3)

    ax.set_title(f"PCA 2D View — Class Imbalance Before SMOTE\n{dataset_name}", pad=16)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)")
    ax.legend(framealpha=0.15, labelcolor=PALETTE["text"], markerscale=3)
    ax.grid(True, zorder=0)

    path = os.path.join(OUTPUT_DIR, f"{dataset_name}_4_pca_scatter.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


def fig5_feature_boxplot(df, y, label_col, dataset_name):
    feature_cols = [c for c in df.columns if c not in {label_col, "Label", "Label_Binary"}]
    X = df[feature_cols].copy()
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.select_dtypes(include=[np.number])

    variances = X.var().sort_values(ascending=False)
    top_feats = variances.head(6).index.tolist()

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.patch.set_facecolor(PALETTE["bg"])
    axes = axes.flatten()

    benign_mask = y.values == 0
    attack_mask = y.values == 1

    for i, feat in enumerate(top_feats):
        ax = axes[i]
        col_data = X[feat].fillna(X[feat].median())
        b_data = col_data.values[benign_mask]
        a_data = col_data.values[attack_mask]

        bp = ax.boxplot(
            [b_data, a_data],
            patch_artist=True,
            widths=0.5,
            medianprops={"color": "white", "linewidth": 2},
            whiskerprops={"color": PALETTE["subtext"]},
            capprops={"color": PALETTE["subtext"]},
            flierprops={"marker": ".", "markersize": 2,
                        "markerfacecolor": PALETTE["subtext"], "alpha": 0.3},
            boxprops={"linewidth": 1.2},
            notch=False,
        )
        bp["boxes"][0].set_facecolor(PALETTE["benign"])
        bp["boxes"][0].set_alpha(0.8)
        bp["boxes"][1].set_facecolor(PALETTE["attack"])
        bp["boxes"][1].set_alpha(0.8)

        ax.set_title(feat[:30], fontsize=10, pad=6)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(["Benign", "Attack"], fontsize=9)
        ax.yaxis.grid(True, zorder=0)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(f"Top 6 Feature Distributions by Class — Before SMOTE\n{dataset_name}",
                 fontsize=14, y=1.01)

    path = os.path.join(OUTPUT_DIR, f"{dataset_name}_5_feature_boxplot.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


def fig6_summary_dashboard(counter, dataset_name):
    benign = counter[0]
    attack = counter[1]
    total  = benign + attack
    ir     = benign / max(attack, 1)
    pct_b  = benign / total * 100
    pct_a  = attack / total * 100

    fig = plt.figure(figsize=(12, 6))
    fig.patch.set_facecolor(PALETTE["bg"])

    ax_bar = fig.add_axes([0.04, 0.15, 0.38, 0.70])
    ax_bar.set_facecolor(PALETTE["surface"])

    classes = ["Benign", "Attack"]
    counts  = [benign, attack]
    colors  = [PALETTE["benign"], PALETTE["attack"]]
    bars = ax_bar.bar(classes, counts, color=colors, width=0.5, zorder=3,
                      edgecolor=PALETTE["bg"], linewidth=1.2)
    for bar, count, pct in zip(bars, counts, [pct_b, pct_a]):
        ax_bar.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + total * 0.01,
                    f"{count:,}\n{pct:.1f}%",
                    ha="center", va="bottom", fontsize=10, fontweight="bold",
                    color=PALETTE["text"])
    ax_bar.set_title("Class Counts", fontsize=12)
    ax_bar.yaxis.grid(True, zorder=0)
    ax_bar.set_axisbelow(True)
    ax_bar.spines[["top", "right"]].set_visible(False)

    ax_pie = fig.add_axes([0.44, 0.12, 0.30, 0.76])
    ax_pie.set_facecolor(PALETTE["surface"])
    wedges, _, autotexts = ax_pie.pie(
        [benign, attack], colors=colors, explode=(0.04, 0.04),
        autopct="%1.1f%%", startangle=140,
        wedgeprops={"edgecolor": PALETTE["bg"], "linewidth": 2},
        pctdistance=0.72,
    )
    for at in autotexts:
        at.set_fontsize(12)
        at.set_fontweight("bold")
        at.set_color(PALETTE["bg"])
    ax_pie.set_title("Proportion", fontsize=12)

    ax_stats = fig.add_axes([0.76, 0.12, 0.22, 0.76])
    ax_stats.set_facecolor(PALETTE["card"])
    ax_stats.axis("off")

    stats = [
        ("Total Samples",  f"{total:,}"),
        ("Benign",         f"{benign:,}"),
        ("Attack",         f"{attack:,}"),
        ("% Benign",       f"{pct_b:.1f}%"),
        ("% Attack",       f"{pct_a:.1f}%"),
        ("Imbalance Ratio",f"{ir:.1f}:1"),
        ("SMOTE Required", "YES" if ir > 2 else "NO"),
    ]

    y_pos = 0.92
    for label, value in stats:
        ax_stats.text(0.08, y_pos, label, transform=ax_stats.transAxes,
                      fontsize=10, color=PALETTE["subtext"])
        color = PALETTE["attack"] if label == "SMOTE Required" and value == "YES" else PALETTE["text"]
        ax_stats.text(0.08, y_pos - 0.07, value, transform=ax_stats.transAxes,
                      fontsize=12, fontweight="bold", color=color)
        y_pos -= 0.125

    fig.suptitle(f"Class Imbalance Summary — Before SMOTE\n{dataset_name}",
                 fontsize=15, fontweight="bold", y=1.01)

    path = os.path.join(OUTPUT_DIR, f"{dataset_name}_6_summary_dashboard.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


def main():
    parser = argparse.ArgumentParser(
        description="Visualize class imbalance BEFORE SMOTE — CSE-CIC-IDS2018"
    )
    parser.add_argument("--dataset-name", type=str, default="02-14-2018",
                        help="Dataset name (default: 02-14-2018)")
    parser.add_argument("--skip-pca", action="store_true",
                        help="Skip PCA scatter plot (slow on large datasets)")
    args = parser.parse_args()

    print("=" * 60)
    print("  Class Imbalance Visualization — BEFORE SMOTE")
    print("=" * 60)
    print(f"  Dataset : {args.dataset_name}")
    print(f"  Output  : {OUTPUT_DIR}")
    print("=" * 60)

    df, label_col = load_dataset(args.dataset_name)
    y, counter, _ = extract_labels(df, label_col)

    print(f"\nClass distribution:")
    print(f"  Benign (0) : {counter[0]:,}")
    print(f"  Attack (1) : {counter[1]:,}")
    print(f"  Total      : {sum(counter.values()):,}")
    print(f"  IR         : {counter[0]/max(counter[1],1):.2f}:1\n")

    print("Generating figures...")
    fig1_bar_chart(counter, args.dataset_name)
    fig2_pie_chart(counter, args.dataset_name)
    fig3_imbalance_ratio_gauge(counter, args.dataset_name)

    if not args.skip_pca:
        fig4_pca_scatter(df, y, label_col, args.dataset_name)
    else:
        print("  Skipped: PCA scatter (--skip-pca)")

    fig5_feature_boxplot(df, y, label_col, args.dataset_name)
    fig6_summary_dashboard(counter, args.dataset_name)

    print(f"\n[OK] All figures saved to: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
