import os
import shutil
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set aesthetics
sns.set_theme(style="whitegrid")
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 12,
        "axes.labelsize": 14,
        "axes.titlesize": 16,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "figure.titlesize": 18,
    }
)


def plot_slice_performance(test_df, output_path):
    """
    Generates the slice performance discovery plot.
    """
    # Sort by accuracy
    test_df = test_df.sort_values("accuracy", ascending=True).reset_index(drop=True)
    test_df["slice_rank"] = range(1, len(test_df) + 1)

    # Calculate overall test accuracy
    overall_acc = (test_df["accuracy"] * test_df["size"]).sum() / test_df["size"].sum()

    # Create plot
    fig, ax1 = plt.subplots(figsize=(12, 7))

    bar_colors = []
    for i, acc in enumerate(test_df["accuracy"]):
        if i == 0:
            bar_colors.append("#EC7063")  # Muted Red
        elif acc < overall_acc:
            bar_colors.append("#5D6D7E")  # Muted Dark Grey
        else:
            bar_colors.append("#D5D8DC")  # Muted Light Grey

    ax1.bar(
        test_df["slice_rank"],
        test_df["accuracy"],
        color=bar_colors,
        alpha=1.0,
        width=0.6,
        label="Slice Accuracy",
        zorder=3,
    )

    ax2 = ax1.twinx()
    ax2.set_ylabel("", alpha=0)
    ax2.set_yticks([])
    ax2.grid(False)

    ax1.set_axisbelow(True)
    ax1.grid(True, zorder=0)

    ax1.axhline(
        y=overall_acc,
        color="#E74C3C",
        linestyle="--",
        linewidth=2.5,
        label="Overall test accuracy (aggregate)",
        zorder=5,
    )

    ax1.set_xlabel("Slice rank (DOMINO-discovered, sorted by accuracy)")
    ax1.set_ylabel("Slice Accuracy")
    plt.title("Slice Performance Discovery (Domino)", pad=20)

    ax1.set_xticks(range(1, len(test_df) + 1))
    ax1.set_xlim(0, len(test_df) + 1)
    ax1.set_ylim(0, 1.05)

    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: f"{x:.0%}"))
    ax1.legend(loc="lower right", frameon=True, shadow=True)

    worst_slice = test_df.iloc[0]
    best_slice = test_df.iloc[-1]
    perf_gap = best_slice["accuracy"] - worst_slice["accuracy"]

    ax2.annotate(
        f"Worst slice: {worst_slice['accuracy']:.1%}\n"
        f"Performance Gap (Δ): {perf_gap:.1%}",
        xy=(1, worst_slice["accuracy"]),
        xycoords=ax1.transData,
        xytext=(0.05, 0.9),
        textcoords="axes fraction",
        arrowprops=dict(
            arrowstyle="->", connectionstyle="arc3,rad=.2", color="black", lw=2
        ),
        fontsize=12,
        fontweight="bold",
        color="#2C3E50",
        bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=1.0),
        zorder=20,
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Slice performance plot saved to: {output_path}")


def plot_error_concentration(test_df, output_path):
    """
    Generates the error mass concentration Pareto plot.
    """
    # Calculate errors per slice
    test_df = test_df.copy()
    test_df["errors"] = (1 - test_df["accuracy"]) * test_df["size"]
    total_errors = test_df["errors"].sum()
    test_df["error_fraction"] = test_df["errors"] / total_errors

    # Sort by error fraction descending
    test_df = test_df.sort_values("error_fraction", ascending=False).reset_index(
        drop=True
    )
    test_df["cumulative_error_fraction"] = test_df["error_fraction"].cumsum()
    test_df["rank"] = range(1, len(test_df) + 1)

    # Create plot
    fig, ax1 = plt.subplots(figsize=(12, 7))

    bar_colors = ["#EC7063"] * min(3, len(test_df)) + ["#D5D8DC"] * max(
        0, len(test_df) - 3
    )

    ax1.bar(
        test_df["rank"],
        test_df["error_fraction"],
        color=bar_colors,
        alpha=1.0,
        width=0.6,
        label="Slice Error Fraction",
        zorder=3,
    )

    ax2 = ax1.twinx()
    ax2.plot(
        test_df["rank"],
        test_df["cumulative_error_fraction"],
        color="#3498DB",
        marker="o",
        linewidth=3,
        markersize=8,
        label="Cumulative % of Total Errors",
    )

    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: f"{x:.0%}"))
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: f"{x:.0%}"))

    ax1.set_ylim(0, max(test_df["error_fraction"]) * 1.2)
    ax2.set_ylim(0, 1.1)

    ax1.set_xlabel("Slice rank (DOMINO-discovered, sorted by error mass)")
    ax1.set_ylabel("Fraction of total model errors", color="#2C3E50")
    ax2.set_ylabel("Cumulative % of total errors", color="#3498DB")
    plt.title("Error Mass Concentration (Pareto of Failures)", pad=30)

    ax1.set_axisbelow(True)
    ax1.grid(True, zorder=0)
    ax2.grid(False)

    ax1.set_xticks(range(1, len(test_df) + 1))
    ax1.set_xlim(0, len(test_df) + 1)

    top_3_fraction = test_df.iloc[min(2, len(test_df) - 1)]["cumulative_error_fraction"]
    ax2.annotate(
        f"Top 3 slices → {top_3_fraction:.1%} of errors",
        xy=(2, test_df.iloc[min(1, len(test_df) - 1)]["error_fraction"]),
        xycoords=ax1.transData,
        xytext=(0.1, 0.9),
        textcoords="axes fraction",
        arrowprops=dict(
            arrowstyle="->", connectionstyle="arc3,rad=.2", color="black", lw=2
        ),
        fontsize=12,
        fontweight="bold",
        color="#2C3E50",
        bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=1.0),
        zorder=20,
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Error concentration plot saved to: {output_path}")


def extract_slice_examples(df_test, test_res_df, output_dir, n_examples=5):
    """
    Extracts example images for the top 3 error-contributing slices.
    """
    # Calculate error mass to identify top slices
    test_res_df = test_res_df.copy()
    test_res_df["errors"] = (1 - test_res_df["accuracy"]) * test_res_df["size"]
    top_slices = test_res_df.sort_values("errors", ascending=False).head(3)

    print(
        f"\nExtracting examples for top {len(top_slices)} error-contributing slices..."
    )

    # We need to match samples in df_test to their slice assignments
    # In 5_run_analysis.py, slice_preds_test is calculated
    # We'll assume df_test has a 'domino_slice' column or we'll need it passed
    if "domino_slice" not in df_test.columns:
        # If not present, we can't extract without it.
        # We'll ensure 5_run_analysis.py adds it.
        print("Error: 'domino_slice' column not found in df_test.")
        return

    os.makedirs(output_dir, exist_ok=True)
    summary_lines = ["# Slice Example Extraction Summary\n"]

    for _, slice_row in top_slices.iterrows():
        slice_id = int(slice_row["slice"])
        slice_error_mass = slice_row["errors"]

        slice_dir = os.path.join(output_dir, f"slice_{slice_id}")
        os.makedirs(slice_dir, exist_ok=True)

        # Filter samples for this slice
        slice_samples = df_test[df_test["domino_slice"] == slice_id]

        # Prioritize errors (prediction != target)
        # We need predicted labels in df_test as well
        if "prediction" not in slice_samples.columns:
            # Fallback if no predictions, just sample
            selected = slice_samples.sample(min(n_examples, len(slice_samples)))
        else:
            errors = slice_samples[
                slice_samples["prediction"] != slice_samples["target"]
            ]
            correct = slice_samples[
                slice_samples["prediction"] == slice_samples["target"]
            ]

            # Take as many errors as possible, then fill with correct ones
            n_errors = min(n_examples, len(errors))
            n_correct = min(n_examples - n_errors, len(correct))

            selected = pd.concat([errors.sample(n_errors), correct.sample(n_correct)])

        print(
            f"  Slice {slice_id}: Selected {len(selected)} images (Expected Errors: {slice_error_mass:.1f})"
        )
        summary_lines.append(
            f"## Slice {slice_id}\n- Expected Errors: {slice_error_mass:.1f}\n- Hidden Artifact Rate: {slice_row['hidden_rate']:.1%}\n- Known Artifact Rate: {slice_row['known_rate']:.1%}\n- Accuracy: {slice_row['accuracy']:.1%}\n"
        )

        for i, (_, sample) in enumerate(selected.iterrows()):
            img_path = sample["image_path"]
            # Assume image_path is relative to project root or absolute
            # Based on 5_run_analysis.py, it seems to be just the filename or relative path
            # Need to verify if we need to prefix with data_root

            # Simple heuristic: if path doesn't exist, try common locations
            full_img_path = img_path
            if not os.path.exists(full_img_path):
                # Try relative to project root
                script_dir = os.path.dirname(os.path.abspath(__file__))
                project_root = os.path.abspath(os.path.join(script_dir, "../../../"))
                full_img_path = os.path.join(
                    project_root, "data/synthetic_imagenette", img_path
                )

            if os.path.exists(full_img_path):
                ext = os.path.splitext(full_img_path)[1]
                target_name = f"example_{i}{ext}"
                if "prediction" in sample:
                    is_correct = (
                        "correct"
                        if sample["prediction"] == sample["target"]
                        else "incorrect"
                    )
                    target_name = f"example_{i}_{is_correct}{ext}"

                shutil.copy2(full_img_path, os.path.join(slice_dir, target_name))
                summary_lines.append(
                    f"  - {target_name} (Source: {os.path.basename(full_img_path)})"
                )
            else:
                print(f"    Warning: Could not find image at {full_img_path}")

        summary_lines.append("\n")

    summary_path = os.path.join(output_dir, "extraction_summary.md")
    with open(summary_path, "w") as f:
        f.writelines("\n".join(summary_lines))
    print(f"Extraction summary saved to: {summary_path}")
