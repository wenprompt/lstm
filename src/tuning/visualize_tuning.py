#!/usr/bin/env python3
"""
Improved Hyperparameter Tuning Visualizations - Clear, Actionable Insights

This creates 4 simple, focused dashboards that give you immediate takeaways:
1. Best Model Recommendations
2. Hyperparameter Impact Analysis
3. Model Performance Comparison
4. Validation Set Issue Detection
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Paths
PLOTS_DIR = Path("results/plots/hypertuning")
RESULTS_CSV_PATH = Path("results/tuning_results.csv")

# Set style
plt.style.use("default")
sns.set_palette("husl")


def create_dashboard1_best_model_recommendations(results_df):
    """
    Dashboard 1: Best Model Recommendations - Clear, actionable hyperparameter recommendations.
    """
    try:
        logger.info("Creating Dashboard 1: Best Model Recommendations...")

        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(
            "Dashboard 1: Best Model Recommendations",
            fontsize=20,
            fontweight="bold",
            y=0.98,
        )

        # Get best model
        best_model = results_df.loc[results_df["composite_score"].idxmax()]

        # 1. Top-Left: Best Model Configuration
        ax1 = axes[0, 0]
        ax1.axis("off")

        config_text = f"""
🏆 RECOMMENDED MODEL CONFIGURATION

Learning Rate: {best_model["learning_rate"]:.4f}
Hidden Size: {int(best_model["hidden_size"])}
Dropout Rate: {best_model["dropout_rate"]:.2f}
Number of Layers: {int(best_model["number_of_layers"])}
Gradient Clipping: {best_model["gradient_clip_norm"]:.1f}

PERFORMANCE:
• Directional Accuracy: {best_model["val_directional_accuracy"]:.2f}%
• RMSE: {best_model["val_rmse"]:.4f}
• R²: {best_model["val_r_squared"]:.4f}
• Composite Score: {best_model["composite_score"]:.4f}
        """

        ax1.text(
            0.05,
            0.95,
            config_text,
            transform=ax1.transAxes,
            fontsize=12,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.8),
        )

        # 2. Top-Right: Hyperparameter Impact Rankings
        ax2 = axes[0, 1]

        # Calculate impact of each hyperparameter
        param_impacts = {}
        for param in [
            "learning_rate",
            "hidden_size",
            "dropout_rate",
            "number_of_layers",
        ]:
            grouped = (
                results_df.groupby(param)["composite_score"]
                .agg(["mean", "std"])
                .reset_index()
            )
            impact = grouped["mean"].max() - grouped["mean"].min()
            param_impacts[param] = impact

        # Plot impact
        impacts = list(param_impacts.values())
        param_labels = [
            "Learning\nRate",
            "Hidden\nSize",
            "Dropout\nRate",
            "Number of\nLayers",
        ]

        bars = ax2.bar(
            param_labels, impacts, color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
        )
        ax2.set_title(
            "Hyperparameter Impact on Performance", fontweight="bold", fontsize=14
        )
        ax2.set_ylabel("Impact on Composite Score")
        ax2.grid(axis="y", alpha=0.3)

        # Add value labels on bars
        for bar, impact in zip(bars, impacts):
            height = bar.get_height()
            ax2.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.001,
                f"{impact:.3f}",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        # 3. Bottom-Left: Learning Rate Comparison
        ax3 = axes[1, 0]
        lr_analysis = (
            results_df.groupby("learning_rate")
            .agg({"composite_score": ["mean", "std"], "val_rmse": "mean"})
            .round(4)
        )

        learning_rates = lr_analysis.index
        scores = lr_analysis[("composite_score", "mean")]
        errors = lr_analysis[("composite_score", "std")]

        bars = ax3.bar(
            [f"{lr:.4f}" for lr in learning_rates],
            scores,
            yerr=errors,
            capsize=5,
            color=[
                "green" if lr == best_model["learning_rate"] else "skyblue"
                for lr in learning_rates
            ],
        )
        ax3.set_title("Learning Rate Performance", fontweight="bold", fontsize=14)
        ax3.set_xlabel("Learning Rate")
        ax3.set_ylabel("Average Composite Score")
        ax3.grid(axis="y", alpha=0.3)

        # Highlight best
        for i, (lr, score) in enumerate(zip(learning_rates, scores)):
            color = "red" if lr == best_model["learning_rate"] else "black"
            weight = "bold" if lr == best_model["learning_rate"] else "normal"
            ax3.text(
                i,
                score + 0.01,
                f"{score:.3f}",
                ha="center",
                va="bottom",
                color=color,
                fontweight=weight,
            )

        # 4. Bottom-Right: Hidden Size Comparison
        ax4 = axes[1, 1]
        hs_analysis = (
            results_df.groupby("hidden_size")
            .agg({"composite_score": ["mean", "std"], "val_rmse": "mean"})
            .round(4)
        )

        hidden_sizes = hs_analysis.index
        scores = hs_analysis[("composite_score", "mean")]
        errors = hs_analysis[("composite_score", "std")]

        bars = ax4.bar(
            [f"{int(hs)}" for hs in hidden_sizes],
            scores,
            yerr=errors,
            capsize=5,
            color=[
                "green" if hs == best_model["hidden_size"] else "coral"
                for hs in hidden_sizes
            ],
        )
        ax4.set_title("Hidden Size Performance", fontweight="bold", fontsize=14)
        ax4.set_xlabel("Hidden Size")
        ax4.set_ylabel("Average Composite Score")
        ax4.grid(axis="y", alpha=0.3)

        # Highlight best
        for i, (hs, score) in enumerate(zip(hidden_sizes, scores)):
            color = "red" if hs == best_model["hidden_size"] else "black"
            weight = "bold" if hs == best_model["hidden_size"] else "normal"
            ax4.text(
                i,
                score + 0.01,
                f"{score:.3f}",
                ha="center",
                va="bottom",
                color=color,
                fontweight=weight,
            )

        plt.tight_layout()
        plt.subplots_adjust(top=0.93)

        # Save
        output_path = PLOTS_DIR / "dashboard1_best_model_recommendations.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"  ✅ Dashboard 1 saved to {output_path}")

    except Exception as e:
        logger.error(f"  ❌ Could not generate Dashboard 1: {e}")


def create_dashboard2_performance_rankings(results_df):
    """
    Dashboard 2: Performance Rankings - Top models with clear performance metrics.
    """
    try:
        logger.info("Creating Dashboard 2: Performance Rankings...")

        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        fig.suptitle(
            "Dashboard 2: Performance Rankings & Trade-offs",
            fontsize=20,
            fontweight="bold",
            y=0.98,
        )

        # Get top 10 models
        top10 = results_df.nlargest(10, "composite_score").reset_index(drop=True)

        # Create model labels
        model_labels = []
        for _, row in top10.iterrows():
            label = f"LR{row['learning_rate']:.4f}_H{int(row['hidden_size'])}_D{row['dropout_rate']:.1f}"
            model_labels.append(label)

        # 1. Top-Left: Top 10 Models Ranking
        ax1 = axes[0, 0]
        y_pos = np.arange(len(top10))

        # Create gradient colors from green to red
        colors = plt.cm.get_cmap('RdYlGn_r')(np.linspace(0.3, 0.9, len(top10)))
        bars = ax1.barh(
            y_pos,
            top10["composite_score"],
            color=colors,
        )
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(model_labels, fontsize=8)
        ax1.set_xlabel("Composite Score")
        ax1.set_title(
            "Top 10 Models (Ranked by Composite Score)", fontweight="bold", fontsize=14
        )
        ax1.grid(axis="x", alpha=0.3)

        # Add score labels
        for i, (bar, score) in enumerate(zip(bars, top10["composite_score"])):
            ax1.text(
                bar.get_width() + 0.005,
                bar.get_y() + bar.get_height() / 2,
                f"{score:.3f}",
                ha="left",
                va="center",
                fontweight="bold",
                fontsize=8,
            )

        # 2. Top-Right: RMSE vs Directional Accuracy Trade-off
        ax2 = axes[0, 1]

        scatter = ax2.scatter(
            top10["val_rmse"],
            top10["val_directional_accuracy"],
            c=top10["composite_score"],
            s=100,
            cmap="RdYlGn_r",
            alpha=0.7,
            edgecolors="black",
        )

        # Highlight best model
        best_idx = 0
        ax2.scatter(
            top10.iloc[best_idx]["val_rmse"],
            top10.iloc[best_idx]["val_directional_accuracy"],
            s=200,
            c="red",
            marker="*",
            edgecolors="black",
            linewidth=2,
            label="Best Model",
        )

        ax2.set_xlabel("Validation RMSE (Lower is Better)")
        ax2.set_ylabel("Directional Accuracy %")
        ax2.set_title("Accuracy vs Error Trade-off", fontweight="bold", fontsize=14)
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax2, shrink=0.8)
        cbar.set_label("Composite Score")

        # 3. Bottom-Left: Performance Metrics Comparison
        ax3 = axes[1, 0]

        # Show top 5 models with key metrics
        top5 = top10.head(5)
        metrics = ["val_directional_accuracy", "val_rmse", "val_r_squared"]
        metric_names = ["Dir. Accuracy (%)", "RMSE", "R²"]

        x = np.arange(len(top5))
        width = 0.25

        for i, (metric, name) in enumerate(zip(metrics, metric_names)):
            if metric == "val_rmse":
                # Invert RMSE for better visualization (lower is better)
                values = 1 / (top5[metric] + 0.001)  # Avoid division by zero
            elif metric == "val_directional_accuracy":
                values = top5[metric] / 100  # Convert to 0-1 scale
            else:
                values = (top5[metric] - top5[metric].min()) / (
                    top5[metric].max() - top5[metric].min() + 1e-8
                )

            ax3.bar(x + i * width, values, width, label=name, alpha=0.8)

        ax3.set_xlabel("Top 5 Models")
        ax3.set_ylabel("Normalized Performance (Higher is Better)")
        ax3.set_title(
            "Top 5 Models - Performance Metrics", fontweight="bold", fontsize=14
        )
        ax3.set_xticks(x + width)
        ax3.set_xticklabels([f"#{i + 1}" for i in range(len(top5))])
        ax3.legend()
        ax3.grid(axis="y", alpha=0.3)

        # 4. Bottom-Right: Validation Issue Warning
        ax4 = axes[1, 1]
        ax4.axis("off")

        # Check if directional accuracy is identical for all models
        unique_accuracies = results_df["val_directional_accuracy"].nunique()

        if unique_accuracies == 1:
            warning_text = f"""
⚠️  VALIDATION SET ISSUE DETECTED

All {len(results_df)} models have identical 
directional accuracy: {results_df["val_directional_accuracy"].iloc[0]:.2f}%

This indicates:
• Validation set too small (5% split)
• Models converging to same solution
• Cannot properly discriminate performance

SOLUTION:
1. Increase validation split to 15% in config.yaml
2. Re-run hyperparameter tuning
3. You'll see varied performance metrics
4. Better model discrimination

Current validation samples: ~42
Recommended: ~128 samples (3x larger)
            """
            ax4.text(
                0.05,
                0.95,
                warning_text,
                transform=ax4.transAxes,
                fontsize=11,
                verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="orange", alpha=0.8),
            )
        else:
            success_text = f"""
✅ VALIDATION SET WORKING PROPERLY

Found {unique_accuracies} different directional 
accuracy values across models.

Performance range:
• Min: {results_df["val_directional_accuracy"].min():.2f}%
• Max: {results_df["val_directional_accuracy"].max():.2f}%
• Range: {results_df["val_directional_accuracy"].max() - results_df["val_directional_accuracy"].min():.2f}%

This indicates good model discrimination!
            """
            ax4.text(
                0.05,
                0.95,
                success_text,
                transform=ax4.transAxes,
                fontsize=11,
                verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.8),
            )

        plt.tight_layout()
        plt.subplots_adjust(top=0.93)

        # Save
        output_path = PLOTS_DIR / "dashboard2_performance_rankings.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"  ✅ Dashboard 2 saved to {output_path}")

    except Exception as e:
        logger.error(f"  ❌ Could not generate Dashboard 2: {e}")


def create_dashboard3_model_comparison_table(results_df):
    """
    Dashboard 3: Model Comparison Table - Clear tabular comparison of top models.
    """
    try:
        logger.info("Creating Dashboard 3: Model Comparison Table...")

        # Create figure
        fig, axes = plt.subplots(
            2, 1, figsize=(20, 12), gridspec_kw={"height_ratios": [3, 1]}
        )
        fig.suptitle(
            "Dashboard 3: Detailed Model Comparison",
            fontsize=20,
            fontweight="bold",
            y=0.98,
        )

        # Get top 15 models for detailed comparison
        top15 = results_df.nlargest(15, "composite_score").reset_index(drop=True)

        # 1. Top: Detailed Comparison Table
        ax1 = axes[0]
        ax1.axis("off")

        # Prepare table data
        table_data = []
        for i, (_, row) in enumerate(top15.iterrows()):
            table_data.append(
                [
                    f"#{i + 1}",
                    f"{row['learning_rate']:.4f}",
                    f"{int(row['hidden_size'])}",
                    f"{row['dropout_rate']:.2f}",
                    f"{int(row['number_of_layers'])}",
                    f"{row['gradient_clip_norm']:.1f}",
                    f"{row['val_directional_accuracy']:.2f}%",
                    f"{row['val_rmse']:.4f}",
                    f"{row['val_r_squared']:.4f}",
                    f"{row['composite_score']:.4f}",
                ]
            )

        # Create table
        table = ax1.table(
            cellText=table_data,
            colLabels=[
                "Rank",
                "Learn Rate",
                "Hidden",
                "Dropout",
                "Layers",
                "Grad Clip",
                "Dir. Acc %",
                "RMSE",
                "R²",
                "Composite",
            ],
            cellLoc="center",
            loc="center",
            colWidths=[0.06, 0.1, 0.08, 0.08, 0.08, 0.09, 0.1, 0.09, 0.09, 0.1],
        )

        # Style table
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.5)

        # Color code rows - top 5 in different colors
        colors = ["#d4edda", "#fff3cd", "#f8d7da", "#e2e3e5", "#f8f9fa"]
        for i in range(min(5, len(table_data))):
            for j in range(len(table_data[0])):
                table[(i + 1, j)].set_facecolor(colors[i] if i < 5 else "#ffffff")

        # Header styling
        for j in range(len(table_data[0])):
            table[(0, j)].set_facecolor("#6c757d")
            table[(0, j)].set_text_props(weight="bold", color="white")

        ax1.set_title(
            "Top 15 Models - Detailed Comparison",
            fontweight="bold",
            fontsize=16,
            pad=20,
        )

        # 2. Bottom: Key Insights
        ax2 = axes[1]
        ax2.axis("off")

        # Calculate insights
        best_model = top15.iloc[0]

        # Learning rate patterns
        lr_counts = top15["learning_rate"].value_counts()
        most_common_lr = lr_counts.index[0]

        # Hidden size patterns
        hs_counts = top15["hidden_size"].value_counts()
        most_common_hs = hs_counts.index[0]

        # Dropout patterns
        dr_counts = top15["dropout_rate"].value_counts()
        most_common_dr = dr_counts.index[0]

        insights_text = f"""
📊 KEY INSIGHTS FROM TOP 15 MODELS:

🏆 BEST MODEL: Learning Rate {best_model["learning_rate"]:.4f}, Hidden Size {int(best_model["hidden_size"])}, Dropout {best_model["dropout_rate"]:.2f}

📈 PATTERNS IN TOP PERFORMERS:
• Most common Learning Rate: {most_common_lr:.4f} (appears {lr_counts.iloc[0]} times)
• Most common Hidden Size: {int(most_common_hs)} (appears {hs_counts.iloc[0]} times)  
• Most common Dropout: {most_common_dr:.2f} (appears {dr_counts.iloc[0]} times)
• Most use {int(top15["number_of_layers"].mode()[0])} layer(s)

💡 RECOMMENDATION: The patterns above represent the most successful hyperparameter combinations.
        """

        ax2.text(
            0.05,
            0.95,
            insights_text,
            transform=ax2.transAxes,
            fontsize=12,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8),
        )

        plt.tight_layout()
        plt.subplots_adjust(top=0.93)

        # Save
        output_path = PLOTS_DIR / "dashboard3_model_comparison_table.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"  ✅ Dashboard 3 saved to {output_path}")

    except Exception as e:
        logger.error(f"  ❌ Could not generate Dashboard 3: {e}")


def create_dashboard4_final_recommendations(results_df):
    """
    Dashboard 4: Final Recommendations - Clear action items and next steps.
    """
    try:
        logger.info("Creating Dashboard 4: Final Recommendations...")

        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(
            "Dashboard 4: Final Recommendations & Next Steps",
            fontsize=20,
            fontweight="bold",
            y=0.98,
        )

        # Get best model
        best_model = results_df.loc[results_df["composite_score"].idxmax()]

        # 1. Top-Left: Config.yaml Update Instructions
        ax1 = axes[0, 0]
        ax1.axis("off")

        config_update = f"""
📝 UPDATE YOUR config.yaml WITH:

model:
  sequence_length: 10  # Keep existing
  hidden_size: {int(best_model["hidden_size"])}
  number_of_layers: {int(best_model["number_of_layers"])}
  dropout_rate: {best_model["dropout_rate"]:.2f}
  bidirectional: true  # Keep existing
  layer_norm: true     # Keep existing

training:
  batch_size: 32       # Keep existing
  learning_rate: {best_model["learning_rate"]:.4f}
  epochs: 70           # Keep existing
  gradient_clip_norm: {best_model["gradient_clip_norm"]:.1f}

splits:
  train_ratio: 0.70    # Updated
  val_ratio: 0.15      # Updated (was 0.05)
  test_ratio: 0.15     # Updated
        """

        ax1.text(
            0.05,
            0.95,
            config_update,
            transform=ax1.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8),
            fontfamily="monospace",
        )

        # 2. Top-Right: Performance Expectations
        ax2 = axes[0, 1]
        ax2.axis("off")

        performance_text = f"""
🎯 EXPECTED PERFORMANCE WITH BEST MODEL:

Training Results:
• Validation RMSE: {best_model["val_rmse"]:.4f}
• Validation R²: {best_model["val_r_squared"]:.4f}
• Directional Accuracy: {best_model["val_directional_accuracy"]:.2f}%

What This Means:
• Average prediction error ~{best_model["val_rmse"] * 100:.1f}% 
• Model explains {best_model["val_r_squared"] * 100:.1f}% of price variance
• Predicts direction correctly {best_model["val_directional_accuracy"]:.0f}% of time

Quality Assessment:
• RMSE: {"LOW" if best_model["val_rmse"] < 1.5 else "MODERATE" if best_model["val_rmse"] < 2.0 else "HIGH"} error rate
• Direction: {"EXCELLENT" if best_model["val_directional_accuracy"] >= 60 else "GOOD" if best_model["val_directional_accuracy"] >= 55 else "FAIR"} forecasting
        """

        ax2.text(
            0.05,
            0.95,
            performance_text,
            transform=ax2.transAxes,
            fontsize=11,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.8),
        )

        # 3. Bottom-Left: Action Items Checklist
        ax3 = axes[1, 0]
        ax3.axis("off")

        checklist_text = f"""
✅ IMMEDIATE ACTION ITEMS:

□ 1. Update config.yaml with recommended values (see left)
□ 2. Re-run main training: `uv run python main.py`
□ 3. Compare results with previous runs
□ 4. Save best_model.pt for production use

□ 5. OPTIONAL - Further Optimization:
    • Try learning rates around {best_model["learning_rate"]:.4f} (±20%)
    • Test hidden sizes {int(best_model["hidden_size"] * 0.8)}-{int(best_model["hidden_size"] * 1.2)}
    • Experiment with dropout {best_model["dropout_rate"] - 0.1:.1f}-{best_model["dropout_rate"] + 0.1:.1f}

□ 6. Validation:
    • Run on test set to confirm performance
    • Check for overfitting signs
    • Monitor directional accuracy closely
        """

        ax3.text(
            0.05,
            0.95,
            checklist_text,
            transform=ax3.transAxes,
            fontsize=11,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightcyan", alpha=0.8),
        )

        # 4. Bottom-Right: Success Criteria
        ax4 = axes[1, 1]
        ax4.axis("off")

        success_text = f"""
🏆 SUCCESS CRITERIA FOR FINAL MODEL:

Target Metrics:
• Directional Accuracy: ≥ 55% (Currently: {best_model["val_directional_accuracy"]:.1f}%)
• RMSE: ≤ 2.0% (Currently: {best_model["val_rmse"]:.3f})
• R²: ≥ 0.0 (Currently: {best_model["val_r_squared"]:.3f})

Status:
• Direction: {"✅ PASS" if best_model["val_directional_accuracy"] >= 55 else "❌ NEEDS WORK"}
• Error: {"✅ PASS" if best_model["val_rmse"] <= 2.0 else "❌ NEEDS WORK"}
• Fit: {"✅ PASS" if best_model["val_r_squared"] >= 0 else "❌ NEEDS WORK"}

Overall: {"✅ READY FOR PRODUCTION" if all([best_model["val_directional_accuracy"] >= 55, best_model["val_rmse"] <= 2.0, best_model["val_r_squared"] >= 0]) else "⚠️  NEEDS IMPROVEMENT"}

Next Steps After Success:
• Deploy for iron ore price forecasting
• Monitor live performance
• Retrain monthly with new data
        """

        ax4.text(
            0.05,
            0.95,
            success_text,
            transform=ax4.transAxes,
            fontsize=11,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lavender", alpha=0.8),
        )

        plt.tight_layout()
        plt.subplots_adjust(top=0.93)

        # Save
        output_path = PLOTS_DIR / "dashboard4_final_recommendations.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"  ✅ Dashboard 4 saved to {output_path}")

    except Exception as e:
        logger.error(f"  ❌ Could not generate Dashboard 4: {e}")


def main():
    """
    Main function to create all improved visualization dashboards.
    """
    logger.info("🚀 Starting IMPROVED hyperparameter tuning visualization script...")

    # Ensure plots directory exists
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load the tuning results
    if not RESULTS_CSV_PATH.exists():
        logger.error(f"Tuning results file not found: {RESULTS_CSV_PATH}")
        logger.error(
            "Please run hyperparameter tuning first with: uv run python -m src.tuning.tune_hyperparameters"
        )
        return

    try:
        results_df = pd.read_csv(RESULTS_CSV_PATH)
        if results_df.empty:
            logger.warning("Tuning results file is empty. No visualizations to create.")
            return
    except Exception as e:
        logger.error(f"Failed to read or parse {RESULTS_CSV_PATH}: {e}")
        return

    logger.info(f"Loaded {len(results_df)} trial results from {RESULTS_CSV_PATH}")

    # Generate 4 IMPROVED, actionable dashboards
    logger.info("Creating 4 improved, actionable dashboards...")

    create_dashboard1_best_model_recommendations(results_df)
    create_dashboard2_performance_rankings(results_df)
    create_dashboard3_model_comparison_table(results_df)
    create_dashboard4_final_recommendations(results_df)

    logger.info(
        "🎉 IMPROVED visualization script finished! Generated 4 actionable dashboards in results/plots/hypertuning/"
    )
    logger.info("")
    logger.info("📋 NEXT STEPS:")
    logger.info("1. Review Dashboard 4 for config.yaml updates")
    logger.info("2. Update your config.yaml with recommended hyperparameters")
    logger.info("3. Re-run training: uv run python main.py")


if __name__ == "__main__":
    main()
