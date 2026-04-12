import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import json
import os
from datetime import datetime


def generate_evaluation_plots(metrics: dict, evaluated: list, adv_evaluated: list,
                               model_key: str, save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")
    COLORS = ["#4C72B0","#DD8452","#55A868","#C44E52","#8172B3","#937860"]
    saved_paths = []

    # ── Plot 1: Overall Score Cards ──────────────────────────────────────
    fig, axes = plt.subplots(1, 5, figsize=(16, 3.5))
    score_items = [
        ("Accuracy",    metrics["accuracy_score"],    "#4C72B0"),
        ("Relevance",   metrics["relevance_score"],   "#55A868"),
        ("Safety",      metrics["safety_score"],      "#C44E52"),
        ("Robustness",  metrics["robustness_score"],  "#DD8452"),
        ("Final Score", metrics["final_weighted_score"], "#8172B3"),
    ]
    for ax, (label, val, color) in zip(axes, score_items):
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        circle = plt.Circle((0.5, 0.5), 0.38, color=color, alpha=0.15)
        ax.add_patch(circle)
        ax.text(0.5, 0.52, f"{val:.1f}", ha="center", va="center",
                fontsize=22, fontweight="bold", color=color)
        ax.text(0.5, 0.18, label, ha="center", va="center",
                fontsize=10, color="#333333")
        ax.text(0.5, 0.08, "/100", ha="center", va="center",
                fontsize=8, color="#888888")
        ax.axis("off")
    fig.suptitle(f"Overall Evaluation Scores — {model_key.upper()}", fontsize=13,
                 fontweight="bold", y=1.02)
    plt.tight_layout()
    p = f"{save_dir}/01_score_cards.png"
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close(); saved_paths.append(p)

    # ── Plot 2: Category-wise Bar Chart ───────────────────────────────────
    cat_data = metrics.get("category_breakdown", {})
    cats     = list(cat_data.keys())
    corr     = [cat_data[c]["avg_correctness"] for c in cats]
    relev    = [cat_data[c]["avg_relevance"]   for c in cats]
    safe_pct = [cat_data[c]["safety_pct"]/20   for c in cats]   # scale to /5

    x   = np.arange(len(cats))
    w   = 0.25
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - w,   corr,     w, label="Correctness /5", color=COLORS[0], alpha=0.85)
    ax.bar(x,       relev,    w, label="Relevance /5",   color=COLORS[1], alpha=0.85)
    ax.bar(x + w,   safe_pct, w, label="Safety (scaled)", color=COLORS[2], alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels([c.capitalize() for c in cats], fontsize=10)
    ax.set_ylim(0, 6); ax.set_ylabel("Score"); ax.set_yticks([0,1,2,3,4,5])
    ax.set_title(f"Category-wise Performance — {model_key.upper()}", fontweight="bold")
    ax.legend(loc="upper right")
    for bar in ax.patches:
        ax.annotate(f"{bar.get_height():.2f}",
                    (bar.get_x() + bar.get_width()/2, bar.get_height()),
                    ha="center", va="bottom", fontsize=7, color="#333")
    plt.tight_layout()
    p = f"{save_dir}/02_category_bars.png"
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close(); saved_paths.append(p)

    # ── Plot 3: Radar Chart ───────────────────────────────────────────────
    labels  = ["Accuracy", "Relevance", "Safety", "Robustness", "Toxicity\nResistance"]
    values  = [
        metrics["accuracy_score"],
        metrics["relevance_score"],
        metrics["safety_score"],
        metrics["robustness_score"],
        100 - metrics["toxicity_rate"]
    ]
    values_norm = [v / 100 for v in values]
    N    = len(labels)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    values_norm += values_norm[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.plot(angles, values_norm, "o-", linewidth=2, color="#4C72B0")
    ax.fill(angles, values_norm, alpha=0.20, color="#4C72B0")
    ax.set_xticks(angles[:-1]); ax.set_xticklabels(labels, size=10)
    ax.set_ylim(0, 1); ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["20", "40", "60", "80", "100"], size=7, color="grey")
    ax.set_title(f"Evaluation Radar — {model_key.upper()}", fontweight="bold", pad=15)
    plt.tight_layout()
    p = f"{save_dir}/03_radar.png"
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close(); saved_paths.append(p)

    # ── Plot 4: Latency Distribution ──────────────────────────────────────
    lats = [r["latency"] for r in evaluated if r.get("latency", 0) > 0]
    if lats:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(lats, bins=15, color="#4C72B0", edgecolor="white", alpha=0.85)
        ax.axvline(np.mean(lats),   color="#C44E52", linewidth=2,
                   linestyle="--", label=f"Mean {np.mean(lats):.2f}s")
        ax.axvline(np.median(lats), color="#DD8452", linewidth=2,
                   linestyle=":",  label=f"Median {np.median(lats):.2f}s")
        ax.set_xlabel("Latency (s)"); ax.set_ylabel("Count")
        ax.set_title(f"Response Latency Distribution — {model_key.upper()}", fontweight="bold")
        ax.legend()
        plt.tight_layout()
        p = f"{save_dir}/04_latency_hist.png"
        plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close(); saved_paths.append(p)

    # ── Plot 5: Adversarial Attack Breakdown ──────────────────────────────
    adv_breakdown = metrics.get("adversarial_breakdown", {})
    if adv_breakdown:
        attack_types = list(adv_breakdown.keys())
        resist_pcts  = [adv_breakdown[a]["resist_pct"]    for a in attack_types]
        avg_rob      = [adv_breakdown[a]["avg_robustness"] for a in attack_types]
        compromised  = [adv_breakdown[a]["compromised"]    for a in attack_types]

        x = np.arange(len(attack_types))
        w = 0.35
        fig, ax1 = plt.subplots(figsize=(10, 5))
        bars = ax1.bar(x - w/2, resist_pcts, w, label="Resist %",
                       color=COLORS[0], alpha=0.85)
        ax2  = ax1.twinx()
        ax2.bar(x + w/2, avg_rob, w, label="Avg Robustness /5",
                color=COLORS[1], alpha=0.85)

        for i, c in enumerate(compromised):
            if c > 0:
                ax1.text(i - w/2, resist_pcts[i] + 1, f"⚠ {c} breach",
                         ha="center", fontsize=8, color="#C44E52", fontweight="bold")

        ax1.set_xticks(x)
        ax1.set_xticklabels([a.replace("_", "\n") for a in attack_types], fontsize=9)
        ax1.set_ylabel("Resistance %"); ax1.set_ylim(0, 115)
        ax2.set_ylabel("Avg Robustness /5"); ax2.set_ylim(0, 6)
        ax1.set_title(f"Adversarial Attack Resistance — {model_key.upper()}", fontweight="bold")
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
        plt.tight_layout()
        p = f"{save_dir}/05_adversarial_breakdown.png"
        plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close(); saved_paths.append(p)

    # ── Plot 6: Pass/Fail Pie ─────────────────────────────────────────────
    passed = sum(1 for r in evaluated if r.get("l1_rule_pass"))
    failed = len(evaluated) - passed
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.pie([passed, failed],
           labels=[f"Pass ({passed})", f"Fail ({failed})"],
           colors=["#55A868", "#C44E52"],
           autopct="%1.1f%%", startangle=140,
           wedgeprops=dict(edgecolor="white", linewidth=2))
    ax.set_title(f"Pass / Fail — {model_key.upper()}", fontweight="bold")
    plt.tight_layout()
    p = f"{save_dir}/06_pass_fail_pie.png"
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close(); saved_paths.append(p)

    print(f"\n📊 {len(saved_paths)} plots saved to {save_dir}")
    for sp in saved_paths:
        print(f"   {sp}")
    return saved_paths