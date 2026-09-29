from pathlib import Path
import csv
import math


ROOT = Path(__file__).resolve().parents[1]
CONSOLIDATED = ROOT / "final_inference" / "consolidated"
FIGURES = ROOT / "paper" / "figures"


MODELS = [
    "GPT-5.5",
    "GPT-5.6 Terra",
    "Qwen3-VL-32B",
    "InternVL3.5-38B",
    "Qwen3-VL-8B",
    "InternVL3.5-8B",
    "Gemma 4 12B IT",
    "MiniCPM-V 4.5",
]

CONDITIONS = [
    ("plain", "Plain"),
    ("decorative", "Decorative"),
    ("islamic_authority", "Islamic\nauthority"),
    ("decorative_islamic_authority", "Decor. +\nIslamic"),
    ("neutral_authority", "Neutral\nauthority"),
    ("decorative_neutral_authority", "Decor. +\nNeutral"),
]


def load_values():
    data = {condition: {} for condition, _ in CONDITIONS}
    with (CONSOLIDATED / "condition_summary.csv").open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["prompt"] != "natural":
                continue
            if row["condition"] in data and row["model"] in MODELS:
                data[row["condition"]][row["model"]] = float(row["invalid_false_accept"])
    return data


def main():
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "axes.titlesize": 9,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    FIGURES.mkdir(parents=True, exist_ok=True)
    data = load_values()

    n = len(MODELS)
    angles = [2 * math.pi * i / n for i in range(n)]
    angles_closed = angles + [angles[0]]
    label_models = [
        "GPT-5.5",
        "GPT-5.6",
        "Qwen3-32B",
        "InternVL-38B",
        "Qwen3-8B",
        "InternVL-8B",
        "Gemma-12B",
        "MiniCPM",
    ]

    fig, axes = plt.subplots(
        2,
        3,
        subplot_kw={"polar": True},
        figsize=(9.0, 5.5),
    )
    axes = axes.ravel()

    for ax, (condition, title) in zip(axes, CONDITIONS):
        vals = [data[condition].get(model, float("nan")) for model in MODELS]
        vals_closed = vals + [vals[0]]
        ax.plot(angles_closed, vals_closed, color="#2a6fbb", linewidth=1.3)
        ax.fill(angles_closed, vals_closed, color="#2a6fbb", alpha=0.18)
        ax.set_title(title, y=1.12, fontweight="bold")
        ax.set_ylim(0, 100)
        ax.set_yticks([50, 100])
        ax.set_yticklabels(["50", "100"], color="gray")
        ax.set_xticks(angles)
        ax.set_xticklabels(label_models)
        ax.grid(color="#bfc7d5", alpha=0.7, linewidth=0.6)
        ax.spines["polar"].set_color("#9aa8b8")
        ax.spines["polar"].set_linewidth(0.7)
        for angle, val in zip(angles, vals):
            ax.scatter([angle], [val], s=9, color="#2a6fbb", zorder=3)

    fig.suptitle("False-accept profiles by visual condition", y=0.99, fontsize=11, fontweight="bold")
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.96])
    fig.savefig(FIGURES / "false_accept_by_condition_selected_natural.pdf")
    fig.savefig(FIGURES / "false_accept_by_condition_selected_natural.png", dpi=260)


if __name__ == "__main__":
    main()
