from pathlib import Path
import csv


ROOT = Path(__file__).resolve().parents[1]
CONSOLIDATED = ROOT / "final_inference" / "consolidated"
FIGURES = ROOT / "paper" / "figures"


ORDERED_MODELS = [
    "GPT-5.5",
    "GPT-5.6 Terra",
    "GPT-4o",
    "GPT-4.1",
    "Qwen3-VL-32B",
    "InternVL3.5-38B",
    "Qwen3-VL-8B",
    "InternVL3.5-8B",
    "Gemma 4 12B IT",
    "MiniCPM-V 4.5",
]


def family_color(model):
    if model.startswith("GPT"):
        return "#2f6fb0"
    if model.startswith("Qwen"):
        return "#d9822b"
    if model.startswith("InternVL"):
        return "#7b5ea7"
    return "#7a7a7a"


def load_values():
    values = {model: {} for model in ORDERED_MODELS}
    with (CONSOLIDATED / "model_prompt_summary.csv").open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            model = row["model"]
            if model in values:
                values[model][row["prompt"]] = float(row["invalid_false_accept"])
    return {m: values[m] for m in ORDERED_MODELS if "natural" in values[m] and "debiased" in values[m]}


def main():
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    FIGURES.mkdir(parents=True, exist_ok=True)
    values = load_values()
    models = sorted(values, key=lambda m: values[m]["natural"])
    y = list(range(len(models)))
    natural = [values[m]["natural"] for m in models]
    debiased = [values[m]["debiased"] for m in models]

    fig, (ax1, ax2) = plt.subplots(
        ncols=2,
        figsize=(10.0, 4.0),
        gridspec_kw={"width_ratios": [1.05, 1.0], "wspace": 0.28},
        constrained_layout=True,
    )

    bars = ax1.barh(y, natural, color=[family_color(m) for m in models], edgecolor="white", linewidth=0.7)
    ax1.set_yticks(y)
    ax1.set_yticklabels(models, fontsize=8)
    ax1.set_xlim(0, 112)
    ax1.set_xlabel("False accept (%)", fontsize=9)
    ax1.set_title("(a) Natural prompt", fontsize=10, weight="bold")
    ax1.grid(axis="x", linestyle="-", alpha=0.16, linewidth=0.7)
    ax1.set_axisbelow(True)
    ax1.tick_params(axis="y", length=0)
    for bar, val in zip(bars, natural):
        ax1.text(val + 1.2, bar.get_y() + bar.get_height() / 2, f"{val:.1f}", va="center", fontsize=7.5)

    for yi, nval, dval in zip(y, natural, debiased):
        ax2.plot([nval, dval], [yi, yi], color="#aab2bd", linewidth=1.5, zorder=1)
    ax2.scatter(natural, y, color="#d65f3a", s=26, label="Natural", zorder=2)
    ax2.scatter(debiased, y, color="#2a9d8f", marker="D", s=28, label="De-biased", zorder=3)
    ax2.set_yticks(y)
    ax2.set_yticklabels([])
    ax2.set_xlim(0, 105)
    ax2.set_xlabel("False accept (%)", fontsize=9)
    ax2.set_title("(b) Prompt sensitivity", fontsize=10, weight="bold")
    ax2.grid(axis="x", linestyle="-", alpha=0.16, linewidth=0.7)
    ax2.axvline(50, color="#808080", linestyle=":", linewidth=1.0, alpha=0.65)
    ax2.text(51, len(models) - 0.35, "50%", fontsize=7, color="#606060", va="top")
    ax2.legend(frameon=False, fontsize=8, loc="lower right")
    ax2.tick_params(axis="y", length=0)

    for ax in (ax1, ax2):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)

    fig.savefig(FIGURES / "main_false_accept_two_panel.pdf")
    fig.savefig(FIGURES / "main_false_accept_two_panel.png", dpi=260)


if __name__ == "__main__":
    main()
