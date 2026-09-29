from pathlib import Path
import csv


ROOT = Path(__file__).resolve().parents[1]
CONSOLIDATED = ROOT / "final_inference" / "consolidated"
FIGURES = ROOT / "paper" / "figures"


def main():
    import matplotlib.pyplot as plt

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

    ordered = [
        "GPT-4o",
        "GPT-4.1",
        "GPT-5.6 Terra",
        "GPT-5.5",
        "Qwen3-VL-8B",
        "Qwen3-VL-32B",
        "InternVL3.5-8B",
        "InternVL3.5-38B",
        "MiniCPM-V 4.5",
        "Gemma 4 12B IT",
    ]
    values = {model: {} for model in ordered}
    with (CONSOLIDATED / "model_prompt_summary.csv").open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["model"] in values:
                values[row["model"]][row["prompt"]] = float(row["invalid_false_accept"])

    labels = [m for m in ordered if "natural" in values[m] and "debiased" in values[m]]
    x = range(len(labels))
    width = 0.38
    plt.figure(figsize=(9.2, 3.9))
    plt.bar([i - width / 2 for i in x], [values[m]["natural"] for m in labels], width, label="Natural")
    plt.bar([i + width / 2 for i in x], [values[m]["debiased"] for m in labels], width, label="De-biased")
    plt.ylabel("Invalid false-accept rate (%)")
    plt.ylim(0, 105)
    plt.xticks(list(x), labels, rotation=35, ha="right")
    plt.legend(frameon=False, ncol=2)
    plt.title("Prompting helps, but does not replace citation verification")
    plt.tight_layout()
    plt.savefig(FIGURES / "false_accept_prompt_comparison.pdf")
    plt.savefig(FIGURES / "false_accept_prompt_comparison.png", dpi=220)
    plt.close()


if __name__ == "__main__":
    main()
