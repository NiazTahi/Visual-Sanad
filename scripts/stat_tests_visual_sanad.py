import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path


def read_rows(path):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def binom_two_sided(k, n, p=0.5):
    if n == 0:
        return 1.0
    probs = [math.comb(n, i) * (p**i) * ((1 - p) ** (n - i)) for i in range(n + 1)]
    obs = probs[k]
    return min(1.0, sum(x for x in probs if x <= obs + 1e-15))


def false_accept(row):
    return row.get("gold_label") == "INVALID" and row.get("prediction") == "VALID"


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", default="final_inference/consolidated/model_prompt_summary.csv")
    parser.add_argument("--out-dir", default="final_inference/consolidated")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    runs = read_rows(args.summary)
    rowsets = {}
    for run in runs:
        path = run.get("source_file")
        if not path or not Path(path).exists() or Path(path).stat().st_size == 0:
            continue
        rowsets[(run["model"], run["prompt"])] = read_rows(path)

    prompt_rows = []
    by_model = defaultdict(dict)
    for (model, prompt), rows in rowsets.items():
        by_model[model][prompt] = {r["trial_id"]: r for r in rows if r.get("gold_label") == "INVALID"}

    for model, prompts in sorted(by_model.items()):
        if "natural" not in prompts or "debiased" not in prompts:
            continue
        natural = prompts["natural"]
        debiased = prompts["debiased"]
        ids = sorted(set(natural) & set(debiased))
        nat_only = deb_only = both = neither = 0
        for tid in ids:
            nfa = false_accept(natural[tid])
            dfa = false_accept(debiased[tid])
            if nfa and dfa:
                both += 1
            elif nfa and not dfa:
                nat_only += 1
            elif dfa and not nfa:
                deb_only += 1
            else:
                neither += 1
        prompt_rows.append(
            {
                "model": model,
                "invalid_pairs": len(ids),
                "natural_only_false_accept": nat_only,
                "debiased_only_false_accept": deb_only,
                "both_false_accept": both,
                "neither_false_accept": neither,
                "net_debiased_rescue": nat_only - deb_only,
                "mcnemar_p": f"{binom_two_sided(min(nat_only, deb_only), nat_only + deb_only):.6g}",
            }
        )

    condition_rows = []
    comparisons = [
        ("plain", "decorative"),
        ("plain", "islamic_authority"),
        ("plain", "decorative_islamic_authority"),
        ("plain", "neutral_authority"),
        ("plain", "decorative_neutral_authority"),
    ]
    for (model, prompt), rows in sorted(rowsets.items()):
        invalid = [r for r in rows if r.get("gold_label") == "INVALID"]
        by_base = defaultdict(dict)
        for r in invalid:
            key = (r.get("verse_id"), r.get("foil_type"))
            by_base[key][r.get("condition")] = r
        for base, comp in comparisons:
            induced = rescued = both = neither = 0
            for conds in by_base.values():
                if base not in conds or comp not in conds:
                    continue
                bfa = false_accept(conds[base])
                cfa = false_accept(conds[comp])
                if bfa and cfa:
                    both += 1
                elif (not bfa) and cfa:
                    induced += 1
                elif bfa and (not cfa):
                    rescued += 1
                else:
                    neither += 1
            condition_rows.append(
                {
                    "model": model,
                    "prompt": prompt,
                    "comparison": f"{base}_vs_{comp}",
                    "pairs": induced + rescued + both + neither,
                    "induced": induced,
                    "rescued": rescued,
                    "both_false_accept": both,
                    "neither_false_accept": neither,
                    "net_induced": induced - rescued,
                    "mcnemar_p": f"{binom_two_sided(min(induced, rescued), induced + rescued):.6g}",
                }
            )

    write_csv(
        out_dir / "prompt_effect_mcnemar.csv",
        prompt_rows,
        [
            "model",
            "invalid_pairs",
            "natural_only_false_accept",
            "debiased_only_false_accept",
            "both_false_accept",
            "neither_false_accept",
            "net_debiased_rescue",
            "mcnemar_p",
        ],
    )
    write_csv(
        out_dir / "condition_effect_mcnemar.csv",
        condition_rows,
        [
            "model",
            "prompt",
            "comparison",
            "pairs",
            "induced",
            "rescued",
            "both_false_accept",
            "neither_false_accept",
            "net_induced",
            "mcnemar_p",
        ],
    )

    md = ["# Paired statistical tests\n"]
    md.append("## Natural vs debiased prompt on INVALID false-accepts\n")
    md.append("| model | invalid pairs | natural-only FA | debiased-only FA | net debias rescue | McNemar p |")
    md.append("|---|---:|---:|---:|---:|---:|")
    for r in prompt_rows:
        md.append(f"| {r['model']} | {r['invalid_pairs']} | {r['natural_only_false_accept']} | {r['debiased_only_false_accept']} | {r['net_debiased_rescue']} | {r['mcnemar_p']} |")
    md.append("")
    md.append("## Plain vs visual cue conditions on INVALID false-accepts\n")
    md.append("See `condition_effect_mcnemar.csv` for the full table.")
    (out_dir / "statistical_tests.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(out_dir / "statistical_tests.md")


if __name__ == "__main__":
    main()
