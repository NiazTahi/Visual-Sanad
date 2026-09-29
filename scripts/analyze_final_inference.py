import argparse
import csv
import math
from collections import Counter, defaultdict
from pathlib import Path


CONDITIONS = [
    "plain",
    "decorative",
    "islamic_authority",
    "decorative_islamic_authority",
    "neutral_authority",
    "decorative_neutral_authority",
]


def read_rows(path):
    with open(path, "r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def pct(value):
    return 100.0 * value


def rate(rows, predicate):
    return sum(1 for row in rows if predicate(row)) / len(rows) if rows else 0.0


def binom_two_sided_p(k, n, p=0.5):
    if n == 0:
        return 1.0
    probs = [math.comb(n, i) * (p**i) * ((1 - p) ** (n - i)) for i in range(n + 1)]
    observed = probs[k]
    return min(1.0, sum(prob for prob in probs if prob <= observed + 1e-15))


def write_csv(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize_condition(rows):
    out = []
    for condition in CONDITIONS:
        group = [row for row in rows if row["condition"] == condition]
        invalid = [row for row in group if row["gold_label"] == "INVALID"]
        valid = [row for row in group if row["gold_label"] == "VALID"]
        out.append(
            {
                "condition": condition,
                "n": len(group),
                "accuracy": f"{pct(rate(group, lambda r: r['prediction'] == r['gold_label'])):.2f}",
                "invalid_n": len(invalid),
                "false_accept_rate": f"{pct(rate(invalid, lambda r: r['prediction'] == 'VALID')):.2f}",
                "valid_n": len(valid),
                "false_reject_rate": f"{pct(rate(valid, lambda r: r['prediction'] == 'INVALID')):.2f}",
                "prediction_counts": dict(Counter(row["prediction"] for row in group)),
            }
        )
    return out


def summarize_foil_condition(rows):
    out = []
    for foil_type in sorted({row["foil_type"] for row in rows if row["gold_label"] == "INVALID"}):
        for condition in CONDITIONS:
            group = [
                row for row in rows
                if row["gold_label"] == "INVALID"
                and row["foil_type"] == foil_type
                and row["condition"] == condition
            ]
            out.append(
                {
                    "foil_type": foil_type,
                    "condition": condition,
                    "n": len(group),
                    "false_accept_rate": f"{pct(rate(group, lambda r: r['prediction'] == 'VALID')):.2f}",
                    "false_accepts": sum(1 for row in group if row["prediction"] == "VALID"),
                }
            )
    return out


def paired_flips(rows):
    invalid = [row for row in rows if row["gold_label"] == "INVALID"]
    by_item = defaultdict(dict)
    for row in invalid:
        key = (row["verse_id"], row["factual_state"], row["foil_type"])
        by_item[key][row["condition"]] = row

    out = []
    for condition in CONDITIONS:
        if condition == "plain":
            continue
        pairs = [item for item in by_item.values() if "plain" in item and condition in item]
        plain_reject = [item for item in pairs if item["plain"]["prediction"] == "INVALID"]
        induced = [item for item in plain_reject if item[condition]["prediction"] == "VALID"]
        rescued = [
            item for item in pairs
            if item["plain"]["prediction"] == "VALID"
            and item[condition]["prediction"] == "INVALID"
        ]
        discordant = len(induced) + len(rescued)
        out.append(
            {
                "condition_vs_plain": condition,
                "paired_invalid_items": len(pairs),
                "plain_correct_invalid": len(plain_reject),
                "induced_false_accepts": len(induced),
                "rescued_plain_false_accepts": len(rescued),
                "net_false_accept_change": len(induced) - len(rescued),
                "mcnemar_exact_p": f"{binom_two_sided_p(min(len(induced), len(rescued)), discordant):.4g}",
            }
        )
    return out


def cue_families(rows):
    invalid = [row for row in rows if row["gold_label"] == "INVALID"]
    families = {
        "no_authority": {"plain", "decorative"},
        "islamic_authority": {"islamic_authority", "decorative_islamic_authority"},
        "neutral_authority": {"neutral_authority", "decorative_neutral_authority"},
        "not_decorative": {"plain", "islamic_authority", "neutral_authority"},
        "decorative": {"decorative", "decorative_islamic_authority", "decorative_neutral_authority"},
    }
    out = []
    for name, conditions in families.items():
        group = [row for row in invalid if row["condition"] in conditions]
        out.append(
            {
                "family": name,
                "invalid_n": len(group),
                "false_accepts": sum(1 for row in group if row["prediction"] == "VALID"),
                "false_accept_rate": f"{pct(rate(group, lambda r: r['prediction'] == 'VALID')):.2f}",
            }
        )
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--model-name", required=True)
    args = parser.parse_args()

    rows = [row for row in read_rows(args.results) if not row.get("error")]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    condition_rows = summarize_condition(rows)
    foil_rows = summarize_foil_condition(rows)
    flip_rows = paired_flips(rows)
    family_rows = cue_families(rows)

    write_csv(out_dir / f"{args.model_name}_by_condition.csv", condition_rows)
    write_csv(out_dir / f"{args.model_name}_by_foil_condition.csv", foil_rows)
    write_csv(out_dir / f"{args.model_name}_paired_flips_vs_plain.csv", flip_rows)
    write_csv(out_dir / f"{args.model_name}_cue_families.csv", family_rows)

    correct = sum(row["prediction"] == row["gold_label"] for row in rows)
    report = [
        f"# Final inference analysis: {args.model_name}",
        "",
        f"- Rows analyzed: {len(rows)}",
        f"- Overall accuracy: {pct(correct / len(rows)):.2f}%",
        "",
        "## By visual condition",
        "",
    ]
    for row in condition_rows:
        report.append(
            f"- {row['condition']}: accuracy {row['accuracy']}%, "
            f"INVALID false-accept {row['false_accept_rate']}%, "
            f"VALID false-reject {row['false_reject_rate']}%"
        )
    report.extend(["", "## Paired INVALID flips vs plain", ""])
    for row in flip_rows:
        report.append(
            f"- {row['condition_vs_plain']}: induced {row['induced_false_accepts']}, "
            f"rescued {row['rescued_plain_false_accepts']}, "
            f"net {row['net_false_accept_change']}, exact McNemar p={row['mcnemar_exact_p']}"
        )
    report.extend(["", "## Cue families on INVALID rows", ""])
    for row in family_rows:
        report.append(f"- {row['family']}: {row['false_accept_rate']}% false-accept ({row['false_accepts']}/{row['invalid_n']})")
    (out_dir / f"{args.model_name}_analysis_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(report))


if __name__ == "__main__":
    main()
