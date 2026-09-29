import argparse
import csv
from collections import defaultdict
from pathlib import Path


def read_csv(path):
    with open(path, "r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def pct(n, d):
    return 100.0 * n / d if d else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="final_analysis/main_cue_bias_results.csv")
    parser.add_argument("--out", default="final_analysis/paired_flip_table.csv")
    args = parser.parse_args()

    rows = [row for row in read_csv(args.results) if row["gold_label"] == "INVALID"]
    by_item = defaultdict(dict)
    for row in rows:
        by_item[(row["verse_id"], row["factual_state"], row["foil_type"])][row["condition"]] = row

    conditions = sorted({row["condition"] for row in rows if row["condition"] != "plain"})
    out_rows = []
    for condition in conditions:
        comparable = [
            item for item in by_item.values()
            if "plain" in item and condition in item
        ]
        plain_invalid = [item for item in comparable if item["plain"]["prediction"] == "INVALID"]
        induced = [
            item for item in plain_invalid
            if item[condition]["prediction"] == "VALID"
        ]
        rescued = [
            item for item in comparable
            if item["plain"]["prediction"] == "VALID" and item[condition]["prediction"] == "INVALID"
        ]
        out_rows.append(
            {
                "condition": condition,
                "paired_items": len(comparable),
                "plain_correct_invalid": len(plain_invalid),
                "induced_false_accepts": len(induced),
                "induced_false_accept_pct_of_plain_correct": f"{pct(len(induced), len(plain_invalid)):.1f}",
                "rescued_plain_false_accepts": len(rescued),
            }
        )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(out_rows[0].keys()))
        writer.writeheader()
        writer.writerows(out_rows)

    for row in out_rows:
        print(",".join(str(value) for value in row.values()))


if __name__ == "__main__":
    main()
