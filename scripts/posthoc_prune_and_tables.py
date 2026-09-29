import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path


MAIN_FOILS = {"same_surah_far_ayah", "same_surah_nearby_ayah"}
BENCHMARK_FOILS = {"cross_surah_lexical_overlap"}


def read_csv(path):
    with open(path, "r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def grouped_flags(rows):
    by_group = defaultdict(list)
    for row in rows:
        by_group[(row["verse_id"], row["factual_state"])].append(row)

    flags = {}
    for key, items in by_group.items():
        invalid_items = [row for row in items if row["gold_label"] == "INVALID"]
        if not invalid_items:
            flags[key] = "valid_control"
            continue
        false_accepts = [row for row in invalid_items if row["prediction"] == "VALID"]
        if len(false_accepts) == len(invalid_items):
            flags[key] = "all_condition_false_accept"
        elif false_accepts:
            flags[key] = "visual_sensitive_false_accept"
        else:
            flags[key] = "clean_rejected"
    return flags


def enrich(rows, flags):
    enriched = []
    for row in rows:
        flag = flags[(row["verse_id"], row["factual_state"])]
        tier = "excluded"
        if row["gold_label"] == "VALID":
            tier = "valid_control"
        elif row["foil_type"] in MAIN_FOILS and flag != "all_condition_false_accept":
            tier = "main_cue_bias"
        elif row["foil_type"] in BENCHMARK_FOILS:
            tier = "hard_benchmark"
        enriched.append({**row, "posthoc_flag": flag, "analysis_tier": tier})
    return enriched


def pct(n, d):
    return 100.0 * n / d if d else 0.0


def summarize(rows, fields):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[field] for field in fields)].append(row)

    out = []
    for key, items in sorted(groups.items()):
        n = len(items)
        correct = sum(row["prediction"] == row["gold_label"] for row in items)
        invalid = [row for row in items if row["gold_label"] == "INVALID"]
        false_accept = sum(row["prediction"] == "VALID" for row in invalid)
        valid = [row for row in items if row["gold_label"] == "VALID"]
        false_reject = sum(row["prediction"] == "INVALID" for row in valid)
        out.append(
            {
                **{field: value for field, value in zip(fields, key)},
                "n": n,
                "accuracy_pct": f"{pct(correct, n):.1f}",
                "false_accept_pct": f"{pct(false_accept, len(invalid)):.1f}",
                "false_reject_pct": f"{pct(false_reject, len(valid)):.1f}",
            }
        )
    return out


def markdown_table(rows, fields):
    lines = []
    lines.append("| " + " | ".join(fields) + " |")
    lines.append("| " + " | ".join(["---"] * len(fields)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(str(row[field]) for field in fields) + " |")
    return "\n".join(lines)


def write_report(path, enriched, main_rows, benchmark_rows):
    condition_rows = summarize(enriched, ["condition"])
    main_condition_rows = summarize(main_rows, ["condition"])
    main_foil_rows = summarize(main_rows, ["foil_type", "condition"])
    benchmark_rows_summary = summarize(benchmark_rows, ["condition"])

    flags = Counter(row["posthoc_flag"] for row in enriched if row["gold_label"] == "INVALID")
    tiers = Counter(row["analysis_tier"] for row in enriched)

    text = [
        "# Posthoc Pruned Analysis Report",
        "",
        "## Dataset Split",
        "",
        f"- total rows: {len(enriched)}",
        f"- main cue-bias rows: {len(main_rows)}",
        f"- hard benchmark rows: {len(benchmark_rows)}",
        "",
        "Analysis tiers:",
        "",
    ]
    for key, value in sorted(tiers.items()):
        text.append(f"- {key}: {value}")
    text.extend(["", "Invalid-group flags:", ""])
    for key, value in sorted(flags.items()):
        text.append(f"- {key}: {value}")

    text.extend(
        [
            "",
            "## Overall By Condition",
            "",
            markdown_table(condition_rows, ["condition", "n", "accuracy_pct", "false_accept_pct", "false_reject_pct"]),
            "",
            "## Main Cue-Bias Tier",
            "",
            markdown_table(main_condition_rows, ["condition", "n", "accuracy_pct", "false_accept_pct", "false_reject_pct"]),
            "",
            "## Main Tier By Foil Type",
            "",
            markdown_table(main_foil_rows, ["foil_type", "condition", "n", "accuracy_pct", "false_accept_pct", "false_reject_pct"]),
            "",
            "## Hard Benchmark Tier",
            "",
            markdown_table(benchmark_rows_summary, ["condition", "n", "accuracy_pct", "false_accept_pct", "false_reject_pct"]),
            "",
            "## Reading",
            "",
            "Use the main cue-bias tier for the central visual-cue claim. Use the hard benchmark tier as evidence that lexical citation verification is difficult, not as the clean causal estimate.",
        ]
    )
    Path(path).write_text("\n".join(text) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="final_candidate/openai_ocr_gpt41.csv")
    parser.add_argument("--out-dir", default="final_analysis")
    args = parser.parse_args()

    rows = [row for row in read_csv(args.results) if not row.get("error")]
    flags = grouped_flags(rows)
    enriched = enrich(rows, flags)

    main_rows = [row for row in enriched if row["analysis_tier"] in {"main_cue_bias", "valid_control"}]
    benchmark_rows = [row for row in enriched if row["analysis_tier"] == "hard_benchmark"]

    out_dir = Path(args.out_dir)
    write_csv(out_dir / "results_with_posthoc_flags.csv", enriched)
    write_csv(out_dir / "main_cue_bias_results.csv", main_rows)
    write_csv(out_dir / "hard_benchmark_results.csv", benchmark_rows)
    write_report(out_dir / "posthoc_pruned_report.md", enriched, main_rows, benchmark_rows)

    print(f"wrote {out_dir / 'posthoc_pruned_report.md'}")
    print(f"main_rows={len(main_rows)} benchmark_rows={len(benchmark_rows)} total={len(enriched)}")


if __name__ == "__main__":
    main()
