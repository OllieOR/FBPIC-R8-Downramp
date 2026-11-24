"""Estimate R8 modal errors at the saved peak of each three-mode diagnostic.

Run from the repository root:
    python3 r8-convergence/analysis/estimate_r8_modal_error.py > modal_error_estimates.csv

The geometric estimate assumes that the absolute changes continue to
shrink at the observed ratio. It measures mode-count sensitivity at
p_nt = 16. All mode counts are compared at the same saved iteration.
Use --at final to reproduce the final-snapshot calculation.
"""

import argparse
import csv
import math
from pathlib import Path
import sys


PAIRS = {
    ("R8_Nm2_nt16", "R8_Nm3_nt16"): "2_to_3",
    ("R8_Nm3_nt16", "R8_Nm4_nt16"): "3_to_4",
}

DEFAULT_INPUT = Path(__file__).resolve().parents[1] / "results/04_convergence/data/convergence_summary.csv"


def load_modal_comparisons(path, by_iteration=False):
    comparisons = {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            pair = row["reference_case"], row["comparison_case"]
            if pair not in PAIRS:
                continue
            key = row["population"], row["metric"]
            if by_iteration:
                key += (int(row["iteration"]),)
            steps = comparisons.setdefault(key, {})
            step = PAIRS[pair]
            if step in steps:
                raise ValueError(f"Duplicate comparison for {key}: {step}")
            steps[step] = row
    if not comparisons:
        raise ValueError("No R8 modal comparisons at p_nt = 16 were found.")
    return comparisons


def number(row, field):
    return float(row.get(field) or "nan")


def peak_snapshots(path):
    peaks = {}
    for key, steps in load_modal_comparisons(path, by_iteration=True).items():
        population, metric, iteration = key
        diagnostic = population, metric
        q3 = number(steps.get("3_to_4", {}), "reference_value")
        if math.isfinite(q3) and (q3, -iteration) > peaks.get(diagnostic, ((-math.inf, 0), {}))[0]:
            peaks[diagnostic] = ((q3, -iteration), steps)
    return {key: value[1] for key, value in peaks.items()}


def estimate(summary_steps, snapshot, at):
    first_summary, second_summary = summary_steps["2_to_3"], summary_steps["3_to_4"]
    if first_summary["unit"] != second_summary["unit"]:
        raise ValueError(f"Units differ for {second_summary['description']}.")
    first = snapshot.get("2_to_3", {})
    second = snapshot.get("3_to_4", {})
    if at == "final":
        iteration_key, time_key = "final_iteration", "final_time_ps"
        before, after = "final_reference", "final_comparison"
    else:
        iteration_key, time_key = "iteration", "time_ps"
        before, after = "reference_value", "comparison_value"
    if first and second and (
        int(first[iteration_key]) != int(second[iteration_key])
        or not math.isclose(float(first[time_key]), float(second[time_key]), rel_tol=0, abs_tol=1e-12)
    ):
        raise ValueError(f"Snapshots differ for {second_summary['description']}.")

    # The archive stores energy spread as a fraction. Report the
    # value in percent and differences in percentage points.
    fraction = second_summary["metric"] == "relative_rms_spread"
    scale = 100 if fraction else 1
    unit = second_summary["unit"]
    q2, q3_first, q3, q4 = (
        number(row, column) * scale
        for row in (first, second)
        for column in (before, after)
    )
    if all(map(math.isfinite, (q3_first, q3))) and q3_first != q3:
        raise ValueError(f"Three-mode values differ for {second_summary['description']}.")
    d23 = abs(q3 - q2)
    d34 = abs(q4 - q3)
    row = {
        "quantity": second_summary["description"],
        "final_iteration": second_summary["final_iteration"],
        "iteration": second.get(iteration_key),
        "time_ps": second.get(time_key),
        "snapshot_selection": at,
        "value_unit": "%" if fraction else unit,
        "difference_unit": "percentage points" if fraction else unit,
        "value_3_modes": q3,
        "observed_change_2_to_3": d23,
        "observed_change_3_to_4": d34,
        "observed_change_3_to_4_percent": 100 * d34 / abs(q3) if q3 else math.nan,
        "observed_decay_ratio": None,
        "predicted_change_4_to_5": None,
        "geometric_error_estimate": None,
        "geometric_error_estimate_percent": None,
        "geometric_multiplier": None,
        "status": "Missing matched result" if snapshot else "No saved three-mode value",
    }
    if not all(map(math.isfinite, (q2, q3, q4))):
        return row
    
    if d23 == 0:
        row["status"] = "Ratio undefined because the 2-to-3 change is zero"
        return row
    ratio = d34 / d23
    row["observed_decay_ratio"] = ratio
    if ratio >= 1:
        row["status"] = "Observed changes do not decay"
        return row

    multiplier = 1 / (1 - ratio)
    error = d34 * multiplier
    
    row.update(
        predicted_change_4_to_5=d34 * ratio,
        geometric_error_estimate=error,
        geometric_error_estimate_percent=100 * error / abs(q3) if q3 else None,
        geometric_multiplier=multiplier,
        status="Assumes geometric decay",
    )
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary_csv", type=Path, nargs="?", default=DEFAULT_INPUT)
    parser.add_argument("--at", choices=("peak", "final"), default="peak")
    parser.add_argument("--matched-csv", type=Path, help="Defaults to matched_differences.csv next to the summary.")
    args = parser.parse_args()
    try:
        comparisons = load_modal_comparisons(args.summary_csv)
        snapshots = comparisons
        if args.at == "peak":
            matched = args.matched_csv or args.summary_csv.with_name("matched_differences.csv")
            snapshots = peak_snapshots(matched)
        results = [estimate(steps, snapshots.get(key, {}), args.at) for key, steps in comparisons.items()]
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))

    writer = csv.DictWriter(sys.stdout, fieldnames=list(results[0]))
    writer.writeheader()
    for row in results:
        writer.writerow({
            key: (f"{value:.12g}" if math.isfinite(value) else "")
            if isinstance(value, float) else value
            for key, value in row.items()
        })


if __name__ == "__main__":
    main()
