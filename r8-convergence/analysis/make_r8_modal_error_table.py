"""Render the R8 modal-error CSV as a PNG table."""

import argparse
import csv
from pathlib import Path

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure


METRICS = {
    "Main accelerated charge ≥20 MeV": "Main accelerated charge, K ≥ 20 MeV",
    "Near-axis main accelerated charge ≥20 MeV": "Near-axis main accelerated charge,\nK ≥ 20 MeV",
    "Main population mean energy": "Main population mean kinetic energy",
    "Main population relative RMS energy spread": "Relative RMS energy spread",
    "Normalised emittance x": "Normalised x-emittance",
    "Normalised emittance y": "Normalised y-emittance",
    "Maximum forward energy": "Maximum forward kinetic energy",
    "99th percentile forward energy": "Charge-weighted 99th-percentile\nforward energy",
}

ARCHIVED_NAMES = {
    "First-bucket charge ≥20 MeV": "Main accelerated charge ≥20 MeV",
    "First-bucket core charge ≥20 MeV": "Near-axis main accelerated charge ≥20 MeV",
    "First-bucket mean energy": "Main population mean energy",
    "First-bucket relative RMS spread": "Main population relative RMS energy spread",
}


def with_unit(value, unit):
    if not value:
        return "—"
    value = f"{float(value):.4g}"
    return value + ("%" if unit == "%" else f" {unit}")


def make_table(rows, output):
    table_rows = []
    at_peak = rows["Main accelerated charge ≥20 MeV"].get("snapshot_selection") == "peak"
    for name, label in METRICS.items():
        row = rows[name]
        unit = row["difference_unit"]
        if unit == "percentage points":
            unit = "pp"
        ratio = row["observed_decay_ratio"]
        estimate = row["geometric_error_estimate"]
        relative = row["geometric_error_estimate_percent"]
        error = "No estimate"
        if estimate:
            error = with_unit(estimate, unit)
            if relative:
                error += f" ({float(relative):.3g}%)"
        table_rows.append((
            label,
            row.get("iteration", row.get("final_iteration")) or "—",
            with_unit(row["value_3_modes"], row["value_unit"]),
            with_unit(row["observed_change_3_to_4"], unit),
            f"{float(ratio):.3f}" if ratio else "—",
            error,
        ))

    figure = Figure(figsize=(12, 4.05), facecolor="white")
    FigureCanvasAgg(figure)
    axes = figure.add_axes((0.015, 0.03, 0.97, 0.94))
    axes.set_axis_off()
    table = axes.table(
        cellText=table_rows,
        colLabels=(
            "Quantity",
            "Peak\niteration" if at_peak else "Iteration",
            "$N_m = 3$ peak" if at_peak else "$N_m = 3$ result",
            "Change magnitude\n3 → 4",
            "Ratio $r$",
            "Modal error\nestimate",
        ),
        colWidths=(0.30, 0.09, 0.14, 0.17, 0.09, 0.21),
        cellLoc="center",
        bbox=(0, 0, 1, 1),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    for (row, column), cell in table.get_celld().items():
        cell.set_edgecolor("#a8b2bd")
        cell.set_linewidth(0.6)
        if row == 0:
            cell.set_facecolor("#e4ebf2")
            cell.get_text().set_weight("bold")
        elif row % 2 == 0:
            cell.set_facecolor("#f4f6f8")
        if column == 0:
            cell.PAD = 0.04
            cell.get_text().set_ha("left")
    figure.savefig(output, dpi=240)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("estimates_csv", type=Path)
    parser.add_argument("output_png", type=Path)
    args = parser.parse_args()
    try:
        with args.estimates_csv.open(encoding="utf-8-sig", newline="") as handle:
            rows = {ARCHIVED_NAMES.get(row["quantity"], row["quantity"]): row
                    for row in csv.DictReader(handle)}
        make_table(rows, args.output_png)
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
