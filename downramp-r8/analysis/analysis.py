"""Redraw the R8/C4 downramp figures from saved results and diagnostics."""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

import plotting
import separatrix
from diagnostics import CASES, load_runs, read_csv

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/downramp"
RESULTS = ROOT / "results"
CASE_FOLDER = "downramp"


def settings(path):
    return json.loads(path.read_text())


def plot_tables(output, configs, laser):
    wake = output / "01_wake"
    laser_dir = output / "02_laser"
    electrons = output / "03_electrons"
    primary = electrons / "primary_beam"
    history = output / "04_history"
    for folder in (wake, laser_dir, primary, history):
        folder.mkdir(parents=True, exist_ok=True)

    plotting.input_profile(configs, wake)
    plotting.laser_figures(laser, laser_dir, configs)

    plotting.electron_figures(read_csv(DATA / "electron_evolution.csv"),
                              read_csv(DATA / "maximum_energy.csv"),
                              settings(DATA / "settings/electron_definitions.json"), electrons)
    beam = read_csv(DATA / "primary_beam/beam_evolution.csv")
    record = settings(DATA / "primary_beam/definitions.json")
    plotting.primary_beam(beam, record, primary)
    plotting.reference_plot(beam, record, primary)
    plotting.main_population_compact_fraction(beam, record, primary)

    bunch = read_csv(DATA / "bunch.csv")
    plotting.matched_histories(bunch, bunch[0]["energy_cut_MeV"], history)
    plotting.plot_origins(read_csv(DATA / "initial_matches.csv"), configs["R8"], history)
    plotting.peak_histories(read_csv(DATA / "fixed_electron.csv"), history)

    data = DATA / "sensitivity"
    sensitivity = output / "06_sensitivity"
    sensitivity.mkdir(parents=True, exist_ok=True)
    threshold_rows = read_csv(data / "beam_evolution.csv")
    threshold_settings = settings(data / "definitions.json")
    plotting.sensitivity_figures(threshold_rows, threshold_settings, sensitivity)
    plotting.r8_divergence(threshold_rows, threshold_settings, sensitivity)
    plotting.longitudinal_emittance(read_csv(data / "longitudinal_emittance.csv"),
                                    threshold_settings, sensitivity)


def plot_snapshots(runs, output):
    wake = output / "01_wake"
    electrons = output / "03_electrons"
    wake.mkdir(parents=True, exist_ok=True)
    electrons.mkdir(parents=True, exist_ok=True)
    wake_settings = settings(DATA / "settings/wake_definitions.json")
    reference = wake_settings["density_reference_m3"]
    field_limit = wake_settings["field_colour_limit_GVm"]
    axis_limit = wake_settings["axis_limit_GVm"]
    for it in wake_settings["iterations"]:
        states = [plotting.read_state(runs[case], it) for case in CASES]
        if not np.isclose(states[0]["frame"]["time_fs"], states[1]["frame"]["time_fs"],
                          rtol=1e-8, atol=1e-6):
            raise ValueError(f"R8 and C4 times differ at iteration {it}.")
        plotting.plot_pair(states, reference, field_limit, axis_limit, wake)

    electron_settings = settings(DATA / "settings/electron_definitions.json")
    states = [plotting.particle_snapshot(runs[case], it, electron_settings)
              for it in electron_settings["iterations"] for case in CASES]
    plotting.phase_plots(states, electron_settings, electrons)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tables-only", action="store_true",
                        help="Redraw curves from the included CSVs without opening HDF5 files.")
    parser.add_argument("--separatrix-only", action="store_true",
                        help="Redraw the five fixed-ID R8/C4 snapshots, full and close-up.")
    parser.add_argument("--raw-separatrix", action="store_true",
                        help="Rebuild the separatrix inputs from the original HDF5 diagnostics.")
    parser.add_argument("--run-root", type=Path, default=ROOT,
                        help="Project containing downramp_comparison/R8 and C4.")
    parser.add_argument("--output", type=Path, default=RESULTS,
                        help="Parent directory for the downramp figure set.")
    args = parser.parse_args()
    if args.tables_only and (args.separatrix_only or args.raw_separatrix):
        parser.error("--tables-only generates the table-based figures only.")

    try:
        configs = {case: settings(ROOT / "records" / case / "run_config.json")
                   for case in CASES}
        laser = read_csv(DATA / "laser.csv")
        need_raw = args.raw_separatrix or not (args.tables_only or args.separatrix_only)
        runs = load_runs(args.run_root.expanduser().resolve()) if need_raw else None
        if runs is not None:
            for case in CASES:
                if configs[case] != runs[case].config:
                    raise ValueError(f"The {case} diagnostic run differs from the saved configuration.")
        output = args.output.expanduser().resolve()
        output.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix=".figures-", dir=output) as tmp:
            stage = Path(tmp) / CASE_FOLDER
            if not args.separatrix_only:
                plot_tables(stage, configs, laser)
                if runs is not None:
                    plot_snapshots(runs, stage)
            if not args.tables_only:
                folder = stage / "05_separatrix"
                separatrix.regenerate(DATA / "separatrix", laser, folder, folder / "data",
                                      runs if args.raw_separatrix else None)
            files = [path for path in stage.rglob("*") if path.is_file()]
            for path in files:
                target = output / CASE_FOLDER / path.relative_to(stage)
                target.parent.mkdir(parents=True, exist_ok=True)
                path.replace(target)
        print(f"Generated {sum(path.suffix == '.png' for path in files)} figures in {output / CASE_FOLDER}")
    except (OSError, ValueError, KeyError, ImportError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
