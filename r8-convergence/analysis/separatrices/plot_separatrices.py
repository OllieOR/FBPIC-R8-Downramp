"""Plot matched R8 snapshots from the archived longitudinal field and particle snapshots."""

import argparse
import csv
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np

from separatrix_model import branches, kinetic, make_model, potential

ROOT = Path(__file__).resolve().parents[2] / "results/06_separatrices"
DATA = ROOT / "data"
INPUTS = DATA / "inputs"
BLUE, GREEN, PURPLE = "#3d6f8c", "#29826c", "#9b76b3"
ITERATIONS = (2860, 4000, 6000, 8000, 10750)
CASES = (
    ("R8_Nm2_nt8", 2, 8),
    ("R8_Nm2_nt16", 2, 16),
    ("R8_Nm3_nt16", 3, 16),
    ("R8_Nm4_nt16", 4, 16),
)
XLIM = (-30., 12.)
X_EDGES = np.linspace(-30., -5., 126)
U_EDGES = np.geomspace(8., 150., 66)
CHARGE_COLORS = LinearSegmentedColormap.from_list(
    "electron_charge", ["#fff5e7", "#f7c174", "#de8040", "#9d3d22", "#5d1e16"]
)
CHARGE_NORM = LogNorm(vmin=0.003, vmax=5.)


def archived_panels():
    panels = {}
    with np.load(INPUTS / "R8_optical_reference.npz") as reference:
        for case, nm, pnt in CASES:
            with np.load(INPUTS / f"{case}_snapshots.npz") as archive:
                for iteration in ITERATIONS:
                    prefix = f"i{iteration}_"
                    beta = float(reference[prefix + "beta_ref"])
                    laser = float(archive[prefix + "laser_z_um"])
                    xi = archive[prefix + "z_um"] - laser
                    ez = archive[prefix + "ez_axis_GVm"]
                    bucket = tuple(
                        float(archive[prefix + f"bucket_{side}_um"]) for side in ("low", "high")
                    )
                    optical_xi = reference[prefix + "xi"]
                    optical_values = reference[prefix + "a2"]
                    if nm == 2 and pnt == 8:
                        np.testing.assert_allclose(optical_xi, xi, atol=1e-8, rtol=0)
                        np.testing.assert_allclose(
                            reference[prefix + "ez_axis_GVm"], ez, atol=1e-6, rtol=0
                        )
                    a2 = np.interp(xi, optical_xi, optical_values)
                    model = make_model(xi, potential(xi, ez), a2, beta, bucket)
                    curve_x = np.unique(
                        np.r_[np.linspace(*XLIM, 12001), model["rear"], model["front"]]
                    )
                    curves = {
                        key: branches(curve_x, model[key], model) for key in ("hr", "hf")
                    }
                    x = archive[prefix + "z"] - laser
                    u = archive[prefix + "uz"]
                    energy = archive[prefix + "K"]
                    q = archive[prefix + "q"]
                    select = (
                        (energy >= 5.) & (u >= 1.) & (x >= bucket[0]) & (x <= bucket[1])
                    )
                    charge, _, _ = np.histogram2d(
                        x[select], u[select], bins=(X_EDGES, U_EDGES), weights=q[select]
                    )
                    np.testing.assert_allclose(charge.sum(), q[select].sum(), rtol=2e-12)
                    inside = (x >= model["left"]) & (x <= model["right"])
                    hamiltonian = kinetic(u, model["q2"](x), beta) - model["psi"](x)
                    inside &= hamiltonian < model["level"] - 1e-8
                    metrics = {
                        "iteration": iteration,
                        "time_ps": float(archive[prefix + "time_ps"]),
                        "N_m": nm,
                        "p_nt": pnt,
                        "beta_ref": beta,
                        "rear_saddle_um": model["rear"],
                        "forward_saddle_um": model["front"],
                        "closed_region_left_um": model["left"],
                        "closed_region_right_um": model["right"],
                        "limiting_saddle": "rear" if model["hr"] <= model["hf"] else "forward",
                        "first_bucket_charge_5MeV_pC": float(q[select].sum()),
                        "first_bucket_charge_20MeV_pC": float(q[select & (energy >= 20)].sum()),
                        "projected_charge_inside_contour_5MeV_pC": float(q[select & inside].sum()),
                        "max_hamiltonian_residual": max(model["residual"], *(v[2] for v in curves.values())),
                    }
                    panels[iteration, case] = (metrics, model, curve_x, curves, charge)
    return panels


def draw(iteration, panels, ylim, figures):
    fig, axes = plt.subplots(2, 2, figsize=(8.6, 6.6), sharex=True, sharey=True)
    fig.subplots_adjust(left=.118, right=.855, bottom=.167, top=.865, hspace=.27, wspace=.12)
    for ax, (case, nm, pnt) in zip(axes.flat, CASES, strict=True):
        _, model, x, curves, charge = panels[iteration, case]
        ax.fill_between(
            model["phase"], model["low"], model["high"], color=PURPLE, alpha=.19, lw=0, zorder=1
        )
        heatmap = ax.pcolormesh(
            X_EDGES, U_EDGES, np.ma.masked_less(charge.T, CHARGE_NORM.vmin),
            cmap=CHARGE_COLORS, norm=CHARGE_NORM, shading="flat", rasterized=True, zorder=2
        )
        for key, colour in (("hr", BLUE), ("hf", GREEN)):
            low, high, _ = curves[key]
            ax.plot(x, low, color=colour, lw=1.55, zorder=3)
            ax.plot(x, high, color=colour, lw=1.55, zorder=3)
        limiting_colour = BLUE if model["hr"] <= model["hf"] else GREEN
        ax.plot(model["phase"], model["low"], color=limiting_colour, lw=1.55, zorder=3)
        ax.plot(model["phase"], model["high"], color=limiting_colour, lw=1.55, zorder=3)
        ax.axhline(0., color="#9aa4af", lw=.7, zorder=2)
        ax.set_title(fr"$N_m = {nm},\ p_{{nt}} = {pnt}$", pad=6)
        ax.set_yscale("symlog", linthresh=1, linscale=.8)
        ax.set(xlim=XLIM, ylim=ylim, xticks=(-30, -20, -10, 0, 10))
        ax.grid(alpha=.2)
    fig.suptitle(f"Longitudinal phase space at iteration {iteration}", y=.976, fontsize=18)
    fig.supxlabel(r"Position relative to the laser, $\xi$ (µm)", x=.486, y=.075, fontsize=15)
    fig.supylabel(r"Longitudinal momentum, $u_z$", x=.018, y=.516, fontsize=15)
    cax = fig.add_axes((.88, .235, .018, .56))
    cbar = fig.colorbar(heatmap, cax=cax, ticks=[.01, .1, 1., 5.])
    cbar.set_ticklabels(["0.01", "0.1", "1", "5"])
    cbar.set_label("Selected charge (pC/bin)", fontsize=15, labelpad=5)
    fig.legend(
        handles=[
            Line2D([], [], color=BLUE, lw=2, label="Rear separatrix"),
            Line2D([], [], color=GREEN, lw=2, label="Forward separatrix"),
            Patch(facecolor=PURPLE, alpha=.19, label="Closed region"),
        ],
        loc="lower center", bbox_to_anchor=(.486, .005), ncol=3, frameon=False,
        handlelength=2., columnspacing=1.4, fontsize=13,
    )
    fig.savefig(figures / f"R8_Separatrices_{iteration:05}.png", dpi=210, facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Redraw the five R8 convergence separatrix figures.")
    parser.add_argument("--output", type=Path, default=ROOT,
                        help="Parent directory for figures and the summary table.")
    args = parser.parse_args()
    output = args.output.expanduser()
    figures = output / "figures"
    summary = output / "data" / "separatrix_summary.csv"
    mpl.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 13,
        "axes.titlesize": 15, "axes.spines.top": False, "axes.spines.right": False,
        "xtick.labelsize": 13, "ytick.labelsize": 13,
    })
    panels = archived_panels()
    valid_branches = [
        v[np.isfinite(v)] for _, _, _, curves, _ in panels.values()
        for pair in curves.values() for v in pair[:2]
    ]
    ylim = (
        min(-1., min(float(v.min()) for v in valid_branches)) * 1.1,
        max(float(v.max()) for v in valid_branches) * 1.14,
    )
    figures.mkdir(parents=True, exist_ok=True)
    summary.parent.mkdir(parents=True, exist_ok=True)
    with summary.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=tuple(next(iter(panels.values()))[0]))
        writer.writeheader()
        for iteration in ITERATIONS:
            draw(iteration, panels, ylim, figures)
            for case, _, _ in CASES:
                writer.writerow(panels[iteration, case][0])

    print(f"Figures: {figures}")
    print(f"Summary: {summary}")


if __name__ == "__main__":
    main()
