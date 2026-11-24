"""Frozen on-axis separatrices and the five archived R8 particle tracks."""

from dataclasses import dataclass
import hashlib
import json

import numpy as np
from scipy.constants import c, e, epsilon_0, m_e
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import CubicSpline, PchipInterpolator
from scipy.optimize import brentq, minimize_scalar
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from diagnostics import (MC2_MEV, XI_LABEL, first_bucket, mesh, optical_profile,
                         read_particles, reference_axis, write_csv)
from plotting import save_figure

COLOURS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00")
BLUE, GREEN, PURPLE = "#3d6f8c", "#29826c", "#9b76b3"


def potential(x, ez):
    # Same m=0 field, integration sign, units and gauge as the existing R8 code.
    psi = -1e-3/MC2_MEV*cumulative_trapezoid(ez, x, initial=0)
    return psi-np.interp(0, x, psi)


def kinetic(u, q, beta):
    u = np.asarray(u)
    gamma = np.sqrt(q+u*u)
    return np.where(u >= 0, q/(gamma+np.abs(u))+(1-beta)*u, gamma-beta*u)


def branches(x, level, model):
    q, beta = model['q2'](x), model['beta']
    a = level+model['psi'](x)
    delta = (1-beta)*(1+beta)
    gap = a-np.sqrt(q*delta)
    valid = (a > 0) & (gap >= -2e-9)
    root = np.sqrt(np.maximum(gap, 0)*(a+np.sqrt(q*delta)).clip(min=0))
    den = beta*a+root
    low = np.full(np.shape(a), np.nan)
    np.divide(q-a*a, den, out=low, where=valid & (den != 0))
    high = np.where(valid, den/delta, np.nan)
    if np.any(valid):
        residual = max(float(np.max(np.abs(kinetic(low[valid],q[valid],beta)
                                          -model['psi'](x[valid])-level))),
                       float(np.max(np.abs(kinetic(high[valid],q[valid],beta)
                                          -model['psi'](x[valid])-level))))
        if residual > 1e-8:
            raise ValueError(f'Contour Hamiltonian residual too large: {residual:g}')
    else:
        residual = 0.
    return low, high, residual


def extremum(g, left, right, maximum):
    sign = -1 if maximum else 1
    fit = minimize_scalar(lambda x: sign*float(g(x)), bounds=(left,right),
                          method='bounded', options={'xatol':1e-12})
    return float(fit.x)


def turn(g, centre, end, level):
    xs = np.linspace(centre,end,15001)
    vals = g(xs)-level
    hits = np.flatnonzero(vals >= 0)
    if len(hits):
        j = int(hits[0])
        if j == 0:
            raise ValueError('Limiting level does not enclose the well centre.')
        return float(brentq(lambda x: float(g(x)-level),
                           *sorted([xs[j-1],xs[j]]),xtol=1e-12))
    if abs(float(g(end)-level)) < 2e-9:
        return float(end)
    raise ValueError('Unresolved turning point: refusing to invent closure.')


def make_model(x, psi_values, a2_values, beta, bounds):
    if not 0 < beta < 1:
        raise ValueError('Reference beta must be subluminal.')
    if not np.all(np.isfinite(np.r_[x,psi_values,a2_values,bounds])):
        raise ValueError('Non-finite profile or first-bucket geometry.')
    psi = CubicSpline(x,psi_values,extrapolate=False)
    q2 = PchipInterpolator(x,1+np.maximum(a2_values,0),extrapolate=False)
    delta = (1-beta)*(1+beta)
    g = lambda xx: np.sqrt(q2(xx)*delta)-psi(xx)
    model = dict(psi=psi, q2=q2, beta=float(beta))
    # The old first-period interval brackets the rear saddle between two wells.
    xx = np.linspace(bounds[0],bounds[1],12001)
    j = int(np.argmax(g(xx)))
    if j in (0,len(xx)-1):
        raise ValueError('No resolved interior rear saddle.')
    rear = extremum(g,xx[j-1],xx[j+1],True)
    model.update(rear=rear,hr=float(g(rear)))
    # The first-bucket well lies in front of the rear saddle, near the front
    # field-period boundary. The optical ponderomotive term matters near laser.
    xx = np.linspace(rear+.05,min(2.,x[-1]-.1),20001)
    minima, _ = find_peaks(-g(xx),prominence=1e-7)
    if not len(minima):
        raise ValueError('No resolved first-bucket well centre.')
    nearby=minima[abs(xx[minima]-bounds[1])<=4.]
    if not len(nearby):
        raise ValueError('No well centre near the established first-bucket boundary.')
    # Choose the actual well bottom, not a tiny ripple nearest a smoothed root.
    j = int(nearby[np.argmin(g(xx[nearby]))])
    centre = extremum(g,xx[j-1],xx[j+1],False)
    xx = np.linspace(centre+.01,x[-1]-.1,30001)
    peaks, _ = find_peaks(g(xx), prominence=1e-7)
    if not len(peaks):
        model.update(front=None,closed=False)
        return model
    k = int(np.argmax(g(xx[peaks])))
    j = int(peaks[k])
    front = extremum(g,xx[j-1],xx[j+1],True)
    hf = float(g(front))
    level = min(model['hr'], hf)
    left = turn(g,centre,rear,level)
    right = turn(g,centre,front,level)
    phase = .5*(left+right)-.5*(right-left)*np.cos(np.linspace(0,np.pi,5001))
    low,high,residual = branches(phase,level,model)
    if not np.all(np.isfinite(np.r_[low,high])):
        raise ValueError('Forbidden interval inside candidate island.')
    for point, ii in [(left,0),(right,-1)]:
        if abs(float(g(point))-level)>2e-8:
            raise ValueError('Endpoint is not a verified turning point.')
        low[ii]=high[ii]=beta*np.sqrt(q2(point)/delta)
    model.update(front=front,hf=hf,level=level,left=left,right=right,
                 phase=phase,low=low,high=high,residual=residual,closed=True)
    return model

@dataclass
class Panel:
    case: str
    iteration: int
    time_ps: float
    xi: np.ndarray
    ez: np.ndarray
    a2: np.ndarray
    beta: float
    particles: list
    model: dict
    status: str


def make_panel(case, iteration, time_ps, xi, ez, a2, beta, bucket, particles):
    model = {}
    status = "First wake period unresolved"
    if bucket is not None:
        try:
            model = make_model(xi, potential(xi, ez), a2, beta, bucket)
            status = "closed" if model["closed"] else "Forward saddle unresolved"
        except ValueError as error:
            status = str(error)
    return Panel(case, iteration, time_ps, xi, ez, a2, beta, particles, model, status)


def matched_reference(run, archive, columns, record):
    """Verify that the C4 IDs refer to the same early states as the R8 IDs."""
    full = run.series(full=True)
    ids = archive["ids"][columns]
    early = (np.all(archive["status"][:, columns] == "observed", axis=1)
             & (archive["source"] == "unfiltered_history")
             & (archive["laser_um"] < record["source_config"]["downramp_start"] * 1e6 - 12))
    matches = np.flatnonzero(early & np.isin(archive["iterations"], full.iterations))
    if not len(matches):
        raise ValueError("No common early unfiltered frame for matching the C4 IDs.")
    i = matches[0]
    iteration = int(archive["iterations"][i])
    p = read_particles(full, iteration, ids, allow_duplicates=True)
    expected_xyz = np.column_stack([archive[f"{k}_um"][i, columns] for k in "xyz"])
    expected_u = np.column_stack([archive[f"u{k}"][i, columns] for k in "xyz"])
    weights = {int(row["id"]): row["native_charge_pC"] for row in record["selection"]}
    rows = []
    for j, pid in enumerate(ids):
        keep = p["id"] == pid
        if keep.sum() != 1:
            raise ValueError(f"C4 ID {pid} has {keep.sum()} records at early iteration {iteration}.")
        xyz = np.array([p[k][keep][0] * 1e6 for k in "xyz"])
        u = np.array([p[f"u{k}"][keep][0] for k in "xyz"])
        dx, du = max(abs(xyz - expected_xyz[j])), max(abs(u - expected_u[j]))
        charge = p["w"][keep][0] * e * 1e12
        same_weight = np.isclose(charge, weights[int(pid)], rtol=1e-7, atol=1e-12)
        if dx > .001 or du > .001 or not same_weight:
            raise ValueError(f"C4 ID {pid} does not match the early R8 state at {iteration}.")
        rows.append(dict(id=int(pid), iteration=iteration, position_error_um=dx,
                         momentum_error=du, native_charge_pC=charge))
    return rows


def r8_panels(archive, columns, iterations, optical):
    for it in iterations:
        i = int(np.flatnonzero(archive["iterations"] == it)[0])
        xi = archive[f"snapshot_{it}_xi"]
        a2 = optical[it]
        particles = [dict(
            id=int(archive["ids"][j]), xi_um=archive["xi_um"][i, j], uz=archive["uz"][i, j],
            radius_um=archive["radius_um"][i, j],
            uperp=np.hypot(archive["ux"][i, j], archive["uy"][i, j]),
            K_MeV=archive["K_MeV"][i, j], status=str(archive["status"][i, j]),
            source=str(archive["source"][i]),
        ) for j in columns]
        yield make_panel("R8", it, archive["time_ps"][i], xi, archive[f"snapshot_{it}_axis_ez"],
                         a2, archive["beta_group_reference"][i], archive["bounds_um"][i, 0], particles)


def saved_c4_panels(saved, ids, iterations):
    if saved["ids"].tolist() != ids or saved["iterations"].tolist() != iterations:
        raise ValueError("The saved C4 tracks differ from the matched R8 IDs or snapshots.")
    for i, it in enumerate(iterations):
        particles = []
        for j, pid in enumerate(ids):
            status = str(saved["status"][i, j])
            observed = status.startswith("observed")
            particles.append(dict(
                id=pid, xi_um=saved["xi_um"][i, j], uz=saved["uz"][i, j],
                radius_um=saved["radius_um"][i, j], uperp=np.hypot(saved["ux"][i, j], saved["uy"][i, j]),
                K_MeV=saved["K_MeV"][i, j], status="observed" if observed else status,
                source=str(saved["observation_source"][i]),
            ))
        bounds = saved["bounds_um"][i]
        yield make_panel("C4", it, saved["time_ps"][i], saved[f"snapshot_{it}_xi"],
                         saved[f"snapshot_{it}_ez"], saved[f"snapshot_{it}_a2"],
                         saved["beta_group_reference"][i],
                         bounds if np.isfinite(bounds).all() else None, particles)


def read_r8_optical(run, archive, iterations):
    series = run.series()
    optical = {}
    for it in iterations:
        i = int(np.flatnonzero(archive["iterations"] == it)[0])
        z, a2 = optical_profile(series, it, run.config)
        if not np.allclose(z * 1e6 - archive["laser_um"][i], archive[f"snapshot_{it}_xi"],
                           rtol=0, atol=1e-6):
            raise ValueError(f"R8 optical grid differs from the archived wake at {it}.")
        optical[it] = a2
    return optical


def c4_panels(run, identity, iterations, laser):
    series, full = run.series(), run.series(full=True)
    ids = np.array([row["id"] for row in identity], dtype=np.uint64)
    nc = epsilon_0 * m_e * (2 * np.pi * c / run.config["lambda0"])**2 / e**2
    density = run.config["n_e"]
    beta = np.sqrt(1 - density / nc)
    previous, missing_full, unreliable = {}, set(), set()
    for it in iterations:
        row = laser[it]
        laser_um, time = row["pulse_peak_z_um"], row["time_ps"]
        if row["boundary_flag"]:
            raise ValueError(f"C4 laser peak is boundary-flagged at {it}.")
        plateau_start = run.config["ramp_start"] + run.config["ramp_length"]
        if not plateau_start < laser_um * 1e-6 < run.config["exit_start"]:
            raise ValueError(f"C4 laser peak is outside the constant-density plateau at {it}.")
        if not np.isclose(it * run.config["dt"] * 1e12, time, rtol=0, atol=1e-9):
            raise ValueError(f"C4 laser table time differs from the diagnostics at {it}.")
        field, r, z = mesh(series, it, "E", "z")
        optical_z, a2 = optical_profile(series, it, run.config)
        if not np.allclose(z, optical_z, rtol=0, atol=1e-14):
            raise ValueError(f"C4 optical and wake grids differ at {it}.")
        xi, ez = z * 1e6 - laser_um, reference_axis(field, r) / 1e9
        try:
            bucket = first_bucket(xi, ez, xi[0], xi[-1], density)
        except ValueError:
            bucket = None

        unfiltered = it in full.iterations
        source = "unfiltered_history" if unfiltered else "filtered_diags"
        p = read_particles(full if unfiltered else series, it, ids, allow_duplicates=True)
        particles = []
        for native in identity:
            pid = native["id"]
            keep = np.flatnonzero(p["id"] == pid)
            particle = dict(id=pid, xi_um=np.nan, uz=np.nan, radius_um=np.nan,
                            uperp=np.nan, K_MeV=np.nan, status="not recorded", source=source)
            particles.append(particle)
            if not len(keep):
                if unfiltered:
                    missing_full.add(pid)
                continue
            if len(keep) > 1:
                unreliable.add(pid)
            if pid in missing_full or pid in unreliable:
                particle["status"] = "identity unverified"
                continue
            j = keep[0]
            xyz = np.array([p[k][j] * 1e6 for k in "xyz"])
            u = np.array([p[f"u{k}"][j] for k in "xyz"])
            charge = p["w"][j] * e * 1e12
            valid = np.isclose(charge, native["native_charge_pC"], rtol=1e-7, atol=1e-12)
            if pid in previous:
                old_time, old_xyz = previous[pid]
                valid &= np.linalg.norm(xyz - old_xyz) <= c * (time - old_time) * 1e-6 + .1
            if not valid:
                unreliable.add(pid)
                particle["status"] = "identity unverified"
                continue
            previous[pid] = time, xyz
            particle.update(xi_um=xyz[2] - laser_um, uz=u[2], radius_um=np.hypot(*xyz[:2]),
                            uperp=np.hypot(*u[:2]), K_MeV=(np.sqrt(1 + u @ u) - 1) * MC2_MEV,
                            status="observed")
        yield make_panel("C4", it, time, xi, ez, a2, beta, bucket, particles)


def draw_panel(ax, panel, zoom=False):
    m = panel.model
    limits = [-1., 160.]
    if m:
        x = np.unique(np.r_[np.linspace(panel.xi[0], panel.xi[-1], 16001), m["rear"],
                            m["front"] if m["closed"] else []])
        for key, colour, label in (("hr", BLUE, "Rear separatrix"), ("hf", GREEN, "Forward separatrix")):
            if key in m:
                lo, hi, _ = branches(x, m[key], m)
                ax.plot(x, lo, color=colour, lw=1.8, label=label)
                ax.plot(x, hi, color=colour, lw=1.8)
                for values in (lo, hi):
                    limits.extend(values[np.isfinite(values)])
        if m["closed"]:
            ax.fill_between(m["phase"], m["low"], m["high"], color=PURPLE,
                            alpha=.18, lw=0, label="Frozen-model trapped region")

    points = [p for p in panel.particles if p["status"] == "observed"]
    for colour, p in zip(COLOURS, panel.particles, strict=True):
        observed = p["status"] == "observed"
        ax.scatter([p["xi_um"]] if observed else [], [p["uz"]] if observed else [],
                   s=44, color=colour, edgecolors="white", linewidths=.6, zorder=6,
                   label=f'ID {p["id"]}' + ("" if observed else "*"))
    if zoom and points:
        px, pu = (np.array([p[key] for p in points]) for key in ("xi_um", "uz"))
        dx, du = max(.12, .25 * np.ptp(px)), max(2., .25 * np.ptp(pu))
        ax.set(xlim=(px.min()-dx, px.max()+dx), ylim=(pu.min()-du, pu.max()+du))
    else:
        ax.set_yscale("symlog", linthresh=1., linscale=.8)
        left = max(panel.xi[0], m.get("rear", -25) - 5)
        right = min(panel.xi[-1], m["front"] + 5 if m.get("closed") else 5.)
        if points:
            left = min(left, min(p["xi_um"] for p in points) - .5)
            right = max(right, max(p["xi_um"] for p in points) + .5)
            limits.extend(p["uz"] for p in points)
        ax.set(xlim=(left, right), ylim=(min(limits)*1.15, max(limits)*1.15))
    ax.set(title=f"{panel.case} · iteration {panel.iteration} · {panel.time_ps:.3f} ps",
           xlabel=XI_LABEL, ylabel=r"Longitudinal momentum $u_z=p_z/(m_e c)$")
    ax.axhline(0, color=".65", lw=.65)
    note = f"On-axis reference · β = {panel.beta:.7f}"
    if panel.status != "closed":
        note += "\nClosed region unresolved"
    if len(points) < len(panel.particles):
        note += " · * unavailable"
    ax.legend(loc="upper center", bbox_to_anchor=(.5, -.13), ncol=3, fontsize=8,
              frameon=False, title=note, title_fontsize=8)


def save_panels(path, panels, zoom=False):
    fig, axes = plt.subplots(1, len(panels), figsize=(10.8 if len(panels) == 1 else 15, 6.5),
                             squeeze=False, layout="constrained")
    axes = axes[0]
    for ax, panel in zip(axes, panels, strict=True):
        draw_panel(ax, panel, zoom)
    if len(panels) > 1:
        xlo, xhi = zip(*(ax.get_xlim() for ax in axes))
        ylo, yhi = zip(*(ax.get_ylim() for ax in axes))
        for ax in axes:
            ax.set(xlim=(min(xlo), max(xhi)), ylim=(min(ylo), max(yhi)))
        axes[1].set_ylabel("")
    path.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, path)


def regenerate(inputs, laser, output, data, runs=None):
    from diagnostics import read_csv

    cache = inputs / "r8_first_bucket_top5.npz"
    record = json.loads((inputs / "selection.json").read_text())
    if hashlib.sha256(cache.read_bytes()).hexdigest() != record["cache_sha256"]:
        raise ValueError("The saved R8 selection and its metadata disagree.")
    if runs is not None and runs["R8"].config != record["source_config"]:
        raise ValueError("The R8 run configuration differs from the archived selection.")
    with np.load(cache) as archive:
        columns = archive["chosen_columns"]
        if archive["ids"][columns].tolist() != record["selected_ids"]:
            raise ValueError("The saved IDs differ from the selection table.")
        iterations = record["settings"]["snapshots"]
        if runs is None:
            r8 = inputs / "R8"
            c4 = inputs / "C4"
            identity = read_csv(c4 / "identity_check.csv")
            optical_meta = json.loads((r8 / "source.json").read_text())
            c4_meta = json.loads((c4 / "definitions.json").read_text())
            if ([row["id"] for row in identity] != record["selected_ids"]
                    or any(not row["passed"] for row in identity)
                    or optical_meta["selection_cache_sha256"] != record["source_cache_sha256"]
                    or c4_meta["source_cache_sha256"] != record["source_cache_sha256"]):
                raise ValueError("Saved optical profiles or C4 identity records differ from the R8 selection.")
            with np.load(r8 / "optical_axis_profiles.npz") as saved:
                optical = {it: saved[f"a2_{it}"] for it in iterations}
            with np.load(c4 / "c4_fixed_ids.npz") as saved:
                control = list(saved_c4_panels(saved, record["selected_ids"], iterations))
        else:
            identity = matched_reference(runs["C4"], archive, columns, record)
            optical = read_r8_optical(runs["R8"], archive, iterations)
            c4_laser = {row["iteration"]: row for row in laser if row["case"] == "C4"}
            control = c4_panels(runs["C4"], identity, iterations, c4_laser)
        paired = zip(r8_panels(archive, columns, iterations, optical), control, strict=True)
        snapshots, markers = [], []
        for pair in paired:
            it = pair[0].iteration
            if it != pair[1].iteration or not np.isclose(pair[0].time_ps, pair[1].time_ps,
                                                         rtol=0, atol=1e-9):
                raise ValueError(f"R8 and C4 times differ at iteration {it}.")
            filename = f"{it:05d}.png"
            save_panels(output / "comparison" / filename, pair)
            for panel in pair:
                save_panels(output / panel.case / filename, [panel])
                save_panels(output / panel.case / "zoom" / filename, [panel], zoom=True)
                m = panel.model
                snapshots.append(dict(case=panel.case, iteration=it, time_ps=panel.time_ps,
                    beta_ref=panel.beta, status=panel.status, rear_xi_um=m.get("rear"),
                    forward_xi_um=m.get("front"), limiting_kappa=m.get("level"),
                    island_left_um=m.get("left"), island_right_um=m.get("right"),
                    contour_residual=m.get("residual")))
                for particle in panel.particles:
                    inside = ""
                    if particle["status"] == "observed" and m.get("closed"):
                        x, u = particle["xi_um"], particle["uz"]
                        inside = bool(m["left"] <= x <= m["right"]
                                      and kinetic(u, m["q2"](x), m["beta"]) - m["psi"](x) < m["level"])
                    markers.append(dict(case=panel.case, iteration=it,
                                        inside_frozen_region=inside, **particle))
                profile = data / "profiles" / f"{panel.case}_{it:05d}.csv"
                profile.parent.mkdir(parents=True, exist_ok=True)
                np.savetxt(profile, np.column_stack((panel.xi, panel.ez, panel.a2)),
                           delimiter=",", header="xi_um,Ez_GVm,cycle_averaged_a2", comments="")
    write_csv(data / "snapshots.csv", snapshots)
    write_csv(data / "particles.csv", markers)
    write_csv(data / "identity_check.csv", identity)
