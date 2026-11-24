"""Read R8/C4 fields and particles with their diagnostic coordinates."""

from dataclasses import dataclass, field
from pathlib import Path
import csv
import json

import numpy as np
from scipy.constants import c, e, epsilon_0, m_e
from scipy.ndimage import gaussian_filter1d
from scipy.signal import hilbert

CASES = ("R8", "C4")
MC2_MEV = m_e * c**2 / e / 1e6
XI_LABEL = "Position relative to the laser, ξ (µm)"
PARTICLES = ("x", "y", "z", "ux", "uy", "uz", "w")


def edges(x):
    return np.r_[x[0]-(x[1]-x[0])/2, (x[:-1]+x[1:])/2, x[-1]+(x[-1]-x[-2])/2]


def density(z, config):
    entrance = np.clip((z-config["ramp_start"])/config["ramp_length"], 0., 1.)
    drop = np.clip((z-config["downramp_start"])/config["downramp_length"], 0., 1.)
    exit_ramp = np.clip((z-config["exit_start"])/config["exit_length"], 0., 1.)
    return config["n_e"]*entrance*(1.-(1.-config["lower_density_factor"])*drop)*(1.-exit_ramp)


def mesh(series, iteration, name, coord=None, mode=0, theta=0.):
    args = dict(field=name, iteration=int(iteration), m=mode, theta=theta)
    if coord is not None:
        args["coord"] = coord
    values, info = series.get_field(**args)
    values, r, z = np.asarray(values, float), np.asarray(info.r), np.asarray(info.z)
    if values.shape == (len(z), len(r)) and values.shape != (len(r), len(z)):
        values = values.T
    if (values.shape != (len(r), len(z)) or min(values.shape) < 2
            or np.any(np.diff(r) <= 0) or np.any(np.diff(z) <= 0)
            or not np.all(np.isfinite(values))):
        raise ValueError(f"Invalid {name} mesh at iteration {iteration}.")
    return values, r, z


def on_axis(values, r):
    if not r[0] <= 0 <= r[-1]:
        raise ValueError("The field plane does not bracket the axis.")
    j = int(np.clip(np.searchsorted(r, 0.), 1, len(r)-1))
    f = -r[j-1]/(r[j]-r[j-1])
    return values[j-1]*(1-f)+values[j]*f


def laser_peak(series, iteration, config):
    """Band-limited m=1 optical envelope; return peak z in um."""
    ex, r, z = mesh(series, iteration, "E", "x", mode=1)
    angle = float(config.get("theta_pol", 0.))
    line = on_axis(ex, r)*np.cos(angle)
    if abs(np.sin(angle)) > 1.e-12:
        ey, _, _ = mesh(series, iteration, "E", "y", mode=1)
        line += on_axis(ey, r)*np.sin(angle)
    dz = float(np.mean(np.diff(z)))
    if not np.allclose(np.diff(z), dz, rtol=1.e-7, atol=1.e-15):
        raise ValueError("Optical filtering requires uniform z spacing.")
    k0 = 2*np.pi/config["lambda0"]
    k = 2*np.pi*np.fft.rfftfreq(len(z), dz)
    spectrum = np.fft.rfft(line-line.mean())
    spectrum[(k < .25*k0) | (k > 2.5*k0)] = 0.
    envelope = abs(hilbert(np.fft.irfft(spectrum, n=len(z))))
    peak = int(np.argmax(envelope))
    if envelope[peak] <= 0 or not z[0]+2.e-6 < z[peak] < z[-1]-2.e-6:
        raise ValueError(f"No usable optical peak at iteration {iteration}.")
    return float(z[peak]*1.e6)


def read_particles(series, iteration, ids=None, allow_duplicates=False):
    """Match integer IDs explicitly and read one full particle record at a time."""
    raw, = series.get_particle(var_list=["id"], species="electrons", iteration=int(iteration))
    raw = np.asarray(raw)
    if raw.ndim != 1 or raw.dtype.kind not in "ui" or np.any(raw < 0):
        raise ValueError(f"Invalid electron IDs at iteration {iteration}.")
    raw = raw.astype(np.uint64, copy=False)
    if ids is None:
        indices = np.argsort(raw)
    else:
        ids = np.asarray(ids, np.uint64)
        if np.any(ids[1:] <= ids[:-1]):
            raise ValueError("Requested IDs must be unique and sorted.")
        pos = np.searchsorted(ids, raw)
        hit = pos < len(ids)
        hit[hit] &= ids[pos[hit]] == raw[hit]
        indices = np.flatnonzero(hit)
    selected = raw[indices]
    if not allow_duplicates and len(np.unique(selected)) != len(selected):
        raise ValueError(f"Duplicate electron IDs at iteration {iteration}.")
    result = {"id": selected}
    for name in PARTICLES:
        if not len(indices):
            result[name] = np.empty(0)
            continue
        values, = series.get_particle(var_list=[name], species="electrons", iteration=int(iteration))
        values = np.asarray(values, float)
        if values.ndim != 1 or len(values) != len(raw):
            raise ValueError(f"Inconsistent {name} record at iteration {iteration}.")
        result[name] = values[indices]
        if not np.all(np.isfinite(result[name])):
            raise ValueError(f"Non-finite {name} at iteration {iteration}.")
    if np.any(result["w"] < 0):
        raise ValueError("Negative particle weights.")
    return result


def derive(p, frame):
    gamma = np.sqrt(1+p["ux"]**2+p["uy"]**2+p["uz"]**2)
    radius, z = np.hypot(p["x"], p["y"])*1.e6, p["z"]*1.e6
    inside = ((radius <= frame["rmax_um"]) & (z >= frame["zmin_um"]) & (z <= frame["zmax_um"]))
    return dict(energy_MeV=(gamma-1)*m_e*c*c/e/1.e6, radius_um=radius,
                xi_um=z-frame["laser_z_um"], x_um=p["x"]*1.e6, y_um=p["y"]*1.e6,
                inside=inside, charge_pC=p["w"]*e*1.e12)


def window(xi, limits):
    return (xi >= limits[0]) & (xi <= limits[1])


def reference_axis(values, r):
    # The R8 model extrapolates the even radial profile in r².
    zero = np.flatnonzero(np.isclose(r, 0, rtol=0, atol=1e-15))
    if len(zero):
        return values[zero[0]]
    positive = np.flatnonzero(r > 0)
    if len(positive) < 2:
        raise ValueError("Need two positive radial cells for the on-axis field.")
    i, j = positive[:2]
    return (r[j]**2 * values[i] - r[i]**2 * values[j]) / (r[j]**2 - r[i]**2)


def optical_profile(series, iteration, config):
    """Cycle-averaged a² and envelope on the axis, on the diagnostic z grid."""
    ex, r, z = mesh(series, iteration, "E", "x", mode=1)
    angle = float(config.get("theta_pol", 0.))
    line = reference_axis(ex, r) * np.cos(angle)
    if abs(np.sin(angle)) > 1e-12:
        ey, ry, zy = mesh(series, iteration, "E", "y", mode=1)
        if not np.array_equal(r, ry) or not np.array_equal(z, zy):
            raise ValueError(f"Optical grids differ at iteration {iteration}.")
        line += reference_axis(ey, r) * np.sin(angle)
    dz = float(np.mean(np.diff(z)))
    if not np.allclose(np.diff(z), dz, rtol=1e-7, atol=1e-15):
        raise ValueError("Optical filtering requires uniform z spacing.")
    k0 = 2 * np.pi / config["lambda0"]
    k = 2 * np.pi * np.fft.rfftfreq(len(z), dz)
    spectrum = np.fft.rfft(line - line.mean())
    spectrum[(k < .25 * k0) | (k > 2.5 * k0)] = 0.
    envelope = abs(hilbert(np.fft.irfft(spectrum, n=len(z))))
    if not np.isfinite(envelope).all() or envelope.max() <= 0:
        raise ValueError(f"Invalid optical envelope at iteration {iteration}.")
    a2 = gaussian_filter1d(.5 * (envelope / (MC2_MEV * 1e6 * k0))**2, .1e-6 / dz)
    return z, a2


def first_bucket(xi_um, ez_gvm, grid_low_um, grid_high_um, reference_ne_m3):
    """Nearest complete period between robust negative-to-positive Ez crossings."""
    xi, ez = np.asarray(xi_um, float), np.asarray(ez_gvm, float)
    dz = float(np.median(np.diff(xi)))
    if dz <= 0 or not np.allclose(np.diff(xi), dz, rtol=1.e-6, atol=1.e-9):
        raise ValueError('Common wake selection requires uniform longitudinal cells.')
    smooth = gaussian_filter1d(ez, .2/dz, mode='nearest')
    amplitude = float(np.max(abs(smooth[xi < 0]), initial=0.))
    if amplitude <= 0 or not np.isfinite(amplitude):
        raise ValueError('No longitudinal wake resolved behind the laser.')
    period = 2*np.pi*c*np.sqrt(epsilon_0*m_e/(reference_ne_m3*e*e))*1.e6
    roots = []
    for j in np.flatnonzero((smooth[:-1] <= 0) & (smooth[1:] > 0))[::-1]:
        root = float(xi[j]-smooth[j]*dz/(smooth[j+1]-smooth[j]))
        left = smooth[(xi >= root-period/8) & (xi < root)]
        right = smooth[(xi > root) & (xi <= root+period/8)]
        if root >= 0 or not len(left) or not len(right):
            continue
        if left.min() >= -.03*amplitude or right.max() <= .03*amplitude:
            continue
        if roots and roots[-1]-root < period/4:
            continue
        roots.append(root)
    if len(roots) < 2:
        raise ValueError('First complete wake period is not resolved in the saved field window.')
    high, low = roots[:2]
    if low < grid_low_um or high > min(0., grid_high_um) or high <= low:
        raise ValueError('The first wake period is clipped by the saved field window.')
    return low, high


@dataclass
class Run:
    case: str
    path: Path
    config: dict
    fields: object = None
    history: object = None
    frames: dict = field(default_factory=dict)

    def series(self, full=False):
        from openpmd_viewer import OpenPMDTimeSeries
        name = "history" if full else "fields"
        if getattr(self, name) is None:
            folder = self.path/("particle_history" if full else "diags")/"hdf5"
            if not folder.is_dir():
                raise FileNotFoundError(f"Missing {folder}. Use the project containing the simulation data.")
            data = OpenPMDTimeSeries(str(folder), check_all_files=False, backend="h5py")
            periods = self.config["diagnostic_schedule"]
            expected = [i for i in range(self.config["N_step"]) if any(
                start <= i < stop and i % (hp if full else fp) == 0 for start, stop, fp, hp in periods)]
            missing = np.setdiff1d(expected, data.iterations)
            if (not len(data.iterations) or len(missing) or np.any(np.diff(data.iterations) <= 0)
                    or not np.allclose(data.t, np.asarray(data.iterations)*self.config["dt"], rtol=1.e-7, atol=1.e-18)):
                raise ValueError(f"Missing snapshots or inconsistent times in {folder}.")
            setattr(self, name, data)
        return getattr(self, name)

    def frame(self, iteration):
        iteration = int(iteration)
        if iteration not in self.frames:
            series = self.series()
            index = np.flatnonzero(series.iterations == iteration)
            if len(index) != 1:
                raise ValueError(f"{self.case}: iteration {iteration} has no saved field.")
            _, r, z = mesh(series, iteration, "E", "z")
            self.frames[iteration] = dict(iteration=iteration, time_fs=float(series.t[index[0]]*1.e15),
                laser_z_um=laser_peak(series, iteration, self.config),
                rmax_um=float(max(abs(edges(r)[[0, -1]]))*1.e6),
                zmin_um=float(edges(z)[0]*1.e6), zmax_um=float(edges(z)[-1]*1.e6))
        return self.frames[iteration]


def load_runs(root):
    runs = {}
    for case in CASES:
        path = Path(root)/"downramp_comparison"/case
        config = json.loads((path/"run_config.json").read_text())
        completed = json.loads((path/"completed.json").read_text())
        if config["case"] != case or completed["case"] != case or completed["iteration"] < config["N_step"]:
            raise ValueError(f"{path} does not contain the completed {case} run.")
        runs[case] = Run(case, path, config)
    different = {k for k in set(runs["R8"].config) | set(runs["C4"].config)
                 if runs["R8"].config.get(k) != runs["C4"].config.get(k)}
    if different-{"case", "lower_density_factor"}:
        raise ValueError(f"Unexpected run differences: {sorted(different)}")
    if not 0 < runs["R8"].config["lower_density_factor"] < 1 or runs["C4"].config["lower_density_factor"] != 1:
        raise ValueError("Expected an R8 density drop and a constant-density C4 control.")
    return runs


def read_csv(path):
    """Preserve integer IDs; blank numeric cells remain gaps, never zeros."""
    rows = []
    text_fields = {'case', 'selection', 'source', 'method', 'seed_iterations', 'status', 'reason'}
    with Path(path).open(newline='', encoding='utf-8') as stream:
        for raw in csv.DictReader(stream):
            row = {}
            for key, value in raw.items():
                if key in text_fields:
                    row[key] = value
                elif value in ('True', 'False'):
                    row[key] = value == 'True'
                elif value == '':
                    row[key] = np.nan
                else:
                    try:
                        row[key] = int(value)
                    except ValueError:
                        try:
                            row[key] = float(value)
                        except ValueError:
                            row[key] = value
            rows.append(row)
    return rows


def write_csv(path, rows):
    rows = list(rows)
    if not rows:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows({k: "" if isinstance(v, (float, np.floating)) and not np.isfinite(v) else v
                         for k, v in row.items()} for row in rows)
