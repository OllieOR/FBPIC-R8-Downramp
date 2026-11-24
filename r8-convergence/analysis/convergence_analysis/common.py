"""Validated inputs and small I/O helpers. Simulation files are opened read-only."""

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.constants import c, e, m_e

CASE_SETTINGS = {'R8_Nm2_nt8': (2, 8), 'R8_Nm2_nt16': (2, 16),
                 'R8_Nm3_nt16': (3, 16), 'R8_Nm4_nt16': (4, 16)}
CASES = tuple(CASE_SETTINGS)
LABELS = {case: f'Nm={nm}, p_nt={nt}' for case, (nm, nt) in CASE_SETTINGS.items()}
COLORS = dict(zip(CASES, ('#28688c', '#d17b25', '#32845b', '#8856a7')))
STYLES = dict(zip(CASES, ('-', '--', '-.', ':')))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f'No rows for {path}')
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: '' if isinstance(v, (float, np.floating)) and not np.isfinite(v) else v
                             for k, v in row.items()})


def read_csv(path):
    with Path(path).open(newline='', encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        for key, value in row.items():
            if value in ('True', 'False'):
                row[key] = value == 'True'
            elif not value:
                row[key] = np.nan
            else:
                try:
                    row[key] = float(value)
                except ValueError:
                    pass
    return rows


def edges(x):
    x = np.asarray(x)
    return np.r_[x[0]-(x[1]-x[0])/2, (x[:-1]+x[1:])/2, x[-1]+(x[-1]-x[-2])/2]


def mesh(series, iteration, name, coord=None, mode=0, theta=0.):
    kwargs = dict(field=name, iteration=int(iteration), m=mode, theta=theta)
    if coord is not None:
        kwargs['coord'] = coord
    values, info = series.get_field(**kwargs)
    values, r, z = np.asarray(values, float), np.asarray(info.r), np.asarray(info.z)
    if values.shape == (len(z), len(r)) and len(r) != len(z):
        values = values.T
    if (values.shape != (len(r), len(z)) or min(values.shape) < 2
            or np.any(np.diff(r) <= 0) or np.any(np.diff(z) <= 0)
            or not np.all(np.isfinite(values))):
        raise ValueError(f'Invalid {name}/{coord} mesh at iteration {iteration}.')
    return values, r, z


def on_axis(values, r):
    if not r[0] < 0 < r[-1]:
        raise ValueError('Expected a signed-r field plane bracketing the axis.')
    j = int(np.searchsorted(r, 0.))
    f = -r[j-1]/(r[j]-r[j-1])
    return values[j-1]*(1-f)+values[j]*f


def same_grid(r, z, rr, zz):
    if not np.array_equal(r, rr) or not np.array_equal(z, zz):
        raise ValueError('Field components do not share one grid.')


def particles(series, iteration):
    names = ('x', 'y', 'z', 'ux', 'uy', 'uz', 'w')
    raw = series.get_particle(var_list=list(names), species='electrons', iteration=int(iteration))
    p = dict(zip(names, (np.asarray(v, float) for v in raw)))
    n = len(p['w'])
    if any(v.ndim != 1 or len(v) != n or not np.all(np.isfinite(v)) for v in p.values()):
        raise ValueError(f'Invalid particle records at iteration {iteration}.')
    if np.any(p['w'] < 0):
        raise ValueError('Negative macroparticle weights.')
    data = {k: p[k]*1.e6 if k in ('x', 'y', 'z') else p[k] for k in names[:-1]}
    data.update(q=p['w']*e*1.e12,
                K=(np.sqrt(1+p['ux']**2+p['uy']**2+p['uz']**2)-1)*m_e*c*c/e/1.e6,
                nitrogen=np.zeros(n, bool))
    # No IDs are used: labels cannot identify corresponding electrons across resolutions.
    return data


@dataclass
class Run:
    case: str
    path: Path
    config: dict
    fields: object = None

    def series(self):
        if self.fields is None:
            from openpmd_viewer import OpenPMDTimeSeries
            self.fields = OpenPMDTimeSeries(str(self.path/'diags'/'hdf5'),
                                           check_all_files=False, backend='h5py')
        return self.fields


def expected_iterations(config):
    return np.array([i for i in range(int(config['N_step'])) if any(
        start <= i < stop and i % fp == 0
        for start, stop, fp, hp in config['diagnostic_schedule'])], dtype=int)


def load_runs(project, baseline):
    paths = (baseline, *(project/'runs'/case for case in CASES[1:]))
    runs = []
    for case, path in zip(CASES, paths):
        nm, nt = CASE_SETTINGS[case]
        if not (path/'run_config.json').is_file():
            raise ValueError(f'Missing {path}/run_config.json. Use --baseline or --project to set the path.')
        config = json.loads((path/'run_config.json').read_text())
        if config.get('case') != 'R8' or config.get('Nm') != nm or config.get('p_nt') != nt:
            raise ValueError(f'{path}: expected R8, Nm={nm}, p_nt={nt}.')
        completed_path = path/'completed.json'
        if not completed_path.is_file():
            raise ValueError(f'{path}: missing completed.json; this package requires completed runs.')
        completed = json.loads(completed_path.read_text())
        if (completed.get('case') != 'R8' or completed.get('iteration') != config['N_step']
                or abs(completed.get('simulation_time_s', -1)-config['N_step']*config['dt']) > .01*config['dt']):
            raise ValueError(f'{path}: completion record does not match the full configured run.')
        runs.append(Run(case, path, config))
    base = runs[0].config
    for run in runs[1:]:
        changed = [k for k in set(base)|set(run.config)
                   if k not in ('Nm', 'p_nt') and base.get(k) != run.config.get(k)]
        if changed:
            raise ValueError(f'{run.case}: unexpected configuration differences: {sorted(changed)}')
    expected = expected_iterations(base)
    if not len(expected) or int(base['N_step']) != 10800:
        raise ValueError('Expected the full 10,800-step R8 study.')
    for run in runs:
        ts = run.series()
        actual = np.asarray(ts.iterations, int)
        missing = np.setdiff1d(expected, actual)
        if len(missing):
            raise ValueError(f'{run.case}: {len(missing)} missing diagnostics, first: {missing[:10].tolist()}.')
        if (np.any(np.diff(actual) <= 0) or
                not np.allclose(ts.t, actual*base['dt'], rtol=1.e-8, atol=1.e-18)):
            raise ValueError(f'{run.case}: inconsistent diagnostic times.')
        if 'electrons' not in (ts.avail_species or []):
            raise ValueError(f'{run.case}: no electrons in selected diagnostics.')
        for field in ('E', 'B', 'rho'):
            if field not in (ts.avail_fields or []):
                raise ValueError(f'{run.case}: missing {field} diagnostics.')
            meta = ts.fields_metadata[field]
            modes = {str(m) for m in meta.get('avail_circ_modes', [])}
            if meta.get('geometry') != 'thetaMode' or not all(str(m) in modes for m in range(run.config['Nm'])):
                raise ValueError(f'{run.case}: {field} modes do not match the saved configuration.')
    return runs, expected


def result_path(output, section, kind, name):
    folder = Path(output)/kind/section if kind == 'settings' else Path(output)/section/kind
    path = folder/name
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
