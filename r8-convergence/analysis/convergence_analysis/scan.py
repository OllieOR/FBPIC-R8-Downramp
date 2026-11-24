"""One pass over each run, retaining tables and five snapshots for replotting."""

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.constants import c, e

from . import beam, laser
from .common import (edges, mesh, on_axis, particles, read_csv, result_path,
                     same_grid, write_csv, write_json)


def fingerprint(run, iterations, snapshots, radius):
    sources = [(p.name, p.stat().st_size, p.stat().st_mtime_ns)
               for p in sorted((run.path/'diags'/'hdf5').glob('*.h5'))]
    implementation = hashlib.sha256(b''.join(p.read_bytes() for p in
        sorted(Path(__file__).parent.glob('*.py')))).hexdigest()
    payload = dict(config=run.config, files=sources, implementation=implementation,
                   iterations=list(map(int, iterations)), snapshots=list(map(int, snapshots)), radius=radius)
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def files(output, case):
    return dict(
        beam=result_path(output, '03_electrons', 'data', f'{case}_beam.csv'),
        maximum=result_path(output, '03_electrons', 'data', f'{case}_maximum_energy.csv'),
        laser=result_path(output, '02_laser', 'data', f'{case}_laser.csv'),
        wake=result_path(output, '01_wake', 'data', f'{case}_wake.csv'),
        snapshots=result_path(output, '01_wake', 'data', f'{case}_snapshots.npz'),
        cache=Path(output)/'settings'/f'{case}_cache.json')


def load_cached(output, case):
    paths = files(output, case)
    tables = {key: read_csv(paths[key]) for key in ('beam', 'maximum', 'laser', 'wake')}
    with np.load(paths['snapshots'], allow_pickle=False) as data:
        tables['snapshots'] = {key: data[key].copy() for key in data.files}
    return tables


def scan(run, iterations, snapshots, output, radius=2., force=False):
    paths = files(output, run.case)
    stamp = fingerprint(run, iterations, snapshots, radius)
    if (not force and all(p.is_file() for p in paths.values())
            and json.loads(paths['cache'].read_text()).get('fingerprint') == stamp):
        print(f'{run.case}: reusing verified analysis tables.', flush=True)
        return load_cached(output, run.case)
    result = dict(beam=[], maximum=[], laser=[], wake=[], snapshots={})
    series = run.series()
    for index, iteration in enumerate(iterations):
        iteration = int(iteration)
        time_ps = float(iteration*run.config['dt']*1.e12)
        ez, r, z = mesh(series, iteration, 'E', 'z', mode=0)
        ex, rr, zz = mesh(series, iteration, 'E', 'x', mode='all')
        same_grid(r, z, rr, zz)
        angle = float(run.config.get('theta_pol', 0.))
        optical = on_axis(ex, r)*np.cos(angle)
        if abs(np.sin(angle)) > 1.e-12:
            ey, rr, zz = mesh(series, iteration, 'E', 'y', mode='all')
            same_grid(r, z, rr, zz)
            optical += on_axis(ey, r)*np.sin(angle)
        wake = dict(z_um=z*1.e6, ez_axis_gvm=on_axis(ez, r)/1.e9, optical_axis=optical)
        frame = dict(rmax_um=float(max(abs(edges(r)[[0, -1]]))*1.e6),
                     zmin_um=float(edges(z)[0]*1.e6), zmax_um=float(edges(z)[-1]*1.e6))
        data = particles(series, iteration)
        validations = {}
        for cut in (20., 5.):
            rows, validation = beam.measure(run.case, iteration, time_ps, run.config,
                                            frame, data, wake, cut=cut, radius=radius)
            result['beam'].extend(rows)
            validations[cut] = validation
        validation = validations[20.]
        laser_z = validation['laser']
        xi = wake['z_um']-laser_z
        # This fixed window is the same one used in the established R8 wake analysis.
        selected_wake = (xi >= -30.) & (xi <= -3.)
        smooth_ez = wake['ez_axis_gvm']
        from scipy.ndimage import gaussian_filter1d
        smooth_ez = gaussian_filter1d(smooth_ez, .2/float(np.median(np.diff(xi))))
        minimum = float(np.min(smooth_ez[selected_wake])) if selected_wake.any() else np.nan
        low, high = validation['low'], validation['high']
        result['wake'].append(dict(case=run.case, iteration=iteration, time_ps=time_ps,
            accelerating_field_GVm=-min(0., minimum), first_bucket_length_um=high-low,
            primary_status=validation['status'], laser_peak_z_um=laser_z))
        mask = validations[5.]['selected']
        # All-forward maximum is independent of the 5/20 MeV beam cuts.
        forward = ((data['uz'] >= 1.) & (data['q'] > 0)
                   & (np.hypot(data['x'], data['y']) <= frame['rmax_um'])
                   & (data['z'] >= frame['zmin_um']) & (data['z'] <= frame['zmax_um']))
        # The percentile is taken from the same forward population as the maximum.
        order = np.argsort(data['K'][forward])
        qf, kf = data['q'][forward][order], data['K'][forward][order]
        percentile = float(kf[min(np.searchsorted(np.cumsum(qf), .99*qf.sum()), len(kf)-1)]) if len(kf) else np.nan
        result['maximum'].append(dict(case=run.case, iteration=iteration, time_ps=time_ps,
            maximum_K_MeV=float(kf.max()) if len(kf) else np.nan, percentile99_K_MeV=percentile))
        row = laser.measure(run, iteration)
        row['energy_fraction_of_initial'] = row['optical_energy_mJ']/(result['laser'][0]['optical_energy_mJ'] if result['laser'] else row['optical_energy_mJ'])
        result['laser'].append(row)
        if iteration in snapshots:
            rho, rr, zz = mesh(series, iteration, 'rho', mode='all', theta=0.)
            same_grid(r, z, rr, zz)
            # Keep the maps and >=5 MeV records only; no full particle histories.
            prefix = f'i{iteration}_'
            snapshot = dict(r_um=r*1.e6, z_um=z*1.e6, density_n0=-rho/e/run.config['n_e'],
                ez_GVm=ez/1.e9, ez_axis_GVm=wake['ez_axis_gvm'],
                laser_z_um=np.array(laser_z), time_ps=np.array(time_ps),
                fixed_origin_um=np.array((run.config['z0']+c*iteration*run.config['dt'])*1.e6),
                bucket_low_um=np.array(low), bucket_high_um=np.array(high))
            snapshot.update({key: data[key][mask] for key in ('x', 'y', 'z', 'ux', 'uy', 'uz', 'K', 'q')})
            result['snapshots'].update({prefix+k: v for k, v in snapshot.items()})
        if index % 20 == 0 or index == len(iterations)-1:
            print(f'{run.case}: {index+1}/{len(iterations)} snapshots; iteration {iteration}', flush=True)
    for key in ('beam', 'maximum', 'laser', 'wake'):
        write_csv(paths[key], result[key])
    np.savez_compressed(paths['snapshots'], **result['snapshots'])
    write_json(paths['cache'], dict(fingerprint=stamp, complete=True))
    return result
