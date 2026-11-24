"""Common instantaneous beam definition, units and plots for all configurations.

This file is identical in both analysis packages. It combines physical
particle records, never particle identities or precomputed species moments.
Wake windows and transverse acceptance are geometric selections, not trapping
conditions. The first complete wake period is the primary quality population.
"""

import numpy as np
from scipy.constants import c, e, epsilon_0, m_e
from scipy.ndimage import gaussian_filter1d

DEFINITION_VERSION = '1.0'
MIN_EFFECTIVE = 20
POPULATIONS = ('all_saved', 'all_saved_compact', 'first_bucket', 'first_bucket_compact', 'outside_first_bucket')
XKEY = 'laser_position_from_entrance_um'
XLABEL = 'Laser position relative to plasma entrance (µm)'


def geometry(config):
    ion = 'n_He' in config
    start = float(config.get('p_zmin', 0.) if ion else config['ramp_start'])*1.e6
    nref = (config.get('helium_level_start', 1)*config['n_He']+
            config.get('nitrogen_level_start', 5)*config['n_N']) if ion else config['n_e']
    return dict(entrance_z_um=start, entrance_length_um=float(config['ramp_length'])*1.e6,
                focus_z_um=float(config.get('z_foc', config.get('zf', 0.)))*1.e6,
                initial_laser_z_um=float(config.get('z0', np.nan))*1.e6, reference_ne_m3=float(nref))


def optical_peak(line, z_um, wavelength_m):
    """Same all-mode, on-axis, polarisation-projected optical peak in both runs."""
    line, z = np.asarray(line, float), np.asarray(z_um, float)
    dz = float(np.mean(np.diff(z)))*1.e-6
    if len(z) < 8 or dz <= 0 or not np.allclose(np.diff(z)*1.e-6, dz, rtol=1.e-7, atol=1.e-15):
        raise ValueError('Common optical reference requires uniform longitudinal cells.')
    ratio = np.fft.fftfreq(len(z), dz)*wavelength_m
    if .5/dz*wavelength_m < 2.5:
        raise ValueError('The saved grid does not resolve the common optical filter.')
    spectrum = np.fft.fft(line)
    spectrum[(ratio < .25) | (ratio > 2.5)] = 0.
    envelope = abs(np.fft.ifft(2*spectrum))
    j = int(np.argmax(envelope))
    if not np.all(np.isfinite(envelope)) or envelope[j] <= 0:
        raise ValueError('No resolved optical envelope for the common reference.')
    return float(z[j]), bool(min(j, len(z)-1-j) < 3)


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


def moment(values, q):
    values, q = np.asarray(values, np.longdouble), np.asarray(q, np.longdouble)
    if q.sum() <= 0:
        return np.nan, np.nan
    mean = np.sum(q*values)/q.sum()
    return float(mean), float(np.sqrt(np.sum(q*(values-mean)**2)/q.sum()))


def emittance(position_um, u, q):
    if len(q) < 2 or np.sum(q) <= 0:
        return np.nan
    x, u, q = (np.asarray(a, np.longdouble) for a in (position_um, u, q))
    x, u = x-np.sum(q*x)/q.sum(), u-np.sum(q*u)/q.sum()
    xx, uu, xu = (np.sum(q*a)/q.sum() for a in (x*x, u*u, x*u))
    return float(np.sqrt(max(np.longdouble(0.), xx*uu-xu*xu)))


def statistics(data, keep):
    q, energy = data['q'][keep], data['K'][keep]
    Q = float(q.sum())
    mean, sigma = moment(energy, q)
    result = dict(N_macroparticles=int(keep.sum()), effective_macroparticles=float(Q*Q/np.sum(q*q)) if Q > 0 else 0.,
                  charge_pC=Q, mean_K_MeV=mean, rms_spread_MeV=sigma,
                  relative_rms_spread=sigma/mean if mean > 0 else np.nan,
                  maximum_K_MeV=float(energy.max()) if len(energy) else np.nan,
                  nitrogen_charge_pC=float(q[data['nitrogen'][keep]].sum()))
    for axis in 'xy':
        x, u = data[axis][keep], data['u'+axis][keep]
        result[f'rms_{axis}_um'] = moment(x, q)[1]
        result[f'rms_divergence_{axis}_mrad'] = moment(np.arctan2(u, data['uz'][keep])*1000, q)[1]
        result[f'normalised_emittance_{axis}_mm_mrad'] = emittance(x, u, q)
    return result


def measure(case, iteration, time_ps, config, frame, data, wake, cut=20., radius=2.):
    """Arrays use µm, MeV, pC and normalised momentum. IDs are not inputs."""
    if not np.isfinite(cut) or cut <= 0 or not np.isfinite(radius) or radius <= 0:
        raise ValueError('Common beam cuts must be finite and positive.')
    n = len(data['q'])
    for key in ('x', 'y', 'z', 'ux', 'uy', 'uz', 'K', 'q', 'nitrogen'):
        if len(data[key]) != n or not np.all(np.isfinite(data[key])):
            raise ValueError(f'Invalid common beam particle array: {key}.')
    if np.any(data['q'] < 0):
        raise ValueError('Common beam analysis cannot use negative charge weights.')
    g = geometry(config)
    laser, edge = optical_peak(wake['optical_axis'], wake['z_um'], config['lambda0'])
    xi = data['z']-laser
    rr = np.hypot(data['x'], data['y'])
    inside = (rr <= frame['rmax_um']) & (data['z'] >= frame['zmin_um']) & (data['z'] <= frame['zmax_um'])
    selected = inside & (data['q'] > 0) & (data['uz'] >= 1.) & (data['K'] >= cut)
    compact = rr <= radius
    low = high = np.nan
    reason, status = '', 'resolved'
    try:
        if edge:
            raise ValueError('Optical reference peak lies at the edge of the saved window.')
        low, high = first_bucket(wake['z_um']-laser, wake['ez_axis_gvm'],
                                frame['zmin_um']-laser, frame['zmax_um']-laser, g['reference_ne_m3'])
    except ValueError as error:
        status, reason = 'unresolved', str(error)
    primary = selected & (xi >= low) & (xi <= high)
    masks = dict(all_saved=selected, all_saved_compact=selected & compact, first_bucket=primary,
                 first_bucket_compact=primary & compact, outside_first_bucket=selected & ~primary)
    base = dict(case=case, iteration=int(iteration), time_ps=float(time_ps), energy_cut_MeV=float(cut),
                acceptance_radius_um=float(radius), laser_peak_z_um=laser,
                laser_position_from_entrance_um=laser-g['entrance_z_um'] if not edge else np.nan,
                laser_position_from_focus_um=laser-g['focus_z_um'],
                laser_travel_from_input_um=laser-g['initial_laser_z_um'], window_travel_um=c*time_ps*1.e-6,
                grid_rear_xi_um=frame['zmin_um']-laser, grid_front_xi_um=frame['zmax_um']-laser,
                primary_lower_xi_um=low, primary_upper_xi_um=high,
                primary_status=status, primary_reason=reason, coordinate_valid=not edge)
    rows = []
    for population in POPULATIONS:
        values = statistics(data, masks[population])
        if status != 'resolved' and population not in ('all_saved', 'all_saved_compact'):
            values = {key: None if key == 'N_macroparticles' else np.nan for key in values}
        rows.append(dict(base, population=population, **values, compact_fraction=np.nan,
                         charge_fraction_of_all_saved=np.nan))
    by = {r['population']: r for r in rows}
    total, small = by['all_saved']['charge_pC'], by['all_saved_compact']['charge_pC']
    if small > total+max(1.e-10, total*1.e-11):
        raise ValueError('Compact charge exceeds the saved selection.')
    by['all_saved']['compact_fraction'] = small/total if total > 0 else np.nan
    if status == 'resolved':
        Q, Qc = by['first_bucket']['charge_pC'], by['first_bucket_compact']['charge_pC']
        if not np.isclose(Q+by['outside_first_bucket']['charge_pC'], total, rtol=1.e-11, atol=1.e-9) or Qc > Q+1.e-9:
            raise ValueError('Common primary-window charge accounting failed.')
        by['first_bucket']['compact_fraction'] = Qc/Q if Q > 0 else np.nan
    for row in rows:
        row['charge_fraction_of_all_saved'] = row['charge_pC']/total if total > 0 else np.nan
    validation = dict(case=case, iteration=int(iteration), laser=laser, low=low, high=high,
                      data=data, selected=selected, wake=wake, coordinate=base[XKEY], status=status)
    return rows, validation

