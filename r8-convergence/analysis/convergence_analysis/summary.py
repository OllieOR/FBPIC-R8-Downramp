"""Compare the R8 runs at matching saved times."""

import numpy as np

from .common import CASES, COLORS, LABELS, STYLES, result_path, write_csv, write_json
from .plots import beam_rows, save, snapshot, values

PAIRS = ((CASES[0], CASES[1], 'Particle sampling: p_nt 8 → 16, Nm=2'),
         (CASES[1], CASES[2], 'Azimuthal modes: Nm 2 → 3, p_nt=16'),
         (CASES[2], CASES[3], 'Azimuthal modes: Nm 3 → 4, p_nt=16'))
PAIR_LABELS = ('p_nt: 8 → 16', 'Nm: 2 → 3', 'Nm: 3 → 4')
METRICS = (
    ('beam', 'charge_pC', 'Main accelerated charge ≥20 MeV', 'pC', False),
    ('core', 'charge_pC', 'Near-axis main accelerated charge ≥20 MeV', 'pC', False),
    ('beam', 'mean_K_MeV', 'Main population mean energy', 'MeV', True),
    ('beam', 'relative_rms_spread', 'Main population relative RMS energy spread', 'fraction', True),
    ('beam', 'normalised_emittance_x_mm_mrad', 'Normalised emittance x', 'mm mrad', True),
    ('beam', 'normalised_emittance_y_mm_mrad', 'Normalised emittance y', 'mm mrad', True),
    ('beam', 'rms_divergence_x_mrad', 'RMS divergence x', 'mrad', True),
    ('beam', 'rms_divergence_y_mrad', 'RMS divergence y', 'mrad', True),
    ('maximum', 'maximum_K_MeV', 'Maximum forward energy', 'MeV', False),
    ('maximum', 'percentile99_K_MeV', '99th percentile forward energy', 'MeV', False),
    ('laser', 'optical_energy_mJ', 'Saved optical energy', 'mJ', False),
    ('laser', 'spot_radius_um', 'Laser spot radius', 'µm', False),
    ('wake', 'accelerating_field_GVm', 'Peak accelerating wake field', 'GV/m', False),
    ('wake', 'first_bucket_length_um', 'First complete wake period length', 'µm', False),
    ('lowcut', 'charge_pC', 'Main accelerated charge ≥5 MeV', 'pC', False),
)


def series(tables, case, table, key, quality):
    if table in ('beam', 'core', 'lowcut'):
        rows = beam_rows(tables, case, 'first_bucket_compact' if table == 'core' else 'first_bucket',
                         5. if table == 'lowcut' else 20.)
    else:
        rows = sorted(tables[case][table], key=lambda row: row['iteration'])
    return np.array([r['iteration'] for r in rows], int), np.array([r['time_ps'] for r in rows]), values(rows, key, quality)


def comparison(t, a, b):
    """Time weights respect the dense downramp diagnostics and missing moments."""
    valid = np.isfinite(a) & np.isfinite(b)
    delta = b-a
    scale = float(np.max(abs(a[np.isfinite(a)]), initial=0.))
    relative = np.full(len(t), np.nan)
    use = valid & (abs(a) >= .01*scale) & (abs(a) > 0.)
    relative[use] = 100.*delta[use]/a[use]
    dt = np.diff(t)
    segments = valid[:-1] & valid[1:]
    error = float(np.sum(.5*(delta[:-1][segments]**2+delta[1:][segments]**2)*dt[segments]))
    reference = float(np.sum(.5*(a[:-1][segments]**2+a[1:][segments]**2)*dt[segments]))
    nrmse = 100*np.sqrt(error/reference) if reference > 0 else np.nan
    duration = float(dt[segments].sum())
    return valid, delta, relative, nrmse, duration/(t[-1]-t[0]) if t[-1] > t[0] else np.nan


def fmt(value, spec='.3g', suffix=''):
    return f'{value:{spec}}{suffix}' if np.isfinite(value) else '—'


def make_summary(tables, output, record):
    import matplotlib.pyplot as plt
    summaries, differences = [], []
    for first, second, label in PAIRS:
        for table, key, name, unit, quality in METRICS:
            its, t, a = series(tables, first, table, key, quality)
            its2, t2, b = series(tables, second, table, key, quality)
            if not np.array_equal(its, its2) or not np.allclose(t, t2, rtol=0, atol=1.e-12):
                raise ValueError('Summary requires exact matched iterations and times.')
            valid, delta, relative, nrmse, coverage = comparison(t, a, b)
            endpoint = bool(valid[-1])
            summaries.append(dict(comparison=label, reference_case=first, comparison_case=second,
                metric=key, population=table, description=name, unit=unit,
                n_valid=int(valid.sum()), n_total=len(t), valid_time_fraction=coverage,
                time_weighted_NRMSE_percent=nrmse, final_iteration=int(its[-1]), final_time_ps=float(t[-1]),
                final_reference=float(a[-1]), final_comparison=float(b[-1]),
                final_absolute_difference=float(delta[-1]) if endpoint else np.nan,
                final_relative_difference_percent=float(relative[-1]) if endpoint else np.nan,
                maximum_absolute_difference=float(np.max(abs(delta[valid]))) if valid.any() else np.nan))
            differences.extend(dict(comparison=label, reference_case=first, comparison_case=second,
                metric=key, population=table, iteration=int(it), time_ps=float(time), reference_value=float(x),
                comparison_value=float(y), absolute_difference=float(d), relative_difference_percent=float(rel))
                for it, time, x, y, d, rel in zip(its, t, a, b, delta, relative))
    write_csv(result_path(output, '04_convergence', 'data', 'convergence_summary.csv'), summaries)
    write_csv(result_path(output, '04_convergence', 'data', 'matched_differences.csv'), differences)
    metrics = [METRICS[i] for i in (0, 2, 4, 5, 10, 11)]
    fig, axs = plt.subplots(3, 2, figsize=(12, 10), sharex=True, layout='constrained')
    for ax, (table, key, name, unit, quality) in zip(axs.flat, metrics):
        for index, (first, second, label) in enumerate(PAIRS):
            _, t, a = series(tables, first, table, key, quality)
            _, _, b = series(tables, second, table, key, quality)
            relative = comparison(t, a, b)[2]
            ax.plot(t, relative, lw=1.8, color=COLORS[second],
                    ls=STYLES[second], label=PAIR_LABELS[index])
        ax.set_title(name, fontsize=11)
        ax.axhline(0, color='grey', lw=.6)
        ax.set_ylabel('Difference (%)')
        ax.grid(alpha=.2)
    axs[0, 0].legend(frameon=False, fontsize=9)
    for ax in axs[-1]:
        ax.set_xlabel('Time (ps)')
    fig.suptitle('Differences at the same saved times\nRelative differences omitted when reference <1% of its peak', fontsize=13)
    save(fig, result_path(output, '04_convergence', 'figures', 'relative_differences.png'))
    wake_errors = []
    for it in record['snapshots']:
        for first, second, label in PAIRS:
            a, b = snapshot(tables, first, it), snapshot(tables, second, it)
            xa, xb = a['z_um']-a['fixed_origin_um'], b['z_um']-b['fixed_origin_um']
            common = (xa >= max(xa[0], xb[0])) & (xa <= min(xa[-1], xb[-1]))
            x, fa = xa[common], a['ez_axis_GVm'][common]
            fb = np.interp(x, xb, b['ez_axis_GVm'])
            if len(x) < 2:
                raise ValueError('No common longitudinal extent for wake lineout comparison.')
            integral = getattr(np, 'trapezoid', None) or np.trapz
            denominator = float(integral(fa*fa, x))
            error = 100*np.sqrt(float(integral((fb-fa)**2, x))/denominator) if denominator > 0 else np.nan
            wake_errors.append(dict(comparison=label, iteration=it, time_ps=float(a['time_ps']),
                unshifted_profile_relative_L2_percent=error, coordinate='z - (z0+c*t)',
                common_xi_min_um=float(x[0]), common_xi_max_um=float(x[-1])))
    write_csv(result_path(output, '04_convergence', 'data', 'wake_profile_errors.csv'), wake_errors)
    issues = []
    for case in CASES:
        rows = beam_rows(tables, case, 'first_bucket')
        issues.append(dict(case=case,
            unresolved_bucket_snapshots=sum(r['primary_status'] != 'resolved' for r in rows),
            low_effective_count_snapshots=sum(not np.isfinite(r['effective_macroparticles']) or r['effective_macroparticles'] < 20 for r in rows),
            optical_boundary_flag_snapshots=sum(bool(r['boundary_flag']) for r in tables[case]['laser']),
            maximum_optical_band_sensitivity=float(max(r['relative_cutoff_spread'] for r in tables[case]['laser']))))
    write_json(result_path(output, '04_convergence', 'settings', 'comparison_settings.json'), dict(
        time_alignment='Exact same saved iterations and physical times; no temporal interpolation.',
        relative_difference='100 * (comparison-reference) / reference, omitted if |reference| <1% of its finite peak or reference is zero.',
        NRMSE='100 * sqrt(integral((comparison-reference)^2 dt) / integral(reference^2 dt)); trapezoids only between adjacent valid diagnostics.',
        quality='Beam moments require N_eff >=20 in each run. No quality cut on charge. CSV includes finite sample counts and valid time coverage.',
        waveforms='L2 difference on common fixed moving-coordinate extent; only spatial interpolation; no phase fitting.',
        comparisons=[dict(reference_case=a, comparison_case=b, description=label) for a, b, label in PAIRS],
        scope='Angular sampling and mode-count sensitivity at fixed grid and timestep, with one random seed.',
        checks=issues))
    text = ['# R8 convergence comparison', '',
        'Four completed R8 runs, with only azimuthal particle sampling and retained modes changed. '
        'The spatial grid, timestep, density profile, input laser and seed are checked for equality.', '',
        'The main accelerated population uses forward electrons in the first complete wake period and K ≥20 MeV. '
        f"Near-axis charge applies r ≤{record['core_radius_um']:g} µm. Moments require at least 20 effective macroparticles. "
        'The particle selection is evaluated independently at each saved iteration.', '',
        'NRMSE uses time-weighted squared differences over adjacent valid diagnostics. '
        'The percentage is normalised by the reference signal RMS. '
        'Missing windows and insufficient particle counts remain undefined.', '']
    for first, second, label in PAIRS:
        text += ['## '+label, '', '| Quantity | Time-weighted NRMSE | Final reference | Final comparison | Final difference | Valid time |',
                 '|---|---:|---:|---:|---:|---:|']
        for row in summaries:
            if row['reference_case'] == first and row['comparison_case'] == second:
                text.append(f"| {row['description']} ({row['unit']}) | {fmt(row['time_weighted_NRMSE_percent'], suffix='%')} | "
                    f"{fmt(row['final_reference'])} | {fmt(row['final_comparison'])} | "
                    f"{fmt(row['final_relative_difference_percent'], suffix='%')} | "
                    f"{fmt(row['valid_time_fraction']*100, suffix='%')} |")
        text.append('')
    text += ['## Input and diagnostic checks', '', '| Setting | Unresolved wake periods | Low effective count | Optical boundary flags |',
             '|---|---:|---:|---:|']
    for row in issues:
        text.append(f"| {LABELS[row['case']]} | {row['unresolved_bucket_snapshots']} | {row['low_effective_count_snapshots']} | {row['optical_boundary_flag_snapshots']} |")
    final_rows = {(row['comparison_case'], row['population'], row['metric']): row for row in summaries}
    charge_23 = final_rows[CASES[2], 'beam', 'charge_pC']
    charge_34 = final_rows[CASES[3], 'beam', 'charge_pC']
    energy_34 = final_rows[CASES[3], 'maximum', 'maximum_K_MeV']
    text += ['', 'Charge is undefined until a complete wake period is resolved. '
             'Resolved empty selections have zero charge and undefined moments. '
             'The optical boundary flags record when part of the pulse approaches the edge of the saved window.', '',
             '## Comparing the final snapshot', '',
             'At fixed pₙₜ = 16, the final charge above 20 MeV in the main accelerated population went from '
             f"{fmt(charge_23['final_reference'], '.2f', ' pC')} with two modes to "
             f"{fmt(charge_23['final_comparison'], '.2f', ' pC')} with three, then to "
             f"{fmt(charge_34['final_comparison'], '.2f', ' pC')} with four. "
             'I used this comparison when choosing three modes for the main beam analysis. '
             'The final maximum forward kinetic energy changed by '
             f"{fmt(energy_34['final_absolute_difference'], '+.2f', ' MeV')} when adding the fourth mode.", '',
             'The particle-sampling test was performed at Nₘ = 2, so the effect of increasing pₙₜ at Nₘ = 3 remains unknown. '
             'The spatial grid, timestep and seed were kept fixed throughout.', '']
    path = output/'04_convergence/summary.md'
    path.write_text('\n'.join(text), encoding='utf-8')
