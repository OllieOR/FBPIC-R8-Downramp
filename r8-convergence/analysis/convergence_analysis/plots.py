"""Consistent overlays and charge-weighted snapshot comparisons."""

import numpy as np

from .common import CASES, COLORS, LABELS, STYLES, result_path, write_csv, write_json


def save(fig, path):
    import matplotlib.pyplot as plt
    fig.savefig(path, dpi=180, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def axes(title, ylabel, panels=1):
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(panels, 1, figsize=(8.5, 3.3*panels+.45),
                            sharex=True, layout='constrained', squeeze=False)
    axs = axs[:, 0]
    fig.suptitle(title, fontsize=14)
    for ax in axs:
        ax.set_ylabel(ylabel)
        ax.grid(alpha=.2)
    axs[-1].set_xlabel('Time (ps)')
    return fig, axs


def values(rows, key, quality=False, factor=1.):
    v = np.array([row[key] for row in rows], dtype=float)*factor
    if quality:
        v[np.array([row['effective_macroparticles'] < 20 for row in rows])] = np.nan
    return v


def beam_rows(tables, case, population, cut=20.):
    return sorted([r for r in tables[case]['beam'] if r['population'] == population
                   and r['energy_cut_MeV'] == cut], key=lambda r: r['iteration'])


def overlay(ax, tables, table, key, population='first_bucket', cut=20., quality=False, factor=1.):
    for case in CASES:
        rows = beam_rows(tables, case, population, cut) if table == 'beam' else tables[case][table]
        ax.plot([row['time_ps'] for row in rows], values(rows, key, quality, factor),
                color=COLORS[case], ls=STYLES[case], lw=2., label=LABELS[case])


def snapshot(tables, case, iteration):
    prefix = f'i{iteration}_'
    return {k[len(prefix):]: v for k, v in tables[case]['snapshots'].items() if k.startswith(prefix)}


def line_plots(tables, output, record):
    fig, axs = axes('Charge above 20 MeV', 'Charge (pC)', 3)
    for ax, pop, title in zip(axs, ('all_saved', 'first_bucket', 'first_bucket_compact'),
            ('Saved window', 'Main accelerated charge',
             f"Near-axis main accelerated charge · r ≤ {record['core_radius_um']:g} µm")):
        overlay(ax, tables, 'beam', 'charge_pC', pop)
        ax.set_title(title, fontsize=11)
        ax.set_ylim(bottom=0)
    axs[0].legend(frameon=False)
    save(fig, result_path(output, '03_electrons', 'figures', 'charge.png'))
    for key, ylabel, name, factor in (
        ('mean_K_MeV', 'Mean kinetic energy (MeV)', 'mean_energy', 1.),
        ('relative_rms_spread', 'RMS energy spread (%)', 'energy_spread', 100.)):
        fig, axs = axes('Main accelerated population · K ≥ 20 MeV', ylabel)
        overlay(axs[0], tables, 'beam', key, quality=True, factor=factor)
        axs[0].legend(frameon=False)
        axs[0].set_ylim(bottom=0)
        save(fig, result_path(output, '03_electrons', 'figures', name+'.png'))
    for prefix, ylabel, name in (
        ('normalised_emittance', 'Normalised emittance (mm mrad)', 'emittance'),
        ('rms_divergence', 'RMS divergence (mrad)', 'divergence')):
        fig, axs = axes('Main accelerated population · K ≥ 20 MeV', ylabel, 2)
        for ax, plane in zip(axs, 'xy'):
            suffix = 'mm_mrad' if name == 'emittance' else 'mrad'
            overlay(ax, tables, 'beam', f'{prefix}_{plane}_{suffix}', quality=True)
            ax.set_title(f'{plane} plane', fontsize=11)
            ax.set_ylim(bottom=0)
        axs[0].legend(frameon=False)
        save(fig, result_path(output, '03_electrons', 'figures', name+'.png'))
    fig, axs = axes('Forward saved electrons · no beam energy cut', 'Kinetic energy (MeV)', 2)
    for ax, key, title in zip(axs, ('maximum_K_MeV', 'percentile99_K_MeV'),
                             ('Maximum energy', 'Charge-weighted 99th percentile')):
        overlay(ax, tables, 'maximum', key)
        ax.set_title(title, fontsize=11)
        ax.set_ylim(bottom=0)
    axs[0].legend(frameon=False)
    save(fig, result_path(output, '03_electrons', 'figures', 'maximum_energy.png'))
    for key, ylabel, name, factor in (
        ('energy_fraction_of_initial', 'Saved optical energy (% of initial)', 'laser_energy', 100.),
        ('spot_radius_um', 'RMS-equivalent spot radius (µm)', 'laser_waist', 1.)):
        fig, axs = axes('Laser evolution', ylabel)
        overlay(axs[0], tables, 'laser', key, factor=factor)
        axs[0].legend(frameon=False)
        axs[0].set_ylim(bottom=0)
        save(fig, result_path(output, '02_laser', 'figures', name+'.png'))
    fig, axs = axes('Wake evolution', '', 2)
    overlay(axs[0], tables, 'wake', 'accelerating_field_GVm')
    overlay(axs[1], tables, 'wake', 'first_bucket_length_um')
    axs[0].set_ylabel('Peak accelerating field (GV/m)')
    axs[0].set_title('Smoothed on-axis m=0 field, −30 ≤ ξ ≤ −3 µm', fontsize=10)
    axs[1].set_ylabel('First complete wake\nperiod length (µm)')
    for ax in axs:
        ax.set_ylim(bottom=0)
    axs[0].legend(frameon=False)
    save(fig, result_path(output, '01_wake', 'figures', 'wake_evolution.png'))


def wake_plots(tables, snapshots, output):
    import matplotlib.pyplot as plt
    cropped_fields = []
    for case in CASES:
        for it in snapshots:
            s = snapshot(tables, case, it)
            xi = s['z_um']-s['laser_z_um']
            crop = s['ez_GVm'][np.ix_(abs(s['r_um']) <= 12, (xi >= -30) & (xi <= -3))]
            if crop.size:
                cropped_fields.append(float(np.max(abs(crop))))
    limit = max(cropped_fields, default=1.) or 1.
    lineouts = []
    for it in snapshots:
        for kind, key, cmap, vmin, vmax, title in (
            ('density', 'density_n0', 'viridis', 0, 5, 'Electron density / initial plateau density'),
            ('field', 'ez_GVm', 'RdBu_r', -limit, limit, 'Longitudinal electric field, m=0 (GV/m)')):
            fig, axs = plt.subplots(len(CASES), 1, figsize=(8.5, 2.6*len(CASES)+.5),
                                    sharex=True, sharey=True, squeeze=False, layout='constrained')
            axs = axs[:, 0]
            for ax, case in zip(axs, CASES):
                s = snapshot(tables, case, it)
                xi = s['z_um']-s['laser_z_um']
                image = ax.pcolormesh(xi, s['r_um'], s[key], cmap=cmap, vmin=vmin, vmax=vmax, shading='auto', rasterized=True)
                ax.set_xlim(-30, -3)
                ax.set_ylim(-12, 12)
                ax.set_ylabel('r (µm)')
                ax.set_title(LABELS[case], fontsize=11)
            fig.colorbar(image, ax=axs, label='nₑ/n₀' if kind == 'density' else 'E_z (GV/m)',
                         extend='max' if kind == 'density' else 'neither', shrink=.85)
            axs[-1].set_xlabel('ξ = z − measured laser peak (µm)')
            fig.suptitle(f'{title}\nIteration {it} · {float(s["time_ps"]):.3f} ps', fontsize=13)
            save(fig, result_path(output, '01_wake', 'figures', f'{kind}_{it:05d}.png'))
        for case in CASES:
            s = snapshot(tables, case, it)
            for z, ez in zip(s['z_um'], s['ez_axis_GVm']):
                lineouts.append(dict(case=case, iteration=it, time_ps=float(s['time_ps']),
                    xi_fixed_um=float(z-s['fixed_origin_um']), xi_laser_um=float(z-s['laser_z_um']), Ez_GVm=float(ez)))
    fig, axs = plt.subplots(len(snapshots), 1, figsize=(8.7, 2.5*len(snapshots)+.5),
                            sharex=True, sharey=True, squeeze=False, layout='constrained')
    for ax, it in zip(axs[:, 0], snapshots):
        for case in CASES:
            s = snapshot(tables, case, it)
            ax.plot(s['z_um']-s['fixed_origin_um'], s['ez_axis_GVm'],
                    color=COLORS[case], ls=STYLES[case], lw=1.8, label=LABELS[case])
        ax.axhline(0, color='grey', lw=.6)
        ax.grid(alpha=.15)
        ax.set_ylabel('E_z (GV/m)')
        ax.set_title(f'Iteration {it} · {float(s["time_ps"]):.3f} ps', fontsize=11)
    axs[0, 0].legend(frameon=False, ncol=2, fontsize=9)
    axs[-1, 0].set_xlabel('z − (input laser centre + ct) (µm) · no phase alignment')
    fig.suptitle('On-axis wake comparison in one common moving coordinate', fontsize=14)
    save(fig, result_path(output, '01_wake', 'figures', 'on_axis_Ez.png'))
    write_csv(result_path(output, '01_wake', 'data', 'wake_lineouts.csv'), lineouts)
    write_json(result_path(output, '01_wake', 'settings', 'wake_settings.json'), dict(
        density_cap=5, reference='initial plateau n_e, identical in all four runs', density_modes='all, theta=0',
        field_mode=0, map_field_limit_GVm=limit, transverse_limit_um=12, laser_relative_window_um=[-30, -3],
        lineout_coordinate='z - (z0 + c*t); exact matched times; no phase shifts or amplitude normalization'))


def particle_snapshots(tables, snapshots, output):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    maximum = max([float(np.max(snapshot(tables, case, it)['K'], initial=5.))
                   for case in CASES for it in snapshots]+[25.])*1.02
    energy_edges = np.linspace(5., maximum, 101)
    xi_edges = np.linspace(-45., 20., 161)
    # Physical charge per bin area, not raw macro counts or peak-normalized density.
    histograms, peak = {}, 0.
    for it in snapshots:
        for case in CASES:
            s = snapshot(tables, case, it)
            hist, _, _ = np.histogram2d(s['z']-s['laser_z_um'], s['K'],
                                      bins=(xi_edges, energy_edges), weights=s['q'])
            hist /= np.diff(xi_edges)[:, None]*np.diff(energy_edges)[None, :]
            histograms[case, it] = hist
            peak = max(peak, float(hist.max(initial=0.)))
    peak = max(peak, 1.e-8)
    norm = LogNorm(vmin=peak*1.e-5, vmax=peak)
    spectra_rows = []
    for it in snapshots:
        fig, axs = plt.subplots(len(CASES), 1, figsize=(8.5, 2.8*len(CASES)+.5),
                                sharex=True, sharey=True, squeeze=False, layout='constrained')
        axs = axs[:, 0]
        for ax, case in zip(axs, CASES):
            s = snapshot(tables, case, it)
            h = np.ma.masked_less_equal(histograms[case, it], 0.)
            image = ax.pcolormesh(xi_edges, energy_edges, h.T, cmap='magma', norm=norm, shading='flat', rasterized=True)
            ax.set_title(LABELS[case], fontsize=11)
            ax.set_ylabel('Kinetic energy (MeV)')
            ax.axhline(20., color='#777777', ls=':', lw=1.)
            for boundary in (s['bucket_low_um'], s['bucket_high_um']):
                if np.isfinite(boundary):
                    ax.axvline(float(boundary), color='#22aacc', ls='--', lw=.9)
        fig.colorbar(image, ax=axs, label='Charge density (pC / µm / MeV)', shrink=.8)
        axs[-1].set_xlabel('ξ = z − measured laser peak (µm)')
        fig.suptitle(f'Forward saved electrons, K ≥ 5 MeV\nIteration {it} · {float(s["time_ps"]):.3f} ps', fontsize=13)
        save(fig, result_path(output, '03_electrons', 'figures', f'phase_space_{it:05d}.png'))
        fig, axs = axes(f'Main population spectrum · iteration {it}', 'dQ/dK (pC/MeV)')
        for case in CASES:
            s = snapshot(tables, case, it)
            xi = s['z']-s['laser_z_um']
            valid = np.isfinite(s['bucket_low_um']) and np.isfinite(s['bucket_high_um'])
            selected = (xi >= s['bucket_low_um']) & (xi <= s['bucket_high_um'])
            counts = np.histogram(s['K'][selected], bins=energy_edges, weights=s['q'][selected])[0]
            spectrum = counts/np.diff(energy_edges) if valid else np.full(len(counts), np.nan)
            axs[0].stairs(spectrum, energy_edges, color=COLORS[case], ls=STYLES[case], lw=1.8, label=LABELS[case])
            spectra_rows.extend(dict(case=case, iteration=it, primary_status='resolved' if valid else 'unresolved',
                lower_K_MeV=float(a), upper_K_MeV=float(b), charge_pC=float(q) if valid else np.nan,
                dQdK_pC_per_MeV=float(v)) for a, b, q, v in zip(energy_edges[:-1], energy_edges[1:], counts, spectrum))
        axs[0].set_xlabel('Kinetic energy (MeV)')
        axs[0].set_xlim(5, maximum)
        axs[0].set_ylim(bottom=0)
        axs[0].axvline(20, color='grey', ls=':', lw=1)
        axs[0].legend(frameon=False)
        save(fig, result_path(output, '03_electrons', 'figures', f'energy_spectrum_{it:05d}.png'))
    write_csv(result_path(output, '03_electrons', 'data', 'energy_spectra.csv'), spectra_rows)
    write_json(result_path(output, '03_electrons', 'settings', 'snapshot_settings.json'), dict(
        phase_space_population='Saved forward records with K >= 5 MeV. Dashed cyan lines mark the selected wake period. The horizontal dotted line marks 20 MeV.',
        phase_space_xi_edges_um=xi_edges.tolist(), energy_edges_MeV=energy_edges.tolist(),
        phase_space_log_colour_range=[peak*1.e-5, peak],
        spectrum_population='Main accelerated population in the selected wake period, K >= 5 MeV, no radial cut or unit-area normalization.',
        charge_weighted=True, all_snapshots_share_energy_bins_and_phase_space_colour_scale=True))


def sensitivity_plots(tables, output):
    for key, ylabel, name, factor, quality in (
        ('charge_pC', 'Main accelerated charge (pC)', 'charge_thresholds', 1., False),
        ('mean_K_MeV', 'Mean kinetic energy (MeV)', 'mean_energy_thresholds', 1., True),
        ('relative_rms_spread', 'RMS energy spread (%)', 'energy_spread_thresholds', 100., True)):
        fig, axs = axes('Main population · energy-threshold sensitivity', ylabel, 2)
        for ax, cut in zip(axs, (5., 20.)):
            overlay(ax, tables, 'beam', key, cut=cut, factor=factor, quality=quality)
            ax.set_title(f'K ≥ {cut:g} MeV', fontsize=11)
            ax.set_ylim(bottom=0)
        axs[0].legend(frameon=False)
        save(fig, result_path(output, '05_sensitivity_check', 'figures', name+'.png'))
    write_csv(result_path(output, '05_sensitivity_check', 'data', 'beam_evolution_5MeV.csv'),
              [row for case in CASES for row in tables[case]['beam'] if row['energy_cut_MeV'] == 5.])


def make_plots(tables, snapshots, output, record):
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False})
    line_plots(tables, output, record)
    wake_plots(tables, snapshots, output)
    particle_snapshots(tables, snapshots, output)
    sensitivity_plots(tables, output)
    write_json(result_path(output, '02_laser', 'settings', 'laser_settings.json'), dict(
        optical_band_k_over_k0=[.25, 2.5], sensitivity_bands=[[.15, 2.5], [.4, 2.5]],
        energy='Optical-band E and B energy in the saved box, all retained modes, cylindrical cell volumes; Parseval checked.',
        radius='Centroid-corrected RMS-equivalent radius at the peak of the longitudinal optical-energy profile.',
        boundary_flag='At least 1% of optical energy within outer 10% of radius or 2 µm of a z boundary. Flags in laser CSV.',
        implementation='analysis/convergence_analysis/laser.py'))
