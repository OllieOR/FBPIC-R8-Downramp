"""Draw R8/C4 comparisons from measured tables or saved snapshots."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import c, e

from diagnostics import CASES, density, edges, mesh, on_axis, read_particles, derive, window

COLORS = {"R8": "#28688c", "C4": "#c45435"}
STYLES = {"R8": "-", "C4": "--"}
MIN_EFFECTIVE = 20
MIN_RADIUS_COVERAGE = .10
XKEY = "laser_position_from_entrance_um"
XLABEL = "Laser position from plasma entrance (µm)"

plt.rcParams.update({
    "font.size": 12, "axes.titlesize": 13, "axes.labelsize": 13,
    "xtick.labelsize": 11, "ytick.labelsize": 11, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.alpha": .16,
    "figure.facecolor": "white", "text.color": "#182433", "axes.labelcolor": "#182433",
})


def line_axes(title, ylabel, panels=1):
    fig, axes = plt.subplots(panels, 1, figsize=(9.5, 5.6 if panels == 1 else 8.2),
                             sharex=True, squeeze=False, layout='constrained')
    for ax in axes[:, 0]:
        ax.set_ylabel(ylabel, fontsize=12)
        ax.tick_params(labelsize=11)
        ax.grid(axis='y', alpha=.15)
    axes[-1, 0].set_xlabel('Simulation time (ps)', fontsize=12, labelpad=9)
    fig.suptitle(title, fontsize=16)
    return fig, axes[0, 0] if panels == 1 else axes[:, 0]


def save_figure(fig, path):
    fig.savefig(path, dpi=200)
    plt.close(fig)


def input_profile(configs, output):
    end = max(cfg.get('window_travel', c*cfg['dt']*cfg['N_step'])+cfg['zmax'] for cfg in configs.values())
    z = np.linspace(0., end*1.e6, 2000)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout='constrained')
    cfg = configs['R8']
    start, length = cfg['downramp_start']*1.e6, cfg['downramp_length']*1.e6
    for ax in axes:
        for case in CASES:
            ax.plot(z, density(z*1.e-6, configs[case])/1.e24, color=COLORS[case],
                    ls=STYLES[case], lw=2.2, label=case)
        ax.axvspan(start, start+length, color='.6', alpha=.12)
        ax.set(xlabel='Initial longitudinal position (µm)', ylabel='Electron density (10²⁴ m⁻³)',
               ylim=(0, max(v['n_e'] for v in configs.values())/1.e24*1.1))
        ax.tick_params(labelsize=11)
    axes[0].set_title('Prescribed plasma')
    axes[1].set(title='Density transition', xlim=(start-20, start+length+20))
    axes[0].legend(frameon=False, fontsize=11)
    fig.suptitle('R8 and C4 input density profiles', fontsize=16)
    save_figure(fig, (output / 'input_density.png'))


def read_state(run, iteration):
    frame = run.frame(iteration)
    rho, r, z = mesh(run.series(), iteration, 'rho', mode='all')
    ez, rr, zz = mesh(run.series(), iteration, 'E', 'z')
    if not (np.array_equal(r, rr) and np.array_equal(z, zz)):
        raise ValueError('Density and longitudinal field meshes differ.')
    return dict(case=run.case, iteration=int(iteration), frame=frame, r=r*1.e6, z=z*1.e6,
                ne=-rho/e, ez=ez/1.e9, ez_axis=on_axis(ez, r)/1.e9,
                ne_axis=-on_axis(rho, r)/e)


def plot_pair(states, reference, field_limit, axis_limit, output):
    left = max(edges(s['z']-s['frame']['laser_z_um'])[0] for s in states)
    right = min(edges(s['z']-s['frame']['laser_z_um'])[-1] for s in states)
    radius = min(12., *[max(abs(edges(s['r'])[[0, -1]])) for s in states])
    iteration, time = states[0]['iteration'], states[0]['frame']['time_fs']/1000
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 7.2), sharex=True, sharey=True, layout='constrained')
    for ax, s in zip(axes, states):
        xi = s['z']-s['frame']['laser_z_um']
        im = ax.pcolormesh(edges(xi), edges(s['r']), s['ne']/reference, cmap='viridis',
                           vmin=0, vmax=5, rasterized=True)
        ax.set(title=s['case'], ylabel='Transverse coordinate (µm)', ylim=(-radius, radius), xlim=(left, right))
        ax.axvline(0, color='.6', ls='--', lw=.7)
        ax.tick_params(labelsize=11)
    axes[-1].set_xlabel(r'$\xi=z-z_{\mathrm{laser}}$ (µm)', fontsize=12)
    fig.colorbar(im, ax=list(axes), label=r'$n_e/n_{\mathrm{ref}}$', ticks=range(6), extend='max', shrink=.9)
    fig.suptitle(f'Electron density | iteration {iteration} · {time:.3f} ps', fontsize=16)
    save_figure(fig, (output / f'density_{iteration:05d}.png'))

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.4), sharex=True, sharey='row',
                             layout='constrained', gridspec_kw={'height_ratios': [1.25, 1.]})
    for col, s in enumerate(states):
        xi = s['z']-s['frame']['laser_z_um']
        im = axes[0, col].pcolormesh(edges(xi), edges(s['r']), s['ez'], cmap='RdBu_r',
                                    vmin=-field_limit, vmax=field_limit, rasterized=True)
        axes[0, col].set(title=s['case'], ylim=(-radius, radius))
        axes[1, col].plot(xi, s['ez_axis'], color=COLORS[s['case']], lw=2.2)
        axes[1, col].axhline(0, color='.5', lw=.7)
        axes[1, col].set(ylim=(-axis_limit, axis_limit), xlabel=r'$\xi=z-z_{\mathrm{laser}}$ (µm)')
        for ax in axes[:, col]:
            ax.set_xlim(left, right)
            ax.axvline(0, color='.6', ls='--', lw=.7)
            ax.tick_params(labelsize=11)
    axes[0, 0].set_ylabel('Transverse coordinate (µm)', fontsize=12)
    axes[1, 0].set_ylabel(r'On-axis $E_z$ (GV/m)', fontsize=12)
    fig.colorbar(im, ax=list(axes[0]), label=r'$E_z$ (GV/m)', extend='both', shrink=.9)
    fig.suptitle(f'Wake electric field | iteration {iteration} · {time:.3f} ps', fontsize=16)
    save_figure(fig, (output / f'electric_field_{iteration:05d}.png'))


def laser_figures(rows, output, configs=None):
    """Two readable figures, preserving the existing optical measurement."""
    fig, ax = line_axes('Laser energy remaining', 'Optical energy (% of first saved value)')
    maximum = 100.
    for case in CASES:
        data = sorted((r for r in rows if r['case'] == case), key=lambda r: r['iteration'])
        if not data:
            continue
        initial = data[0]['optical_energy_mJ']
        values = np.array([r['optical_energy_mJ']/initial*100 for r in data])
        ax.plot([r['time_ps'] for r in data], values, color=COLORS[case], ls=STYLES[case], lw=2.3, label=case)
        maximum = max(maximum, float(values.max()))
    ax.set_ylim(0, maximum*1.05)
    ax.legend(frameon=False, fontsize=11)
    save_figure(fig, (output / 'laser_energy.png'))
    fig, ax = line_axes('Laser spot size', 'Spot radius (µm)')
    for case in CASES:
        data = sorted((r for r in rows if r['case'] == case), key=lambda r: r['iteration'])
        ax.plot([r['time_ps'] for r in data], [r['spot_radius_um'] for r in data],
                color=COLORS[case], ls=STYLES[case], lw=2.3, label=case)
    if configs is not None:
        cfg = configs['R8']
        data = sorted((r for r in rows if r['case'] == 'R8'), key=lambda r: r['iteration'])
        zr = np.pi*cfg['w0']**2/cfg['lambda0']
        z = np.array([r['pulse_peak_z_um'] for r in data])*1.e-6
        vacuum = cfg['w0']*np.sqrt(1+((z-cfg['zf'])/zr)**2)*1.e6
        ax.plot([r['time_ps'] for r in data], vacuum, color='#777777', ls=':', lw=1.9, label='Vacuum Gaussian reference')
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, fontsize=11)
    save_figure(fig, (output / 'laser_waist.png'))


def particle_snapshot(run, iteration, settings):
    frame = run.frame(iteration)
    # Physical particle records remain valid even when an ID label is reused.
    p = read_particles(run.series(), iteration, allow_duplicates=True)
    d = derive(p, frame)
    good = d['inside'] & (p['uz'] >= 1.) & (p['w'] > 0)
    masks = dict(forward=good, beam=good & (d['energy_MeV'] >= settings["beam_energy_cut_MeV"]),
                 core=good & (d['radius_um'] <= settings["central_radius_um"]) & window(d['xi_um'], settings["central_xi_bounds_um"]))
    return dict(case=run.case, iteration=int(iteration), frame=frame, p=p, d=d, masks=masks)


def draw(ax, rows, selection, key, scale=1., quality=True):
    for case in CASES:
        data = sorted((r for r in rows if r['case'] == case and r['selection'] == selection), key=lambda r: r['iteration'])
        if not data:
            continue
        x = np.array([r['time_ps'] for r in data])
        values = np.array([r.get(key, np.nan) for r in data], float)*scale
        if quality:
            counts = np.array([r.get('effective_macroparticles', np.nan) for r in data], float)
            values[np.isfinite(counts) & (counts < MIN_EFFECTIVE)] = np.nan
        marker = 'o' if len(data) <= 8 else None
        ax.plot(x, values, color=COLORS[case], ls=STYLES[case], lw=2.3, marker=marker, ms=4, label=case)
    ax.set_ylim(bottom=0)


def electron_figures(rows, peaks, settings, output):
    for selection, key, scale, title, ylabel, filename in [
        ('energetic', 'charge_pC', 1., f'Energetic electron charge · K ≥ {settings["energetic_cut_MeV"]:g} MeV', 'Charge (pC)', 'charge.png'),
        ('beam', 'charge_pC', 1., f'Beam charge · K ≥ {settings["beam_energy_cut_MeV"]:g} MeV', 'Charge (pC)', 'beam_charge.png'),
        ('beam', 'mean_K_MeV', 1., f'Mean beam energy · K ≥ {settings["beam_energy_cut_MeV"]:g} MeV', 'Mean kinetic energy (MeV)', 'mean_energy.png'),
        ('beam', 'relative_rms_spread', 100., f'Beam energy spread · K ≥ {settings["beam_energy_cut_MeV"]:g} MeV', 'Relative RMS energy spread (%)', 'energy_spread.png')]:
        fig, ax = line_axes(title, ylabel)
        draw(ax, rows, selection, key, scale, quality=key != 'charge_pC')
        ax.legend(frameon=False, fontsize=11)
        save_figure(fig, (output / filename))
    for name, key, ylabel, title in [
        ('emittance', 'normalised_emittance_{}_mm_mrad', 'Normalised emittance (mm mrad)', 'Beam transverse emittance'),
        ('divergence', 'rms_divergence_{}_mrad', 'RMS divergence (mrad)', 'Beam divergence')]:
        fig, axes = line_axes(f'{title} · K ≥ {settings["beam_energy_cut_MeV"]:g} MeV', ylabel, panels=2)
        for axis, ax in zip('xy', axes):
            draw(ax, rows, 'beam', key.format(axis))
            ax.set_title(f'{axis} plane', fontsize=13)
        axes[0].legend(frameon=False, fontsize=11)
        save_figure(fig, (output / f'{name}.png'))
    maximum_plot(peaks, output)


def maximum_plot(peaks, output):
    fig, ax = line_axes('Maximum electron energy', 'Maximum kinetic energy (MeV)')
    for case in CASES:
        data = sorted((r for r in peaks if r['case'] == case), key=lambda r: r['iteration'])
        if data:
            ax.plot([r['time_ps'] for r in data], [r['maximum_energy_MeV'] for r in data],
                    color=COLORS[case], ls=STYLES[case], lw=2.3, label=case)
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, fontsize=11)
    save_figure(fig, (output / 'maximum_energy.png'))


def phase_plots(states, settings, output):
    from matplotlib.colors import LogNorm
    lookup = {(s['case'], s['iteration']): s for s in states}
    ordered = [lookup[case, it] for it in settings["iterations"] for case in CASES]
    xmin = min(s['frame']['zmin_um']-s['frame']['laser_z_um'] for s in states)
    xmax = max(s['frame']['zmax_um']-s['frame']['laser_z_um'] for s in states)
    maximum = max([2.] + [float(s['p']['uz'][s['masks']['forward']].max()) for s in states if s['masks']['forward'].any()])
    xb, ub = np.linspace(xmin, xmax, 221), np.linspace(1., maximum*1.02, 161)
    histograms = {}
    for s in states:
        for name in ('forward', 'core'):
            mask = s['masks'][name]
            q = s['d']['charge_pC'][mask]
            h = np.histogram2d(s['d']['xi_um'][mask], s['p']['uz'][mask], (xb, ub), weights=q)[0]
            if not np.isclose(h.sum(), q.sum(), rtol=1.e-9, atol=1.e-10):
                raise ValueError('Longitudinal phase space does not conserve charge.')
            histograms[s['case'], s['iteration'], name] = h
    positive = np.concatenate([h[h > 0] for h in histograms.values()])
    vmin, vmax = (float(positive.min()), float(positive.max())) if len(positive) else (1.e-6, 1.)
    norm = LogNorm(vmin=max(vmin, vmax*1.e-6), vmax=max(vmax, vmin*10))
    for name, title in [('forward', 'All saved forward electrons'), ('core', 'Central geometric selection')]:
        fig, axes = plt.subplots(len(settings["iterations"]), 2, figsize=(11.5, 2.8*len(settings["iterations"])+.7),
                                 squeeze=False, sharex=True, sharey=True, layout='constrained')
        for ax, s in zip(axes.flat, ordered):
            h = histograms[s['case'], s['iteration'], name]
            im = ax.pcolormesh(xb, ub, np.ma.masked_less_equal(h.T, 0), cmap='viridis', norm=norm, rasterized=True)
            ax.set_facecolor('#f3f5f6')
            ax.set_title(f"{s['case']} · iteration {s['iteration']} · {h.sum():.3g} pC", fontsize=11)
            ax.axvline(0, color='.6', ls='--', lw=.7)
            if name == 'core':
                ax.set_xlim(*settings["central_xi_bounds_um"])
        for ax in axes[:, 0]:
            ax.set_ylabel(r'$u_z=p_z/(m_ec)$', fontsize=12)
        for ax in axes[-1]:
            ax.set_xlabel(r'$\xi=z-z_{\mathrm{laser}}$ (µm)', fontsize=12)
        fig.colorbar(im, ax=list(axes.flat), label='Charge per bin (pC)', shrink=.85)
        fig.suptitle(f'Longitudinal phase space | {title}', fontsize=16)
        save_figure(fig, (output / 'phase_space.png' if name == 'forward' else 'phase_space_core.png'))
    transverse_plots(ordered, settings, output)


def transverse_plots(states, settings, output):
    from matplotlib.colors import LogNorm
    panels = []
    for s in states:
        keep = s['masks']['beam']
        for axis in 'xy':
            panels.append(dict(case=s['case'], iteration=s['iteration'], axis=axis,
                position=s['d'][f'{axis}_um'][keep],
                angle=np.arctan2(s['p'][f'u{axis}'][keep], s['p']['uz'][keep])*1000,
                q=s['d']['charge_pC'][keep]))
    extent = max([1.] + [float(abs(p['position']).max()) for p in panels if len(p['q'])])*1.03
    angle = max([1.] + [float(abs(p['angle']).max()) for p in panels if len(p['q'])])*1.03
    xb, ab = np.linspace(-extent, extent, 101), np.linspace(-angle, angle, 101)
    for p in panels:
        p['hist'] = np.histogram2d(p['position'], p['angle'], (xb, ab), weights=p['q'])[0]
        if not np.isclose(p['hist'].sum(), p['q'].sum(), rtol=1.e-9, atol=1.e-10):
            raise ValueError('Transverse phase space does not conserve charge.')
    positive = np.concatenate([p['hist'][p['hist'] > 0] for p in panels])
    vmin, vmax = (float(positive.min()), float(positive.max())) if len(positive) else (1.e-6, 1.)
    norm = LogNorm(vmin=max(vmin, vmax*1.e-6), vmax=max(vmax, vmin*10))
    for case in CASES:
        fig, axes = plt.subplots(len(settings["iterations"]), 2, figsize=(11.5, 2.8*len(settings["iterations"])+.7),
                                 squeeze=False, sharex=True, sharey=True, layout='constrained')
        for ax, p in zip(axes.flat, [p for p in panels if p['case'] == case]):
            im = ax.pcolormesh(xb, ab, np.ma.masked_less_equal(p['hist'].T, 0), cmap='viridis', norm=norm, rasterized=True)
            ax.set_facecolor('#f3f5f6')
            ax.set_title(f"{p['axis']} plane · iteration {p['iteration']} · {p['q'].sum():.3g} pC", fontsize=11)
            ax.set_ylabel(f"θ{p['axis']} (mrad)", fontsize=12)
        for ax, axis in zip(axes[-1], 'xy'):
            ax.set_xlabel(f'{axis} (µm)', fontsize=12)
        fig.colorbar(im, ax=list(axes.flat), label='Charge per bin (pC)', shrink=.85)
        fig.suptitle(f'{case} transverse phase space · K ≥ {settings["beam_energy_cut_MeV"]:g} MeV', fontsize=16)
        save_figure(fig, (output / f'transverse_phase_space_{case}.png'))


def peak_histories(rows, output):
    for key, title, ylabel, filename in [
        ('fixed_energy_MeV', 'Fixed peak-energy electron histories', 'Kinetic energy (MeV)', 'fixed_electron_energy.png'),
        ('fixed_radius_um', 'Fixed peak-energy electron radii', 'Radius (µm)', 'fixed_electron_radius.png'),
        ('fixed_xi_um', 'Fixed peak-energy electron positions in the wake', r'$\xi=z-z_{\mathrm{laser}}$ (µm)', 'fixed_electron_phase.png')]:
        fig, ax = line_axes(title, ylabel)
        for case in CASES:
            data = sorted((r for r in rows if r['case'] == case), key=lambda r: r['iteration'])
            ax.plot([r['time_ps'] for r in data], [r.get(key, np.nan) for r in data],
                    color=COLORS[case], ls=STYLES[case], lw=2.3, label=case)
        if key != 'fixed_xi_um':
            ax.set_ylim(bottom=0)
        ax.legend(frameon=False, fontsize=11)
        save_figure(fig, (output / filename))


def plot_origins(matches, config, output):
    known = [r for r in matches if np.isfinite(r['origin_z_um'])]
    if not known:
        return
    q = np.array([r['R8_charge_pC'] for r in known])
    paired = np.array([bool(r['paired']) for r in known])
    for field, title, xlabel, filename in [
        ('origin_z_um', 'Initial longitudinal positions of the R8-selected cohort', 'Initial longitudinal position (µm)', 'origins_z.png'),
        ('origin_radius_um', 'Initial radii of the R8-selected cohort', 'Initial radius (µm)', 'origins_radius.png')]:
        values = np.array([r[field] for r in known])
        low, high = values.min(), values.max()
        padding = max(.05, (high-low)*.03)
        bins = np.linspace(max(0., low-padding), high+padding, 41)
        fig, ax = plt.subplots(figsize=(9.5, 5.6), layout='constrained')
        for keep, label, colour in [(paired, 'Verified counterpart', '#28688c'), (~paired, 'Unmatched', '#777777')]:
            if keep.any():
                h = np.histogram(values[keep], bins=bins, weights=q[keep])[0]/np.diff(bins)
                ax.stairs(h, bins, color=colour, lw=2.2, label=label)
        if field == 'origin_z_um':
            start = config['downramp_start']*1.e6
            ax.axvspan(start, start+config['downramp_length']*1.e6, color='.6', alpha=.12, label='R8 downramp')
        ax.set(xlabel=xlabel, ylabel='R8 charge per µm (pC/µm)', ylim=(0, None))
        ax.tick_params(labelsize=11)
        ax.legend(frameon=False, fontsize=11)
        ax.grid(axis='y', alpha=.15)
        fig.suptitle(title, fontsize=15)
        save_figure(fig, (output / filename))


def matched_histories(rows, energy_cut, output):
    for field, title, filename in [
        ('Q_inside_pC', 'Matched cohort charge inside the grid', 'matched_charge.png'),
        ('Q_energetic_core_pC', f'Matched energetic central charge · K ≥ {energy_cut:g} MeV', 'matched_core_charge.png')]:
        fig, ax = line_axes(title, 'Native charge (pC)')
        for case in CASES:
            data = [r for r in rows if r['case'] == case and r['source'] == 'unfiltered_history']
            ax.plot([r['time_ps'] for r in data], [r[field] for r in data],
                    color=COLORS[case], ls='-' if case == 'R8' else '--', lw=2.3, label=case)
            filtered = [r for r in rows if r['case'] == case and r['source'] == 'filtered_diags']
            ax.scatter([r['time_ps'] for r in filtered], [r[field] for r in filtered],
                       color=COLORS[case], s=24, zorder=3)
        ax.set_ylim(bottom=0)
        ax.legend(frameon=False, fontsize=11)
        save_figure(fig, (output / filename))
    fig, ax = line_axes('Matched cohort transverse extent', 'Radius (µm)')
    for case in CASES:
        data = [r for r in rows if r['case'] == case and r['source'] == 'unfiltered_history']
        x = np.array([r['time_ps'] for r in data])
        keep = np.array([r['reference_inside_fraction'] >= MIN_RADIUS_COVERAGE for r in data])
        values = lambda key: np.where(keep, [r[key] for r in data], np.nan)
        ax.plot(x, values('radius_median_um'), color=COLORS[case], ls='-' if case == 'R8' else '--', lw=2.3, label=case)
        ax.fill_between(x, values('radius_p10_um'), values('radius_p90_um'), color=COLORS[case], alpha=.14)
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, fontsize=11)
    save_figure(fig, (output / 'matched_radius.png'))


def geometry(config):
    ion = 'n_He' in config
    start = float(config.get('p_zmin', 0.) if ion else config['ramp_start'])*1.e6
    nref = (config.get('helium_level_start', 1)*config['n_He']+
            config.get('nitrogen_level_start', 5)*config['n_N']) if ion else config['n_e']
    return dict(entrance_z_um=start, entrance_length_um=float(config['ramp_length'])*1.e6,
                focus_z_um=float(config.get('z_foc', config.get('zf', 0.)))*1.e6,
                initial_laser_z_um=float(config.get('z0', np.nan))*1.e6, reference_ne_m3=float(nref))


def axes(title, ylabel, panels=1):
    fig, aa = plt.subplots(panels, 1, figsize=(9.5, 5.6 if panels == 1 else 8.2),
                           sharex=True, squeeze=False, layout='constrained')
    for ax in aa[:, 0]:
        ax.set_ylabel(ylabel, fontsize=12)
        ax.tick_params(labelsize=11)
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', alpha=.15)
    aa[-1, 0].set_xlabel(XLABEL, fontsize=12, labelpad=9)
    fig.suptitle(title, fontsize=15)
    return fig, aa[:, 0]


def curves(ax, rows, cases, population, key, scale=1., quality=False):
    for case in cases:
        data = sorted((r for r in rows if r['case'] == case and r['population'] == population), key=lambda r: r['iteration'])
        values = np.array([r[key] for r in data], float)*scale
        if quality:
            effective = np.array([r['effective_macroparticles'] for r in data], float)
            values[~np.isfinite(effective) | (effective < MIN_EFFECTIVE)] = np.nan
        x = np.array([r[XKEY] for r in data], float)
        ax.plot(x, values, color=COLORS[case], ls=STYLES[case], lw=2.3, label=case)
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, fontsize=11)


def primary_beam(rows, record, folder):
    cut, radius, cases = record['energy_cut_MeV'], record['acceptance_radius_um'], record['cases']
    for parent, title, filename in [('all_saved', 'Saved energetic charge', 'saved_charge.png'),
                                    ('first_bucket', 'Main accelerated charge', 'main_accelerated_charge.png')]:
        fig, aa = axes(f'{title} · K ≥ {cut:g} MeV', 'Charge (pC)', panels=2)
        for ax, pop, subtitle in zip(aa, (parent, parent+'_compact'), ('Total in this selection', f'Within r ≤ {radius:g} µm')):
            curves(ax, rows, cases, pop, 'charge_pC')
            ax.set_title(subtitle, fontsize=13)
        fig.savefig(folder/filename, dpi=180)
        plt.close(fig)
    fig, aa = axes(f'Near-axis fraction · K ≥ {cut:g} MeV · r ≤ {radius:g} µm', 'Fraction of charge (%)', panels=2)
    for ax, parent, subtitle in zip(aa, ('all_saved', 'first_bucket'),
                                    ('All saved energetic electrons', 'Main accelerated population')):
        curves(ax, rows, cases, parent, 'compact_fraction', 100., quality=True)
        ax.set_title(subtitle, fontsize=13)
        ax.set_ylim(0, 105)
    fig.savefig(folder/'compact_fraction.png', dpi=180)
    plt.close(fig)
    for key, scale, title, ylabel, filename in [
        ('mean_K_MeV', 1., 'Main population mean energy', 'Mean kinetic energy (MeV)', 'mean_energy.png'),
        ('relative_rms_spread', 100., 'Main population energy spread', 'Relative RMS energy spread (%)', 'energy_spread.png')]:
        fig, aa = axes(f'{title} · K ≥ {cut:g} MeV', ylabel)
        curves(aa[0], rows, cases, 'first_bucket', key, scale, quality=True)
        fig.savefig(folder/filename, dpi=180)
        plt.close(fig)
    for prefix, suffix, title, ylabel, filename in [
        ('normalised_emittance', 'mm_mrad', 'Main population emittance', 'Normalised emittance (mm mrad)', 'emittance.png'),
        ('rms_divergence', 'mrad', 'Main population divergence', 'RMS divergence (mrad)', 'divergence.png')]:
        fig, aa = axes(f'{title} · K ≥ {cut:g} MeV', ylabel, panels=2)
        for ax, plane in zip(aa, 'xy'):
            curves(ax, rows, cases, 'first_bucket', f'{prefix}_{plane}_{suffix}', quality=True)
            ax.set_title(f'{plane} plane', fontsize=13)
        fig.savefig(folder/filename, dpi=180)
        plt.close(fig)


def reference_plot(rows, record, folder):
    fig, aa = plt.subplots(2, 1, figsize=(10, 7.5), layout='constrained')
    end = max([float(r[XKEY]) for r in rows if np.isfinite(r[XKEY])] or [100.])
    for case, config in record['configurations'].items():
        g = geometry(config)
        z = np.linspace(min(-25., g['focus_z_um']-g['entrance_z_um']-5), max(end+5, 120.), 2000)
        f = np.clip(z/g['entrance_length_um'], 0., 1.)
        if 'n_He' in config:
            f = np.sin(.5*np.pi*f)**2
        if config.get('lower_density_factor', 1.) < 1.:
            start = config['downramp_start']*1.e6-g['entrance_z_um']
            length = config['downramp_length']*1.e6
            f *= 1.-(1.-config['lower_density_factor'])*np.clip((z-start)/length, 0., 1.)
        laboratory_z = z+g['entrance_z_um']
        if 'exit_start' in config:
            f *= 1.-np.clip((laboratory_z-config['exit_start']*1.e6)/(config['exit_length']*1.e6), 0., 1.)
        f = np.where(laboratory_z <= config.get('p_zmax', float('inf'))*1.e6, f, 0.)
        for ax in aa:
            ax.plot(z, f*g['reference_ne_m3']/1.e24, color=COLORS[case], ls=STYLES[case], lw=2.2, label=case+' initial')
            if 'n_He' in config:
                ax.plot(z, f*(2*config['n_He']+7*config['n_N'])/1.e24,
                        color='#777777', ls='--', lw=1.8, label='Fully stripped limit')
        focus = g['focus_z_um']-g['entrance_z_um']
        aa[1].axvline(focus, color='.45', ls=':', lw=1.)
    # All cases in a package share entrance/focus; R8 alone has a density drop.
    cfg = next(iter(record['configurations'].values()))
    g = geometry(cfg)
    focus = g['focus_z_um']-g['entrance_z_um']
    aa[1].axvspan(0, g['entrance_length_um'], color='.7', alpha=.13, label='Entrance ramp')
    ramp_ends = []
    for case, cfg in record['configurations'].items():
        if cfg.get('lower_density_factor', 1.) < 1.:
            start = cfg['downramp_start']*1.e6-g['entrance_z_um']
            stop = start+cfg['downramp_length']*1.e6
            ramp_ends.append(stop)
            aa[1].axvspan(start, stop, color='#28688c', alpha=.12, label=case+' downramp')
    aa[1].plot([], [], color='.45', ls=':', label='Input focus')
    aa[0].set_title('Density along the analysed propagation range', fontsize=13)
    aa[0].set_xlim(min(-25., focus-5), max(end+5, 120.))
    aa[1].set(title='Entrance, focus and density transition', xlim=(min(-25., focus-5), max([g['entrance_length_um'], focus, *ramp_ends])+25.))
    for ax in aa:
        ax.axvline(0, color='.55', lw=.8)
        ax.set(xlabel=XLABEL, ylabel='Electron density (10²⁴ m⁻³)', ylim=(0, None))
        ax.tick_params(labelsize=11)
        ax.spines[['top', 'right']].set_visible(False)
    aa[0].legend(frameon=False, fontsize=10)
    aa[1].legend(frameon=False, fontsize=10, loc='best')
    fig.suptitle('Propagation coordinate and prescribed plasma', fontsize=15)
    fig.savefig(folder/'propagation_reference.png', dpi=180)
    plt.close(fig)


def status(row):
    if row['population'].startswith('first_bucket') and row['primary_status'] != 'resolved':
        return 'unresolved_bucket'
    if not np.isfinite(row['charge_pC']):
        return 'unresolved_selection'
    if row['charge_pC'] == 0:
        return 'empty_selection'
    if not np.isfinite(row['effective_macroparticles']) or row['effective_macroparticles'] < MIN_EFFECTIVE:
        return 'below_count_threshold'
    return 'reported'


def sensitivity_values(rows, case, cut, key, scale=1., quality=False):
    selected = sorted((r for r in rows if r['case'] == case and r['energy_cut_MeV'] == cut), key=lambda r: r['iteration'])
    x = np.array([r['time_ps'] for r in selected])
    y = np.array([r[key] for r in selected], float)*scale
    if quality:
        y[np.array([status(r) != 'reported' for r in selected])] = np.nan
    return x, y


def sensitivity_figures(rows, record, folder):
    metrics = [('charge_pC', 1., 'Main accelerated charge', 'Charge (pC)', 'charge.png'),
        ('mean_K_MeV', 1., 'Main population mean kinetic energy', 'Mean kinetic energy (MeV)', 'mean_energy.png'),
        ('relative_rms_spread', 100., 'Main population energy spread', 'Relative RMS energy spread (%)', 'energy_spread.png'),
        ('normalised_emittance', 1., 'Main population normalised emittance', 'Normalised emittance (mm mrad)', 'emittance.png'),
        ('rms_divergence', 1., 'Main population RMS divergence', 'RMS divergence (mrad)', 'divergence.png')]
    with plt.rc_context({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False,
                         'savefig.facecolor': 'white'}):
        for key, scale, title, ylabel, name in metrics:
            planes = ('x', 'y') if key in ('normalised_emittance', 'rms_divergence') else (None,)
            fig, aa = plt.subplots(len(planes), 2, figsize=(11.5, 4.5 if len(planes) == 1 else 7.2),
                sharex=True, sharey='row', squeeze=False, layout='constrained')
            fig.suptitle(title, fontsize=15)
            for i, plane in enumerate(planes):
                metric = f'{key}_{plane}_'+('mm_mrad' if key == 'normalised_emittance' else 'mrad') if plane else key
                limits = []
                for j, cut in enumerate(record["energy_cuts_MeV"]):
                    ax = aa[i, j]
                    ax.set_title(f'K ≥ {cut:g} MeV'+(f' · {plane} plane' if plane else ''), fontsize=12)
                    for case in CASES:
                        x, y = sensitivity_values(rows, case, cut, metric, scale, quality=key != 'charge_pC')
                        ax.plot(x, y, color=COLORS[case], ls=STYLES[case], lw=2.1, label=case)
                        finite = np.isfinite(y)
                        isolated = finite & ~(np.r_[False, finite[:-1]] | np.r_[finite[1:], False])
                        if isolated.any(): ax.scatter(x[isolated], y[isolated], color=COLORS[case], s=15, zorder=3)
                        limits.extend(y[finite].tolist())
                    ax.grid(axis='y', alpha=.15)
                    ax.tick_params(labelsize=10)
                    if i == len(planes)-1: ax.set_xlabel('Simulation time (ps)', fontsize=11)
                aa[i, 0].set_ylabel(ylabel, fontsize=11)
                aa[i, 0].set_ylim(0, max(limits)*1.08 if limits and max(limits) > 0 else 1.)
            aa[0, 0].legend(frameon=False, fontsize=10, loc='best')
            all_times = [r['time_ps'] for r in rows if np.isfinite(r['time_ps'])]
            if all_times:
                width = max(all_times)-min(all_times)
                aa[-1, 0].set_xlim(min(all_times)-.015*max(width, .1), max(all_times)+.015*max(width, .1))
            save_figure(fig, folder / name)


def main_population_compact_fraction(rows, record, output):
    fig, ax = plt.subplots(figsize=(9.5, 5.6), layout="constrained")
    curves(ax, rows, record["cases"], "first_bucket", "compact_fraction", 100., quality=True)
    ax.set(xlabel=XLABEL, ylabel="Fraction of selected charge (%)", ylim=(0, 105),
           title=f'K ≥ {record["energy_cut_MeV"]:g} MeV · r ≤ {record["acceptance_radius_um"]:g} µm')
    fig.suptitle("Main population near-axis charge fraction")
    save_figure(fig, output / "main_population_compact_fraction.png")


def r8_divergence(rows, record, output):
    fig, panels = axes("R8 · main population RMS divergence", "RMS divergence (mrad)", panels=2)
    for cut, style, colour in zip(record["energy_cuts_MeV"], ("-", "--"), COLORS.values(), strict=True):
        selected = sorted((row for row in rows if row["case"] == "R8"
                           and row["energy_cut_MeV"] == cut),
                          key=lambda row: row["iteration"])
        x = [row[XKEY] for row in selected]
        for ax, plane in zip(panels, "xy", strict=True):
            y = [row[f"rms_divergence_{plane}_mrad"] if status(row) == "reported" else np.nan
                 for row in selected]
            ax.plot(x, y, color=colour, ls=style, label=f"K ≥ {cut:g} MeV")
            ax.set(title=f"{plane} plane", ylim=(0, None))
    panels[0].legend(frameon=False)
    save_figure(fig, output / "divergence_R8.png")


def longitudinal_emittance(rows, record, output):
    fig, panels = plt.subplots(1, 2, figsize=(11.5, 4.5), sharex=True, sharey=True,
                               layout="constrained")
    fig.suptitle("Main population longitudinal RMS emittance")
    for ax, cut in zip(panels, record["energy_cuts_MeV"], strict=True):
        ax.set(title=f"K ≥ {cut:g} MeV", xlabel="Simulation time (ps)")
        for case in CASES:
            selected = sorted((row for row in rows if row["case"] == case
                               and row["energy_cut_MeV"] == cut),
                              key=lambda row: row["iteration"])
            x = np.array([row["time_ps"] for row in selected])
            y = np.array([row["longitudinal_rms_emittance_MeV_um"]
                          if row["moment_status"] == "reported" else np.nan for row in selected])
            ax.plot(x, y, color=COLORS[case], ls=STYLES[case], label=case)
            finite = np.isfinite(y)
            isolated = finite & ~(np.r_[False, finite[:-1]] | np.r_[finite[1:], False])
            ax.scatter(x[isolated], y[isolated], color=COLORS[case], s=15, zorder=3)
    panels[0].set(ylabel="Longitudinal RMS emittance (MeV µm)", ylim=(0, None))
    panels[0].legend(frameon=False)
    save_figure(fig, output / "longitudinal_emittance.png")
