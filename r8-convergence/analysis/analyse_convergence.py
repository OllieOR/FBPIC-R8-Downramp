"""Compare four R8 runs: angular particle sampling and Nm=2,3,4 at p_nt=16."""

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, default=ROOT)
    parser.add_argument('--baseline', type=Path,
                        default=ROOT.parent/'downramp-r8/downramp_comparison/R8',
                        help='R8 run directory. Defaults to the sibling downramp-r8 project.')
    parser.add_argument('--output', type=Path, help='Default: PROJECT/results')
    parser.add_argument('--snapshots', nargs='+', type=int, default=[2860, 4000, 6000, 8000, 10750])
    parser.add_argument('--core-radius', type=float, default=2.)
    parser.add_argument('--force', action='store_true', help='Recompute cached run tables.')
    parser.add_argument('--replot', action='store_true', help='Use saved tables and snapshots; no diagnostic reads.')
    parser.add_argument('--replot-from', type=Path,
                        help='Saved results directory for --replot. Defaults to the output directory.')
    parser.add_argument('--check-only', action='store_true', help='Validate inputs without generating results.')
    args = parser.parse_args()
    if args.replot_from is not None and not args.replot:
        parser.error('--replot-from requires --replot.')
    args.project, args.baseline = args.project.expanduser().resolve(), args.baseline.expanduser().resolve()
    output = (args.output or args.project/'results').expanduser().resolve()
    try:
        import matplotlib
        matplotlib.use('Agg')
        import numpy as np
        from convergence_analysis.common import CASES, load_runs, result_path, write_csv, write_json
        from convergence_analysis.scan import load_cached, scan
        from convergence_analysis.plots import make_plots
        from convergence_analysis.summary import make_summary
        if not np.isfinite(args.core_radius) or args.core_radius <= 0:
            raise ValueError('--core-radius must be finite and positive.')
        if args.replot:
            source = (args.replot_from or output).expanduser().resolve()
            record = json.loads((source/'settings/analysis_settings.json').read_text())
            missing = [case for case in CASES if case not in record.get('sources', {})]
            if missing:
                raise ValueError('Saved results do not include '+', '.join(missing)+
                                 '. Run once without --replot to analyse all four runs.')
            snapshots = record['snapshots']
            tables = {case: load_cached(source, case) for case in CASES}
        else:
            runs, iterations = load_runs(args.project, args.baseline)
            snapshots = sorted(set(args.snapshots))
            if not snapshots or not set(snapshots).issubset(set(map(int, iterations))):
                raise ValueError(f'Each snapshot must be an exact saved iteration. Requested: {snapshots}')
            # Protect raw inputs from an accidentally chosen output path.
            for run in runs:
                if output == run.path or run.path in output.parents or output in run.path.parents:
                    raise ValueError('Choose a results output outside each simulation run directory.')
            print(f'Validated {len(runs)} completed R8 runs; only Nm and p_nt vary.', flush=True)
            print(f'{len(iterations)} matched snapshots; last saved iteration {iterations[-1]}.', flush=True)
            if args.check_only:
                return 0
            record = dict(version='1.1', study='R8 angular particle sampling and azimuthal modes',
                baseline=str(args.baseline), sources={r.case: str(r.path) for r in runs},
                configurations={r.case: r.config for r in runs},
                snapshots=snapshots, iterations=list(map(int, iterations)),
                energy_cuts_MeV=[20., 5.], core_radius_um=args.core_radius,
                moment_min_effective_macroparticles=20,
                beam_definition='Centred charge-weighted moments in the first complete wake period, with electrons selected independently at each saved iteration.',
                status='running')
            write_json(output/'settings/analysis_settings.json', record)
            tables = {r.case: scan(r, iterations, snapshots, output, args.core_radius, args.force) for r in runs}
        make_plots(tables, snapshots, output, record)
        make_summary(tables, output, record)
        write_csv(output/'03_electrons/data/beam_evolution.csv',
                  [row for case in CASES for row in tables[case]['beam'] if row['energy_cut_MeV'] == 20.])
        for table, section, name in (('maximum', '03_electrons', 'maximum_energy'),
                                     ('laser', '02_laser', 'laser'), ('wake', '01_wake', 'wake_evolution')):
            write_csv(output/section/'data'/(name+'.csv'),
                      [row for case in CASES for row in tables[case][table]])
        write_json(result_path(output, '03_electrons', 'settings', 'beam_settings.json'), dict(
            energy_cut_MeV=20., sensitivity_cut_MeV=5., core_radius_um=record['core_radius_um'],
            population='Main accelerated population in the first complete wake period between the nearest two robust negative-to-positive m=0 Ez crossings behind the measured laser peak.',
            selection='Saved positive-charge records inside the grid with uz>=1 and K>=cut. Primary moments have no radial cut; the compact population additionally has r<=core_radius about the simulation axis.',
            moments='Centred charge-weighted population moments. Normalised emittance uses x,ux and y,uy. RMS projected divergence uses atan2(ux,uz) and atan2(uy,uz).',
            display='Moment curves require at least 20 effective macroparticles. Every selected positive-charge record contributes to charge.',
            empty_or_unresolved='Resolved empty selections have zero charge and undefined moments. Unresolved wake windows have undefined charge and moments.',
            identity='Instantaneous physical records; IDs are not matched, deduplicated or tracked across runs.',
            implementation='analysis/convergence_analysis/beam.py'))
        write_json(result_path(output, '05_sensitivity_check', 'settings', 'sensitivity_settings.json'), dict(
            energy_cuts_MeV=[5., 20.], population='Main accelerated population in the selected wake period, no radial cut.',
            purpose='Check whether angular-setting differences depend on the analysis energy threshold; not a separate numerical convergence test.'))
        record['status'] = 'complete'
        write_json(output/'settings/analysis_settings.json', record)
        print(f'Analysis complete: {output}', flush=True)
        print(f'Read the quantitative comparison in {output}/04_convergence/summary.md', flush=True)
        return 0
    except KeyboardInterrupt:
        print('\nAnalysis stopped. Rerun the same command; completed run tables will be reused.', file=sys.stderr)
        return 130
    except ImportError as exc:
        print(f'Missing dependency: {exc}. Activate your existing fbpic environment.\n'
              'Required: numpy scipy matplotlib h5py openPMD-viewer.', file=sys.stderr)
        return 1
    except (OSError, ValueError, KeyError) as exc:
        print(f'Cannot analyse: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
