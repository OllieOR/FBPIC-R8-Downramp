"""R8 density-downramp simulation and C4 constant-density control.

    python lwfa_r8.py --case R8 --dry-run
    python -u lwfa_r8.py --case R8
    python -u lwfa_r8.py --case C4

R8: entrance 30--70 um, downramp 110--118 um from 4e18 to 3e18 cm^-3.
C4: the same setup with the 4e18 cm^-3 plateau retained. Both target 10800
steps. The exit ramp at 900--1000 um is beyond the simulated propagation.

Run settings and the original input are saved beside the diagnostics.
The FBPIC licence is retained below.
"""

import argparse
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import shutil

import numpy as np
from scipy.constants import c, e, m_e


# Grid: retain dz=50 nm and dr=400 nm while extending the physical box.
use_cuda = False
n_order = -1
Nz = 1200
zmax = 30.e-6
zmin = -30.e-6
Nr = 100
rmax = 40.e-6
Nm = 2
dt = (zmax - zmin) / Nz / c
boundaries = {'z': 'open', 'r': 'open'}

# Pre-ionised plasma, with the same immobile-background treatment as the source.
p_zmin = 30.e-6
p_zmax = 1.e-3
p_rmax = 38.e-6
n_e = 4.e24
p_nz = 2
p_nr = 2
p_nt = 8  # FBPIC recommends at least 4*Nm azimuthal loading positions.

# Retain the original laser; specify its previous defaults explicitly.
a0 = 4.
w0 = 5.e-6
tau = 16.e-15
z0 = 15.e-6
lambda0 = .8e-6
zf = z0
theta_pol = 0.
v_window = c

ramp_start = 30.e-6
ramp_length = 40.e-6
downramp_start = 110.e-6
downramp_length = 8.e-6
lower_density_factor = .75
exit_start = 900.e-6
exit_length = 100.e-6

# Specify actual window travel directly, independently of the box length.
window_travel = 540.e-6
N_step = round(window_travel / (v_window * dt))
T_interact = N_step * dt
checkpoint_period = 2000
random_seed = 1729

# (inclusive start, exclusive end, field/selected period, unfiltered period)
diagnostic_schedule = (
    (0, 1500, 50, 200),
    (1500, 3000, 10, 20),
    (3000, N_step + 1, 50, 200),
)


def density_for_case(case):
    """Build an FBPIC-compatible two-argument density function."""
    if case not in ('R8', 'C4'):
        raise ValueError(f'Unknown density case: {case}')
    factor = lower_density_factor if case == 'R8' else 1.

    def dens_func(z, r):
        """Return initial density / n_e; r is unused in the loaded radius."""
        z = np.asarray(z, dtype=float)
        n = np.clip((z - ramp_start) / ramp_length, 0., 1.)
        drop = np.clip((z - downramp_start) / downramp_length, 0., 1.)
        n *= 1. - (1. - factor) * drop
        n *= 1. - np.clip((z - exit_start) / exit_length, 0., 1.)
        return n

    return dens_func


def configuration(case='R8'):
    """Settings that must agree when restarting this run."""
    keys = (
        'use_cuda', 'n_order', 'Nz', 'zmax', 'zmin', 'Nr', 'rmax', 'Nm',
        'dt', 'boundaries', 'p_zmin', 'p_zmax', 'p_rmax', 'n_e',
        'p_nz', 'p_nr', 'p_nt', 'a0', 'w0', 'tau', 'z0', 'lambda0',
        'zf', 'theta_pol', 'v_window', 'ramp_start', 'ramp_length',
        'downramp_start', 'downramp_length', 'lower_density_factor',
        'exit_start', 'exit_length', 'window_travel', 'N_step',
        'checkpoint_period', 'random_seed', 'diagnostic_schedule',
    )
    # Round-trip makes tuples match their saved JSON-list representation.
    result = json.loads(json.dumps({key: globals()[key] for key in keys}))
    result['case'] = case
    result['lower_density_factor'] = lower_density_factor if case == 'R8' else 1.
    return result


def validate_setup():
    """Check the intended physical profile and key numerical invariants."""
    positions = np.array([0., 30., 50., 70., 110., 114., 118., 500.,
                          900., 950., 1000.]) * 1.e-6
    expected = {
        'R8': [0., 0., .5, 1., 1., .875, .75, .75, .75, .375, 0.],
        'C4': [0., 0., .5, 1., 1., 1., 1., 1., 1., .5, 0.],
    }
    for case in ('R8', 'C4'):
        np.testing.assert_allclose(
            density_for_case(case)(positions, np.zeros_like(positions)),
            expected[case], rtol=1.e-12, atol=1.e-14)
    np.testing.assert_allclose((zmax-zmin)/Nz, 50.e-9, rtol=1.e-12)
    np.testing.assert_allclose(rmax/Nr, 400.e-9, rtol=1.e-12)
    np.testing.assert_allclose(N_step*dt*v_window, 540.e-6, rtol=1.e-12)
    if N_step != 10800 or p_nt < 4*Nm or p_rmax >= rmax:
        raise ValueError('Unexpected run length, angular sampling or plasma radius.')
    if zmax+window_travel >= exit_start:
        raise ValueError('Target exit intrudes into the simulated window.')
    if [x[:2] for x in diagnostic_schedule] != [(0,1500),(1500,3000),(3000,10801)]:
        raise ValueError('Diagnostic intervals do not cover the complete run.')


def environment():
    versions = {}
    for package in ('fbpic', 'numpy', 'scipy', 'numba', 'h5py', 'openPMD-viewer'):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = 'not found'
    return {
        'python': platform.python_version(), 'platform': platform.platform(),
        'packages': versions,
        'thread_environment': {key: os.environ.get(key) for key in (
            'NUMBA_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')},
    }


def check_output(parser, run_dir, restart, case='R8'):
    """Avoid mixing this experiment with another run or changed parameters."""
    if restart:
        config_path = run_dir / 'run_config.json'
        if not config_path.is_file():
            parser.error('No run_config.json found in this output directory.')
        if json.loads(config_path.read_text()) != configuration(case):
            parser.error('Settings differ from the saved case configuration. '
                         'Restore that setup or use a new output directory.')
        checkpoint_dir = run_dir / 'checkpoints'
        if not any(checkpoint_dir.rglob('*.h5')):
            parser.error(f'No checkpoints found in {checkpoint_dir}')
    elif run_dir.exists() and (not run_dir.is_dir() or any(run_dir.iterdir())):
        parser.error('Output path is occupied. Use --restart to continue this case, '
                     'or choose a new --output-dir.')


def configure_diagnostics(sim, elec, run_dir, FieldDiagnostic, ParticleDiagnostic):
    sim.diags = []
    for start, stop, period, history_period in diagnostic_schedule:
        limits = {'iteration_min': start, 'iteration_max': stop}
        sim.diags.extend([
            FieldDiagnostic(period, sim.fld, comm=sim.comm,
                            write_dir=str(run_dir / 'diags'), **limits),
            ParticleDiagnostic(
                period, {'electrons': elec}, select={'uz': [1., None]},
                particle_data=['position', 'momentum', 'weighting', 'E', 'B'],
                comm=sim.comm, write_dir=str(run_dir / 'diags'), **limits),
            ParticleDiagnostic(
                history_period, {'electrons': elec}, comm=sim.comm,
                write_dir=str(run_dir / 'particle_history'), **limits),
        ])


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                    formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--case', choices=('R8', 'C4'), default='R8')
    parser.add_argument('--output-dir', type=Path,
                        help='Default: downramp_comparison/CASE beside this script.')
    parser.add_argument('--restart', action='store_true',
                        help='Resume this case from its latest checkpoint.')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print settings without importing FBPIC or writing files.')
    args = parser.parse_args()
    run_dir = (args.output_dir or Path(__file__).resolve().parent /
               'downramp_comparison' / args.case).resolve()
    target = N_step
    validate_setup()

    print(f'{args.case} output: {run_dir}', flush=True)
    print(f'{target:,} steps; {target*dt*1.e12:.3f} ps; '
          f'{target*dt*v_window*1.e6:.0f} um window travel.', flush=True)
    print(f'Grid: {Nz} x {Nr}, Nm={Nm}; dz={(zmax-zmin)/Nz*1.e9:.0f} nm, '
          f'dr={rmax/Nr*1.e9:.0f} nm.', flush=True)
    factor = lower_density_factor if args.case == 'R8' else 1.
    if args.case == 'R8':
        print(f'Density: {n_e/1.e24:g} -> {n_e*factor/1.e24:g} x 10^18 cm^-3 over '
              f'{downramp_start*1.e6:.0f}--{(downramp_start+downramp_length)*1.e6:.0f} um.', flush=True)
    else:
        print(f'Density: {n_e/1.e24:g} x 10^18 cm^-3 plateau; no internal transition.', flush=True)
    if args.dry_run:
        return

    check_output(parser, run_dir, args.restart, args.case)
    completed = run_dir / 'completed.json'
    if args.restart and completed.is_file():
        if json.loads(completed.read_text()).get('iteration', 0) >= N_step:
            print('This case has already completed all target iterations.', flush=True)
            return
    # Keep imports here so --dry-run works without FBPIC installed.
    from fbpic.main import Simulation
    from fbpic.lpa_utils.laser import add_laser_pulse
    from fbpic.lpa_utils.laser.laser_profiles import GaussianLaser
    from fbpic.openpmd_diag import (
        FieldDiagnostic, ParticleDiagnostic,
        set_periodic_checkpoint, restart_from_checkpoint,
    )
    from fbpic.utils.random_seed import set_random_seed

    set_random_seed(random_seed)
    run_dir.mkdir(parents=True, exist_ok=True)
    if not args.restart:
        (run_dir / 'run_config.json').write_text(
            json.dumps(configuration(args.case), indent=2) + '\n')
        (run_dir / 'run_environment.json').write_text(
            json.dumps(environment(), indent=2) + '\n')
        shutil.copyfile(Path(__file__).resolve(), run_dir / 'run_input.py')
        profile_z = np.linspace(0., p_zmax, 4001)
        factor = density_for_case(args.case)(profile_z, np.zeros_like(profile_z))
        np.savetxt(run_dir / 'prescribed_density.csv',
                   np.column_stack((profile_z*1.e6, factor, factor*n_e,
                                    factor*n_e/1.e6)), delimiter=',',
                   header='z_um,density_factor,n_e_m-3,n_e_cm-3', comments='')

    sim = Simulation(Nz, zmax, Nr, rmax, Nm, dt, zmin=zmin,
                     n_order=n_order, use_cuda=use_cuda, boundaries=boundaries)
    elec = sim.add_new_species(
        q=-e, m=m_e, n=n_e, dens_func=density_for_case(args.case),
        p_zmin=p_zmin, p_zmax=p_zmax, p_rmax=p_rmax,
        p_nz=p_nz, p_nr=p_nr, p_nt=p_nt,
    )
    checkpoint_dir = run_dir / 'checkpoints'
    if args.restart:
        # Restore first: the moving window infers injection positions afterwards.
        restart_from_checkpoint(sim, checkpoint_dir=str(checkpoint_dir))
    else:
        add_laser_pulse(sim, GaussianLaser(
            a0, w0, tau, z0, zf=zf, lambda0=lambda0, theta_pol=theta_pol))
        elec.track(sim.comm)

    sim.set_moving_window(v=v_window)
    configure_diagnostics(sim, elec, run_dir, FieldDiagnostic, ParticleDiagnostic)
    set_periodic_checkpoint(sim, checkpoint_period,
                            checkpoint_dir=str(checkpoint_dir))
    remaining = max(0, target - int(sim.iteration))
    print(f'Starting at iteration {sim.iteration}; {remaining:,} cycles to target.',
          flush=True)
    if remaining:
        sim.step(remaining)
    print(f'Finished at iteration {sim.iteration}.', flush=True)
    if int(sim.iteration) >= N_step:
        completed.write_text(json.dumps({
            'case': args.case, 'iteration': int(sim.iteration),
            'simulation_time_s': float(sim.time),
            'window_travel_m': float(sim.iteration*dt*v_window),
        }, indent=2) + '\n')


if __name__ == '__main__':
    main()


# FBPIC licence and attribution
# =============================
# 
# This input is adapted from the standard
# laser-wakefield-acceleration example distributed with FBPIC:
# 
# https://fbpic.github.io/example_input/lwfa_script.html
# 
# The adapted input remains subject to the following upstream notice and licence.
# 
# “Fourier-Bessel Particle-In-Cell (FBPIC)”, Copyright (c) 2016,
# University of Hamburg and The Regents of the University of California,
# through Lawrence Berkeley National Laboratory (subject to receipt of
# any required approvals from the U.S. Dept. of Energy). All rights
# reserved.
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
# 
# (1) Redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer.
# 
# (2) Redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the distribution.
# 
# (3) Neither the name of the University of Hamburg, University of
# California, Lawrence Berkeley National Laboratory, U.S. Dept. of Energy
# nor the names of its contributors may be used to endorse or promote
# products derived from this software without specific prior written
# permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# “AS IS” AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# 
# You are under no obligation whatsoever to provide any bug fixes,
# patches, or upgrades to the features, functionality or performance of
# the Software (“Enhancements”) to anyone; however, if you choose to make
# your Enhancements available either publicly, or directly to Lawrence
# Berkeley National Laboratory, without imposing a separate written
# license agreement for such Enhancements, then you hereby grant the
# following license: a non-exclusive, royalty-free perpetual license to
# install, use, modify, prepare derivative works, incorporate into other
# computer software, distribute, and sublicense such enhancements or
# derivative works thereof, in binary and source code form.
# 
# Official upstream licence:
# https://github.com/fbpic/fbpic/blob/dev/LICENSE.txt
