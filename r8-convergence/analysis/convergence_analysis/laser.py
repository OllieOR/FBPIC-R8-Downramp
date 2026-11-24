"""Saved optical energy and spot size, run through analyse.py."""

import numpy as np
from scipy.constants import epsilon_0, mu_0

from .common import edges, mesh

BANDS = {"main": (.25, 2.5), "wide": (.15, 2.5), "strict": (.40, 2.5)}


def geometry(r, z, wavelength, nplanes):
    """Signed-r planes represent opposite azimuths; integrate real cell volumes."""
    dr, dz = np.mean(np.diff(r)), np.mean(np.diff(z))
    if (not np.allclose(np.diff(r), dr, rtol=1.e-7, atol=1.e-15)
            or not np.allclose(np.diff(z), dz, rtol=1.e-7, atol=1.e-15)
            or not np.allclose(r, -r[::-1], rtol=1.e-7, atol=1.e-15)
            or not r[0] < 0 < r[-1]):
        raise ValueError("Expected uniform z and symmetric signed-r cell centres.")
    radial_edges, z_edges = edges(r), edges(z)
    k0 = 2*np.pi/wavelength
    ratio = 2*np.pi*np.fft.fftfreq(len(z), dz)/k0
    if np.pi/dz <= BANDS["main"][1]*k0:
        raise ValueError("The saved z grid does not resolve the requested optical band.")
    masks = {name: (ratio >= low) & (ratio <= high) for name, (low, high) in BANDS.items()}
    # Each plane supplies theta and theta+pi. This is the exact annular
    # cell area for one angular sample, including the near-axis cells.
    area = np.pi/nplanes * .5*abs(np.diff(radial_edges**2))
    rmax = float(max(abs(radial_edges)))
    return dict(area=area, dz=float(dz), masks=masks, rmax=rmax,
                radial_edge=abs(r) >= .9*rmax,
                z_edge=np.minimum(z-z_edges[0], z_edges[-1]-z) < 2.e-6)


def measure(run, iteration):
    """Optical-band E/B energy and a centroid-corrected RMS spot radius.

    Reconstruct all saved cylindrical modes. With Nm modes the squared
    fields have angular order at most 2*(Nm-1), so 2*Nm equally spaced
    azimuths integrate their energy and first transverse moments exactly.
    Negative-r samples supply half of those azimuths.

    A one-sided optical FFT gives the complex analytic field. Its squared
    magnitude gives cycle-averaged energy density with coefficients eps0/4
    and 1/(4*mu0). Spatial filtering separates the optical field from the
    slowly varying wake; the result is saved optical energy inside the box.
    """
    series = run.series()
    nmodes = int(run.config["Nm"])
    if nmodes < 1:
        raise ValueError("No saved azimuthal modes.")
    nplanes = max(2, nmodes)
    energies = {name: 0. for name in BANDS}
    profile = radial_moment = x_moment = y_moment = None
    edge_radial_energy = 0.
    g = r = z = None
    for angle in np.arange(nplanes)*np.pi/nplanes:
        density = None
        for field in ("E", "B"):
            coefficient = epsilon_0/4 if field == "E" else 1/(4*mu_0)
            for coord in ("r", "t", "z"):
                values, rr, zz = mesh(series, iteration, field, coord, mode="all", theta=float(angle))
                if g is None:
                    r, z = rr, zz
                    g = geometry(r, z, run.config["lambda0"], nplanes)
                    profile, radial_moment, x_moment, y_moment = [np.zeros(len(z)) for _ in range(4)]
                if (values.shape != (len(r), len(z))
                        or not np.allclose(rr, r, rtol=1.e-9, atol=1.e-15)
                        or not np.allclose(zz, z, rtol=1.e-9, atol=1.e-15)):
                    raise ValueError("Electric and magnetic fields must share the same grid.")
                spectrum = np.fft.fft(values, axis=1)
                spectral_energy = (4*coefficient*g["dz"]/len(z)
                                   * np.sum(g["area"][:, None]*abs(spectrum)**2, axis=0))
                for name, mask in g["masks"].items():
                    energies[name] += float(spectral_energy[mask].sum())
                spectrum[:, ~g["masks"]["main"]] = 0.
                analytic = np.fft.ifft(2*spectrum, axis=1)
                component = coefficient*abs(analytic)**2
                density = component if density is None else density+component
        weighted = density*g["area"][:, None]
        profile += weighted.sum(axis=0)
        radial_moment += (weighted*r[:, None]**2).sum(axis=0)
        x_moment += (weighted*r[:, None]*np.cos(angle)).sum(axis=0)
        y_moment += (weighted*r[:, None]*np.sin(angle)).sum(axis=0)
        edge_radial_energy += float(weighted[g["radial_edge"]].sum()*g["dz"])

    energy = float(profile.sum()*g["dz"])
    if not np.isfinite(energy) or energy <= 0:
        raise ValueError(f"{run.case} {iteration}: no measurable optical-band energy.")
    if not np.isclose(energy, energies["main"], rtol=1.e-10, atol=0.):
        raise ValueError("Optical energy disagrees with its Parseval integral.")
    peak = int(np.argmax(profile))
    xbar, ybar = x_moment[peak]/profile[peak], y_moment[peak]/profile[peak]
    variance = radial_moment[peak]/profile[peak]-xbar*xbar-ybar*ybar
    radial_fraction = edge_radial_energy/energy
    longitudinal_fraction = float(profile[g["z_edge"]].sum()*g["dz"]/energy)
    return dict(case=run.case, iteration=int(iteration),
                time_ps=float(iteration*run.config["dt"]*1.e12),
                optical_energy_mJ=energy*1000,
                wide_band_energy_mJ=energies["wide"]*1000,
                strict_band_energy_mJ=energies["strict"]*1000,
                relative_cutoff_spread=(energies["wide"]-energies["strict"])/energy,
                spot_radius_um=float(np.sqrt(2*max(0., variance))*1.e6),
                pulse_peak_z_um=float(z[peak]*1.e6),
                centroid_x_um=float(xbar*1.e6), centroid_y_um=float(ybar*1.e6),
                radial_edge_energy_fraction=radial_fraction,
                longitudinal_edge_energy_fraction=longitudinal_fraction,
                boundary_flag=bool(max(radial_fraction, longitudinal_fraction) >= .01))

