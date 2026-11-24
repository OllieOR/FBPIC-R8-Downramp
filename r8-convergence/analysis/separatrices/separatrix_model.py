"""Frozen on-axis Hamiltonian solver used by the downramp R8 analysis."""

import numpy as np
from scipy.constants import c, e, m_e
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import CubicSpline, PchipInterpolator
from scipy.optimize import brentq, minimize_scalar
from scipy.signal import find_peaks

MC2_MEV = m_e * c**2 / e / 1e6


def potential(x, ez):
    # Same m=0 field, integration sign, units and gauge as the existing R8 code.
    psi = -1e-3/MC2_MEV*cumulative_trapezoid(ez, x, initial=0)
    return psi-np.interp(0, x, psi)


def kinetic(u, q, beta):
    u = np.asarray(u)
    gamma = np.sqrt(q+u*u)
    return np.where(u >= 0, q/(gamma+np.abs(u))+(1-beta)*u, gamma-beta*u)


def branches(x, level, model):
    q, beta = model['q2'](x), model['beta']
    a = level+model['psi'](x)
    delta = (1-beta)*(1+beta)
    gap = a-np.sqrt(q*delta)
    valid = (a > 0) & (gap >= -2e-9)
    root = np.sqrt(np.maximum(gap, 0)*(a+np.sqrt(q*delta)).clip(min=0))
    den = beta*a+root
    low = np.full(np.shape(a), np.nan)
    np.divide(q-a*a, den, out=low, where=valid & (den != 0))
    high = np.where(valid, den/delta, np.nan)
    if np.any(valid):
        residual = max(float(np.max(np.abs(kinetic(low[valid],q[valid],beta)
                                          -model['psi'](x[valid])-level))),
                       float(np.max(np.abs(kinetic(high[valid],q[valid],beta)
                                          -model['psi'](x[valid])-level))))
        if residual > 1e-8:
            raise ValueError(f'Contour Hamiltonian residual too large: {residual:g}')
    else:
        residual = 0.
    return low, high, residual


def extremum(g, left, right, maximum):
    sign = -1 if maximum else 1
    fit = minimize_scalar(lambda x: sign*float(g(x)), bounds=(left,right),
                          method='bounded', options={'xatol':1e-12})
    return float(fit.x)


def turn(g, centre, end, level):
    xs = np.linspace(centre,end,15001)
    vals = g(xs)-level
    hits = np.flatnonzero(vals >= 0)
    if len(hits):
        j = int(hits[0])
        if j == 0:
            raise ValueError('Limiting level does not enclose the well centre.')
        return float(brentq(lambda x: float(g(x)-level),
                           *sorted([xs[j-1],xs[j]]),xtol=1e-12))
    if abs(float(g(end)-level)) < 2e-9:
        return float(end)
    raise ValueError('Unresolved turning point: refusing to invent closure.')


def make_model(x, psi_values, a2_values, beta, bounds, require_front=True):
    if not 0 < beta < 1:
        raise ValueError('Reference beta must be subluminal.')
    if not np.all(np.isfinite(np.r_[x,psi_values,a2_values,bounds])):
        raise ValueError('Non-finite profile or first-bucket geometry.')
    psi = CubicSpline(x,psi_values,extrapolate=False)
    q2 = PchipInterpolator(x,1+np.maximum(a2_values,0),extrapolate=False)
    delta = (1-beta)*(1+beta)
    g = lambda xx: np.sqrt(q2(xx)*delta)-psi(xx)
    model = dict(x=x,psi=psi,q2=q2,beta=float(beta),g=g)
    # The old first-period interval brackets the rear saddle between two wells.
    xx = np.linspace(bounds[0],bounds[1],12001)
    j = int(np.argmax(g(xx)))
    if j in (0,len(xx)-1):
        raise ValueError('No resolved interior rear saddle.')
    rear = extremum(g,xx[j-1],xx[j+1],True)
    model.update(rear=rear,hr=float(g(rear)))
    # The first-bucket well lies in front of the rear saddle, near the front
    # field-period boundary. The optical ponderomotive term matters near laser.
    xx = np.linspace(rear+.05,min(2.,x[-1]-.1),20001)
    minima, _ = find_peaks(-g(xx),prominence=1e-7)
    if not len(minima):
        raise ValueError('No resolved first-bucket well centre.')
    nearby=minima[abs(xx[minima]-bounds[1])<=4.]
    if not len(nearby):
        raise ValueError('No well centre near the established first-bucket boundary.')
    # Choose the actual well bottom, not a tiny ripple nearest a smoothed root.
    j = int(nearby[np.argmin(g(xx[nearby]))])
    centre = extremum(g,xx[j-1],xx[j+1],False)
    model['centre'] = centre
    xx = np.linspace(centre+.01,x[-1]-.1,30001)
    peaks, props = find_peaks(g(xx),prominence=1e-7)
    if not len(peaks):
        if require_front:
            raise ValueError('No resolved forward saddle. No closed region was drawn.')
        model.update(front=None,closed=False)
        return model
    k = int(np.argmax(g(xx[peaks])))
    j = int(peaks[k])
    front = extremum(g,xx[j-1],xx[j+1],True)
    hf = float(g(front))
    level = min(model['hr'], hf)
    left = turn(g,centre,rear,level)
    right = turn(g,centre,front,level)
    phase = .5*(left+right)-.5*(right-left)*np.cos(np.linspace(0,np.pi,5001))
    low,high,residual = branches(phase,level,model)
    if not np.all(np.isfinite(np.r_[low,high])):
        raise ValueError('Forbidden interval inside candidate island.')
    for point, ii in [(left,0),(right,-1)]:
        if abs(float(g(point))-level)>2e-8:
            raise ValueError('Endpoint is not a verified turning point.')
        low[ii]=high[ii]=beta*np.sqrt(q2(point)/delta)
    model.update(front=front,hf=hf,level=level,left=left,right=right,
                 phase=phase,low=low,high=high,residual=residual,
                 prominence=float(props['prominences'][k]),closed=True)
    return model
