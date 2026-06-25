"""Bias sweeps and derived observables for self-consistent dc transport."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import CONST
from .stack import Device

_TINY = 1e-30


@dataclass
class DCBiasSweep:
    """Result of a self-consistent bias sweep."""

    V: np.ndarray
    I_dc: np.ndarray
    T: float
    I_spin: np.ndarray | None = None
    I_plus: np.ndarray | None = None
    I_minus: np.ndarray | None = None
    eta_internal: np.ndarray | None = None
    iterations: np.ndarray | None = None
    residual: np.ndarray | None = None
    converged: np.ndarray | None = None

    @property
    def eta(self) -> np.ndarray:
        """Photocurrent-derived recombination/extraction proxy.

        Uses the maximum current in the sweep as the extraction-dominated reference.
        If `eta_internal` is available, prefer it for rate-balance analysis because it
        is computed from generation and extracted current.
        """
        return self.I_dc.max() / np.maximum(self.I_dc, _TINY) - 1.0

    @property
    def didv(self) -> np.ndarray:
        """Differential conductance dI_dc/dV."""
        return np.gradient(self.I_dc, self.V)

    def maximum_current_bias(self) -> float:
        """Bias at which the extracted dc photocurrent is maximum."""
        return float(self.V[np.argmax(self.I_dc)])

    def minimum_current_bias(self) -> float:
        """Bias at which the extracted dc photocurrent is minimum."""
        return float(self.V[np.argmin(self.I_dc)])


def _active_generation_flux(device: Device, dx: float, T: float) -> float:
    """Photogeneration flux (cm^-2 s^-1) before the first blocking barrier layer."""
    from .transport import generation_profile

    z = device.stack.grid(dx)
    z_barrier = device.stack.thickness
    cum = 0.0
    for layer in device.stack.layers:
        if layer.role == "barrier":
            z_barrier = cum
            break
        cum += layer.thickness
    G = generation_profile(device, dx, T)
    active = z < z_barrier
    return float(np.trapezoid(G[active], dx=dx * 1e-7))


def _gauss_smooth_bias(V, y, sigma):
    """Convolve y(V) with a normalized Gaussian of width `sigma` in volts."""
    out = np.empty_like(y)
    for i in range(len(V)):
        w = np.exp(-0.5 * ((V - V[i]) / sigma) ** 2)
        out[i] = float(np.dot(w, y) / w.sum())
    return out


def bias_sweep_sc(
    device: Device,
    biases,
    T: float = 100.0,
    dx: float = 1.0,
    intensity_scale: float = 1.0,
    phiB_sigma: float = 0.0,
    compute_spin_current: bool = True,
    spin_lifetime: float = 1.4e-10,
    generation_polarization: float | None = None,
    current_asymmetry: float | None = None,
    **solve_kw,
) -> DCBiasSweep:
    """Run a self-consistent photocurrent sweep.

    Each bias point is solved with `spinpd.transport_sc.solve_sc`, which couples
    nonlinear Poisson, electron/hole continuity, optical generation, and
    recombination. When `compute_spin_current` is true, the sweep also evaluates
    the macroscopic helicity-current observable on each converged solution.
    `phiB_sigma` optionally applies Gaussian bias-domain smoothing to represent
    lateral contact-barrier inhomogeneity at the device level.
    """
    from .transport_sc import solve_sc, spin_current

    biases = np.asarray(biases, dtype=float)
    I_dc = np.empty_like(biases)
    I_spin = np.full_like(biases, np.nan)
    I_plus = np.full_like(biases, np.nan)
    I_minus = np.full_like(biases, np.nan)
    iterations = np.empty_like(biases, dtype=int)
    residual = np.empty_like(biases)
    converged = np.empty_like(biases, dtype=bool)

    for i, V in enumerate(biases):
        result = solve_sc(
            device,
            bias_V=float(V),
            T=T,
            dx=dx,
            intensity_scale=intensity_scale,
            **solve_kw,
        )
        if compute_spin_current:
            spin = spin_current(
                device,
                result,
                T=T,
                dx=dx,
                spin_lifetime=spin_lifetime,
                generation_polarization=generation_polarization,
                current_asymmetry=current_asymmetry,
            )
            I_dc[i] = spin.I_dc
            I_spin[i] = spin.I_spin
            I_plus[i] = spin.I_plus
            I_minus[i] = spin.I_minus
        else:
            I_dc[i] = result.I_dc
        iterations[i] = result.iterations
        residual[i] = result.residual
        converged[i] = result.converged

    if phiB_sigma > 0 and biases.size > 2:
        I_dc = _gauss_smooth_bias(biases, I_dc, phiB_sigma)
        if compute_spin_current:
            I_spin = _gauss_smooth_bias(biases, I_spin, phiB_sigma)
            I_plus = _gauss_smooth_bias(biases, I_plus, phiB_sigma)
            I_minus = _gauss_smooth_bias(biases, I_minus, phiB_sigma)

    G_active = _active_generation_flux(device, dx, T) * intensity_scale
    extraction = I_dc / (CONST.q * device.pump.spot_area_cm2)
    eta_internal = G_active / np.maximum(extraction, _TINY) - 1.0

    return DCBiasSweep(
        V=biases,
        I_dc=I_dc,
        T=T,
        I_spin=I_spin if compute_spin_current else None,
        I_plus=I_plus if compute_spin_current else None,
        I_minus=I_minus if compute_spin_current else None,
        eta_internal=eta_internal,
        iterations=iterations,
        residual=residual,
        converged=converged,
    )
