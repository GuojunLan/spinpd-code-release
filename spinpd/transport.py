"""Shared optical-generation and Scharfetter-Gummel helper functions."""
from __future__ import annotations

import numpy as np

from .stack import Device


def bernoulli(x):
    """Bernoulli function B(x) = x / (exp(x) - 1), evaluated stably."""
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    small = np.abs(x) < 1e-10
    out[small] = 1.0 - x[small] / 2.0
    xs = x[~small]
    out[~small] = xs / np.expm1(xs)
    return out


def generation_profile(device: Device, dx: float, T: float, alpha_cm: float = 1e4) -> np.ndarray:
    """Photogeneration rate G(z), in cm^-3 s^-1.

    The optical model is intentionally simple: Beer-Lambert absorption from the
    illuminated side, with absorption enabled only in layers whose band gap is below
    the pump photon energy.
    """
    stack = device.stack
    z = stack.grid(dx)
    E_ph = device.pump.photon_energy_eV
    absorbs = stack.material_property(dx, "Eg", T) < E_ph
    alpha = np.where(absorbs, alpha_cm, 0.0)
    dxc = dx * 1e-7
    order = slice(None) if device.illuminated_from_left else slice(None, None, -1)
    a = alpha[order]
    tau_opt = np.cumsum(a * dxc) - a * dxc
    flux = device.pump.photon_flux() * np.exp(-tau_opt)
    G = (a * flux)[order]
    return G


def _bernoulli_pair(potential, kT):
    """Return Scharfetter-Gummel Bernoulli arrays B(delta), B(-delta)."""
    delta = (potential[1:] - potential[:-1]) / kT
    return bernoulli(delta), bernoulli(-delta)
