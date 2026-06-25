"""Scalar contact-extraction boundary condition for transport solves.

The self-consistent transport solver needs a surface extraction velocity `v_ext`
(cm/s) at the collecting contact:

    J_extract = v_ext * (n(0) - n_eq(0))

This module provides a compact phenomenological implementation for that boundary
condition.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import CONST
from .stack import Device


def emission_velocity(m_rel: float, T: float) -> float:
    """Thermal emission velocity v_R = sqrt(kT / (2 pi m*)), cm/s."""
    m = m_rel * CONST.m0
    v_si = np.sqrt(CONST.kB * T / (2.0 * np.pi * m))
    return v_si * 100.0


def barrier_transmission(
    barrier_height_eV: float,
    thickness_nm: float,
    m_rel_ox: float = 0.4,
    field_V_per_nm: float = 0.0,
) -> float:
    """WKB-like barrier transmission used for the scalar extraction velocity.

    A positive `field_V_per_nm` lowers the effective barrier using a simple
    triangular-barrier approximation.
    """
    d = thickness_nm * 1e-9
    Phi_eff = barrier_height_eV - 0.5 * field_V_per_nm * thickness_nm
    Phi_eff = max(Phi_eff, 1e-3)
    m = m_rel_ox * CONST.m0
    kappa = np.sqrt(2.0 * m * Phi_eff * CONST.q) / CONST.hbar
    return float(np.exp(-2.0 * kappa * d))


@dataclass
class TunnelExtraction:
    """Scalar extraction velocity at the collecting contact."""

    device: Device
    m_rel: float = 0.067
    m_rel_ox: float = 0.4

    def effective_barrier(
        self,
        bias_V: float = 0.0,
        barrier_height: float | None = None,
    ) -> float:
        """Bias-dependent effective barrier height, in eV."""
        c = self.device.left
        Phi0 = c.barrier_height if barrier_height is None else barrier_height
        return max(Phi0 + c.barrier_modulation * bias_V, 0.05)

    def v_ext(
        self,
        T: float,
        bias_V: float = 0.0,
        interface_field: float = 0.0,
        barrier_height: float | None = None,
    ) -> float:
        """Extraction velocity, cm/s."""
        c = self.device.left
        Phi = self.effective_barrier(bias_V, barrier_height)
        v_R = emission_velocity(self.m_rel, T)
        Tt = barrier_transmission(Phi, c.barrier_thickness, self.m_rel_ox, interface_field)
        return v_R * Tt * c.gamma0

    def rate(
        self,
        T: float,
        active_width_nm: float,
        bias_V: float = 0.0,
        interface_field: float = 0.0,
        barrier_height: float | None = None,
    ) -> float:
        """Lumped extraction rate gamma = v_ext / d_active, 1/s."""
        v_cm_s = self.v_ext(T, bias_V, interface_field, barrier_height)
        return v_cm_s / (active_width_nm * 1e-7)
