"""Recombination models: radiative, Shockley-Read-Hall, Auger, and QD capture.

All rates are net recombination rates in cm^-3 s^-1 (positive = net loss of
carriers), vanishing at equilibrium (n=n0, p=p0).  The InAs QD sheet is treated
as a *capture-residence trap* with TWO time-scales: a fast CAPTURE into the dot
(``qd_capture_rate``, ~ps -- the dot's large capture cross-section) followed by
the slower TRION recombination ceiling ``R_max = N_t / tau_trion`` (650 ps, the
TRPL value).  The fast capture localizes the recombination AT the dot (on the
time axis, with no band offset), raising the effective recombination rate ``r``
and thus the competition parameter ``eta = r / gamma``.  This is the physically
correct way to put a QD in a 1-D classical drift-diffusion model: a confining
*potential well* would over-trap and break the Scharfetter-Gummel flux, whereas
a fast trap holds the carrier by *residence time*, exactly as a real dot does.

These z-resolved rates feed the drift-diffusion continuity equations
(:mod:`spinpd.transport`).  In the 0-D / rate-equation limit the volume-averaged
``r`` is what competes with the tunneling extraction ``gamma``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .stack import Device


def radiative_rate(n, p, n0, p0, B):
    """Net radiative (bimolecular) recombination, cm^-3 s^-1.  B in cm^3/s."""
    return B * (n * p - n0 * p0)


def srh_rate(n, p, ni, tau_n, tau_p):
    """Net Shockley-Read-Hall recombination (midgap traps), cm^-3 s^-1.

    R = (n p - ni^2) / (tau_p (n + ni) + tau_n (p + ni)).
    """
    return (n * p - ni**2) / (tau_p * (n + ni) + tau_n * (p + ni))


def auger_rate(n, p, n0, p0, C):
    """Net Auger (band-to-band, 3-carrier) recombination, cm^-3 s^-1.

    R = (C_n n + C_p p)(n p - n0 p0), here with C_n = C_p = ``C``.  Scales as n^3
    at high injection (n=p), so it is negligible at ordinary injection and only
    competes at very high carrier density (for GaAs, n >~ 1e20).  Vanishes at
    equilibrium (n p = n0 p0).
    """
    return C * (n + p) * (n * p - n0 * p0)


def qd_capture_rate(n, n0, capture_rate, mask, R_max=None):
    """Non-radiative capture of excess electrons in the QD layer, cm^-3 s^-1.

    ``capture_rate`` (1/s) is the FAST capture into the dot (~ps; physically
    sigma*v_th*N_t, the dot's capture cross-section), NOT the recombination time.
    Low-injection: ``R_qd = capture_rate * (n - n0)``.

    With a finite ``R_max`` (A4 -- the dot can only process N_t/tau_trion
    electrons per unit volume and time, the slow TRION recombination, 650 ps) the
    rate SATURATES at the residence-limited ceiling:

        R_qd = R_max * (k*dn) / (k*dn + R_max),  k = capture_rate, dn = n - n0,

    which is ``k*dn`` at low injection and ``R_max`` at high injection.  ``R_max``
    is +inf (or None) -> the original linear capture.  The two time-scales matter:
    a fast capture (small 1/k) with a 650 ps trion (R_max) localizes ~90% of the
    recombination at the dot without over-trapping the reverse photocurrent.
    """
    dn = np.maximum(n - n0, 0.0)
    lin = capture_rate * dn
    if R_max is None:
        R = lin
    else:
        with np.errstate(divide="ignore", invalid="ignore"):
            R = np.where(np.isfinite(R_max), R_max * lin / (lin + R_max), lin)
    return np.where(mask, R, 0.0)


@dataclass
class RecombinationModel:
    """Assembles z-resolved recombination rates for a device.

    Precomputes per-node material arrays (B, SRH lifetimes, ni) and the QD
    capture profile; :meth:`rates` then evaluates the channels for given carrier
    densities.
    """

    device: Device
    dx: float
    T: float

    def __post_init__(self):
        s = self.device.stack
        self.B = s.material_property(self.dx, "B_rad")
        self.tau_srh = s.material_property(self.dx, "tau_srh")
        self.C_auger = s.material_property(self.dx, "C_auger")
        self.ni = s.material_property(self.dx, "ni", self.T)
        self.qd_rate = s.map_layer(self.dx, lambda l: l.params.get("qd_capture_rate", 0.0))
        self.qd_mask = self.qd_rate > 0.0
        # A4: optional saturable QD trap -> R_max = N_t / tau_trion (else +inf = linear)
        Nt = s.map_layer(self.dx, lambda l: l.params.get("qd_trap_density", 0.0))
        tau_tr = s.map_layer(self.dx, lambda l: l.params.get("qd_trion_time", 0.0))
        self.qd_Rmax = np.where((Nt > 0) & (tau_tr > 0), Nt / np.where(tau_tr > 0, tau_tr, 1.0), np.inf)

    def rates(self, n, p, n0, p0) -> dict:
        """Return the recombination channels and total (cm^-3 s^-1)."""
        R_rad = radiative_rate(n, p, n0, p0, self.B)
        R_srh = srh_rate(n, p, self.ni, self.tau_srh, self.tau_srh)
        R_qd = qd_capture_rate(n, n0, self.qd_rate, self.qd_mask, R_max=self.qd_Rmax)
        R_aug = auger_rate(n, p, n0, p0, self.C_auger)
        return {"rad": R_rad, "srh": R_srh, "qd": R_qd, "auger": R_aug,
                "total": R_rad + R_srh + R_qd + R_aug}

    def tau_eff(self, n, p, n0, p0):
        """Effective excess-electron lifetime tau = (n - n0) / R_total, s.

        Returned per node; large where recombination is weak.  Guards against
        division by zero at/near equilibrium.
        """
        R = self.rates(n, p, n0, p0)["total"]
        dn = n - n0
        with np.errstate(divide="ignore", invalid="ignore"):
            tau = np.where(np.abs(R) > 0, dn / R, np.inf)
        return tau

    def average_rate(self, n, p, n0, p0) -> float:
        """Volume-averaged recombination rate r (1/s): <R>/<excess n>.

        This is the lumped ``r`` that competes with the extraction rate gamma in
        the rate-equation picture (eta = r / gamma).
        """
        R = self.rates(n, p, n0, p0)["total"]
        dn = n - n0
        num = np.trapezoid(R, dx=self.dx)
        den = np.trapezoid(dn, dx=self.dx)
        return float(num / den) if den != 0 else 0.0
