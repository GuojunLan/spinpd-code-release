"""Heterostructure band profile via the nonlinear Poisson equation.

We solve, on the 1-D grid built by :mod:`spinpd.stack`,

    d/dz [ eps_r(z) dpsi/dz ] = -(q/eps0) * rho(z),
    rho(z) = p(z) - n(z) + ND(z) - NA(z),      (cm^-3)

with carriers in local equilibrium (Boltzmann) relative to a flat Fermi level
``EF = 0`` (equilibrium) or to contact Fermi levels offset by the applied bias:

    n(z) = Nc(z) exp[(EF - Ec(z)) / kT],   Ec(z) = Ec0(z) - psi(z)
    p(z) = Nv(z) exp[(Ev(z) - EF) / kT],   Ev(z) = Ev0(z) - psi(z)

Band edges ``Ec0/Ev0`` are referenced so that the GaAs conduction edge is 0 eV
(heterojunction offsets retained), keeping all energies O(1).  The equation is
solved by damped Newton iteration with Dirichlet potentials at the two contacts;
the built-in potential then emerges as ``psi(0) - psi(L)``.

Working units: nm, V, eV, cm^-3.

Caveat: Boltzmann statistics are used; the heavily doped p+ buffer (2e18 vs
Nv~1.8e18 at 100 K) is mildly degenerate and treated approximately as a contact
reservoir.  Fermi-Dirac can be added later if needed.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

from .config import CONST
from .stack import Device

#: Poisson coefficient so that eps_r * d2psi[V/nm^2] = -POISSON_C * N[cm^-3].
#: POISSON_C = (q/eps0) * (cm^-3 -> m^-3) * (nm^2 -> m^2) = (q/eps0)*1e6*1e-18.
POISSON_C = CONST.q / CONST.eps0 * 1e-12   # V * nm^2 * cm^3  (~1.81e-20)

#: reference electron affinity (GaAs); band edges are shifted by +CHI_REF.
CHI_REF = 4.07


@dataclass
class BandProfile:
    """Result of a band-profile solve."""

    z: np.ndarray            # nm
    psi: np.ndarray          # electrostatic potential, V (EF = 0 reference)
    Ec: np.ndarray           # conduction edge, eV (EF = 0)
    Ev: np.ndarray           # valence edge, eV
    Efield: np.ndarray       # electric field -dpsi/dz, V/nm
    n: np.ndarray            # equilibrium electron density, cm^-3
    p: np.ndarray            # equilibrium hole density, cm^-3
    Ec0: np.ndarray          # flat-band conduction edge (offsets only), eV
    Ev0: np.ndarray
    bias_V: float
    T: float
    iterations: int
    residual: float

    @property
    def Vbi(self) -> float:
        """Built-in potential psi(0) - psi(L), V."""
        return float(self.psi[0] - self.psi[-1])


def _shifted_edges(device: Device, dx: float, T: float):
    Ec0, Ev0 = device.stack.band_edges(dx, T)        # vacuum-referenced (~ -4 eV)
    return Ec0 + CHI_REF, Ev0 + CHI_REF                # GaAs CB -> 0 eV


def _carriers(psi, Ec0s, Ev0s, Nc, Nv, kT, EFn=0.0, EFp=0.0):
    """Boltzmann carrier densities given the potential (cm^-3).

    Electrons reference the electron quasi-Fermi level ``EFn`` (pinned to the
    n/left contact), holes the hole quasi-Fermi level ``EFp`` (pinned to the
    p/right contact).  In equilibrium EFn = EFp = 0; under applied bias V the
    split is EFn - EFp = V, modelled here as EFn = 0, EFp = -V (flat-QFL /
    depletion approximation, valid when recombination in the depletion region is
    weak).
    """
    Ec = Ec0s - psi
    Ev = Ev0s - psi
    n = Nc * np.exp(np.clip((EFn - Ec) / kT, -400, 400))
    p = Nv * np.exp(np.clip((Ev - EFp) / kT, -400, 400))
    return n, p, Ec, Ev


def _neutral_psi(Ec0s, Ev0s, Nc, Nv, ND, NA, kT, EFn=0.0, EFp=0.0):
    """Per-node charge-neutral potential (good Newton initial guess)."""
    psi = np.zeros_like(Ec0s)
    for _ in range(80):
        n, p, _, _ = _carriers(psi, Ec0s, Ev0s, Nc, Nv, kT, EFn, EFp)
        f = p - n + (ND - NA)
        df = (-p - n) / kT            # d(p-n)/dpsi = -p/kT - n/kT
        step = np.clip(f / df, -0.1, 0.1)
        psi = psi - step
        if np.max(np.abs(step)) < 1e-10:
            break
    return psi


def solve_equilibrium(device: Device, T: float = 100.0, dx: float = 1.0,
                      bias_V: float = 0.0, max_iter: int = 200,
                      tol: float = 1e-8) -> BandProfile:
    """Solve the nonlinear Poisson equation for the band profile.

    ``bias_V`` splits the quasi-Fermi levels (EFn = 0, EFp = -bias_V): the band
    bending becomes ``Vbi - bias_V`` and, unlike a single-Fermi-level shift, the
    applied voltage correctly modulates the junction depletion field rather than
    being screened by the neutral bulk.  Forward bias (> 0) flattens the bands.
    """
    stack = device.stack
    z = stack.grid(dx)
    N = z.size
    kT = CONST.kT_eV(T)
    EFn, EFp = 0.0, -bias_V

    eps = stack.material_property(dx, "eps_r")
    Nc = stack.material_property(dx, "Nc", T)
    Nv = stack.material_property(dx, "Nv", T)
    Ec0s, Ev0s = _shifted_edges(device, dx, T)
    net = stack.net_doping(dx)
    ND = np.where(net > 0, net, 0.0)
    NA = np.where(net < 0, -net, 0.0)

    # Dirichlet contact potentials from neutral bulk values under the QFL split.
    psi_neutral = _neutral_psi(Ec0s, Ev0s, Nc, Nv, ND, NA, kT, EFn, EFp)
    phi_B = getattr(device.left, "schottky_barrier", None)
    if phi_B is not None:
        # Fermi-level pinning at the metal/SC surface: Ec(0) = phi_B above the
        # n-contact level (EFn=0).  Ec(0)=Ec0s[0]-psi(0)=phi_B => psi_left=Ec0s[0]-phi_B.
        psi_left = Ec0s[0] - phi_B
    else:
        psi_left = psi_neutral[0]            # flat-band (neutral) surface
    psi_right = psi_neutral[-1]

    # half-node permittivities
    eps_half = 0.5 * (eps[:-1] + eps[1:])     # length N-1

    psi = psi_neutral.copy()
    psi[0], psi[-1] = psi_left, psi_right

    residual = np.inf
    it = 0
    for it in range(1, max_iter + 1):
        n, p, _, _ = _carriers(psi, Ec0s, Ev0s, Nc, Nv, kT, EFn, EFp)
        rho = p - n + ND - NA                  # cm^-3

        # residual F_i = eps_{i+1/2}(psi_{i+1}-psi_i) - eps_{i-1/2}(psi_i-psi_{i-1})
        #               + POISSON_C * rho_i * dx^2     (interior nodes)
        flux = eps_half * np.diff(psi)         # length N-1
        F = np.zeros(N)
        F[1:-1] = flux[1:] - flux[:-1] + POISSON_C * rho[1:-1] * dx**2

        # Jacobian (tridiagonal). d rho/d psi = -(p+n)/kT.
        drho = -(p + n) / kT
        diag = np.ones(N)                      # Dirichlet rows -> identity
        diag[1:-1] = -(eps_half[:-1] + eps_half[1:]) + POISSON_C * drho[1:-1] * dx**2
        # sub-diagonal a[i] = J[i+1, i]; super-diagonal c[i] = J[i, i+1]
        a = np.zeros(N - 1)
        a[:-1] = eps_half[:-1]                 # rows 1..N-2 couple to i-1; last row Dirichlet
        c = np.zeros(N - 1)
        c[1:] = eps_half[1:]                   # rows 1..N-2 couple to i+1; first row Dirichlet
        # build sparse and solve J dpsi = -F (Dirichlet rows: F=0, identity)
        J = diags([a, diag, c], [-1, 0, 1], format="csr")
        dpsi = spsolve(J, -F)
        # damp the step (limit to a few kT for robustness)
        dpsi = np.clip(dpsi, -0.5, 0.5)
        psi = psi + dpsi
        psi[0], psi[-1] = psi_left, psi_right

        residual = float(np.max(np.abs(F[1:-1])))
        if np.max(np.abs(dpsi)) < tol:
            break

    n, p, Ec, Ev = _carriers(psi, Ec0s, Ev0s, Nc, Nv, kT, EFn, EFp)
    Efield = -np.gradient(psi, z)              # V/nm
    return BandProfile(z=z, psi=psi, Ec=Ec, Ev=Ev, Efield=Efield, n=n, p=p,
                       Ec0=Ec0s, Ev0=Ev0s, bias_V=bias_V, T=T,
                       iterations=it, residual=residual)
