"""Self-consistent high-injection drift-diffusion (Poisson <-> e/h continuity).

At high illumination, photo-generated carriers form a space charge that screens
the junction field.  Equivalently, the device develops an internal photovoltage.
The band profile, carrier densities, recombination, and extracted photocurrent
therefore need to be solved as a coupled problem rather than as transport on a
fixed dark band.

This module solves the *coupled* problem for the TOTAL carrier densities::

    d/dz[eps dpsi/dz] = -(q/eps0)(p - n + ND - NA)        (Poisson)
    dFn/dz =  G - R,   Fn = -mu_n n dEFn/dz               (electron continuity)
   -dFp/dz =  G - R,   Fp = -mu_p p dEFp/dz               (hole continuity)

by Gummel iteration: nonlinear Poisson for psi (carriers at fixed quasi-Fermi
levels), then Scharfetter-Gummel continuity for n and p, then update R and
repeat.  Boundary conditions:

* electrons: extracting left contact -> Fn(0) = v_ext (n(0) - n_eq(0));
             p+ ohmic right -> n = n_eq (equilibrium, Dirichlet).
* holes:     left contact -> blocked Fp(0) = 0;
             p+ ohmic right -> p = p_eq (Dirichlet).

Working units: nm grid, V, eV, cm^-3, cm (transport), s, cm/s.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import spsolve

from .bands import (CHI_REF, POISSON_C, _carriers, _neutral_psi, _shifted_edges,
                    solve_equilibrium)
from .config import CONST
from .recombination import RecombinationModel
from .stack import Device
from .transport import _bernoulli_pair, generation_profile


@dataclass
class SCResult:
    """Result of a self-consistent high-injection solve."""

    z: np.ndarray
    psi: np.ndarray          # self-consistent potential (V)
    Ec: np.ndarray           # conduction edge under illumination (eV)
    Ev: np.ndarray
    n: np.ndarray            # TOTAL electron density, cm^-3
    p: np.ndarray            # TOTAL hole density, cm^-3
    n0: np.ndarray           # dark-equilibrium electron density
    p0: np.ndarray
    G: np.ndarray            # generation, cm^-3 s^-1
    R: np.ndarray            # recombination, cm^-3 s^-1
    I_dc: float              # extracted photocurrent (A), = q A v_ext (n(0)-n_eq(0))
    bias_V: float
    T: float
    iterations: int
    residual: float
    converged: bool

    @property
    def band_bending(self) -> float:
        """Conduction-band bending across the active region (eV)."""
        return float(self.Ec.max() - self.Ec.min())


@dataclass
class SpinCurrentResult:
    """Macroscopic helicity-current observable on a converged transport solution."""

    I_plus: float
    I_minus: float
    I_dc: float
    I_spin: float
    spin_lifetime: float
    generation_polarization: float
    current_asymmetry: float


def _poisson_qfl(EFn, EFp, psi_init, psi_left, psi_right,
                 Ec0s, Ev0s, Nc, Nv, ND, NA, kT, eps_half, dx,
                 max_iter=200, tol=1e-9):
    """Nonlinear Poisson for psi with position-dependent quasi-Fermi levels.

    Carriers follow Boltzmann at the (fixed) quasi-Fermi levels EFn(z), EFp(z):
    n = Nc exp((EFn - Ec0 + psi)/kT), p = Nv exp((Ev0 - psi - EFp)/kT).  Dirichlet
    psi at the two contacts.  Damped Newton.
    """
    N = psi_init.size
    psi = psi_init.copy()
    psi[0], psi[-1] = psi_left, psi_right
    for _ in range(max_iter):
        n = Nc * np.exp(np.clip((EFn - Ec0s + psi) / kT, -400, 400))
        p = Nv * np.exp(np.clip((Ev0s - psi - EFp) / kT, -400, 400))
        rho = p - n + ND - NA
        flux = eps_half * np.diff(psi)
        F = np.zeros(N)
        F[1:-1] = flux[1:] - flux[:-1] + POISSON_C * rho[1:-1] * dx**2
        drho = -(p + n) / kT
        diag = np.ones(N)
        diag[1:-1] = -(eps_half[:-1] + eps_half[1:]) + POISSON_C * drho[1:-1] * dx**2
        a = np.zeros(N - 1); a[:-1] = eps_half[:-1]
        c = np.zeros(N - 1); c[1:] = eps_half[1:]
        J = diags([a, diag, c], [-1, 0, 1], format="csr")
        dpsi = np.clip(spsolve(J, -F), -0.3, 0.3)
        psi = psi + dpsi
        psi[0], psi[-1] = psi_left, psi_right
        if np.max(np.abs(dpsi)) < tol:
            break
    n = Nc * np.exp(np.clip((EFn - Ec0s + psi) / kT, -400, 400))
    p = Nv * np.exp(np.clip((Ev0s - psi - EFp) / kT, -400, 400))
    return psi, n, p


def _link_harmonic(node_vals) -> np.ndarray:
    """Per-LINK transport coefficient from per-NODE ``node_vals`` by symmetric harmonic mean.

    At a heterojunction the coefficient on the link BETWEEN two materials is better
    represented by the series (harmonic) combination of the two adjacent nodes than by
    either node alone::

        x_link = 2 x_l x_r / (x_l + x_r)

    This is symmetric (no arbitrary left/right pick) and reduces EXACTLY to the common
    value in a homogeneous region (x_l == x_r), so single-material devices are numerically
    unchanged.  Used for BOTH the carrier mobility and the saturation velocity that feed
    :func:`diffusion_half`. Returns an array of length ``len(node_vals) - 1``.
    """
    v = np.asarray(node_vals, dtype=float)
    l, r = v[:-1], v[1:]
    s = l + r
    return np.where(s > 0, 2.0 * l * r / np.where(s > 0, s, 1.0), 0.0)


def diffusion_half(edge, mu0, kT, dx, v_sat=None):
    """Per-link diffusion coefficient D = mu(E) kT with velocity saturation (A1).

    Field-dependent mobility mu(E) = mu0 / (1 + mu0|E|/v_sat) (Caughey-Thomas,
    beta=1), so the drift velocity saturates at v_sat for strong field.  ``edge``
    is the band edge used for transport (Ec for electrons, -Ev for holes); the
    field at each link is |d(edge)/dz|.  ``v_sat`` None or <=0 -> constant mu0.
    Returns an array of length len(edge)-1.

    ``mu0`` and ``v_sat`` may be scalars for a homogeneous device or per-link arrays
    of length len(edge)-1 for a heterostructure.
    """
    h = dx * 1e-7                                       # cm
    nlink = edge.size - 1
    mu0 = np.asarray(mu0, dtype=float)
    E = np.abs(np.diff(edge)) / h                       # |d(edge)/dz|, V/cm
    if v_sat is None:
        mu = np.broadcast_to(mu0, (nlink,))             # constant mobility
    else:
        v_sat = np.asarray(v_sat, dtype=float)
        vs = np.where(v_sat > 0, v_sat, np.inf)         # v_sat<=0 -> no saturation (constant mu0)
        mu = mu0 / (1.0 + mu0 * E / vs)
    return np.broadcast_to(mu * kT, (nlink,)).copy()


def _solve_n_total(Ec, G, rec, D_half, dx, kT, v_ext, n_eq):
    """Total electron density via Scharfetter-Gummel continuity.

    BCs: left tunnel extraction Fn(0)=v_ext (n(0)-n_eq(0)); right ohmic n=n_eq.
    ``rec`` is the per-node effective inverse lifetime R/n (1/s).  ``D_half`` is
    the per-link diffusion coefficient (length N-1; allows field-dependent mu).
    """
    N = Ec.size
    h = dx * 1e-7
    DD = D_half / h**2                                  # array, length N-1
    Bp, Bm = _bernoulli_pair(Ec, kT)
    main = np.zeros(N); lo = np.zeros(N - 1); up = np.zeros(N - 1); b = np.zeros(N)
    for i in range(N):
        if i == N - 1:                          # right ohmic Dirichlet
            main[i] = 1.0; b[i] = n_eq[-1]; continue
        if i == 0:                              # left extraction
            main[0] = DD[0] * Bp[0] + v_ext / h + rec[0]
            up[0] = -DD[0] * Bm[0]
            b[0] = G[0] + v_ext * n_eq[0] / h
        else:
            main[i] = DD[i] * Bp[i] + DD[i - 1] * Bm[i - 1] + rec[i]
            up[i] = -DD[i] * Bm[i]
            lo[i - 1] = -DD[i - 1] * Bp[i - 1]
            b[i] = G[i]
    A = diags([lo, main, up], [-1, 0, 1], format="csr")
    return np.maximum(spsolve(A, b), 0.0)


def _solve_p_total(Ev, G, rec, D_half, dx, kT, p_eq):
    """Total hole density via SG continuity.  BCs: left blocked; right ohmic p=p_eq.
    ``D_half`` is the per-link diffusion coefficient (length N-1)."""
    N = Ev.size
    h = dx * 1e-7
    DD = D_half / h**2
    Bp, Bm = _bernoulli_pair(-Ev, kT)
    main = np.zeros(N); lo = np.zeros(N - 1); up = np.zeros(N - 1); b = np.zeros(N)
    for i in range(N):
        if i == N - 1:                          # right ohmic Dirichlet
            main[i] = 1.0; b[i] = p_eq[-1]; continue
        if i == 0:                              # left blocked: Fp(0)=0
            main[0] = DD[0] * Bp[0] + rec[0]
            up[0] = -DD[0] * Bm[0]
            b[0] = G[0]
        else:
            main[i] = DD[i] * Bp[i] + DD[i - 1] * Bm[i - 1] + rec[i]
            up[i] = -DD[i] * Bm[i]
            lo[i - 1] = -DD[i - 1] * Bp[i - 1]
            b[i] = G[i]
    A = diags([lo, main, up], [-1, 0, 1], format="csr")
    return np.maximum(spsolve(A, b), 0.0)


def solve_sc(device: Device, bias_V: float = 0.0, T: float = 100.0, dx: float = 1.0,
             intensity_scale: float = 1.0, max_outer: int = 300, tol: float = 1e-5,
             damping: float = 0.3, velocity_saturation: bool = True,
             verbose: bool = False) -> SCResult:
    """Self-consistently solve coupled Poisson + e/h drift-diffusion (total carriers).

    ``intensity_scale`` multiplies the generation profile (the device pump sets the
    base level).  ``velocity_saturation`` (A1): if True the drift velocity saturates
    at the material v_sat (field-dependent mobility), so mu can be kept at its real
    measured value instead of being fudged low.  Returns the photocurrent and
    self-consistent band/carriers.
    """
    from .tunneling import TunnelExtraction

    kT = CONST.kT_eV(T)
    q, A = CONST.q, device.pump.spot_area_cm2
    stack = device.stack
    z = stack.grid(dx); N = z.size

    eps = stack.material_property(dx, "eps_r")
    Nc = stack.material_property(dx, "Nc", T)
    Nv = stack.material_property(dx, "Nv", T)
    Ec0s, Ev0s = _shifted_edges(device, dx, T)
    net = stack.net_doping(dx)
    ND = np.where(net > 0, net, 0.0); NA = np.where(net < 0, -net, 0.0)
    eps_half = 0.5 * (eps[:-1] + eps[1:])

    # dark equilibrium: initial guess + contact (Dirichlet) values + n_eq/p_eq.
    dark = solve_equilibrium(device, T=T, dx=dx, bias_V=bias_V)
    psi = dark.psi.copy()
    psi_left, psi_right = dark.psi[0], dark.psi[-1]
    n_eq, p_eq = dark.n.copy(), dark.p.copy()
    n0, p0 = dark.n.copy(), dark.p.copy()
    n, p = dark.n.copy(), dark.p.copy()

    G = generation_profile(device, dx, T) * intensity_scale
    # Per-node mobility / saturation velocity -> per-link transport coefficients.
    # The harmonic mean handles material interfaces symmetrically.
    li = stack.layer_index(z)
    mu_e_link = _link_harmonic(np.array([l.material.mu_e for l in stack.layers])[li])
    mu_h_link = _link_harmonic(np.array([l.material.mu_h for l in stack.layers])[li])
    if velocity_saturation:
        vsat_e = _link_harmonic(np.array([l.material.v_sat_e for l in stack.layers])[li])
        vsat_h = _link_harmonic(np.array([l.material.v_sat_h for l in stack.layers])[li])
    else:
        vsat_e = vsat_h = None
    recomb = RecombinationModel(device, dx, T)

    ext = TunnelExtraction(device)
    v_ext = ext.v_ext(T, bias_V=bias_V, interface_field=abs(dark.Efield[0]))

    I = 0.0; residual = np.inf; converged = False
    for it in range(1, max_outer + 1):
        # --- Poisson: carriers at fixed quasi-Fermi levels from current n,p,psi ---
        Ec, Ev = Ec0s - psi, Ev0s - psi
        EFn = Ec + kT * np.log(np.clip(n, 1e-30, None) / Nc)
        EFp = Ev - kT * np.log(np.clip(p, 1e-30, None) / Nv)
        psi, n_b, p_b = _poisson_qfl(EFn, EFp, psi, psi_left, psi_right,
                                     Ec0s, Ev0s, Nc, Nv, ND, NA, kT, eps_half, dx)
        Ec, Ev = Ec0s - psi, Ev0s - psi

        # --- recombination from current densities (R>0; 100K ni negligible) ---
        R = recomb.rates(n, p, n0, p0)["total"]
        R = np.maximum(R, 0.0)
        rec_n = R / np.clip(n, 1e-30, None)
        rec_p = R / np.clip(p, 1e-30, None)

        # --- continuity (SG) for total n, p, then under-relax in log space ---
        # field-dependent diffusion (velocity saturation, A1) from the current band,
        # with per-link mu/v_sat (each layer's own material, harmonic-mean at interfaces)
        Dn_half = diffusion_half(Ec, mu_e_link, kT, dx, vsat_e)
        Dp_half = diffusion_half(-Ev, mu_h_link, kT, dx, vsat_h)
        n_new = _solve_n_total(Ec, G, rec_n, Dn_half, dx, kT, v_ext, n_eq)
        p_new = _solve_p_total(Ev, G, rec_p, Dp_half, dx, kT, p_eq)
        # Convergence on the carrier-profile change (relative to the peak density).
        # This tracks the full solution -- including the contact density that sets
        # I -- and stays finite/robust as the observable I -> 0 at flat-band, where
        # the old |dI/I| criterion either diverged or stopped prematurely.
        residual = max(float(np.max(np.abs(n_new - n))) / max(float(np.max(n)), 1.0),
                       float(np.max(np.abs(p_new - p))) / max(float(np.max(p)), 1.0))
        n = np.exp((1 - damping) * np.log(np.clip(n, 1e-30, None))
                   + damping * np.log(np.clip(n_new, 1e-30, None)))
        p = np.exp((1 - damping) * np.log(np.clip(p, 1e-30, None))
                   + damping * np.log(np.clip(p_new, 1e-30, None)))
        I = q * A * v_ext * max(n[0] - n_eq[0], 0.0)
        if verbose and (it % 20 == 0 or it == 1):
            print(f"  it={it:3d} I={I*1e6:9.3f}uA bend={Ec.max()-Ec.min():.3f} res={residual:.1e}")
        if residual < tol and it > 5:
            converged = True; break

    R = np.maximum(recomb.rates(n, p, n0, p0)["total"], 0.0)
    return SCResult(z=z, psi=psi, Ec=Ec0s - psi, Ev=Ev0s - psi, n=n, p=p, n0=n0, p0=p0,
                    G=G, R=R, I_dc=I, bias_V=bias_V, T=T, iterations=it,
                    residual=residual, converged=converged)


def spin_current(
    device: Device,
    scr: SCResult,
    T: float = 100.0,
    dx: float = 1.0,
    spin_lifetime: float = 1.4e-10,
    generation_polarization: float | None = None,
    current_asymmetry: float | None = None,
    velocity_saturation: bool = True,
) -> SpinCurrentResult:
    """Compute the macroscopic helicity current on a self-consistent band.

    This is an observable-level calculation. It reuses the converged electrostatic
    potential, carrier densities, generation profile, and recombination profile from
    :func:`solve_sc`, then solves two coupled electron-population continuity equations
    for the two circular-excitation states. The parameters `spin_lifetime`,
    `generation_polarization`, and `current_asymmetry` are effective macroscopic
    parameters; this function is not a microscopic contact calculation.
    """
    from .tunneling import TunnelExtraction

    kT = CONST.kT_eV(T)
    q, A = CONST.q, device.pump.spot_area_cm2
    Ec = scr.Ec
    N = Ec.size
    h = dx * 1e-7
    stack = device.stack

    P = device.pump.helicity_polarization if generation_polarization is None else generation_polarization
    asym = device.left.current_asymmetry if current_asymmetry is None else current_asymmetry
    P = float(np.clip(P, -1.0, 1.0))
    asym = float(np.clip(asym, -0.999, 0.999))

    rec = np.maximum(scr.R, 0.0) / np.clip(scr.n, 1e-30, None)
    n_eq = scr.n0

    ext = TunnelExtraction(device)
    interface_field = abs(np.gradient(scr.psi, scr.z)[0])
    v_avg = ext.v_ext(T, bias_V=scr.bias_V, interface_field=interface_field)
    v_up = v_avg * (1.0 + asym)
    v_down = v_avg * (1.0 - asym)

    li = stack.layer_index(scr.z)
    mu_link = _link_harmonic(np.array([l.material.mu_e for l in stack.layers])[li])
    if velocity_saturation:
        vsat_link = _link_harmonic(np.array([l.material.v_sat_e for l in stack.layers])[li])
    else:
        vsat_link = None
    DD = diffusion_half(Ec, mu_link, kT, dx, vsat_link) / h**2

    tau_s = np.broadcast_to(np.asarray(spin_lifetime, dtype=float), (N,))
    tau_s = np.clip(tau_s, 1e-15, np.inf)
    flip = 1.0 / (2.0 * tau_s)
    Bp, Bm = _bernoulli_pair(Ec, kT)

    G_up = scr.G * (1.0 + P) / 2.0
    G_down = scr.G * (1.0 - P) / 2.0

    rows, cols, vals = [], [], []

    def add(r, c, v):
        rows.append(r)
        cols.append(c)
        vals.append(v)

    rhs_plus = np.zeros(2 * N)
    rhs_minus = np.zeros(2 * N)
    channels = (
        (0, v_up, G_up, G_down),
        (N, v_down, G_down, G_up),
    )
    for offset, v_ext, G_sigma_plus, G_sigma_minus in channels:
        for i in range(N):
            row = offset + i
            other = (N + i) if offset == 0 else i
            if i == N - 1:
                add(row, row, 1.0)
                rhs_plus[row] = n_eq[-1] / 2.0
                rhs_minus[row] = n_eq[-1] / 2.0
                continue
            if i == 0:
                add(row, row, DD[0] * Bp[0] + v_ext / h + rec[0] + flip[0])
                add(row, offset + 1, -DD[0] * Bm[0])
                rhs_plus[row] = G_sigma_plus[0] + v_ext * (n_eq[0] / 2.0) / h
                rhs_minus[row] = G_sigma_minus[0] + v_ext * (n_eq[0] / 2.0) / h
            else:
                add(row, row, DD[i] * Bp[i] + DD[i - 1] * Bm[i - 1] + rec[i] + flip[i])
                add(row, offset + i + 1, -DD[i] * Bm[i])
                add(row, offset + i - 1, -DD[i - 1] * Bp[i - 1])
                rhs_plus[row] = G_sigma_plus[i]
                rhs_minus[row] = G_sigma_minus[i]
            add(row, other, -flip[i])

    mat = csr_matrix((vals, (rows, cols)), shape=(2 * N, 2 * N))
    sol = spsolve(mat, np.column_stack([rhs_plus + rhs_minus, rhs_plus - rhs_minus]))
    x_sum, x_diff = sol[:, 0], sol[:, 1]
    x_plus = 0.5 * (x_sum + x_diff)
    x_minus = 0.5 * (x_sum - x_diff)

    def current(x):
        n_up0 = max(float(x[0] - n_eq[0] / 2.0), 0.0)
        n_down0 = max(float(x[N] - n_eq[0] / 2.0), 0.0)
        return q * A * (v_up * n_up0 + v_down * n_down0)

    I_plus = current(x_plus)
    I_minus = current(x_minus)
    return SpinCurrentResult(
        I_plus=I_plus,
        I_minus=I_minus,
        I_dc=0.5 * (I_plus + I_minus),
        I_spin=I_plus - I_minus,
        spin_lifetime=float(np.mean(tau_s)),
        generation_polarization=P,
        current_asymmetry=asym,
    )
