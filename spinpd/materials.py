"""Semiconductor material database for layered semiconductor simulations.

Each :class:`Material` stores the parameters needed by the band, recombination
and transport modules.  Band edges are referenced to the vacuum level via the
electron affinity ``chi`` (Anderson's rule); heterojunction offsets then follow
automatically in :mod:`spinpd.stack`.

Temperature dependence of the gap uses the Varshni relation
``Eg(T) = Eg0 - alpha*T^2/(T+beta)``.

The database is deliberately small but extensible: add a factory function that
returns a :class:`Material`.  Parametric alloys (AlGaAs, GaNAs) are functions of
composition.  Values are room-temperature literature numbers unless noted; treat
mobilities/lifetimes as editable defaults rather than precise constants.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

from .config import CONST


@dataclass(frozen=True)
class Material:
    """Bulk semiconductor parameters (working units: eV, cm^-3, cm^2/V/s, s)."""

    name: str
    Eg0: float            # bandgap at T=0 K, eV
    chi: float            # electron affinity (vacuum -> CB edge), eV
    eps_r: float          # static relative permittivity
    m_e: float            # electron DOS effective mass, units of m0
    m_h: float            # hole DOS effective mass, units of m0
    mu_e: float           # electron mobility, cm^2/V/s
    mu_h: float           # hole mobility, cm^2/V/s
    B_rad: float          # radiative recombination coefficient, cm^3/s
    tau_srh: float        # Shockley-Read-Hall lifetime, s
    v_sat_e: float = 1.0e7   # electron saturation drift velocity, cm/s (A1)
    v_sat_h: float = 8.0e6   # hole saturation drift velocity, cm/s
    C_auger: float = 0.0     # Auger coeff (per carrier), cm^6/s; R=C(n+p)(np-ni^2), Cn=Cp=C
    varshni_alpha: float = 5.405e-4   # eV/K
    varshni_beta: float = 204.0       # K
    is_direct: bool = True
    note: str = ""

    # --- temperature-dependent quantities -------------------------------------
    def Eg(self, T: float) -> float:
        """Bandgap at temperature T (eV), Varshni relation."""
        return self.Eg0 - self.varshni_alpha * T**2 / (T + self.varshni_beta)

    def Nc(self, T: float) -> float:
        """Conduction-band effective DOS, cm^-3."""
        return CONST.effective_dos(self.m_e, T)

    def Nv(self, T: float) -> float:
        """Valence-band effective DOS, cm^-3."""
        return CONST.effective_dos(self.m_h, T)

    def ni(self, T: float) -> float:
        """Intrinsic carrier density, cm^-3."""
        kT = CONST.kT_eV(T)
        return (self.Nc(T) * self.Nv(T)) ** 0.5 * 2.718281828 ** (-self.Eg(T) / (2 * kT))

    # --- band edges referenced to vacuum (flat-band, before electrostatics) ---
    def Ec0(self, T: float) -> float:
        """Conduction-band edge relative to vacuum (eV), = -chi."""
        return -self.chi

    def Ev0(self, T: float) -> float:
        """Valence-band edge relative to vacuum (eV), = -chi - Eg(T)."""
        return -self.chi - self.Eg(T)


# ---------------------------------------------------------------------------
# Factory functions (composition- and temperature-aware where relevant)
# ---------------------------------------------------------------------------
def GaAs() -> Material:
    return Material(
        name="GaAs",
        Eg0=1.519, chi=4.07, eps_r=12.9,
        m_e=0.067, m_h=0.51,
        mu_e=8000.0, mu_h=400.0,
        B_rad=7.2e-10, tau_srh=1e-9,
        C_auger=5e-31,     # Auger: Cn=Cp=5e-31 -> ambipolar (n=p) C~1e-30 cm^6/s, the
                           # consensus GaAs value (lit. ~1e-31..7e-30). Negligible at lab
                           # injection (<0.2% of R for n<1e18); only enters at extreme
                           # injection (n~1e20), where Boltzmann is also invalid (n>>Nc).
        is_direct=True,
    )


def AlGaAs(x: float) -> Material:
    """Al_x Ga_(1-x) As (direct for x < 0.45).

    Gap: Eg0(x) = 1.519 + 1.247 x (Gamma valley).
    Conduction-band offset to GaAs follows the 60:40 rule, baked into the
    electron affinity: chi(x) = chi_GaAs - 0.6 * (Eg(x) - Eg_GaAs).
    """
    if not 0.0 <= x <= 1.0:
        raise ValueError(f"Al fraction x must be in [0,1], got {x}")
    dEg = 1.247 * x
    return Material(
        name=f"Al{x:.2f}Ga{1 - x:.2f}As",
        Eg0=1.519 + dEg,
        chi=4.07 - 0.6 * dEg,
        eps_r=12.9 - 2.84 * x,
        m_e=0.067 + 0.083 * x,
        m_h=0.51 + 0.25 * x,
        mu_e=max(8000.0 - 22000.0 * x, 200.0),
        mu_h=max(400.0 - 900.0 * x, 50.0),
        B_rad=7.2e-10, tau_srh=1e-9,
        is_direct=(x < 0.45),
        note="60:40 CBO rule via electron affinity",
    )


def InAs_QD() -> Material:
    """Effective InAs/InGaAs quantum-dot sheet for a 1-D continuum model.

    A 1-D semiclassical drift-diffusion CANNOT represent a quantum dot as a
    classical potential well: (i) the confined states are quantum, not a
    classical band dip, and (ii) an abrupt ~0.4 eV band step over one thin grid
    cell breaks the Scharfetter-Gummel flux (it assumes a linear band between
    nodes) and is strongly grid-dependent / non-conserving.

    We therefore give the QD **the host (GaAs) band edges** (chi, Eg) so it is
    band-flat for transport, and represent its physics as a localized
    **localized recombination centre**: the capture rate is supplied
    per-layer (``params['qd_capture_rate']``) and the holes via the layer's
    p-doping (Be delta-doping -> X+ trion). Faster radiative recombination and
    localized capture distinguish it from bulk GaAs.
    """
    return Material(
        name="InAs-QD",
        Eg0=1.519, chi=4.07, eps_r=13.0,   # band-flat to GaAs (no spurious classical well)
        m_e=0.067, m_h=0.51,               # GaAs DOS mass: the dot is modelled band-flat-to-GaAs
        mu_e=2000.0, mu_h=100.0,
        B_rad=1e-9, tau_srh=1e-11,
        is_direct=True,
        note="band-flat (GaAs edges) recombination/capture centre; not a classical well",
    )


def InGaAs(x: float = 0.20) -> Material:
    """Bulk In_x Ga_(1-x) As alloy (relaxed-alloy, parabolic).

    Parameters interpolated GaAs<->InAs (Vurgaftman, Meyer & Ram-Mohan,
    JAP 89, 5815 (2001)):

      * ``m_e(x) = 0.067 - 0.044 x`` (linear DOS mass; ~0.058 at x=0.20).
      * ``Eg0(x) = 1.519(1-x) + 0.417 x - 0.477 x(1-x) = 1.519 - 1.579 x + 0.477 x^2``
        (GaAs 1.519 -> InAs 0.417 eV at 0 K with bowing C=0.477; Vurgaftman et al.).
      * ``chi(x) = 4.07 + 0.83 x`` (electron affinity GaAs 4.07 -> InAs 4.90).
    NOTE: a pseudomorphically strained QW on GaAs has a heavier in-plane mass and
    nonparabolicity; 0.058 is the relaxed-alloy floor.
    """
    if not 0.0 <= x <= 1.0:
        raise ValueError(f"In fraction x must be in [0,1], got {x}")
    return Material(
        name=f"In{x:.2f}Ga{1 - x:.2f}As",
        Eg0=1.519 - 1.579 * x + 0.477 * x ** 2,   # GaAs 1.519 -> InAs 0.417 eV, bowing 0.477
        chi=4.07 + 0.83 * x,
        eps_r=12.9 + 2.25 * x,
        m_e=0.067 - 0.044 * x,
        m_h=0.51 - 0.10 * x,
        mu_e=8000.0 + 6000.0 * x,   # InGaAs is higher-mobility than GaAs (editable default)
        mu_h=400.0 - 200.0 * x,
        B_rad=7.2e-10, tau_srh=1e-9,
        is_direct=True,
        note="relaxed-alloy parabolic InGaAs transport material",
    )


def GaNAs(y: float = 0.022) -> Material:
    """Dilute-nitride GaN_y As_(1-y).

    Nitrogen strongly reduces the gap via band anticrossing (~0.1 eV per %N here).
    """
    return Material(
        name=f"GaN{y:.3f}As",
        Eg0=1.519 - 8.0 * y, chi=4.07 + 2.0 * y, eps_r=12.9,
        m_e=0.08, m_h=0.51,
        mu_e=2000.0, mu_h=200.0,
        B_rad=5e-10, tau_srh=1e-10,
        is_direct=True,
        note="dilute nitride",
    )


def Ge() -> Material:
    """Germanium (indirect L-valley gap)."""
    return Material(
        name="Ge",
        Eg0=0.744, chi=4.00, eps_r=16.0,
        m_e=0.55, m_h=0.37,
        mu_e=3900.0, mu_h=1900.0,
        B_rad=5.2e-14, tau_srh=1e-7,
        varshni_alpha=4.77e-4, varshni_beta=235.0,
        is_direct=False,
        note="indirect gap",
    )


#: Convenience registry of zero-argument materials.
REGISTRY = {m.name: m for m in (GaAs(), InAs_QD(), Ge())}


def with_overrides(mat: Material, **overrides) -> Material:
    """Return a copy of ``mat`` with selected fields replaced (extensibility hook)."""
    return replace(mat, **overrides)
