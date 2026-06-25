"""Physical constants, unit conventions, and simulation configuration.

Unit convention used throughout ``spinpd``
------------------------------------------
We keep human-readable, O(1)-magnitude units at the interfaces and document them
explicitly (the Octave reference suffered from scattered ad-hoc unit factors):

==================  ==========================================
quantity            unit
==================  ==========================================
length / position   nm
energy / potential  eV  (band edges, quasi-Fermi levels) / V
density             cm^-3   (doping, carriers)
mobility            cm^2 V^-1 s^-1
time / lifetime     s
temperature         K
==================  ==========================================

Physical constants are stored in SI in :data:`CONST`; helper methods return
quantities in the working units above.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math


@dataclass(frozen=True)
class PhysicalConstants:
    """Fundamental constants (SI units)."""

    q: float = 1.602176634e-19        # elementary charge, C
    kB: float = 1.380649e-23          # Boltzmann constant, J/K
    eps0: float = 8.8541878128e-12    # vacuum permittivity, F/m
    h: float = 6.62607015e-34         # Planck constant, J*s
    hbar: float = 1.054571817e-34     # reduced Planck constant, J*s
    m0: float = 9.1093837015e-31      # free electron mass, kg
    c: float = 2.99792458e8           # speed of light, m/s
    mu_B: float = 9.2740100783e-24    # Bohr magneton, J/T

    def kT_eV(self, T: float) -> float:
        """Thermal energy kB*T in eV."""
        return self.kB * T / self.q

    def VT(self, T: float) -> float:
        """Thermal voltage kB*T/q in V (numerically equal to kT_eV)."""
        return self.kB * T / self.q

    def effective_dos(self, m_rel: float, T: float) -> float:
        """Effective density of states (cm^-3) for a parabolic band.

        Nc/Nv = 2 (2*pi*m* kB T / h^2)^{3/2}, with m* = m_rel * m0.
        Returned in cm^-3 (SI value / 1e6).
        """
        m = m_rel * self.m0
        n_si = 2.0 * (2.0 * math.pi * m * self.kB * T / self.h**2) ** 1.5  # m^-3
        return n_si * 1e-6  # -> cm^-3


CONST = PhysicalConstants()


@dataclass(frozen=True)
class Numerics:
    """Discretization and self-consistent-solver controls."""

    dx: float = 1.0            # grid spacing, nm
    max_iter: int = 200        # max self-consistent iterations
    tol: float = 1e-6          # relative convergence tolerance on the observable
    damping: float = 0.3       # mixing factor for updated quantities (0..1)


@dataclass(frozen=True)
class OpticalPump:
    """Continuous-wave optical excitation."""

    power_W: float = 5e-3              # incident laser power, W
    wavelength_nm: float = 785.0       # excitation wavelength, nm  (1.58 eV)
    spot_diameter_um: float = 500.0    # illuminated spot diameter, um
    transmission: float = 0.3          # fraction transmitted into the semiconductor
    helicity_polarization: float = 0.5 # effective carrier polarization under circular excitation

    @property
    def photon_energy_eV(self) -> float:
        return CONST.h * CONST.c / (self.wavelength_nm * 1e-9) / CONST.q

    @property
    def spot_area_cm2(self) -> float:
        r_cm = (self.spot_diameter_um * 1e-4) / 2.0
        return math.pi * r_cm**2

    def photon_flux(self) -> float:
        """Incident photon flux entering the SC, photons cm^-2 s^-1."""
        e_ph_J = self.photon_energy_eV * CONST.q
        return self.power_W * self.transmission / (e_ph_J * self.spot_area_cm2)


@dataclass(frozen=True)
class OperatingPoint:
    """A single bias / temperature working point."""

    bias_V: float = 0.0
    temperature_K: float = 100.0   # ~ liquid nitrogen


@dataclass(frozen=True)
class SimConfig:
    """Top-level configuration bundle passed around the solver."""

    numerics: Numerics = field(default_factory=Numerics)
    pump: OpticalPump = field(default_factory=OpticalPump)
    operating: OperatingPoint = field(default_factory=OperatingPoint)

    @property
    def T(self) -> float:
        return self.operating.temperature_K
