"""Extensible one-dimensional semiconductor layer-stack abstraction.

A :class:`Device` is an ordered list of semiconductor :class:`Layer` s (the
drift-diffusion domain) plus the two contacts. Position ``z = 0`` is the
collecting contact; ``z`` increases into the semiconductor toward the
substrate/ohmic contact.

The stack is the single place that knows the geometry: it maps the layer
sequence onto a 1-D grid and produces z-resolved arrays (material properties,
doping, flat-band edges) consumed by the band/recombination/transport modules.
Building a different device (QW instead of QD, different doping, another
material family) is just a different list of layers -- nothing downstream
changes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

import numpy as np

from .config import OpticalPump
from .materials import Material, GaAs, AlGaAs, InAs_QD, with_overrides


class Role(str, Enum):
    """Semantic role of a layer (free strings also accepted)."""

    CONTACT_N = "contact_n"
    ACTIVE = "active"
    QD = "qd"
    QW = "qw"
    BARRIER = "barrier"
    BUFFER = "buffer"
    SUBSTRATE = "substrate"


@dataclass(frozen=True)
class Doping:
    """Ionized dopant specification (densities in cm^-3)."""

    kind: str = "i"          # 'n', 'p', or 'i'
    density: float = 0.0     # magnitude of the ionized dopant density, cm^-3

    @property
    def ND(self) -> float:
        return self.density if self.kind == "n" else 0.0

    @property
    def NA(self) -> float:
        return self.density if self.kind == "p" else 0.0

    @property
    def net(self) -> float:
        """Net doping ND - NA (cm^-3); positive = n-type."""
        return self.ND - self.NA


@dataclass(frozen=True)
class Layer:
    """A single epitaxial layer in the drift-diffusion domain."""

    material: Material
    thickness: float                       # nm
    doping: Doping = field(default_factory=Doping)
    role: str = Role.ACTIVE.value
    name: str = ""
    params: dict = field(default_factory=dict)  # role-specific extras, e.g. qd_capture_rate (1/s)

    def label(self) -> str:
        return self.name or f"{self.doping.kind}-{self.material.name}"


@dataclass(frozen=True)
class TunnelContact:
    """Collecting contact represented by a scalar extraction boundary condition.

    The barrier is not part of the drift-diffusion domain. It enters as a
    boundary condition through the scalar extraction velocity computed in
    :mod:`spinpd.tunneling`.
    """

    barrier_name: str = "MgO"
    barrier_thickness: float = 2.5     # nm
    barrier_height: float = 1.0        # effective tunnel barrier height at zero bias, eV
    kappa: float = 1.0e10              # imaginary wavevector in barrier, 1/m
    fm_name: str = "CoFeB"
    gamma0: float = 1.0                # base extraction prefactor (tuned in tunneling.py)
    current_asymmetry: float = 0.2      # phenomenological helicity-to-current asymmetry
    barrier_modulation: float = 1.0    # d(barrier)/d(bias), eV/V: the share of the applied
                                       # bias that drops across the tunnel barrier (tunnel-
                                       # limited extraction -> exponential gamma(V)).
                                       # Ignored when schottky_barrier is set (TFE mode).
    schottky_barrier: float | None = None  # metal/n-GaAs Schottky barrier height (eV). If set,
                                       # the band solver pins Ec(0)=phi_B (Fermi-level pinning),
                                       # lowering the built-in potential so the I_dc collapse is
                                       # band-flattening at the (low) device Vbi. DEVICE-SPECIFIC.
    # NOTE: the former `s_interface` (interface-recombination) knob was removed.
    # The self-consistent engine reproduces the collapse as band flattening.


@dataclass(frozen=True)
class OhmicContact:
    """Ideal ohmic contact (right, substrate side)."""

    name: str = "p+"


class Stack:
    """Ordered semiconductor layers mapped onto a 1-D grid."""

    def __init__(self, layers: list[Layer]):
        if not layers:
            raise ValueError("Stack needs at least one layer")
        self.layers = list(layers)

    # --- geometry -------------------------------------------------------------
    @property
    def thickness(self) -> float:
        """Total semiconductor thickness, nm."""
        return float(sum(l.thickness for l in self.layers))

    def edges(self) -> np.ndarray:
        """Cumulative layer boundaries [0, t1, t1+t2, ...], nm."""
        return np.concatenate([[0.0], np.cumsum([l.thickness for l in self.layers])])

    def grid(self, dx: float) -> np.ndarray:
        """Node positions 0, dx, ..., thickness (nm)."""
        n = int(round(self.thickness / dx))
        return np.linspace(0.0, n * dx, n + 1)

    def layer_index(self, z: np.ndarray) -> np.ndarray:
        """Index of the layer containing each position in ``z``."""
        edges = self.edges()
        idx = np.searchsorted(edges, z, side="right") - 1
        return np.clip(idx, 0, len(self.layers) - 1)

    # --- z-resolved arrays ----------------------------------------------------
    def map_layer(self, dx: float, fn: Callable[[Layer], float]) -> np.ndarray:
        """Evaluate ``fn(layer)`` per node on the grid (piecewise-constant)."""
        z = self.grid(dx)
        li = self.layer_index(z)
        values = np.array([fn(self.layers[i]) for i in range(len(self.layers))])
        return values[li]

    def material_property(self, dx: float, attr: str, T: float | None = None) -> np.ndarray:
        """z-resolved material attribute (e.g. 'eps_r', or a T-method like 'Eg')."""
        def fn(layer: Layer) -> float:
            a = getattr(layer.material, attr)
            return a(T) if callable(a) else a
        return self.map_layer(dx, fn)

    def net_doping(self, dx: float) -> np.ndarray:
        """Net ionized doping ND - NA per node, cm^-3 (positive = n)."""
        return self.map_layer(dx, lambda l: l.doping.net)

    def band_edges(self, dx: float, T: float) -> tuple[np.ndarray, np.ndarray]:
        """Flat-band conduction/valence edges (eV, vacuum-referenced) per node."""
        Ec0 = self.map_layer(dx, lambda l: l.material.Ec0(T))
        Ev0 = self.map_layer(dx, lambda l: l.material.Ev0(T))
        return Ec0, Ev0

    def role_mask(self, dx: float, role: str) -> np.ndarray:
        """Boolean mask of nodes belonging to layers with the given role."""
        return self.map_layer(dx, lambda l: 1.0 if l.role == role else 0.0) > 0.5

    def summary(self) -> str:
        lines = [f"Stack: {len(self.layers)} layers, total {self.thickness:.1f} nm"]
        z = 0.0
        for l in self.layers:
            lines.append(
                f"  [{z:6.1f}-{z + l.thickness:6.1f}] nm  {l.label():14} "
                f"{l.role:10} doping={l.doping.kind}:{l.doping.density:.1e}"
            )
            z += l.thickness
        return "\n".join(lines)


@dataclass
class Device:
    """A complete one-dimensional device: semiconductor stack + contacts + illumination."""

    stack: Stack
    left: TunnelContact = field(default_factory=TunnelContact)
    right: OhmicContact = field(default_factory=OhmicContact)
    pump: OpticalPump = field(default_factory=OpticalPump)
    illuminated_from_left: bool = True
    name: str = "device"

    def summary(self) -> str:
        return (
            f"Device '{self.name}'\n"
            f"  left contact : {self.left.fm_name}/{self.left.barrier_name}"
            f" ({self.left.barrier_thickness} nm)\n"
            f"  right contact: {self.right.name} (ohmic)\n"
            f"  illumination : from {'left (contact)' if self.illuminated_from_left else 'right'} side\n"
            f"{self.stack.summary()}"
        )


# ---------------------------------------------------------------------------
# Baseline example: CoFeB / MgO / n-GaAs / InGaAs-QD / p-AlGaAs / p-GaAs stack
# ---------------------------------------------------------------------------
def baseline_device() -> Device:
    """Example CoFeB/MgO/GaAs + InGaAs-QD layered photodiode.

    Ordered from the MgO contact (z=0) into the substrate.  Structure:
    n-GaAs:Si(1e16)/30nm i-GaAs/InGaAs QD/30nm i-GaAs with Be delta-doping/
    400nm p-Al0.3Ga0.7As:Be(graded 2e18->5e17)/300nm p-GaAs:Be(2e18)/p+ substrate.
    The Be delta-doping forms *p-doped* InGaAs QDs (~1 dopant/QD): the QD carries
    holes and acts as a localized recombination centre. The QD carrier lifetime
    from TRPL is ~38 ps at 300 K (~650 ps at 100 K; the 100 K value is used in
    :func:`baseline_calibrated`).
    """
    gaas = GaAs()
    algaas = AlGaAs(0.30)
    qd = InAs_QD()
    layers = [
        Layer(gaas, 50.0, Doping("n", 1e16), Role.CONTACT_N.value, "n-GaAs"),
        Layer(gaas, 30.0, Doping("i"), Role.ACTIVE.value, "i-GaAs"),
        # p-doped InGaAs QD (holes from Be delta-doping): recombination centre.
        Layer(qd, 3.0, Doping("p", 2e17), Role.QD.value, "InGaAs-QD(p)",
              params={"qd_capture_rate": 2.6e10,    # 1/s low-injection capture, TRPL tau~38ps
                      "qd_trap_density": 1e17,      # A4: trap density N_t (cm^-3)
                      "qd_trion_time": 38e-12}),    # A4: trion recomb time -> R_max=N_t/tau_trion
        Layer(gaas, 30.0, Doping("i"), Role.ACTIVE.value, "i-GaAs(delta)"),
        Layer(algaas, 400.0, Doping("p", 5e17), Role.BARRIER.value, "p-AlGaAs"),
        Layer(gaas, 300.0, Doping("p", 2e18), Role.BUFFER.value, "p-GaAs"),
    ]
    return Device(
        stack=Stack(layers),
        left=TunnelContact(barrier_name="MgO", barrier_thickness=2.5,
                           fm_name="CoFeB",
                           gamma0=30.0, current_asymmetry=0.2,
                           barrier_modulation=0.0,
                           # FM Fermi-pinning -> Ec(0)=phi_B; gives Vbi~0.45V, matching
                           # THIS sample's measured I-V flat-band (+0.46V, the intensity-
                           # independent photocurrent zero). DEVICE-SPECIFIC calibration
                           # (ideally from C-V), NOT a universal constant -- recalibrate
                           # per device. The bulk-doping/Anderson Vbi (1.48V) is wrong here.
                           # The I_dc collapse is then pure band-flattening (no gamma(V)
                           # modulation, no interface-recombination fudge needed).
                           schottky_barrier=1.05),
        right=OhmicContact("p+"),
        pump=OpticalPump(),
        illuminated_from_left=True,
        name="CoFeB/MgO/GaAs+InGaAs-QD layered photodiode",
    )


def baseline_calibrated() -> Device:
    """Baseline tuned to the reference device (PIG050 Device U, ~100 K, LN2).

    On top of :func:`baseline_device` (Fermi-pinned Vbi from the I-V flat-band)
    this uses PHYSICAL material parameters -- real GaAs mobility and the measured
    100 K carrier lifetime -- because, with the latest understanding, the
    photocurrent's gradual, intensity-dependent collapse is **photovoltaic
    band-flattening**: at the device's real (focused) illumination the photo-
    carriers screen the built-in field / build an open-circuit photovoltage
    Voc(P) that grows ~ln(P) (directly measured: Voc -> ~Vbi at high power,
    ideality n~1.6), so a smaller applied forward bias suffices to flatten the
    band. This is high-injection physics, NOT a sub-ps collection length (the
    old tau~0.5 ps was an artifact of fitting a photovoltaic collapse with a
    transport model). Series resistance is measured small (~0.3 kOhm) and is a
    minor correction, not the mechanism.

    Device-specific anchors: Vbi (from the I-V flat-band, ~+0.43 V for Device U)
    and the contact extraction scale. Material inputs are physical: mu_e (Hall);
    a physical MBE bulk lifetime (i-GaAs tau_SRH ~ ns); and the QD modelled as a
    capture-residence TRAP -- a fast ~ps CAPTURE into the dot followed by the
    measured 650 ps TRION recombination (TRPL at 100 K; the 38 ps often quoted is
    the 300 K value).  The fast capture localizes ~88% of the recombination AT the
    dot on the time axis (no band offset, so no Scharfetter-Gummel break and no
    reverse-current over-trap) -- the physically correct picture, and it leaves the
    measured V50(P) collapse unchanged vs the old distributed-SRH proxy.
    Lateral inhomogeneity of the FM barrier broadens the transition (see
    ``observables.bias_sweep_sc(phiB_sigma=...)``); the photovoltaic regime is
    reached at high ``intensity_scale``.
    """
    # PHYSICAL parameters (no sub-ps fudge): real GaAs mobility, and a PHYSICAL
    # MBE bulk lifetime -- the i-GaAs is a high-quality undoped layer, tau_SRH ~ ns
    # (here 5 ns; radiative-limited ~14-140 ns at these densities).  The fast,
    # QD-set recombination is NOT smeared into the bulk: it is carried by the dot
    # as a trap (capture + trion), see below.
    g = with_overrides(GaAs(), mu_e=8000.0, tau_srh=5e-9)   # real mu; PHYSICAL MBE i-GaAs (~ns)
    al = with_overrides(AlGaAs(0.30), mu_e=8000.0)
    # QD as a CAPTURE-RESIDENCE TRAP (not a confining well): a real dot holds a
    # carrier on TWO time-scales -- a fast CAPTURE (~ps; here 10 ps) into the dot,
    # then the slower TRION recombination (650 ps, the TRPL value, via the
    # saturable R_max = N_t/tau_trion).  A fast capture localizes the recombination
    # AT the dot (~88% of int R) on the time axis WITHOUT a band offset -- so it
    # neither breaks the Scharfetter-Gummel flux nor over-traps the reverse
    # photocurrent (verified: reverse I and the V50(P) collapse are unchanged vs
    # the old distributed-SRH parametrization; the bulk is now physical).  The old
    # qd_capture_rate=1.5e9 wrongly used the 650 ps *recombination* time as the
    # *capture* time, so the thin dot caught almost nothing (16% -> artifact).
    qd = with_overrides(InAs_QD(), tau_srh=1e30)
    layers = [
        Layer(g, 50.0, Doping("n", 1e16), Role.CONTACT_N.value, "n-GaAs"),
        Layer(g, 30.0, Doping("i"), Role.ACTIVE.value, "i-GaAs"),
        Layer(qd, 3.0, Doping("p", 2e17), Role.QD.value, "InGaAs-QD(p)",
              params={"qd_capture_rate": 1e11, "qd_trap_density": 1e17,
                      "qd_trion_time": 650e-12}),   # 10 ps capture -> trion 650 ps
        Layer(g, 30.0, Doping("i"), Role.ACTIVE.value, "i-GaAs(delta)"),
        Layer(al, 400.0, Doping("p", 5e17), Role.BARRIER.value, "p-AlGaAs"),
        Layer(g, 300.0, Doping("p", 2e18), Role.BUFFER.value, "p-GaAs"),
    ]
    dev = baseline_device()
    dev.stack = Stack(layers)
    dev.name = "CoFeB/MgO/GaAs+InGaAs-QD layered photodiode (calibrated)"
    return dev
