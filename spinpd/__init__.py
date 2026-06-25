"""spinpd: 1-D band and self-consistent transport simulation tools."""
from __future__ import annotations

from . import materials
from .bands import BandProfile, solve_equilibrium
from .config import CONST, Numerics, OpticalPump, OperatingPoint, SimConfig
from .observables import DCBiasSweep, bias_sweep_sc
from .stack import (
    Device,
    Doping,
    Layer,
    OhmicContact,
    Role,
    Stack,
    TunnelContact,
    baseline_calibrated,
    baseline_device,
)
from .transport import bernoulli, generation_profile
from .transport_sc import SCResult, SpinCurrentResult, solve_sc, spin_current

__all__ = [
    "materials",
    "CONST",
    "Numerics",
    "OpticalPump",
    "OperatingPoint",
    "SimConfig",
    "BandProfile",
    "Device",
    "Doping",
    "Layer",
    "OhmicContact",
    "Role",
    "Stack",
    "TunnelContact",
    "baseline_device",
    "baseline_calibrated",
    "solve_equilibrium",
    "generation_profile",
    "bernoulli",
    "solve_sc",
    "SCResult",
    "spin_current",
    "SpinCurrentResult",
    "bias_sweep_sc",
    "DCBiasSweep",
]

__version__ = "0.1.0"
