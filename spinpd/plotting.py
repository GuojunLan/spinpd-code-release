"""Diagnostic plotting helpers for band and dc transport simulations."""
from __future__ import annotations

import matplotlib.pyplot as plt

from .bands import solve_equilibrium
from .stack import Device


def _shade_layers(ax, device: Device):
    z0 = 0.0
    colors = {
        "contact_n": "#e8eef7",
        "active": "#edf6ef",
        "qd": "#f6d79e",
        "qw": "#f6d79e",
        "barrier": "#f3e3e3",
        "buffer": "#eaecef",
        "substrate": "#eaecef",
    }
    for layer in device.stack.layers:
        z1 = z0 + layer.thickness
        ax.axvspan(z0, z1, color=colors.get(layer.role, "#f4f4f4"), alpha=0.35, lw=0)
        z0 = z1


def plot_band_diagram(
    device: Device,
    biases=(0.0,),
    T: float = 100.0,
    dx: float = 1.0,
    savepath: str | None = None,
):
    """Plot equilibrium conduction and valence band profiles."""
    fig, ax = plt.subplots(figsize=(8, 4.8))
    _shade_layers(ax, device)
    for V in biases:
        profile = solve_equilibrium(device, T=T, dx=dx, bias_V=float(V))
        ax.plot(profile.z, profile.Ec, label=fr"$E_c$, V={V:+.2f} V")
        ax.plot(profile.z, profile.Ev, "--", label=fr"$E_v$, V={V:+.2f} V")
    ax.axhline(0.0, color="k", lw=0.8, alpha=0.5)
    ax.set_xlabel("Depth z (nm)")
    ax.set_ylabel("Energy (eV)")
    ax.set_title("Equilibrium band profiles")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=150)
        plt.close(fig)
    return fig


def plot_carrier_profile(z, n, p, device: Device, savepath: str | None = None):
    """Plot electron and hole density profiles."""
    fig, ax = plt.subplots(figsize=(8, 4.8))
    _shade_layers(ax, device)
    ax.semilogy(z, n, label="n")
    ax.semilogy(z, p, label="p")
    ax.set_xlabel("Depth z (nm)")
    ax.set_ylabel(r"Carrier density (cm$^{-3}$)")
    ax.set_title("Self-consistent carrier profiles")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=150)
        plt.close(fig)
    return fig


def plot_idc_vs_bias(sweep, savepath: str | None = None):
    """Plot extracted dc photocurrent versus applied bias."""
    fig, ax = plt.subplots(figsize=(6.2, 4.5))
    ax.plot(sweep.V, sweep.I_dc * 1e6, "-o", ms=3)
    ax.set_xlabel("Bias voltage (V)")
    ax.set_ylabel(r"$I_{dc}$ ($\mu$A)")
    ax.set_title("Self-consistent dc bias sweep")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=150)
        plt.close(fig)
    return fig
