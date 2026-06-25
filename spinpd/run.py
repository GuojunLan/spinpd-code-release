"""Basic driver for band and self-consistent transport diagnostics.

Run with:

    python -m spinpd.run

The driver writes equilibrium band profiles, illuminated carrier profiles, and a
dc photocurrent bias sweep into `outputs/`.
"""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .observables import bias_sweep_sc
from .plotting import plot_band_diagram, plot_carrier_profile, plot_idc_vs_bias
from .stack import baseline_calibrated
from .transport_sc import solve_sc


def main(outdir: str = "outputs") -> None:
    os.makedirs(outdir, exist_ok=True)
    dev = baseline_calibrated()
    print(dev.summary())

    plot_band_diagram(
        dev,
        biases=(-0.4, -0.2, 0.0, 0.2, 0.4),
        T=100,
        dx=1.0,
        savepath=os.path.join(outdir, "band_diagram.png"),
    )

    r0 = solve_sc(dev, bias_V=0.0, T=100, dx=1.0)
    plot_carrier_profile(
        r0.z,
        r0.n,
        r0.p,
        dev,
        savepath=os.path.join(outdir, "carrier_profile.png"),
    )

    biases = np.linspace(-0.5, 0.45, 28)
    sweep = bias_sweep_sc(
        dev,
        biases,
        T=100,
        dx=1.0,
        phiB_sigma=0.12,
        intensity_scale=100.0,
    )
    plot_idc_vs_bias(sweep, savepath=os.path.join(outdir, "idc_vs_bias.png"))
    if sweep.I_spin is not None:
        fig, ax = plt.subplots(figsize=(6.2, 4.5))
        ax.plot(sweep.V, sweep.I_spin * 1e9, "-o", ms=3)
        ax.set_xlabel("Bias voltage (V)")
        ax.set_ylabel(r"$I_{spin}$ (nA)")
        ax.set_title("Macroscopic spin-current observable")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, "spin_current_vs_bias.png"), dpi=150)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 4.5))
    ax.plot(sweep.V, sweep.eta_internal, "-o", ms=3)
    ax.set_xlabel("Bias voltage (V)")
    ax.set_ylabel(r"Internal $\eta$ = recombination / extraction")
    ax.set_yscale("log")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "eta_vs_bias.png"), dpi=150)
    plt.close(fig)

    print("\nSelf-consistent dc transport sweep")
    print(f"I_dc max: {sweep.I_dc.max()*1e6:.2f} uA")
    print(f"I_dc min: {sweep.I_dc.min()*1e6:.2f} uA")
    if sweep.I_spin is not None:
        print(f"I_spin max: {sweep.I_spin.max()*1e9:.2f} nA")
    print(f"minimum-current bias: {sweep.minimum_current_bias():+.2f} V")
    print(f"figures written to {outdir}/")


if __name__ == "__main__":
    main()
