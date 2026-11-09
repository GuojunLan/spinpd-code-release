# Source data: helicity-dependent CoFeB/MgO/GaAs spin photodiode (Spin PD)

These files are Data S1 to S3 of the Supplementary Materials of the manuscript on the CoFeB/MgO/GaAs spin photodiode
(Science Advances, revised version, November 2026). They contain the numerical values plotted in every data panel of the
main-text and supplementary figures. They were exported directly from the scripts that draw the published figures.

| File | Figures | Content |
|---|---|---|
| `Data_S1.csv` | Fig. 1B, 1C | Photocurrent–voltage curves at ~300 K and ~100 K (0 to 60 mW, 0 mW = dark) and the saturated photocurrent `I_sat` versus laser power |
| `Data_S2.csv` | Fig. 3, Fig. 5, figs. S3–S6 | Bias dependence of `I_dc`, rate ratio `eta`, spin photocurrent `I_spin`, MCD, differential conductance; model curves; normalized literature comparison (Fig. 5) |
| `Data_S3.csv` | Fig. 1F, Fig. 2, Fig. 4, figs. S2, S7, S8 | Out-of-plane (inverted Hanle) and in-plane (Hanle) field scans at fixed bias, Lorentzian fits, half-widths, time scales and spin survival |

Schematic panels (Fig. 1A, 1D, 1E and Fig. 2F) contain no data. The calculated band profiles of fig. S1 can be regenerated with the code in this repository (see the example in the main README). Figs. S9–S12 replot published data (references in the
Supplementary Materials); the values used in Fig. 5 are in `Data_S2.csv`.

## Format

All three files use the same long ("tidy") format, UTF-8, comma-separated, one plotted point per row:

| Column | Meaning |
|---|---|
| `figure` | Figure in the manuscript (`Fig. 1` … `Fig. 5`, `fig. S2` … `fig. S8`) |
| `panel` | Panel letter (empty for single-panel figures) |
| `axes_note` | Text printed inside the panel (temperature, bias, laser power, fitted half-width, …), identifies sub-plots |
| `x_label`, `y_label` | Axis quantity and unit, as on the figure (`description, symbol (unit)`) |
| `series` | Data series (temperature and laser power, magnetization state, model parameter, literature device with its reference number) |
| `kind` | `points` / `line+points`: measured data; `line`: model or fit curve (exception: Fig. 1B, where the lines are measured I–V curves); `errorbar`: uncertainty given in `y_lo`, `y_hi`; `band`: confidence band between `y_lo` and `y_hi` |
| `marker` | `filled symbol` / `open symbol`, with the meaning given in the figure caption (for example, inside or outside the fitting window) |
| `x`, `y` | Plotted values in the units of `x_label` and `y_label` |
| `y_lo`, `y_hi` | Lower and upper limits for `errorbar` and `band` rows |

Conventions: ~100 K means the liquid-nitrogen cryostat (sample temperature not measured), ~300 K means room temperature.
`I_dc` is the photocurrent (light minus dark), `I_sat` its reverse-bias plateau, and `eta = I_sat/I_dc - 1` the rate ratio.
For twin-axis panels (Fig. 3A, 3E), the `series` name says which axis applies. Model curves are sampled at up to 400 points.

## Reading the data

```python
import pandas as pd
d = pd.read_csv("Data_S2.csv")
fig3b = d[(d.figure == "Fig. 3") & (d.panel == "B")]
for name, g in fig3b.groupby("series"):
    print(name, len(g))
```
