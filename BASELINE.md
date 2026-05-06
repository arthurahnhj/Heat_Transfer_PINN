# Baseline Chat Memory

This file records the project baseline so future work starts from the same assumptions.

## Chosen Topic

Recommended Korean title:

`PINN을 이용한 등온 평판 층류 강제대류의 Thermal Boundary Layer 및 Local Nusselt Number 예측`

Recommended English title:

`Physics-Informed Neural Network for Laminar Forced Convection over an Isothermal Flat Plate`

Optional inverse-problem title:

`Inverse PINN for Estimating Local Convective Heat Transfer Coefficient over a Heated Flat Plate`

## Core Idea

A hot isothermal flat plate is exposed to a uniform external flow. A thermal boundary layer grows in the streamwise direction. Near the leading edge, the boundary layer is thin, the wall temperature gradient is large, and the local heat-transfer coefficient is high. Farther downstream, the thermal boundary layer thickens, the wall gradient decreases, and `h_x` decreases.

The PINN predicts the temperature field, then computes the wall-normal temperature gradient at the wall:

```text
q_s'' = -k_f (dT/dy)|_wall

h_x = -k_f (dT/dy)|_wall / (T_s - T_inf)

Nu_x = h_x x / k_f
```

## Recommended Implementation

Use a safe but still meaningful Version A plus Version C path.

Version A:
- Use the Blasius velocity field as a known reference.
- Train the PINN only for the thermal boundary-layer equation.
- Network input: `(x, y)`.
- Network output: nondimensional temperature `theta(x, y)`.

Version C:
- Add sparse temperature observations.
- Recover the full temperature field.
- Use wall gradients to infer `h_x` and `Nu_x`.
- Optionally make `Pr` trainable with a positive parameterization such as `Pr = exp(lambda)`.

## Physical Model

Laminar forced convection over an isothermal flat plate:

```text
T_s = 75 deg C
T_inf = 25 deg C
U_inf = 1 to 5 m/s
Pr ~= 0.7
Re_L < 5e5
```

Use a domain starting slightly downstream of the leading edge:

```text
x_min / L ~= 0.02
```

## Energy Equation

For Version A, the known velocity field is `(u, v)` and the PINN solves:

```text
u dT/dx + v dT/dy = alpha d2T/dy2
```

Equivalently, for nondimensional temperature:

```text
theta = (T - T_inf) / (T_s - T_inf)
u dtheta/dx + v dtheta/dy = alpha d2theta/dy2
```

Boundary conditions:

```text
theta(x, 0) = 1
theta(x, y_top) = 0
```

Optional inlet or reference-data constraints can be added as needed.

## Validation Correlations

For laminar isothermal flat-plate flow:

```text
Nu_x = 0.332 Re_x^(1/2) Pr^(1/3)

Nu_L_avg = 0.664 Re_L^(1/2) Pr^(1/3)
```

Report line:

The wall gradient from the PINN-predicted temperature field was used to compute `h_x` and `Nu_x`, and the result was compared against the laminar flat-plate similarity correlation.

## Backup Topic

Thermal contact resistance PINN remains the backup topic. It is safer but has lower presentation impact than the flat-plate convection PINN.

## Development Workflow Memory

Current working assumption:

- Code development, cleanup, documentation, and small smoke tests are done on the MacBook.
- Final or heavy training runs are done on the desktop PC with an RTX 5060 Ti 16GB GPU.
- GitHub is used as the bridge between machines: push work from the MacBook, then pull updates on the desktop before running CUDA training.

GPU/Colab decision:

- Use the local RTX 5060 Ti 16GB desktop environment as the main training environment.
- Use Colab only as a backup, sharing/demo environment, or temporary fallback if the local CUDA setup is unavailable.
- The local GPU is preferred because it avoids Colab runtime limits, random GPU allocation, session interruptions, and repeated setup friction.

Implementation habits to preserve:

- Write device-agnostic PyTorch code so the same scripts can run on MacBook and desktop:

```python
device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
```

- Avoid absolute local paths such as `/Users/.../AI coding/...` in project code.
- Keep project paths relative to the repository, for example `data/heat_transfer_experiments.csv`.
- Keep heavy generated files out of Git, including checkpoints, run logs, large arrays, and intermediate outputs.
- Track source code, configuration files, small reference data, final plots, reports, and environment documentation.
- Maintain `requirements.txt`, `environment.yml`, or README setup notes so the desktop CUDA environment can be recreated cleanly.
- For MacBook tests, use small epochs, reduced collocation points, and fast smoke-test settings.
- For desktop runs, use CUDA-oriented settings and the RTX 5060 Ti 16GB as the main training target.
