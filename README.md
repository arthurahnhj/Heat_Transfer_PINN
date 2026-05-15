# Heat Transfer PINN

Final project for MEN310 Heat Transfer.

## Topic

`Inverse PINN for Transient Thermal Parameter Estimation in a Robot Joint Motor Housing`

This project models a robot joint motor housing as a one-dimensional transient heat transfer system. During operation, motor current generates internal Joule heating near the winding region. Heat then conducts along the aluminum housing and is dissipated to ambient air through convection from the outer cylindrical surface.

The goal is to use a Physics-Informed Neural Network (PINN) to predict the transient temperature field and estimate unknown thermal parameters from sparse temperature sensor data.

## Physical Model

Unknown temperature field:

```text
T = T(x,t)
```

Governing equation:

```text
rho*cp*dT/dt
= k*d2T/dx2
  - (h*P/A)*(T - T_inf)
  + q0*I(t)^2*g(x)
```

Unknown inverse parameters:

```text
h  = convective heat transfer coefficient
q0 = Joule heating coefficient
```

Boundary and initial conditions:

```text
T(x,0) = T_inf
dT/dx(0,t) = 0
dT/dx(L,t) = 0
```

## Geometry and Material

```text
L  = 0.15 m
Ro = 0.04 m
Ri = 0.025 m

rho = 2700 kg/m^3
cp  = 900 J/(kg*K)
k   = 167 W/(m*K)
T_inf = 25 deg C
```

The effective axial conduction area and external convection perimeter are:

```text
A = pi*(Ro^2 - Ri^2)
P = 2*pi*Ro
```

## Project References

- `BASELINE.md`: current project memory and implementation direction
- `Phase1/Phase1_Proposal_안형준.pdf`: Phase 1 proposal
- `2026 MEN310 Final Project.pdf`: final project instruction
- `2026 MEN310 Final Project_KR.pdf`: Korean final project instruction

## Current Code Status

The current `main.py` is an early PINN skeleton. It will be updated from the previous flat-plate starter into the motor housing inverse PINN described in the Phase 1 proposal.

Planned implementation path:

1. Define nondimensional variables `X`, `tau`, and `theta`.
2. Implement Gaussian heat generation `g(X)`.
3. Implement a time-varying current profile `i(tau)`.
4. Build a neural network for `theta(X,tau)`.
5. Add PDE, initial condition, boundary condition, and sensor data losses.
6. Learn the inverse parameters `beta` and `S`.
7. Convert learned `beta`, `S` back to physical `h`, `q0`.

## Environment

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install basic packages:

```bash
pip install numpy matplotlib scipy torch
```

## Run

```bash
python main.py
```
