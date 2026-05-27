

"""Finite-difference reference solver for the robot joint motor PINN project.

This file is intentionally separated from ``main_deepxde.py``.

Role of this file:
    1. Solve the nondimensional 1D transient heat equation using FDM.
    2. Generate a full-field reference temperature map.
    3. Extract sparse synthetic sensor data for inverse PINN training.
    
    1. FDM으로 full reference field 생성
    2. sensor 위치에서 synthetic data 추출
    3. main_deepxde.py에서 import해서 사용 가능하게 함수 제공

Nondimensional PDE:
    theta_tau = theta_XX - beta * theta + S * i(tau)^2 * g(X)

Boundary conditions:
    dtheta/dX = 0 at X = 0 and X = 1

Initial condition:
    theta(X, 0) = 0
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

import numpy as np


# -------------------------------------------------------------------
# 1. Physical constants and reference scales
# -------------------------------------------------------------------
@dataclass(frozen=True)
class MotorThermalConfig:
    """Physical and nondimensional constants used in the project."""

    # Geometry [m]
    L: float = 0.15
    R_OUTER: float = 0.04
    R_INNER: float = 0.025

    # Aluminum alloy material properties
    RHO: float = 2700.0  # kg/m^3
    CP: float = 900.0  # J/(kg K)
    K: float = 167.0  # W/(m K)

    # Ambient and reference scales
    T_INF: float = 25.0  # deg C
    I_REF: float = 5.0  # A
    DELTA_T_REF: float = 50.0  # deg C
    T_FINAL: float = 1200.0  # s

    # True inverse parameters used for synthetic data
    H_TRUE: float = 40.0  # W/(m^2 K)
    Q0_TRUE: float = 8.0e4  # W/(m^3 A^2), for Gaussian g(X)

    # Gaussian heat source parameters in nondimensional coordinate X
    X0: float = 0.30
    SIGMA: float = 0.10

    @property
    def AREA(self) -> float:
        """Effective cross-sectional area for axial conduction [m^2]."""
        return np.pi * (self.R_OUTER**2 - self.R_INNER**2)

    @property
    def PERIMETER(self) -> float:
        """Outer perimeter exposed to ambient air [m]."""
        return 2.0 * np.pi * self.R_OUTER

    @property
    def ALPHA(self) -> float:
        """Thermal diffusivity [m^2/s]."""
        return self.K / (self.RHO * self.CP)

    @property
    def TAU_FINAL(self) -> float:
        """Final nondimensional time."""
        return self.ALPHA * self.T_FINAL / self.L**2

    @property
    def BETA_TRUE(self) -> float:
        """True nondimensional cooling parameter."""
        return self.H_TRUE * self.PERIMETER * self.L**2 / (self.K * self.AREA)

    @property
    def S_TRUE(self) -> float:
        """True nondimensional Joule heating parameter."""
        return self.Q0_TRUE * self.I_REF**2 * self.L**2 / (self.K * self.DELTA_T_REF)


CFG = MotorThermalConfig()

# Common aliases for convenient import from main_deepxde.py
L = CFG.L
T_INF = CFG.T_INF
I_REF = CFG.I_REF
DELTA_T_REF = CFG.DELTA_T_REF
ALPHA = CFG.ALPHA
TAU_FINAL = CFG.TAU_FINAL
BETA_TRUE = CFG.BETA_TRUE
S_TRUE = CFG.S_TRUE
H_TRUE = CFG.H_TRUE
Q0_TRUE = CFG.Q0_TRUE


# -------------------------------------------------------------------
# 2. Basic helper functions
# -------------------------------------------------------------------
def temperature_to_theta(temperature_c: np.ndarray | float, cfg: MotorThermalConfig = CFG) -> np.ndarray:
    """Convert physical temperature [deg C] to nondimensional temperature theta."""
    return (np.asarray(temperature_c, dtype=np.float64) - cfg.T_INF) / cfg.DELTA_T_REF


def theta_to_temperature(theta: np.ndarray | float, cfg: MotorThermalConfig = CFG) -> np.ndarray:
    """Convert nondimensional temperature theta to physical temperature [deg C]."""
    return cfg.T_INF + cfg.DELTA_T_REF * np.asarray(theta, dtype=np.float64)


def tau_to_time(tau: np.ndarray | float, cfg: MotorThermalConfig = CFG) -> np.ndarray:
    """Convert nondimensional time tau to physical time [s]."""
    return np.asarray(tau, dtype=np.float64) * cfg.L**2 / cfg.ALPHA


def time_to_tau(time_s: np.ndarray | float, cfg: MotorThermalConfig = CFG) -> np.ndarray:
    """Convert physical time [s] to nondimensional time tau."""
    return cfg.ALPHA * np.asarray(time_s, dtype=np.float64) / cfg.L**2


# -------------------------------------------------------------------
# 3. Current input and heat source profile
# -------------------------------------------------------------------
def current_profile_seconds(time_s: np.ndarray | float) -> np.ndarray:
    """Piecewise current profile I(t) in amperes.

    Recommended profile:
        0-100 s     : 0 A
        100-400 s   : 5 A
        400-650 s   : 0 A
        650-850 s   : 3.5 A
        850-1200 s  : 0 A
    """
    t = np.asarray(time_s, dtype=np.float64)
    current = np.zeros_like(t, dtype=np.float64)

    current[(100.0 <= t) & (t < 400.0)] = 5.0
    current[(650.0 <= t) & (t < 850.0)] = 3.5

    return current


def current_profile_tau(tau: np.ndarray | float, cfg: MotorThermalConfig = CFG) -> np.ndarray:
    """Nondimensional current profile i(tau) = I(tau) / I_REF."""
    time_s = tau_to_time(tau, cfg)
    return current_profile_seconds(time_s) / cfg.I_REF


def current_profile(tau: np.ndarray | float, cfg: MotorThermalConfig = CFG) -> np.ndarray:
    """Alias used by main_deepxde.py: returns nondimensional current i(tau)."""
    return current_profile_tau(tau, cfg)


def gaussian_heat_source(x: np.ndarray | float, cfg: MotorThermalConfig = CFG) -> np.ndarray:
    """Gaussian spatial heat source g(X).

    X is nondimensional position in [0, 1].
    The source is centered near the winding region at X0 = 0.30.
    """
    x_arr = np.asarray(x, dtype=np.float64)
    return np.exp(-((x_arr - cfg.X0) ** 2) / (2.0 * cfg.SIGMA**2))


# -------------------------------------------------------------------
# 4. FDM solver
# -------------------------------------------------------------------
def _compute_second_derivative_neumann(theta: np.ndarray, dx: float) -> np.ndarray:
    """Compute theta_XX with zero-gradient Neumann BC at both ends."""
    theta_xx = np.zeros_like(theta)

    # Interior nodes
    theta_xx[1:-1] = (theta[2:] - 2.0 * theta[1:-1] + theta[:-2]) / dx**2

    # Adiabatic boundaries: ghost-node approach
    # theta[-1 ghost] = theta[1], theta[N ghost] = theta[N-2]
    theta_xx[0] = 2.0 * (theta[1] - theta[0]) / dx**2
    theta_xx[-1] = 2.0 * (theta[-2] - theta[-1]) / dx**2

    return theta_xx


def _rhs(
    theta: np.ndarray,
    x_values: np.ndarray,
    dx: float,
    tau: float,
    beta: float,
    s_value: float,
    cfg: MotorThermalConfig,
) -> np.ndarray:
    """Right-hand side of theta_tau = theta_XX - beta theta + S i^2 g."""
    theta_xx = _compute_second_derivative_neumann(theta, dx)
    i_tau = current_profile_tau(tau, cfg)
    source = s_value * (i_tau**2) * gaussian_heat_source(x_values, cfg)
    return theta_xx - beta * theta + source


def solve_full_fdm_reference(
    nx: int = 101,
    n_time: int = 201,
    beta: float | None = None,
    s_value: float | None = None,
    cfg: MotorThermalConfig = CFG,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solve the full nondimensional PDE using explicit FDM.

    Parameters
    ----------
    nx:
        Number of spatial grid points.
    n_time:
        Number of saved time points.
    beta:
        Nondimensional cooling parameter. If None, use BETA_TRUE.
    s_value:
        Nondimensional Joule heating parameter. If None, use S_TRUE.
    cfg:
        Motor thermal configuration.

    Returns
    -------
    x_values:
        Nondimensional spatial grid X, shape = (nx,).
    tau_values:
        Nondimensional time grid tau, shape = (n_time,).
    theta_map:
        Nondimensional temperature field, shape = (n_time, nx).
    """
    if nx < 3:
        raise ValueError("nx must be at least 3 for second-derivative FDM.")
    if n_time < 2:
        raise ValueError("n_time must be at least 2.")

    beta = cfg.BETA_TRUE if beta is None else float(beta)
    s_value = cfg.S_TRUE if s_value is None else float(s_value)

    x_values = np.linspace(0.0, 1.0, nx, dtype=np.float64)
    tau_values = np.linspace(0.0, cfg.TAU_FINAL, n_time, dtype=np.float64)
    dx = x_values[1] - x_values[0]

    # Explicit Euler stability guideline for 1D diffusion:
    # dtau <= 0.5 dx^2. Use a conservative value.
    dtau_max = 0.4 * dx**2

    theta = np.zeros(nx, dtype=np.float64)
    theta_map = np.zeros((n_time, nx), dtype=np.float64)
    theta_map[0] = theta

    tau_now = 0.0

    for save_idx in range(1, n_time):
        tau_target = tau_values[save_idx]

        while tau_now < tau_target - 1.0e-14:
            dtau = min(dtau_max, tau_target - tau_now)
            theta = theta + dtau * _rhs(theta, x_values, dx, tau_now, beta, s_value, cfg)
            tau_now += dtau

        theta_map[save_idx] = theta

    return x_values, tau_values, theta_map


# -------------------------------------------------------------------
# 5. Sparse synthetic sensor data generation
# -------------------------------------------------------------------
def sample_sensor_data_from_field(
    x_values: np.ndarray,
    tau_values: np.ndarray,
    theta_map: np.ndarray,
    sensor_locations: tuple[float, ...] = (0.25, 0.55, 0.85),
    noise_std_theta: float = 0.0,
    seed: int | None = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract sparse sensor data from a full FDM field.

    Returns
    -------
    sensor_xtau:
        Sensor input coordinates, shape = (N_sensor * N_time, 2).
        Column 0 is X, column 1 is tau.
    sensor_theta:
        Sensor target values, shape = (N_sensor * N_time, 1).
    """
    rng = np.random.default_rng(seed)

    xtau_list: list[list[float]] = []
    theta_list: list[float] = []

    for tau_idx, tau in enumerate(tau_values):
        theta_at_time = theta_map[tau_idx]

        for x_sensor in sensor_locations:
            theta_sensor = float(np.interp(x_sensor, x_values, theta_at_time))

            if noise_std_theta > 0.0:
                theta_sensor += float(rng.normal(loc=0.0, scale=noise_std_theta))

            xtau_list.append([float(x_sensor), float(tau)])
            theta_list.append(theta_sensor)

    sensor_xtau = np.asarray(xtau_list, dtype=np.float32)
    sensor_theta = np.asarray(theta_list, dtype=np.float32).reshape(-1, 1)

    return sensor_xtau, sensor_theta


def make_synthetic_sensor_data_numpy(
    sensor_locations: tuple[float, ...] = (0.25, 0.55, 0.85),
    sensor_time_points: int = 41,
    fdm_nx: int = 201,
    noise_std_deg_c: float = 0.0,
    seed: int | None = 0,
    beta: float | None = None,
    s_value: float | None = None,
    cfg: MotorThermalConfig = CFG,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate sparse synthetic sensor data for DeepXDE PointSetBC.

    The returned arrays can be used directly in DeepXDE:
        dde.icbc.PointSetBC(sensor_xtau, sensor_theta, component=0)

    Parameters
    ----------
    noise_std_deg_c:
        Standard deviation of additive sensor noise in physical degrees Celsius.
        It is internally converted to nondimensional theta.
    """
    x_values, tau_values, theta_map = solve_full_fdm_reference(
        nx=fdm_nx,
        n_time=sensor_time_points,
        beta=beta,
        s_value=s_value,
        cfg=cfg,
    )

    noise_std_theta = noise_std_deg_c / cfg.DELTA_T_REF

    return sample_sensor_data_from_field(
        x_values=x_values,
        tau_values=tau_values,
        theta_map=theta_map,
        sensor_locations=sensor_locations,
        noise_std_theta=noise_std_theta,
        seed=seed,
    )


# Shorter alias if preferred in main_deepxde.py
make_synthetic_sensor_data = make_synthetic_sensor_data_numpy


# -------------------------------------------------------------------
# 6. Optional file export helpers
# -------------------------------------------------------------------
def save_sensor_csv(path: str, sensor_xtau: np.ndarray, sensor_theta: np.ndarray) -> None:
    """Save sparse sensor data as CSV."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    data = np.column_stack([sensor_xtau, sensor_theta.reshape(-1)])
    header = "X,tau,theta"
    np.savetxt(path, data, delimiter=",", header=header, comments="")


def save_reference_npz(path: str, x_values: np.ndarray, tau_values: np.ndarray, theta_map: np.ndarray) -> None:
    """Save full FDM reference field as compressed NPZ."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez_compressed(path, x=x_values, tau=tau_values, theta=theta_map)


def print_reference_summary() -> None:
    """Print key nondimensional and physical reference values."""
    print("FDM reference configuration")
    print(f"L              = {CFG.L:.6g} m")
    print(f"A              = {CFG.AREA:.6e} m^2")
    print(f"P              = {CFG.PERIMETER:.6e} m")
    print(f"alpha          = {CFG.ALPHA:.6e} m^2/s")
    print(f"tau_final      = {CFG.TAU_FINAL:.6e}")
    print(f"h_true         = {CFG.H_TRUE:.6e} W/(m^2 K)")
    print(f"q0_true        = {CFG.Q0_TRUE:.6e} W/(m^3 A^2)")
    print(f"beta_true      = {CFG.BETA_TRUE:.6e}")
    print(f"S_true         = {CFG.S_TRUE:.6e}")


# -------------------------------------------------------------------
# 7. Command-line execution for standalone testing
# -------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate FDM reference data for the motor housing PINN project.")
    parser.add_argument("--nx", type=int, default=201)
    parser.add_argument("--n-time", type=int, default=401)
    parser.add_argument("--sensor-time-points", type=int, default=41)
    parser.add_argument("--noise-std-deg-c", type=float, default=0.0)
    parser.add_argument("--output-dir", type=str, default="outputs")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print_reference_summary()

    x_values, tau_values, theta_map = solve_full_fdm_reference(nx=args.nx, n_time=args.n_time)
    reference_path = os.path.join(args.output_dir, "fdm_reference.npz")
    save_reference_npz(reference_path, x_values, tau_values, theta_map)

    sensor_xtau, sensor_theta = make_synthetic_sensor_data_numpy(
        sensor_time_points=args.sensor_time_points,
        fdm_nx=args.nx,
        noise_std_deg_c=args.noise_std_deg_c,
    )
    sensor_path = os.path.join(args.output_dir, "synthetic_sensor_data.csv")
    save_sensor_csv(sensor_path, sensor_xtau, sensor_theta)

    print()
    print("Saved FDM reference and sensor data")
    print(f"Full reference field : {reference_path}")
    print(f"Sensor data CSV      : {sensor_path}")
    print(f"theta max            : {np.max(theta_map):.6e}")
    print(f"temperature max      : {float(np.max(theta_to_temperature(theta_map))):.3f} deg C")
    print(f"sensor data shape    : xtau={sensor_xtau.shape}, theta={sensor_theta.shape}")


if __name__ == "__main__":
    main()