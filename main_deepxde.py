"""Phase 1 제안서 기준 DeepXDE inverse PINN

주제:
    로봇 관절 모터 하우징의 1D 비정상 열전달 inverse PINN

이 파일의 목표:
    main.py에 직접 PyTorch로 작성한 PINN 구조를 DeepXDE 형식으로 옮김
    main.py는 baseline으로 남겨두고, DeepXDE 실험은 이 파일에서 진행
    이 파일 하나만 실행해도 되도록 물리 상수와 helper 함수도 내부에 직접 정의함

큰 그림:
    1. DeepXDE의 GeometryXTime으로 (X, tau) 해석 영역을 만듦
    2. DeepXDE의 TimePDE에 PDE residual을 넘김
    3. IC, adiabatic BC, sensor data 조건을 DeepXDE BC 객체로 넣음
    4. neural network는 dde.nn.FNN으로 생성
    5. beta와 S는 dde.Variable로 두고 신경망 weight와 함께 학습
    6. Adam optimizer로 먼저 학습, 원하면 L-BFGS로 한 번 더 다듬기
    7. 학습 과정에서 beta, S를 출력하는 callback 사용
    8. 학습이 끝나면 beta, S를 실제 h, q0로 변환해서 출력

주의:
    현재 sensor data는 full PDE finite difference solver로 생성함
    synthetic_theta_no_conduction()은 비교용 local ODE 함수로 남겨둠
"""

from __future__ import annotations

import argparse
import math
import os
import platform
import sys

# DeepXDE는 import될 때 backend를 결정한다.
# 그래서 deepxde를 import하기 전에 PyTorch backend를 먼저 지정해야 한다.
os.environ.setdefault("DDE_BACKEND", "pytorch")

import numpy as np
import torch
from torch.nn import functional as F


# 패키지 설치 여부 확인 코드
try:
    import deepxde as dde
except ModuleNotFoundError as exc:
    raise SystemExit(
        "DeepXDE is not installed.\n"
    ) from exc

# -------------------------------------------------------------------
# 1. 사용자가 정하는 입력값 모음

# -----------------------------
# 1-1. 형상값
L = 0.15                        # 모터 하우징 길이 [m]
R_OUTER = 0.04                  # 외반지름[m]
R_INNER = 0.025                 # 내반지름[m]
X0 = 0.30                       # Gaussian heat source 중심 위치 (x0=0.3L 의 무차원 위치)
SIGMA_X = 0.10                  # Gaussian heat source 폭 (SIGMA_X=0.1L 의 무차원 폭)
SENSOR_X = (0.25, 0.55, 0.85)   # Sparse temperature sensor 위치

# -----------------------------
# 1-2. 재료 물성치 (알루미늄 합금 물성)
RHO = 2700.0                    # 밀도 [kg/m^3]         [알루미늄 합금]
CP = 900.0                      # 비열 [J/(kg*K)]       [알루미늄 합금]
K = 167.0                       # 열전도율 [W/(m*K)]     [알루미늄 합금]
T_INF = 25.0                    # 주변 공기 온도 [deg C]

# -----------------------------
# 1-3. 무차원화 기준값과 해석 시간
DELTA_T_REF = 50.0              # 온도 무차원화 - 사용자가 지정한 값: 50 deg C [무차원 온도 1은 섭씨 50도에 해당]
I_REF = 5.0                     # 전류 무차원화 - 사용자가 지정한 값: 5.0 A [무차원 전류 1은 5 A에 해당]
T_FINAL = 1200.0                # 전체 해석 시간 [s]

# -----------------------------
# 1-4. Synthetic data용 true parameter
H_TRUE = 40.0                   # 대류 열전달계수 [W/(m^2*K)]
Q0_TRUE = 8.0e4                 # Joule heating coefficient [W/(m^3*A^2)]


# -------------------------------------------------------------------
# 2. 위 입력값으로부터 자동 계산되는 값들 (수정안함)
ALPHA = K / (RHO * CP)                      # 열확산계수 alpha = k / (rho*cp)
AREA = math.pi * (R_OUTER**2 - R_INNER**2)  # 유효 단면적 A = pi*(Ro^2 - Ri^2) (축방향 전도에 쓰이는 면적)
PERIMETER = 2.0 * math.pi * R_OUTER         # 외부 대류 둘레 P = 2*pi*Ro
TAU_FINAL = ALPHA * T_FINAL / L**2          # 실제 시간 t를 무차원 시간 tau로 바꾼 최종 시간

# 무차원 PDE:
#   dtheta/dtau = d2theta/dX2 - beta*theta + S*i(tau)^2*g(X)
# 왼쪽으로 모두 넘기면 residual은:
#   f = theta_tau - theta_XX + beta*theta - S*i(tau)^2*g(X)
# PINN은 이 f가 0에 가까워지도록 학습


# -------------------------------------------------------------------
# 3. 차원/무차원 변환 함수
# -------------------------------------------------------------------
def tau_from_time(t_seconds: float | torch.Tensor) -> float | torch.Tensor:
    """실제 시간 t [s] --> 무차원 시간 tau"""
    return ALPHA * t_seconds / L**2


def time_from_tau(tau: torch.Tensor) -> torch.Tensor:
    """무차원 시간 tau --> 실제 시간 t [s]"""
    return tau * L**2 / ALPHA


def beta_from_physical_h(h_value: float | torch.Tensor) -> float | torch.Tensor:
    """실제 h --> 무차원 beta       [  beta = h*P*L^2/(k*A)  ]"""
    return h_value * PERIMETER * L**2 / (K * AREA)


def s_from_physical_q0(q0_value: float | torch.Tensor) -> float | torch.Tensor:
    """실제 q0 --> 무차원 S         [  S = q0*I_ref^2*L^2/(k*DeltaT_ref)  ]"""
    return q0_value * I_REF**2 * L**2 / (K * DELTA_T_REF)


# Synthetic data에 들어갈 진짜 무차원 파라미터
BETA_TRUE = beta_from_physical_h(H_TRUE)
S_TRUE = s_from_physical_q0(Q0_TRUE)


# -------------------------------------------------------------------
# 4. 기본 함수들
# -------------------------------------------------------------------
def gaussian_heat_source(x: torch.Tensor) -> torch.Tensor:
    """g(X)를 계산
    g(X)는 열이 어디에서 많이 생기는지 알려주는 함수
    X = 0.3 근처:
        winding 근처라서 Joule heating이 큼
    X = 0.85 근처:
        heat source에서 멀어서 Joule heating이 거의 없음 """
    return torch.exp(-((x - X0) ** 2) / (2.0 * SIGMA_X**2))     # Gaussian 함수


def current_profile(tau: torch.Tensor) -> torch.Tensor:
    """무차원 전류 i(tau)를 만듦
    사용자가 지정한 current profile:
        두 번 켜고 끄는 duty cycle
    현재 구현:
        0~300 s      on
        300~600 s    off
        600~900 s    on
        900~1200 s   off
    on 구간:
        Joule heating이 켜짐
    off 구간:
        I(t) = 0 이라서 Joule heating이 사라지고 냉각만 남음
    이 on/off 구조가 있어야 q0와 h를 분리해서 추정하기 쉬워짐 """
    t_seconds = time_from_tau(tau)
    first_on = (t_seconds >= 0.0) & (t_seconds < 300.0)
    second_on = (t_seconds >= 600.0) & (t_seconds < 900.0)
    is_on = first_on | second_on
    return torch.where(is_on, torch.ones_like(tau), torch.zeros_like(tau))


def physical_h_from_beta(beta: torch.Tensor) -> torch.Tensor:
    """무차원 beta를 실제 h [W/(m^2*K)]로 바꿈
    beta = h*P*L^2/(k*A)    -->    h = beta*k*A/(P*L^2) """
    return beta * K * AREA / (PERIMETER * L**2)


def physical_q0_from_s(s_value: torch.Tensor) -> torch.Tensor:
    """무차원 S를 실제 q0로 바꿈
    S = q0*I_ref^2*L^2/(k*DeltaT_ref)   -->   q0 = S*k*DeltaT_ref/(I_ref^2*L^2) """
    return s_value * K * DELTA_T_REF / (I_REF**2 * L**2)

def synthetic_theta_no_conduction(x: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
    """비교용 local ODE synthetic data 함수

    주의:
        이것은 full PDE reference solver가 아님
        축방향 전도항 theta_XX가 빠져 있음

    하는 일:
        각 센서 위치에서 local heating/cooling ODE만 풀어서
        전도항 없는 비교용 synthetic data를 만듦

    사용한 단순 모델:
        dtheta/dtau = -beta_true*theta + S_true*i(tau)^2*g(X)

    현재 main 학습 데이터:
        solve_synthetic_theta_finite_difference()에서 만드는
        full PDE finite difference synthetic data를 사용함
    """

    beta = torch.as_tensor(BETA_TRUE, dtype=x.dtype, device=x.device)
    s_value = torch.as_tensor(S_TRUE, dtype=x.dtype, device=x.device)
    source = s_value * gaussian_heat_source(x)

    tau_300 = torch.as_tensor(tau_from_time(300.0), dtype=x.dtype, device=x.device)
    tau_600 = torch.as_tensor(tau_from_time(600.0), dtype=x.dtype, device=x.device)
    tau_900 = torch.as_tensor(tau_from_time(900.0), dtype=x.dtype, device=x.device)

    theta = torch.zeros_like(tau)

    # 1구간: 0~300 s, 전류 on
    mask_1 = tau < tau_300
    theta_1 = (source / beta) * (1.0 - torch.exp(-beta * tau))
    theta = torch.where(mask_1, theta_1, theta)

    # 300초 끝의 온도
    theta_300 = (source / beta) * (1.0 - torch.exp(-beta * tau_300))

    # 2구간: 300~600 s, 전류 off
    mask_2 = (tau >= tau_300) & (tau < tau_600)
    theta_2 = theta_300 * torch.exp(-beta * (tau - tau_300))
    theta = torch.where(mask_2, theta_2, theta)

    # 600초 시작 온도
    theta_600 = theta_300 * torch.exp(-beta * (tau_600 - tau_300))

    # 3구간: 600~900 s, 전류 on
    mask_3 = (tau >= tau_600) & (tau < tau_900)
    tau_since_600 = tau - tau_600
    theta_3 = (
        theta_600 * torch.exp(-beta * tau_since_600)
        + (source / beta) * (1.0 - torch.exp(-beta * tau_since_600))
    )
    theta = torch.where(mask_3, theta_3, theta)

    # 900초 끝의 온도
    tau_since_600_end = tau_900 - tau_600
    theta_900 = (
        theta_600 * torch.exp(-beta * tau_since_600_end)
        + (source / beta) * (1.0 - torch.exp(-beta * tau_since_600_end))
    )

    # 4구간: 900~1200 s, 전류 off
    mask_4 = tau >= tau_900
    theta_4 = theta_900 * torch.exp(-beta * (tau - tau_900))
    theta = torch.where(mask_4, theta_4, theta)

    return theta


def solve_synthetic_theta_finite_difference(n_time: int = 41) -> tuple[torch.Tensor, torch.Tensor]:
    """Full PDE를 finite difference로 풀어서 sensor data를 만듦.

    반환값:
        xtau:
            DeepXDE PointSetBC에 넣을 sensor 위치/시간 쌍
        theta_sensor:
            각 sensor 위치/시간에서의 synthetic theta 값

    푸는 식:
        theta_tau = theta_XX - BETA_TRUE*theta + S_TRUE*i(tau)^2*g(X)
    """

    nx = 101
    x = torch.linspace(0.0, 1.0, nx, dtype=torch.float32)
    dx = x[1] - x[0]
    dtau = 0.4 * dx**2

    sensor_tau = torch.linspace(0.0, TAU_FINAL, n_time, dtype=torch.float32)
    sensor_indices = torch.tensor([round(xs * (nx - 1)) for xs in SENSOR_X], dtype=torch.long)

    theta = torch.zeros(nx, dtype=torch.float32)
    theta_history = torch.zeros(n_time, len(SENSOR_X), dtype=torch.float32)

    g_x = gaussian_heat_source(x.reshape(-1, 1)).reshape(-1)

    save_idx = 0
    n_steps = math.ceil(TAU_FINAL / dtau)

    for step in range(n_steps + 1):
        tau_now = step * dtau

        while save_idx < n_time and tau_now >= sensor_tau[save_idx]:
            theta_history[save_idx] = theta[sensor_indices]
            save_idx += 1

        if tau_now >= TAU_FINAL:
            break

        theta_xx = torch.zeros_like(theta)
        theta_xx[1:-1] = (theta[2:] - 2.0 * theta[1:-1] + theta[:-2]) / dx**2
        theta_xx[0] = 2.0 * (theta[1] - theta[0]) / dx**2
        theta_xx[-1] = 2.0 * (theta[-2] - theta[-1]) / dx**2

        tau_tensor = torch.tensor([[tau_now]], dtype=torch.float32)
        i_tau = current_profile(tau_tensor).reshape(())
        source = S_TRUE * (i_tau**2) * g_x

        step_dtau = min(dtau, TAU_FINAL - tau_now)
        theta = theta + step_dtau * (theta_xx - BETA_TRUE * theta + source)

    sensor_x = torch.tensor(SENSOR_X, dtype=torch.float32).reshape(-1, 1)
    tau_line = sensor_tau.reshape(1, -1)

    x_grid = sensor_x.repeat(1, n_time).reshape(-1, 1)
    tau_grid = tau_line.repeat(len(SENSOR_X), 1).reshape(-1, 1)

    xtau = torch.cat([x_grid, tau_grid], dim=1)
    theta_sensor = theta_history.T.reshape(-1, 1)

    return xtau, theta_sensor



# -------------------------------------------------------------------
# 5. inverse parameter 양수 제약
# -------------------------------------------------------------------
def positive_parameter(raw_value: torch.Tensor) -> torch.Tensor:
    """raw parameter를 항상 양수인 물리 파라미터로 바꿈.

    beta, S, h, q0는 물리적으로 음수가 되면 이상하다.
    그래서 raw_beta, raw_s를 직접 쓰지 않고 softplus를 통과시킨다.
    """

    # softplus(x) = log(1 + exp(x))
    # 어떤 x가 들어와도 출력은 항상 양수
    return F.softplus(raw_value)


# -------------------------------------------------------------------
# 6. DeepXDE용 synthetic sensor data 생성
# -------------------------------------------------------------------
def make_synthetic_sensor_data_numpy(n_time: int = 41) -> tuple[np.ndarray, np.ndarray]:
    """DeepXDE PointSetBC에 넣을 full PDE sensor data를 만듦.

    solve_synthetic_theta_finite_difference()는 torch.Tensor를 반환한다.
    하지만 DeepXDE의 PointSetBC는 sensor point와 value를 NumPy 배열로 받는다.
    따라서 마지막에 torch.Tensor를 NumPy 배열로 변환한다.
    """

    # full PDE finite difference solver가 sensor 위치/시간과 theta 값을 함께 만든다.
    xtau, theta_sensor = solve_synthetic_theta_finite_difference(n_time)
    
    # Mac MPS를 쓰는 경우 tensor가 mps:0 장치에 올라갈 수 있다.
    # NumPy는 CPU memory만 직접 볼 수 있으므로 cpu()로 옮긴 뒤 numpy()를 호출한다.
    return (
        xtau.detach().cpu().numpy().astype("float32"),
        theta_sensor.detach().cpu().numpy().astype("float32"),
    )


# -------------------------------------------------------------------
# 7. command line option
# -------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    """터미널에서 학습 설정을 바꿀 수 있게 argument parser를 만듦."""

    parser = argparse.ArgumentParser(description="Train Phase 1 inverse PINN with DeepXDE.")

    # DeepXDE에서는 보통 epoch 대신 iterations라는 이름을 쓴다.
    # 예: python main_deepxde.py --iterations 10000
    parser.add_argument("--iterations", type=int, default=5000)

    # Adam optimizer learning rate
    parser.add_argument("--lr", type=float, default=1.0e-3)

    # 신경망 구조:
    # 기본값 depth=3, width=32이면 [2, 32, 32, 32, 1]
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--width", type=int, default=32)

    # DeepXDE가 뽑을 학습점 개수
    parser.add_argument("--num-domain", type=int, default=2560)
    parser.add_argument("--num-boundary", type=int, default=256)
    parser.add_argument("--num-initial", type=int, default=128)
    parser.add_argument("--num-test", type=int, default=1000)

    # sensor 하나당 시간점 개수
    # 기본값 41이면 3 sensors * 41 = 123 points
    parser.add_argument("--sensor-time-points", type=int, default=41)

    # loss 출력 간격과 random seed
    parser.add_argument("--display-every", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)

    # Adam 학습 후 L-BFGS로 한 번 더 다듬고 싶을 때 사용
    parser.add_argument("--lbfgs", action="store_true", help="Run an L-BFGS polish after Adam.")
    return parser


# -------------------------------------------------------------------
# 8. DeepXDE model 구성
# -------------------------------------------------------------------
def build_deepxde_model(args: argparse.Namespace) -> tuple[dde.Model, torch.Tensor, torch.Tensor]:
    """DeepXDE에 필요한 geometry, PDE, 조건, network를 한 번에 구성."""

    # beta와 S는 inverse problem에서 찾아야 하는 미지수다.
    # 여기서는 raw 값을 학습하고, 실제 PDE에는 softplus(raw)를 넣는다.
    raw_beta = dde.Variable(0.0)
    raw_s = dde.Variable(0.0)

    def pde(xtau: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        """무차원 열전달 PDE residual을 계산.

        제안서의 무차원 PDE:
            theta_tau = theta_XX - beta*theta + S*i(tau)^2*g(X)

        왼쪽으로 모두 넘긴 residual:
            f = theta_tau - theta_XX + beta*theta - S*i(tau)^2*g(X)
        """

        # DeepXDE 자동미분:
        #   xtau[:, 0] = X
        #   xtau[:, 1] = tau
        # 따라서 j=1은 tau 미분, j=0은 X 미분이다.
        theta_tau = dde.grad.jacobian(theta, xtau, i=0, j=1)
        theta_xx = dde.grad.hessian(theta, xtau, component=0, i=0, j=0)

        x = xtau[:, 0:1]
        tau = xtau[:, 1:2]

        # trainable raw parameter를 양수 parameter로 변환
        beta = positive_parameter(raw_beta)
        s_value = positive_parameter(raw_s)

        # 위에서 직접 정의한 current profile, Gaussian heat source 사용
        i_tau = current_profile(tau)
        g_x = gaussian_heat_source(x)

        return theta_tau - theta_xx + beta * theta - s_value * (i_tau**2) * g_x

    def zero_numpy(x: np.ndarray) -> np.ndarray:
        """초기조건 theta(X, 0) = 0에 사용할 zero 함수."""

        return np.zeros((len(x), 1), dtype=np.float32)

    def on_initial(_: np.ndarray, is_initial: bool) -> bool:
        """DeepXDE가 준 point가 initial boundary인지 알려주는 filter."""

        return is_initial

    def on_left_boundary(x: np.ndarray, is_boundary: bool) -> bool:
        """X=0 왼쪽 경계만 고르는 filter."""

        return is_boundary and np.isclose(x[0], 0.0)

    def on_right_boundary(x: np.ndarray, is_boundary: bool) -> bool:
        """X=1 오른쪽 경계만 고르는 filter."""

        return is_boundary and np.isclose(x[0], 1.0)

    def adiabatic_bc(inputs: torch.Tensor, outputs: torch.Tensor, _: np.ndarray) -> torch.Tensor:
        """단열 경계조건 dtheta/dX = 0.

        OperatorBC는 이 함수의 반환값이 0에 가까워지도록 loss를 만든다.
        여기서 j=0은 입력 (X, tau) 중 X에 대한 미분이다.
        """

        return dde.grad.jacobian(outputs, inputs, i=0, j=0)

    # 해석 영역:
    #   0 <= X <= 1
    #   0 <= tau <= TAU_FINAL
    geom = dde.geometry.Interval(0.0, 1.0)
    time_domain = dde.geometry.TimeDomain(0.0, TAU_FINAL)
    geomtime = dde.geometry.GeometryXTime(geom, time_domain)

    # 초기조건:
    #   theta(X, 0) = 0
    ic = dde.icbc.IC(geomtime, zero_numpy, on_initial)

    # 양끝 단열 경계조건:
    #   dtheta/dX(0, tau) = 0
    #   dtheta/dX(1, tau) = 0
    left_bc = dde.icbc.OperatorBC(geomtime, adiabatic_bc, on_left_boundary)
    right_bc = dde.icbc.OperatorBC(geomtime, adiabatic_bc, on_right_boundary)

    # Sparse sensor data 조건:
    #   sensor point에서 network 예측 theta가 synthetic theta와 가까워지도록 함
    sensor_xtau, sensor_theta = make_synthetic_sensor_data_numpy(args.sensor_time_points)
    sensor_bc = dde.icbc.PointSetBC(sensor_xtau, sensor_theta, component=0)

    # DeepXDE data 객체:
    # PDE, IC, BC, sensor data 조건을 모두 묶어서 학습 문제를 정의한다.
    data = dde.data.TimePDE(
        geomtime,
        pde,
        [ic, left_bc, right_bc, sensor_bc],
        num_domain=args.num_domain,
        num_boundary=args.num_boundary,
        num_initial=args.num_initial,
        num_test=args.num_test,
        train_distribution="Hammersley",
    )

    # DeepXDE neural network:
    # 기본 layer_sizes = [2, 32, 32, 32, 1]
    # 입력 2개: X, tau
    # 출력 1개: theta
    layer_sizes = [2] + [args.width] * args.depth + [1]
    net = dde.nn.FNN(layer_sizes, "tanh", "Glorot normal")

    # DeepXDE model = data + neural network
    model = dde.Model(data, net)
    return model, raw_beta, raw_s


# -------------------------------------------------------------------
# 9. 출력 helper
# -------------------------------------------------------------------
def print_problem_summary(sensor_points: int) -> None:
    """학습 시작 전에 문제 설정을 출력."""

    print(f"Python      : {sys.version.split()[0]}")
    print(f"Platform    : {platform.machine()}")
    print(f"PyTorch     : {torch.__version__}")
    print(f"DeepXDE     : {dde.__version__}")
    print()
    print("Phase 1 DeepXDE inverse PINN")
    print(f"L           : {L:.3f} m")
    print(f"T_final     : {T_FINAL:.1f} s")
    print(f"tau_final   : {TAU_FINAL:.6f}")
    print(f"A           : {AREA:.6e} m^2")
    print(f"P           : {PERIMETER:.6e} m")
    print(f"I_ref       : {I_REF:.1f} A")
    print(f"DeltaT_ref  : {DELTA_T_REF:.1f} deg C")
    print(f"sensor data : {sensor_points} points")
    print()
    print("Synthetic-data true parameters")
    print(f"h_true      : {H_TRUE:.6f} W/(m^2*K)")
    print(f"q0_true     : {Q0_TRUE:.6e} W/(m^3*A^2)")
    print(f"beta_true   : {BETA_TRUE:.6f}")
    print(f"S_true      : {S_TRUE:.6f}")
    print()


def print_learned_parameters(raw_beta: torch.Tensor, raw_s: torch.Tensor) -> None:
    """학습된 beta, S를 실제 h, q0로 변환해서 출력."""

    # 출력할 때는 gradient가 필요 없으므로 no_grad 사용
    with torch.no_grad():
        beta = positive_parameter(raw_beta).detach()
        s_value = positive_parameter(raw_s).detach()

        # 위에서 직접 정의한 변환식을 사용
        h_value = physical_h_from_beta(beta)
        q0_value = physical_q0_from_s(s_value)

    print()
    print("Learned inverse parameters")
    print(f"beta        : {beta.item():.6f}    true: {BETA_TRUE:.6f}")
    print(f"S           : {s_value.item():.6f}    true: {S_TRUE:.6f}")
    print(f"h           : {h_value.item():.6f} W/(m^2*K)")
    print(f"q0          : {q0_value.item():.6e} W/(m^3*A^2)")


# -------------------------------------------------------------------
# 10. beta, S 학습 과정 출력 callback
# -------------------------------------------------------------------
class PositiveParameterPrinter(dde.callbacks.Callback):
    def __init__(self, raw_beta, raw_s, period):
        super().__init__()
        self.raw_beta = raw_beta
        self.raw_s = raw_s
        self.period = period

    def on_epoch_end(self):
        step = self.model.train_state.step
        if step % self.period != 0:
            return

        with torch.no_grad():
            beta = positive_parameter(self.raw_beta).item()
            s_value = positive_parameter(self.raw_s).item()

        print(f"step {step}: beta={beta:.6f}, S={s_value:.6f}")

# -------------------------------------------------------------------
# 11. 실행부
# -------------------------------------------------------------------
def main() -> None:
    """DeepXDE inverse PINN 학습 실행."""

    # 터미널 인자 읽기
    args = build_parser().parse_args()

    # 재현성을 위한 random seed
    dde.config.set_random_seed(args.seed)

    # sensor data 개수 출력용
    sensor_points = len(SENSOR_X) * args.sensor_time_points
    print_problem_summary(sensor_points)

    # DeepXDE model 구성
    model, raw_beta, raw_s = build_deepxde_model(args)

    # Adam optimizer로 먼저 학습
    # external_trainable_variables를 넣어야 신경망 weight뿐 아니라
    # raw_beta, raw_s도 같이 업데이트된다.
    model.compile(
        "adam",
        lr=args.lr,
        loss_weights=[1.0, 10.0, 10.0, 10.0, 100.0],
        external_trainable_variables=[raw_beta, raw_s],
    )

    param_printer = PositiveParameterPrinter(raw_beta, raw_s, args.display_every)

    model.train(
    iterations=args.iterations,
    display_every=args.display_every,
    callbacks=[param_printer],
)


    # 선택 사항:
    # Adam 이후 L-BFGS로 loss를 한 번 더 줄이고 싶을 때 --lbfgs 사용
    if args.lbfgs:
        model.compile("L-BFGS", external_trainable_variables=[raw_beta, raw_s])
        model.train()

    # 학습된 inverse parameter 출력
    print_learned_parameters(raw_beta, raw_s)


if __name__ == "__main__":
    main()
