"""Phase 1 제안서 기준 PINN 기본 틀

주제:
    로봇 관절 모터 하우징의 1D 비정상 열전달 inverse PINN

이 파일의 목표:
    아직 최종 학습 코드는 아님
    Phase 1 제안서에 나온 물리 모델을 코드 뼈대로 옮긴 상태

큰 그림:
    1. 신경망에 위치 X와 시간 tau를 넣음
    2. 신경망은 무차원 온도 theta(X, tau)를 예측
    3. 예측한 theta가 열전달 PDE를 만족하는지 검사
    4. 초기조건, 단열 경계조건, 센서 데이터 조건도 검사
    5. 나중에 beta와 S를 학습해서 h와 q0를 역추정
"""

from __future__ import annotations

import math
import platform
import sys

import torch
from torch import nn
from torch.nn import functional as F


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
    """Synthetic sensor data를 먼저 만들기 위한 간단한 온도 함수

    주의:
        이것은 최종 정답 solver가 아님
        아직 1D 전도항 theta_XX를 포함한 full reference solver가 없음

    지금 하는 일:
        각 센서 위치에서 local heating/cooling ODE만 풀어서
        synthetic data의 첫 버전을 만듦

    사용한 단순 모델:
        dtheta/dtau = -beta_true*theta + S_true*i(tau)^2*g(X)

    나중에 업그레이드:
        finite difference solver로 full PDE synthetic data를 만들면 됨
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


def select_device() -> torch.device:
    """계산 장치를 고름

    데스크탑 NVIDIA GPU에서는 cuda,
    Mac에서 MPS가 가능하면 mps,
    아니면 CPU를 사용
    """

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# -------------------------------------------------------------------
# 5. 신경망
# -------------------------------------------------------------------
class MotorHousingPINN(nn.Module):
    """theta(X, tau), beta, S를 함께 다루는 PINN 모델."""

    def __init__(self) -> None:
        super().__init__()

        # 입력은 2개
        #   X   : 무차원 위치
        #   tau : 무차원 시간
        #
        # 출력은 1개
        #   theta : 무차원 온도
        self.net = nn.Sequential(
            nn.Linear(2, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 1),
        )

        # beta와 S는 inverse problem에서 찾아야 할 미지수
        #
        # 바로 beta를 학습시키지 않고 raw_beta를 학습시킨 뒤,
        # softplus(raw_beta)를 beta로 사용
        #
        # 이유:
        #   h, q0, beta, S는 물리적으로 음수가 되면 이상함
        #   softplus를 쓰면 항상 양수로 만들 수 있음
        self.raw_beta = nn.Parameter(torch.tensor(0.0))
        self.raw_s = nn.Parameter(torch.tensor(0.0))

    def forward(self, xtau: torch.Tensor) -> torch.Tensor:
        """신경망에 (X, tau)를 넣어 theta를 예측."""

        return self.net(xtau)

    def beta(self) -> torch.Tensor:
        """항상 양수인 beta를 반환."""

        return F.softplus(self.raw_beta)

    def s_value(self) -> torch.Tensor:
        """항상 양수인 S를 반환."""

        return F.softplus(self.raw_s)


# -------------------------------------------------------------------
# 6. 점 샘플링
# -------------------------------------------------------------------
def sample_collocation_points(n_points: int, device: torch.device) -> torch.Tensor:
    """PDE를 검사할 내부 점들을 뽑음

    X:
        0~1 사이

    tau:
        0~TAU_FINAL 사이
    """

    xtau = torch.rand(n_points, 2, device=device)
    xtau[:, 1] = TAU_FINAL * xtau[:, 1]
    return xtau


def sample_initial_points(n_points: int, device: torch.device) -> torch.Tensor:
    """초기조건을 검사할 점들을 뽑음

    초기조건은 tau = 0에서 theta = 0.
    """

    x = torch.rand(n_points, 1, device=device)
    tau = torch.zeros_like(x)
    return torch.cat([x, tau], dim=1)


def sample_boundary_points(n_points: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """양쪽 끝단 X=0, X=1의 경계 점들을 뽑음

    제안서의 경계조건은 adiabatic

    즉:
        dtheta/dX(0, tau) = 0
        dtheta/dX(1, tau) = 0
    """

    tau = TAU_FINAL * torch.rand(n_points, 1, device=device)
    left = torch.cat([torch.zeros_like(tau), tau], dim=1)
    right = torch.cat([torch.ones_like(tau), tau], dim=1)
    return left, right


def make_synthetic_sensor_data(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """사용자가 지정한 조건으로 synthetic sensor data를 먼저 만듦

    센서 위치:
        X = 0.25, 0.55, 0.85

    시간:
        0~1200 s

    현재 synthetic data 생성 방식:
        전도항을 뺀 local ODE 근사
    """

    n_time = 41
    sensor_x = torch.tensor(SENSOR_X, dtype=torch.float32, device=device).reshape(-1, 1)
    tau_line = torch.linspace(0.0, TAU_FINAL, n_time, device=device).reshape(1, -1)

    x_grid = sensor_x.repeat(1, n_time).reshape(-1, 1)
    tau_grid = tau_line.repeat(len(SENSOR_X), 1).reshape(-1, 1)

    xtau = torch.cat([x_grid, tau_grid], dim=1)
    theta_sensor = synthetic_theta_no_conduction(x_grid, tau_grid)
    return xtau, theta_sensor


# -------------------------------------------------------------------
# 7. 자동미분 도우미
# -------------------------------------------------------------------
def gradient(output: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
    """output을 inputs로 미분

    PINN은 신경망 출력 theta를 X와 tau로 미분해야 함
    이때 PyTorch 자동미분을 사용
    """

    return torch.autograd.grad(
        output,
        inputs,
        grad_outputs=torch.ones_like(output),
        create_graph=True,
    )[0]


# -------------------------------------------------------------------
# 8. Loss 함수들
# -------------------------------------------------------------------
def pde_residual(model: MotorHousingPINN, xtau: torch.Tensor) -> torch.Tensor: ##pde_residual이 0에 가까워지면 학습이 잘 되는 것
    """Phase 1 제안서의 무차원 PDE residual을 계산

    무차원 PDE:

        theta_tau = theta_XX - beta*theta + S*i(tau)^2*g(X)

    residual:

        f = theta_tau - theta_XX + beta*theta - S*i(tau)^2*g(X)

    f가 0에 가까우면 물리 방정식을 잘 만족한다는 뜻
    """

    xtau = xtau.clone().detach().requires_grad_(True)

    theta = model(xtau)
    grad_theta = gradient(theta, xtau)

    theta_x = grad_theta[:, 0:1]
    theta_tau = grad_theta[:, 1:2]

    grad_theta_x = gradient(theta_x, xtau)
    theta_xx = grad_theta_x[:, 0:1]

    x = xtau[:, 0:1]
    tau = xtau[:, 1:2]

    beta = model.beta()
    s_value = model.s_value()
    i_tau = current_profile(tau)
    g_x = gaussian_heat_source(x)

    return theta_tau - theta_xx + beta * theta - s_value * (i_tau**2) * g_x


def pde_loss(model: MotorHousingPINN, device: torch.device) -> torch.Tensor:
    """PDE residual의 제곱 평균."""

    xtau = sample_collocation_points(256, device)
    residual = pde_residual(model, xtau)
    return torch.mean(residual**2)


def initial_condition_loss(model: MotorHousingPINN, device: torch.device) -> torch.Tensor:
    """초기조건 theta(X, 0) = 0 loss."""

    xtau = sample_initial_points(128, device)
    theta = model(xtau)
    return torch.mean(theta**2)


def boundary_condition_loss(model: MotorHousingPINN, device: torch.device) -> torch.Tensor:
    """양끝 단열조건 dtheta/dX = 0 loss."""

    left, right = sample_boundary_points(128, device)
    left = left.clone().detach().requires_grad_(True)
    right = right.clone().detach().requires_grad_(True)

    theta_left = model(left)
    theta_right = model(right)

    theta_x_left = gradient(theta_left, left)[:, 0:1]
    theta_x_right = gradient(theta_right, right)[:, 0:1]

    left_loss = torch.mean(theta_x_left**2)
    right_loss = torch.mean(theta_x_right**2)
    return left_loss + right_loss


def sensor_data_loss(
    model: MotorHousingPINN,
    sensor_xtau: torch.Tensor,
    sensor_theta: torch.Tensor,
) -> torch.Tensor:
    """센서 데이터와 신경망 예측값의 차이를 계산."""

    theta_pred = model(sensor_xtau)
    return torch.mean((theta_pred - sensor_theta) ** 2)


# -------------------------------------------------------------------
# 9. 실행 확인
# -------------------------------------------------------------------
def main() -> None:
    """아직 오래 학습하지 않고, Phase 1 구조가 돌아가는지만 확인."""

    device = select_device()
    torch.manual_seed(0)

    model = MotorHousingPINN().to(device)

    loss_pde = pde_loss(model, device)
    loss_ic = initial_condition_loss(model, device)
    loss_bc = boundary_condition_loss(model, device)

    sensor_xtau, sensor_theta = make_synthetic_sensor_data(device)
    loss_data = sensor_data_loss(model, sensor_xtau, sensor_theta)

    total_loss = loss_pde + loss_ic + loss_bc + loss_data

    beta_guess = model.beta().detach()
    s_guess = model.s_value().detach()
    h_guess = physical_h_from_beta(beta_guess)
    q0_guess = physical_q0_from_s(s_guess)

    print(f"Python      : {sys.version.split()[0]}")
    print(f"Platform    : {platform.machine()}")
    print(f"PyTorch     : {torch.__version__}")
    print(f"Device      : {device}")
    print()
    print("Phase 1 model")
    print(f"L           : {L:.3f} m")
    print(f"T_final     : {T_FINAL:.1f} s")
    print(f"tau_final   : {TAU_FINAL:.6f}")
    print(f"A           : {AREA:.6e} m^2")
    print(f"P           : {PERIMETER:.6e} m")
    print(f"alpha       : {ALPHA:.6e} m^2/s")
    print(f"I_ref       : {I_REF:.1f} A")
    print(f"DeltaT_ref  : {DELTA_T_REF:.1f} deg C")
    print()
    print("Synthetic-data true parameters")
    print(f"h_true      : {H_TRUE:.6f} W/(m^2*K)")
    print(f"q0_true     : {Q0_TRUE:.6e} W/(m^3*A^2)")
    print(f"beta_true   : {BETA_TRUE:.6f}")
    print(f"S_true      : {S_TRUE:.6f}")
    print(f"sensor data : {sensor_xtau.shape[0]} points")
    print()
    print("Trainable inverse parameters")
    print(f"beta guess  : {beta_guess.item():.6f}")
    print(f"S guess     : {s_guess.item():.6f}")
    print(f"h guess     : {h_guess.item():.6f} W/(m^2*K)")
    print(f"q0 guess    : {q0_guess.item():.6f}")
    print()
    print("Smoke-test losses")
    print(f"PDE loss    : {loss_pde.item():.3e}")
    print(f"IC loss     : {loss_ic.item():.3e}")
    print(f"BC loss     : {loss_bc.item():.3e}")
    print(f"Data loss   : {loss_data.item():.3e}")
    print(f"Total loss  : {total_loss.item():.3e}")


if __name__ == "__main__":
    main()
