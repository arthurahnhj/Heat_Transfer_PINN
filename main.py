"""평판 대류 PINN의 가장 기본 뼈대.

이 파일은 아직 완성된 연구 코드가 아닙니다.
목표는 PINN이 어떤 순서로 생겼는지 아주 쉽게 보는 것입니다.

큰 그림:
1. 신경망에게 위치 (x, y)를 준다.
2. 신경망은 그 위치의 무차원 온도 theta를 맞혀 본다.
3. 그 theta가 열전달 방정식을 잘 만족하는지 검사한다.
4. 벽면 온도 조건도 잘 맞는지 검사한다.
5. 나중에는 이 오차들을 줄이도록 학습시킨다.
"""

from __future__ import annotations

# platform, sys는 지금 내 컴퓨터와 파이썬 정보를 출력할 때 씁니다.
import platform
import sys

# numpy는 일반 숫자 계산에 씁니다.
# 여기서는 Re, Nu 같은 열전달 값을 계산할 때 사용합니다.
import numpy as np

# torch는 PINN의 핵심입니다.
# 신경망을 만들고, 자동미분으로 dtheta/dx 같은 미분값을 구합니다.
import torch
from torch import nn


# -------------------------------------------------------------------
# 1. 물리 문제 설정
# -------------------------------------------------------------------
# 우리는 "뜨거운 평판 위로 공기가 지나가는 상황"을 다룹니다.
#
# 공기가 왼쪽에서 오른쪽으로 흐른다고 생각하면:
#
#   공기 흐름 방향 x ->
#   --------------------------------  뜨거운 평판
#
# y는 평판에서 위쪽으로 떨어진 거리입니다.

# 평판 표면 온도입니다.
# 단위는 섭씨(deg C)입니다.
T_WALL = 75.0

# 멀리 떨어진 공기 온도입니다.
# 평판에서 충분히 멀면 공기는 이 온도를 가집니다.
T_INF = 25.0

# 공기가 평판 위로 흐르는 속도입니다.
# 단위는 m/s입니다.
U_INF = 2.0

# 평판 길이입니다.
# 단위는 m입니다.
L = 0.25

# x = 0 바로 앞쪽은 수식이 불안정해질 수 있습니다.
# 그래서 x = 0에서 바로 시작하지 않고, 조금 뒤에서 시작합니다.
X_MIN = 0.02 * L

# 계산 영역의 높이입니다.
# y = 0은 벽면이고, y = Y_MAX는 자유류 쪽 경계라고 생각합니다.
Y_MAX = 0.02

# 공기의 동점성계수 nu입니다.
# 쉽게 말하면 "공기가 얼마나 끈적하게 움직이는가"와 관련된 값입니다.
NU = 1.568e-5

# Prandtl number입니다.
# 운동량 경계층과 열 경계층의 상대적인 두께를 알려주는 무차원 수입니다.
PR = 0.71

# 열확산계수 alpha입니다.
# 열이 공기 안에서 얼마나 잘 퍼지는지 나타냅니다.
# Pr = nu / alpha 이므로 alpha = nu / Pr 입니다.
ALPHA = NU / PR


# -------------------------------------------------------------------
# 2. 신경망 만들기
# -------------------------------------------------------------------
class ThermalPINN(nn.Module):
    """위치 (x, y)를 넣으면 온도 theta를 예측하는 작은 신경망."""

    def __init__(self) -> None:
        # nn.Module을 상속받는 클래스에서는 이 줄이 필요합니다.
        # "부모 클래스의 기본 준비를 먼저 해라"라는 뜻입니다.
        super().__init__()

        # nn.Sequential은 여러 층을 순서대로 쌓는 도구입니다.
        #
        # 입력:
        #   x, y 두 개 숫자
        #
        # 출력:
        #   theta 한 개 숫자
        #
        # theta는 무차원 온도입니다.
        #
        #   theta = 1  -> 벽면 온도 T_WALL
        #   theta = 0  -> 자유류 온도 T_INF
        self.net = nn.Sequential(
            # 첫 번째 Linear 층입니다.
            # 숫자 2개(x, y)를 받아서 숫자 32개로 바꿉니다.
            nn.Linear(2, 32),

            # Tanh는 신경망이 부드러운 곡선을 만들 수 있게 해줍니다.
            # PINN은 미분을 많이 쓰므로 부드러운 함수가 좋습니다.
            nn.Tanh(),

            # 두 번째 Linear 층입니다.
            # 숫자 32개를 다시 숫자 32개로 바꿉니다.
            nn.Linear(32, 32),

            # 다시 Tanh를 넣어 신경망이 더 복잡한 모양을 배울 수 있게 합니다.
            nn.Tanh(),

            # 마지막 Linear 층입니다.
            # 숫자 32개를 최종 출력 theta 1개로 바꿉니다.
            nn.Linear(32, 1),
        )

    def forward(self, xy: torch.Tensor) -> torch.Tensor:
        """신경망에 (x, y)를 넣고 theta를 받는 함수."""

        # PyTorch에서는 model(xy)를 부르면 내부적으로 forward가 실행됩니다.
        return self.net(xy)


# -------------------------------------------------------------------
# 3. 계산할 점들 뽑기
# -------------------------------------------------------------------
# PINN은 격자 전체를 반드시 만들 필요가 없습니다.
# 대신 계산 영역 안에서 점들을 여러 개 랜덤으로 뽑고,
# 그 점들에서 물리 방정식이 맞는지 검사합니다.


def sample_interior_points(n_points: int, device: torch.device) -> torch.Tensor:
    """평판 위 공기 영역 내부의 점들을 랜덤으로 뽑습니다."""

    # torch.rand(n_points, 2)는 0과 1 사이의 랜덤 숫자를 만듭니다.
    #
    # 모양은 다음과 같습니다.
    #
    #   [[x 후보, y 후보],
    #    [x 후보, y 후보],
    #    ...]
    #
    # 아직 실제 길이 단위가 아니라 0~1 사이 숫자입니다.
    xy = torch.rand(n_points, 2, device=device)

    # 첫 번째 열은 x 좌표로 씁니다.
    # 0~1 사이 숫자를 X_MIN~L 사이 숫자로 바꿉니다.
    xy[:, 0] = X_MIN + (L - X_MIN) * xy[:, 0]

    # 두 번째 열은 y 좌표로 씁니다.
    # 0~1 사이 숫자를 0~Y_MAX 사이 숫자로 바꿉니다.
    xy[:, 1] = Y_MAX * xy[:, 1]

    return xy


def sample_wall_points(n_points: int, device: torch.device) -> torch.Tensor:
    """뜨거운 벽면 y = 0 위의 점들을 랜덤으로 뽑습니다."""

    # x는 X_MIN부터 L까지 랜덤으로 뽑습니다.
    x = X_MIN + (L - X_MIN) * torch.rand(n_points, 1, device=device)

    # 벽면은 y = 0이므로 y는 전부 0입니다.
    y = torch.zeros_like(x)

    # x와 y를 옆으로 붙여서 (x, y) 점으로 만듭니다.
    return torch.cat([x, y], dim=1)


def sample_freestream_points(n_points: int, device: torch.device) -> torch.Tensor:
    """평판에서 멀리 떨어진 위쪽 경계 y = Y_MAX의 점들을 뽑습니다."""

    # x는 X_MIN부터 L까지 랜덤으로 뽑습니다.
    x = X_MIN + (L - X_MIN) * torch.rand(n_points, 1, device=device)

    # 위쪽 경계는 y = Y_MAX입니다.
    y = Y_MAX * torch.ones_like(x)

    # x와 y를 붙여서 (x, y) 점으로 만듭니다.
    return torch.cat([x, y], dim=1)


# -------------------------------------------------------------------
# 4. PINN에서 가장 중요한 부분: 물리 방정식 오차
# -------------------------------------------------------------------
def energy_residual(model: ThermalPINN, xy: torch.Tensor) -> torch.Tensor:
    """열전달 에너지 방정식이 얼마나 안 맞는지 계산합니다.

    지금은 가장 쉬운 출발 버전입니다.

    현재 임시 방정식:

        U_inf * dtheta/dx = alpha * d2theta/dy2

    말로 풀면:

        "공기가 오른쪽으로 열을 데려가는 효과"
        =
        "열이 위아래 방향으로 퍼지는 효과"

    나중에 더 정확하게 만들면 다음 식이 됩니다.

        u dtheta/dx + v dtheta/dy = alpha d2theta/dy2

    여기서 u, v는 Blasius 해로부터 얻을 속도장입니다.
    """

    # xy를 복사한 뒤 requires_grad_(True)를 켭니다.
    #
    # 이유:
    #   PINN은 theta를 x, y로 미분해야 합니다.
    #   PyTorch에게 "이 입력값에 대한 미분을 나중에 계산해줘"라고 알려주는 줄입니다.
    xy = xy.clone().detach().requires_grad_(True)

    # 신경망이 각 점에서 theta를 예측합니다.
    theta = model(xy)

    # theta를 x와 y에 대해 한 번 미분합니다.
    #
    # 결과 grad_theta의 모양:
    #   첫 번째 열: dtheta/dx
    #   두 번째 열: dtheta/dy
    grad_theta = torch.autograd.grad(
        theta,
        xy,
        grad_outputs=torch.ones_like(theta),
        create_graph=True,
    )[0]

    # dtheta/dx입니다.
    theta_x = grad_theta[:, 0:1]

    # dtheta/dy입니다.
    theta_y = grad_theta[:, 1:2]

    # dtheta/dy를 다시 y에 대해 미분해서 d2theta/dy2를 구합니다.
    #
    # 즉:
    #   theta_y  = dtheta/dy
    #   theta_yy = d2theta/dy2
    grad_theta_y = torch.autograd.grad(
        theta_y,
        xy,
        grad_outputs=torch.ones_like(theta_y),
        create_graph=True,
    )[0]

    # 두 번째 열이 y에 대한 미분입니다.
    theta_yy = grad_theta_y[:, 1:2]

    # 방정식을 왼쪽 - 오른쪽 형태로 씁니다.
    #
    # 원래 식:
    #   U_inf * theta_x = alpha * theta_yy
    #
    # residual:
    #   U_inf * theta_x - alpha * theta_yy
    #
    # residual이 0에 가까우면 물리 방정식을 잘 만족한다는 뜻입니다.
    return U_INF * theta_x - ALPHA * theta_yy


def boundary_loss(model: ThermalPINN, device: torch.device) -> torch.Tensor:
    """경계조건이 얼마나 안 맞는지 계산합니다.

    경계조건은 문제의 약속입니다.

    벽면 y = 0:
        평판이 뜨거우므로 theta = 1

    위쪽 y = Y_MAX:
        평판에서 멀리 떨어진 공기는 차가우므로 theta = 0
    """

    # 벽면 점들을 뽑습니다.
    wall_xy = sample_wall_points(64, device)

    # 위쪽 자유류 경계 점들을 뽑습니다.
    far_xy = sample_freestream_points(64, device)

    # 벽면에서는 theta가 1이어야 합니다.
    # 예측값 model(wall_xy)가 1에서 멀어질수록 loss가 커집니다.
    wall_loss = torch.mean((model(wall_xy) - 1.0) ** 2)

    # 위쪽 경계에서는 theta가 0이어야 합니다.
    # 예측값 model(far_xy)가 0에서 멀어질수록 loss가 커집니다.
    far_loss = torch.mean(model(far_xy) ** 2)

    # 두 경계조건 오차를 더합니다.
    return wall_loss + far_loss


def select_device() -> torch.device:
    """계산을 어디서 할지 고릅니다."""

    # NVIDIA GPU가 있으면 cuda를 씁니다.
    # 나중에 데스크탑 RTX GPU에서 돌릴 때 여기가 선택됩니다.
    if torch.cuda.is_available():
        return torch.device("cuda")

    # Mac GPU를 쓸 수 있으면 mps를 씁니다.
    # 지금 환경에서는 False일 수 있습니다. 그래도 코드는 준비해둡니다.
    if torch.backends.mps.is_available():
        return torch.device("mps")

    # GPU를 못 쓰면 CPU로 계산합니다.
    # 초반 작은 테스트는 CPU로도 충분합니다.
    return torch.device("cpu")


# -------------------------------------------------------------------
# 5. 실행 확인용 main 함수
# -------------------------------------------------------------------
def main() -> None:
    """아직 학습은 하지 않고, 기본 계산이 되는지만 확인합니다."""

    # 사용할 계산 장치를 고릅니다.
    device = select_device()

    # 랜덤 숫자를 항상 비슷하게 나오게 고정합니다.
    # 이렇게 하면 실행할 때마다 결과가 너무 달라지지 않습니다.
    torch.manual_seed(0)

    # Reynolds number입니다.
    #
    # Re_L이 작으면 층류, 너무 커지면 난류가 될 수 있습니다.
    # 우리는 laminar flat plate만 다룰 것이므로 Re_L < 5e5를 목표로 합니다.
    re_l = U_INF * L / NU

    # 교과서의 평균 Nusselt number correlation입니다.
    #
    # Nu_L = 0.664 * Re_L^(1/2) * Pr^(1/3)
    #
    # 나중에 PINN 결과와 비교할 기준값입니다.
    nu_l_corr = 0.664 * np.sqrt(re_l) * PR ** (1.0 / 3.0)

    # 신경망 모델을 만들고 선택한 장치로 보냅니다.
    model = ThermalPINN().to(device)

    # 계산 영역 내부 점 128개를 뽑습니다.
    interior_xy = sample_interior_points(128, device)

    # 각 점에서 물리 방정식 residual을 계산합니다.
    residual = energy_residual(model, interior_xy)

    # PDE loss입니다.
    # residual이 0에 가까울수록 물리 방정식을 잘 만족합니다.
    pde_loss = torch.mean(residual**2)

    # 경계조건 loss입니다.
    # 벽면 theta = 1, 위쪽 theta = 0을 얼마나 잘 지키는지 봅니다.
    bc_loss = boundary_loss(model, device)

    # 전체 loss입니다.
    # 지금은 아주 단순하게 PDE loss와 BC loss를 그냥 더합니다.
    total_loss = pde_loss + bc_loss

    # 아래 출력들은 "지금 코드가 잘 돌아가는지" 확인하기 위한 정보입니다.
    print(f"Python      : {sys.version.split()[0]}")
    print(f"Platform    : {platform.machine()}")
    print(f"PyTorch     : {torch.__version__}")
    print(f"Device      : {device}")
    print(f"Re_L        : {re_l:.3e}")
    print(f"Nu_L corr   : {nu_l_corr:.3f}")
    print(f"PDE loss    : {pde_loss.item():.3e}")
    print(f"BC loss     : {bc_loss.item():.3e}")
    print(f"Total loss  : {total_loss.item():.3e}")


# 이 파일을 직접 실행했을 때만 main()을 실행합니다.
#
# 예:
#   python main.py
#
# 나중에 다른 파일에서 이 코드를 import할 때는 main()이 자동 실행되지 않습니다.
if __name__ == "__main__":
    main()
