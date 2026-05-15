# Baseline Chat Memory

이 파일은 앞으로 이 프로젝트의 기준 기억용이다.

최신 기준 문서:

`Phase1/Phase1_Proposal_안형준.pdf`

주의:
- 예전 flat plate convection 주제는 폐기한다.
- 이제 프로젝트는 로봇 관절 모터 하우징의 inverse PINN이다.

## 최종 프로젝트 방향

주제:

`Inverse PINN for Transient Thermal Parameter Estimation in a Robot Joint Motor Housing`

한국어로 정리하면:

`로봇 관절 모터 하우징의 비정상 열전달 해석 및 미지 열전달 파라미터 역추정 PINN`

핵심 목표:
- 모터 하우징의 1D 비정상 온도장 `T(x,t)` 예측
- sparse temperature sensor data를 이용한 inverse parameter estimation
- 미지 파라미터:
  - convective heat transfer coefficient `h`
  - Joule heating coefficient `q0`

## 물리 시스템

로봇 관절 모터가 작동하면 winding 및 내부 부품에서 Joule heating이 발생한다.
발생한 열은 모터 하우징을 따라 축방향으로 전도되고, 외부 공기로 대류 방출된다.

과열은 효율 저하, 절연 손상, 로봇 관절 신뢰성 저하로 이어질 수 있으므로,
모터 하우징의 transient thermal behavior를 예측하고 unknown thermal parameter를 추정하는 것이 목표다.

## 단순화 모델

모터 하우징은 축방향 1D 원통형 body로 단순화한다.

온도장은 단면 평균 온도:

```text
T = T(x,t)
```

계산 영역:

```text
0 <= x <= L
```

대표 형상:

```text
L  = 0.15 m
Ro = 0.04 m
Ri = 0.025 m
```

유효 단면적:

```text
A = pi * (Ro^2 - Ri^2)
```

외부 대류 표면 perimeter:

```text
P = 2*pi*Ro
```

내부 표면은 motor internals를 향하므로 ambient air와 직접 대류하지 않는다고 본다.
따라서 대류 냉각은 외부 원통 표면에서만 발생한다.

## 재료 물성

모터 하우징 재료:

```text
Aluminum alloy
```

물성:

```text
rho = 2700 kg/m^3
cp  = 900 J/(kg*K)
k   = 167 W/(m*K)
```

주변 온도:

```text
T_inf = 25 deg C
```

## 지배 방정식

1D transient conduction-convection with internal Joule heating:

```text
rho*cp*dT/dt
= k*d2T/dx2
  - (h*P/A)*(T - T_inf)
  + q0*I(t)^2*g(x)
```

각 항의 의미:
- `rho*cp*dT/dt`: 시간에 따른 열 저장
- `k*d2T/dx2`: 축방향 전도
- `-(h*P/A)*(T - T_inf)`: 외부 표면 대류 냉각
- `q0*I(t)^2*g(x)`: 전류에 의한 Joule heating

`I(t)`는 motor controller에서 측정 가능한 known input signal로 둔다.
current-on 구간에서는 Joule heating이 작동하고,
current-off 구간에서는 `I(t)=0`이므로 냉각 응답이 `h`에 민감해진다.

## Heat Generation Shape

Joule heating의 공간 분포는 winding 근처에 집중된 Gaussian으로 둔다.

```text
g(x) = exp( - (x - x0)^2 / (2*sigma^2) )
```

위치:

```text
x0    = 0.3L  = 0.045 m
sigma = 0.1L  = 0.015 m
```

이 설정은 온도장을 공간적으로 non-uniform하게 만들어서
sparse sensor data 기반 inverse problem이 의미 있게 된다.

## Initial and Boundary Conditions

초기 조건:

```text
T(x,0) = T_inf
```

의미:
- 모터 작동 전 전체 하우징이 주변 공기와 열평형 상태라고 가정한다.

경계 조건:

```text
dT/dx(0,t) = 0
dT/dx(L,t) = 0
```

의미:
- 양쪽 축방향 끝단은 adiabatic
- 축 끝으로 빠져나가는 열유속은 0
- 주요 냉각은 이미 PDE 내부의 distributed convection sink term으로 들어가 있다.

## PINN Residual

신경망 예측:

```text
T_phi(x,t)
```

PDE residual:

```text
f_PDE =
rho*cp*dT_phi/dt
- k*d2T_phi/dx2
+ (h*P/A)*(T_phi - T_inf)
- q0*I(t)^2*g(x)
```

목표:

```text
f_PDE -> 0
```

Loss 구성:

```text
L_total =
w_PDE  * L_PDE
+ w_IC   * L_IC
+ w_BC   * L_BC
+ w_data * L_data
```

각 loss:
- `L_PDE`: collocation points에서 PDE residual 제곱 평균
- `L_IC`: 초기조건 `T(x,0)=T_inf`
- `L_BC`: 양끝 adiabatic 조건
- `L_data`: sparse sensor temperature data와 예측값 차이

## Nondimensional Form

학습 안정성을 위해 SI unit 그대로 쓰기보다 nondimensional form을 쓴다.

정의:

```text
X     = x/L
tau   = alpha*t/L^2
theta = (T - T_inf)/DeltaT_ref
alpha = k/(rho*cp)
i(tau)= I(t)/I_ref
```

무차원 PDE:

```text
dtheta/dtau =
d2theta/dX2
- beta*theta
+ S*i(tau)^2*g(X)
```

무차원 inverse parameters:

```text
beta = h*P*L^2/(k*A)
S    = q0*I_ref^2*L^2/(k*DeltaT_ref)
```

PINN에서는 우선 `beta`, `S`를 trainable parameter로 학습하고,
나중에 물리 파라미터 `h`, `q0`로 변환한다.

## Sensor Placement

Sparse sensor locations:

```text
X1 = 0.25
X2 = 0.55
X3 = 0.85
```

해석:

`S1 at X=0.25`
- Gaussian heat source peak `X0=0.3` 근처
- Joule heating coefficient `S` 또는 `q0` 추정에 민감

`S2 at X=0.55`
- 중간 conduction zone
- 전체 온도장 재구성에 도움
- S1/S3만으로 생길 수 있는 parameter ambiguity를 줄임

`S3 at X=0.85`
- heat source에서 멀리 떨어진 위치
- local heat generation 거의 없음
- cooling response가 커서 `beta` 또는 `h` 추정에 민감

## Current Profile Strategy

전류 입력 `I(t)`는 heating/cooling phase가 구분되도록 설계한다.

current-on:
- `q0*I(t)^2*g(x)`가 작동
- heat source 근처 S1이 `q0` 추정에 유리

current-off:
- `I(t)=0`
- Joule heating term이 사라짐
- temperature decay가 주로 convection parameter `h`에 의해 결정됨
- S3가 `h` 추정에 유리

## 구현 방향

Step-by-step implementation plan:

1. `main.py`를 flat plate skeleton에서 motor housing inverse PINN skeleton으로 변경
2. 물리 상수와 geometry constants 정의
3. nondimensional variables `X`, `tau`, `theta` 기준으로 코드 구성
4. current profile `i(tau)` 함수 작성
5. Gaussian heat source `g(X)` 함수 작성
6. neural network input: `(X, tau)`
7. neural network output: `theta(X,tau)`
8. PDE residual 구현
9. IC/BC loss 구현
10. synthetic sensor data 생성
11. `beta`, `S`를 trainable parameter로 두고 inverse PINN 학습
12. 학습된 `beta`, `S`를 `h`, `q0`로 변환
13. 결과 plot:
    - temperature field
    - sensor data fit
    - parameter convergence
    - predicted vs true temperature

## Development Workflow Memory

Current working assumption:

- Code development, cleanup, documentation, and small smoke tests are done on the MacBook.
- Final or heavy training runs are done on the desktop PC with an RTX 5060 Ti 16GB GPU.
- GitHub is used as the bridge between machines: push work from the MacBook, then pull updates on the desktop before running CUDA training.

GPU/Colab decision:

- Use the local RTX 5060 Ti 16GB desktop environment as the main training environment.
- Use Colab only as a backup, sharing/demo environment, or temporary fallback if the local CUDA setup is unavailable.

Implementation habits:

- Write device-agnostic PyTorch code:

```python
device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
```

- Avoid absolute local paths in project code.
- Keep project paths relative to the repository.
- Keep heavy generated files out of Git.
- Track source code, configuration files, small reference data, final plots, reports, and environment documentation.
- Maintain setup notes so the desktop CUDA environment can be recreated cleanly.
