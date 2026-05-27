# Claude Handoff: MEN310 Heat Transfer PINN Project

Use this file as the starting context if this project is continued in Claude.

Suggested opening prompt for Claude:

> I am working on a MEN310 Heat Transfer final project using a DeepXDE inverse PINN for a 1D transient robot joint motor housing heat-transfer problem. Please read this handoff and continue in Korean unless I ask otherwise. Prioritize practical report/code help, and always distinguish loss-weight tuning from true ablation.

## 1. User and Communication Preference

- User: Hyungjun Ahn / 안형준.
- Course/project: MEN310 Heat Transfer, Spring 2026, final project.
- Preferred language: Korean explanations with English technical terms when useful.
- Preferred style: direct, practical, no unnecessary encouragement.
- Important standing instruction from user: when writing project documents, refer to files in `Codex_read/`.

## 2. Workspace

Project root:

```text
/Users/arthurahn/Desktop/UNIST/2026-1/Heat_Transfer/Final_Project_PINN
```

Important files:

```text
main.py                                      # original PyTorch baseline PINN
main_deepxde.py                             # standalone DeepXDE inverse PINN script
main_deepxde.ipynb                          # current main working notebook
fdm_reference.py                            # separated full-PDE FDM reference solver
fdm_vs_pinn_full_field_comparison.ipynb     # separate FDM vs PINN comparison notebook
Phase2/Phase2_Midterm_template.docx         # official Phase 2 Word template
Phase2/Phase2_Midterm_Draft.docx            # current Word draft
Phase2/Ablation study plan.pdf              # planned ablation structure from user
Codex_read/2026 MEN310 Final Project.pdf    # official project requirements
Codex_read/Phase1_Proposal_안형준.pdf        # Phase 1 proposal
Codex_read/Phase1_prof_feedback.png         # professor feedback
```

## 3. Official Project Requirements

From `Codex_read/2026 MEN310 Final Project.pdf`:

- Final project must use PINNs, not only FDM or data-only ML.
- Project must include at least one nontrivial heat-transfer feature:
  - internal heat generation,
  - transient analysis,
  - combined convection-conduction,
  - inverse parameter estimation.
- DeepXDE is allowed and gives extra credit.
- Phase 2 Midterm Report requires:
  - source code,
  - loss convergence plots,
  - ablation study.

Current decision:

- Source code will be submitted separately as a Colab/notebook.
- Word report should focus on setup, loss analysis, ablation plan/results, troubleshooting, and interpretation.
- Do not paste large source-code blocks into the Word report unless the user explicitly changes this decision.

## 4. Physical Problem

System:

- Robot joint motor housing.
- Simplified as a 1D cylindrical body along axial coordinate.
- Unknown field: temperature `T(x,t)`, nondimensionalized as `theta(X,tau)`.

Geometry and material:

```text
L       = 0.15 m
R_OUTER = 0.04 m
R_INNER = 0.025 m
rho     = 2700 kg/m^3       # aluminum alloy
cp      = 900 J/(kg K)      # aluminum alloy
k       = 167 W/(m K)       # aluminum alloy
T_inf   = 25 deg C
```

Reference scales:

```text
DELTA_T_REF = 50 deg C
I_REF       = 5 A
T_FINAL     = 1200 s
```

True synthetic-data parameters:

```text
H_TRUE  = 40 W/(m^2 K)
Q0_TRUE = 8.0e4 W/(m^3 A^2)
BETA_TRUE ≈ 0.442193
S_TRUE    ≈ 5.389222
```

## 5. Governing Equation

The nondimensional PDE is:

```text
theta_tau = theta_XX - beta theta + S i(tau)^2 g(X)
```

PINN residual form:

```text
f = theta_tau - theta_XX + beta theta - S i(tau)^2 g(X)
```

Boundary and initial conditions:

```text
theta(X, 0) = 0
theta_X(0, tau) = 0
theta_X(1, tau) = 0
```

Important interpretation:

- This is a 1D transient conduction-convection problem with internal Joule heating.
- The axial ends are adiabatic: `theta_X = 0`.
- Convection is not imposed at `X=0` or `X=1`.
- Convection to ambient is modeled as a distributed sink term, `- beta theta`, because the side surface of the motor housing exchanges heat with air along the whole length.

Heat source:

```text
g(X) = Gaussian centered near X = 0.30
```

## 6. Inverse Parameters

The trainable nondimensional parameters are:

```text
beta = h P L^2 / (k A)
S    = q0 I_ref^2 L^2 / (k DeltaT_ref)
```

After training:

```text
h  = beta k A / (P L^2)
q0 = S k DeltaT_ref / (I_ref^2 L^2)
```

`raw_beta` and `raw_s` are not the physical parameters. They are unconstrained trainable variables. The model applies `softplus(raw_value)` to obtain positive `beta` and `S`.

Why `softplus` instead of `abs`:

- `abs(x)` is nondifferentiable at 0 and can create unstable gradient behavior.
- `softplus(x) = log(1 + exp(x))` is smooth and always positive.
- This is useful because `h`, `q0`, `beta`, and `S` must be physically positive.

## 7. Synthetic Data

Earlier idea:

- `synthetic_theta_no_conduction()`
- Local ODE model:

```text
dtheta/dtau = - beta theta + S i(tau)^2 g(X)
```

- This omits `theta_XX`, so it is not a full 1D conduction solution.
- It remains only as a comparison/helper concept.

Current intended data:

- Full PDE finite-difference synthetic data.
- `main_deepxde.py` currently contains `solve_synthetic_theta_finite_difference()`.
- `fdm_reference.py` also provides a separated FDM reference solver and should be treated as the cleaner long-term direction.

Important current-profile issue:

- `main_deepxde.py` currently has an older on/off profile:

```text
0-300 s     on
300-600 s   off
600-900 s   on
900-1200 s  off
```

- `fdm_reference.py` has the ablation-plan profile:

```text
0-100 s     0 A
100-400 s   5 A
400-650 s   0 A
650-850 s   3.5 A
850-1200 s  0 A
```

Claude should check and align the current profile before running final ablation cases.

## 8. DeepXDE Code Status

Main file:

```text
main_deepxde.py
```

Key implementation points:

- Uses PyTorch backend:

```python
os.environ.setdefault("DDE_BACKEND", "pytorch")
```

- Defaults to CPU on Mac because MPS caused Jupyter/Python kernel crashes:

```python
PINN_FORCE_CPU=1
```

- DeepXDE data object:

```text
dde.data.TimePDE
```

- Network:

```text
dde.nn.FNN([2] + [width] * depth + [1], "tanh", "Glorot normal")
```

- Default script parser values in `main_deepxde.py`:

```text
iterations   = 200000
lr           = 1e-3
depth        = 3
width        = 32
num_domain   = 2560
num_boundary = 256
num_initial  = 128
num_test     = 1000
sensor_time_points = 41
display_every = 500
```

Earlier notebook/report runs used lighter settings such as:

```text
iterations   = 100000
depth x width = 3 x 32
num_domain   = 512
num_boundary = 64
num_initial  = 64
num_test     = 200
```

Run example:

```bash
source .venv/bin/activate
python main_deepxde.py --iterations 100000 --num-domain 512 --num-boundary 64 --num-initial 64 --num-test 200
```

Optional L-BFGS:

```bash
python main_deepxde.py --iterations 100000 --lbfgs
```

L-BFGS is optional polishing after Adam. It may reduce loss but can be slower and is not always necessary for the report.

## 9. Mac/MPS Issue

The user's Mac is a MacBook Pro M1 2020.

Observed issue:

- Jupyter kernel crashed with a Metal/MPS stack trace.
- This happened during Torch/MPS memory operations.

Current mitigation:

- Force CPU for DeepXDE/PyTorch.
- Use smaller collocation counts while testing.
- Avoid heavy MPS use in VSCode Jupyter unless carefully tested.

## 10. Loss Weights and Results

Important conceptual correction:

- Changing loss weights is not a strict ablation study.
- It is better described as:

```text
loss-weight balancing
training objective balance sensitivity study
hyperparameter sensitivity
```

True ablation should remove/change measurement information, data noise, input profile, or parameter-identification setup.

The Word report has moved A1-A5 loss-weight cases into:

```text
1. Initial Training & Loss Analysis
```

Approximate A1-A5 summary used in the current report draft:

```text
A1: weights [5, 10, 10, 10, 100]
    train/test loss 1.84e-03 / 5.85e-03
    beta error -0.199%, S error -0.777%
    MAE 2.012e-03, Rel L2 5.463e-03

A2: weights [5, 10, 20, 20, 100]
    train/test loss 1.55e-03 / 1.58e-02
    beta error -0.355%, S error -0.571%
    MAE 5.444e-03, Rel L2 7.320e-03

A3: weights [10, 10, 10, 10, 100]
    train/test loss 1.30e-03 / 1.54e-01
    beta error +0.472%, S error +0.042%
    MAE 2.175e-03, Rel L2 4.410e-03
    Best full-field relative L2 among tested runs, but mismatch remains near switching time.

A4: weights [5, 10, 10, 10, 200]
    train/test loss 1.97e-03 / 8.37e-03
    beta error -0.005%, S error -0.351%
    MAE 1.534e-03, Rel L2 4.604e-03
    Best beta/h accuracy and lowest MAE.

A5: weights [10, 10, 10, 10, 200]
    train/test loss 2.65e-03 / 4.12e-02
    beta error +0.508%, S error -0.442%
    MAE 1.643e-03, Rel L2 5.762e-03
```

A3 vs A4 comparison currently used:

```text
A3:
beta error +0.472%
S error    +0.042%
h error    +0.472%
q0 error   +0.042%
MAE        2.175e-03
RMSE       4.314e-03
Max error  5.120e-02
Rel L2     4.410e-03

A4:
beta error -0.005%
S error    -0.351%
h error    -0.005%
q0 error   -0.351%
MAE        1.534e-03
RMSE       4.504e-03
Max error  8.391e-02
Rel L2     4.604e-03
```

Interpretation:

- A3 is better if full-field relative L2/RMSE/max error is prioritized.
- A4 is better if parameter recovery and MAE are prioritized.
- Report should avoid claiming that loss weights are ablation.

## 11. Required Loss Curve Section

The official template asks:

```text
Show your current Loss_Total, Loss_BC, and Loss_PDE curves.
Analyze whether the loss is decreasing smoothly or showing instability.
```

Current Word draft now has a section:

```text
1.3 Loss convergence curves and stability analysis
```

Need to eventually insert actual plots from the notebook:

- `Loss_Total`
- `Loss_PDE`
- `Loss_BC`

DeepXDE loss order:

```text
[PDE, IC, left BC, right BC, sensor data]
```

So `Loss_BC` should probably be:

```text
Loss_BC = left BC loss + right BC loss
```

or plotted as two curves with a clear note.

Suggested analysis wording:

- Use log scale.
- Smooth decrease: overall decreasing trend after early Adam transient.
- Mild Adam oscillations are acceptable.
- Large repeated spikes, increasing trend, or high plateau indicate instability or poor loss balancing.

## 12. Planned Ablation Study

From `Phase2/Ablation study plan.pdf`, the planned ablation structure is:

```text
0. FDM vs PINN full-field comparison
1. Sensor number ablation
2. Noise robustness test
3. Current profile ablation
4. One-parameter inverse vs two-parameter inverse
5. Unseen current profile validation
```

### 12.1 FDM vs PINN Full-Field Comparison

Report:

- FDM heatmap
- PINN heatmap
- absolute error heatmap
- relative L2 error

Purpose:

- Check whether accurate inverse parameters also produce a physically correct full temperature field.

### 12.2 Sensor Number Ablation

Cases:

```text
1 sensor  : X = 0.25
2 sensors : X = 0.25, 0.85
3 sensors : X = 0.25, 0.55, 0.85
```

Question:

```text
How many sensors are needed to estimate inverse parameters stably?
```

Metrics:

- h error
- q0 error
- relative L2 temperature error

### 12.3 Noise Robustness Test

Cases:

```text
noise = 0 deg C
noise = 0.5 deg C
noise = 1.0 deg C
noise = 2.0 deg C
```

Question:

```text
Is the inverse PINN robust to realistic temperature-sensor noise?
```

### 12.4 Current Profile Ablation

Cases from the user's plan:

```text
Case A: Constant current
0-1200 s: 5 A

Case B: One on-off
0-100 s: 0 A
100-600 s: 5 A
600-1200 s: 0 A

Case C: Two on-off
0-100 s: 0 A
100-400 s: 5 A
400-650 s: 0 A
650-850 s: 3.5 A
850-1200 s: 0 A
```

Why this matters:

- `q0 I(t)^2 g(x)` controls heating when current is ON.
- `h` controls cooling decay when current is OFF.
- ON and OFF segments are needed to distinguish `q0` and `h`.

Question:

```text
How should the current input be designed to identify inverse parameters well?
```

### 12.5 One-Parameter vs Two-Parameter Inverse

Cases:

```text
Case 1: q0 unknown, h known
Case 2: h unknown, q0 known
Case 3: h and q0 both unknown
```

Question:

```text
How much harder is simultaneous two-parameter estimation than estimating only one parameter?
```

### 12.6 Unseen Current Profile Validation

Plan:

- Estimate `h` and `q0` using the training profile.
- Fix the learned parameters.
- Test on a completely different current profile.
- Compare PINN or FDM forward prediction with a reference field.

Question:

```text
Are the learned h and q0 physical parameters that generalize to other motor trajectories,
or did they only fit the training current profile?
```

## 13. Current Phase 2 Word Report

Current file:

```text
Phase2/Phase2_Midterm_Draft.docx
```

Current structure:

```text
Phase 2: Midterm Progress & Troubleshooting
1. Initial Training & Loss Analysis
1.1 Baseline PINN setup
1.2 Loss components
1.3 Loss convergence curves and stability analysis
1.4 Loss-weight balancing
1.5 Selected baseline case and convergence summary
1.6 FDM vs. PINN full-field baseline comparison
2. Ablation Study: Planned Experiments
2.1 Definition and common evaluation metrics
2.2 Planned ablation matrix
2.3 Sensor number ablation
2.4 Noise robustness test
2.5 Current profile ablation
2.6 One-parameter inverse vs. two-parameter inverse
2.7 Unseen current profile validation
3. Troubleshooting Experience (Flexible Reporting)
3.1 Main issue: scaling and large current terms
3.2 Synthetic data issue: local ODE vs. full PDE reference
3.3 Training issue: switching-time mismatch
3.4 Final submission note
```

Notes:

- The title/name line has Hyungjun Ahn / 안형준 right-aligned.
- The report intentionally excludes long code blocks.
- Placeholders remain for figures:
  - loss convergence curves,
  - selected-case parameter/loss summary,
  - FDM/PINN/error heatmaps,
  - unseen-profile validation plot.
- Rendering with `render_docx.py` failed because LibreOffice `soffice` is not installed.
- Quick Look first-page preview was checked and did not show obvious first-page breakage.

## 14. Code Guide Files

Existing LaTeX/PDF guide files:

```text
code_guide/PHASE1_CODE_GUIDE.tex
code_guide/PHASE1_DEEPXDE_CODE_GUIDE.tex
code_guide/FDM_VS_PINN_NOTEBOOK_CODE_GUIDE.tex
```

These were created/edited earlier to explain the code. User may ask to update them, but current submission direction is Word report + Colab notebook.

## 15. Known Problems / Open Decisions

1. Current profile inconsistency:
   - `main_deepxde.py` still has old 0-300/600-900 profile.
   - `fdm_reference.py` has the newer ablation-plan profile.
   - Before final runs, align current-profile definitions.

2. Notebook vs script may differ:
   - `main_deepxde.ipynb` is the working basis.
   - `main_deepxde.py` has default `LOSS_WEIGHTS = [1.0, 10.0, 10.0, 10.0, 100.0]`.
   - Notebook currently showed A5 active at one point: `[10, 10, 10, 10, 200]`.
   - Always inspect the current notebook cell before interpreting results.

3. A3 mismatch near 600 s:
   - User noticed FDM vs PINN time slices do not match perfectly around 600/1200 s.
   - This is acceptable to report as a limitation, especially near current switching times.
   - Do not hide the mismatch. Explain it as sharp input switching / PINN smoothing / collocation limitations.

4. Loss curve plots still need actual insertion:
   - Current Word report has the section and placeholder but not the final plot image.

5. True ablation cases are planned, not all completed:
   - Do not imply completed ablation results unless the notebook output exists.

## 16. Recommended Next Steps

1. Decide and freeze the baseline current profile for Phase 2.
2. Align `main_deepxde.ipynb`, `main_deepxde.py`, and `fdm_reference.py` current profiles.
3. Generate loss convergence plots:
   - `Loss_Total`
   - `Loss_PDE`
   - `Loss_BC = left BC + right BC`
4. Insert those plots into `Phase2/Phase2_Midterm_Draft.docx`.
5. Use A3/A4 comparison in Section 1 as loss-weight balancing, not ablation.
6. For true ablation, start with the current profile ablation because it is physically strongest:
   - constant current,
   - one on-off,
   - two on-off.
7. Then run sensor-number and noise robustness cases if time allows.
8. Keep the report honest:
   - baseline and loss-weight tuning are completed,
   - ablation plan is defined,
   - final ablation results should only be added after actual runs.

## 17. Short Korean Summary for Claude

이 프로젝트는 로봇 관절 모터 하우징의 1D 비정상 열전달 inverse PINN이다. 무차원 PDE는 `theta_tau = theta_XX - beta theta + S i(tau)^2 g(X)`이고, `beta`와 `S`를 학습한 뒤 실제 `h`, `q0`로 변환한다. 양 끝은 축방향 단열이고, convection은 경계조건이 아니라 `-beta theta` sink term으로 들어간다.

지금까지 A1-A5 loss weight case를 돌렸지만, 이것은 ablation이 아니라 loss-weight balancing이다. 보고서에서는 1장 `Initial Training & Loss Analysis`에 넣어야 한다. 진짜 ablation은 센서 개수, 노이즈, current profile, one/two parameter inverse, unseen current profile validation으로 계획되어 있다.

현재 Word 보고서 초안은 `Phase2/Phase2_Midterm_Draft.docx`에 있고, 템플릿 기반으로 구조가 잡혀 있다. 아직 실제 loss curve plot과 일부 ablation 결과 figure는 넣어야 한다. Claude는 이 내용을 기준으로 이어받으면 된다.
