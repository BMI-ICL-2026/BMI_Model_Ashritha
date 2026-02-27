# BMI Neural Decoder — Kalman Filter with PCA Subspace

A neural decoder for the Brain-Machine Interface (BMI) competition that predicts hand position from monkey motor cortex spike trains. Achieves **~18.30 mm RMSE** on the competition test harness.

## Results

| Metric | Value |
|--------|-------|
| **Final RMSE** | **~18.30 mm** (seed 2013) |
| Neurons | 98 |
| Reaching angles | 8 |
| Training trials | 50 per angle |
| Test trials | ~50 per angle |
| Decoding starts at | t = 320 ms |
| Decode step size | 20 ms |

### RMSE Progression

| Version | Approach | RMSE (mm) |
|---------|----------|-----------|
| v0 | Population Vector Algorithm (PVA) | 57.93 |
| v1 | Kalman Filter (98-dim observations) | 46.18 |
| v2 | + Per-angle models, velocity damping, 100 ms spike window | 29.61 |
| v3 | + Endpoint velocity suppression, trajectory blending | 29.23 |
| **v4** | **+ PCA subspace (10-dim), nearest centroid classifier, confidence blending** | **~18.30** |

## Architecture

```
                        TRAINING PIPELINE
                        =================

  training_data(n,k).spikes  ──────────────────────────────┐
       (98 x T binary)                                     │
              │                                             │
              v                                             v
  ┌───────────────────────┐                    ┌────────────────────────┐
  │  Spike Binning (20ms) │                    │  320ms Spike Features  │
  │  + 100ms Causal Window│                    │  (98 x 16 bins = 1568) │
  │  + Exp Smoothing a=0.5│                    │  per trial             │
  └───────────┬───────────┘                    └────────────┬───────────┘
              │                                             │
              v                                             v
  ┌───────────────────────┐                    ┌────────────────────────┐
  │  Smoothed Rates       │                    │  Z-score + Nearest     │
  │  (98 x N_bins)        │                    │  Centroid Classifier   │
  └───────────┬───────────┘                    │  (8 angle centroids)   │
              │                                └────────────────────────┘
              v
  ┌───────────────────────┐
  │  PCA via SVD          │
  │  98-dim  ──>  10-dim  │
  │  PC = U(:, 1:10)      │
  └───────────┬───────────┘
              │
              v
  ┌───────────────────────────────────────────────────────┐
  │              Per-Angle Kalman Model Fitting            │
  │                                                       │
  │  For each angle k = 1..8:                             │
  │    State transition:  A_k (4x4), W_k (4x4)           │
  │    Observation model: C_k (10x4), Q_k (10x10)        │
  │    Velocity damping:  A_k(3,3) = A_k(4,4) = 0.85     │
  └───────────────────────────────────────────────────────┘
              │
              v
  ┌───────────────────────────────────────────────────────┐
  │           Mean Trajectory Templates (50 pts)          │
  │  Per-angle, covering decode window (t=320 to end)     │
  │  + Mean velocity at t=320ms per angle                 │
  └───────────────────────────────────────────────────────┘



                       DECODING PIPELINE
                       =================

  test_data.spikes(:, 1:t)
              │
              v
  ┌───────────────────────┐     t = 320 (first call)
  │  Smoothed Firing Rate │─────────────────────────────┐
  │  (100ms window, a=0.5)│                             │
  └───────────┬───────────┘                             v
              │                            ┌────────────────────────┐
              │                            │  Angle Classification  │
              │                            │  Nearest centroid on   │
              │                            │  z-scored 320ms feats  │
              │                            │  ──> angle k, conf c   │
              │                            └────────────┬───────────┘
              │                                         │
              │                            Load per-angle model:
              │                            A_k, C_k, Q_k, W_k
              │                                         │
              v                                         v
  ┌───────────────────────┐          ┌──────────────────────────┐
  │  PCA Projection       │          │  Init Kalman State       │
  │  obs = PC'*(sr-mu)    │          │  z = [startPos; vel_320] │
  │  (98-dim ──> 10-dim)  │          │  P = P0                  │
  └───────────┬───────────┘          └──────────────────────────┘
              │
              v
  ┌───────────────────────────────────────────────────────┐
  │                  KALMAN FILTER STEP                    │
  │                                                       │
  │  PREDICT:                                             │
  │    z_pred = A_k * z                                   │
  │    P_pred = A_k * P * A_k' + W_eff                    │
  │                                                       │
  │  UPDATE (10-dim PCA space):                           │
  │    innovation = obs_pca - C_k * z_pred     (10x1)     │
  │    S = C_k * P_pred * C_k' + Q_eff        (10x10)    │
  │    K = P_pred * C_k' / S                   (4x10)     │
  │    z = z_pred + K * innovation             (4x1)      │
  │    P = (I - K * C_k) * P_pred              (4x4)      │
  └───────────┬───────────────────────────────────────────┘
              │
              v
  ┌───────────────────────────────────────────────────────┐
  │              TEMPORAL MODIFICATIONS                    │
  │                                                       │
  │  Early trial (step <= 5):  Q_eff = Q * 1.5            │
  │  Late trial (progress > 0.7):                         │
  │    - W_eff = W * 0.3                                  │
  │    - P velocity rows/cols *= 0.5                      │
  │  Velocity lock: if ||vel|| < 0.5 for 3 steps ──> 0   │
  └───────────┬───────────────────────────────────────────┘
              │
              v
  ┌───────────────────────────────────────────────────────┐
  │         CONFIDENCE-BASED TEMPLATE BLENDING            │
  │                                                       │
  │  progress = (t - 320) / (mean_T - 320)               │
  │  template_pos = meanTraj(:, bin, k)                   │
  │                                                       │
  │  conf > 0.6:  pos = 0.4*kalman + 0.6*template        │
  │  conf 0.3-0.6: pos = 0.6*kalman + 0.4*template       │
  │  conf < 0.3:  pos = 0.8*kalman + 0.2*template        │
  └───────────┬───────────────────────────────────────────┘
              │
              v
         [x, y] output
```

## Key Design Decisions

- **PCA Dimensionality Reduction**: SVD on centred firing rates reduces 98 neurons to 10 principal components. This shrinks the Kalman observation space from 98x98 covariance inversions to 10x10, improving both speed and numerical stability.

- **Per-Angle Kalman Models**: Separate A, C, Q, W matrices for each of the 8 reaching angles capture direction-specific neural tuning and kinematics.

- **Nearest Centroid Classifier**: At t=320ms, spike counts across 16 bins (20ms each) form a 1568-dim feature vector. Z-scored and compared against 8 trained centroids via Euclidean distance. Confidence score derived from gap between best and second-best match.

- **Template Blending**: Mean trajectory templates (50 timepoints per angle) act as a prior. Blending weight scales with classification confidence — high confidence trusts the template more.

- **Velocity Damping**: A(3,3) = A(4,4) = 0.85 applies natural deceleration, preventing velocity runaway.

- **Late-Trial Suppression**: After 70% of expected trial length, process noise W is shrunk (x0.3) and velocity covariance in P is squeezed (x0.5), allowing the filter to smoothly decelerate to the endpoint.

## Toolboxes Used

**None.** The entire decoder uses only base MATLAB functions:

| Function | Purpose |
|----------|---------|
| `svd(X, 'econ')` | PCA via singular value decomposition |
| `cov()` | Covariance matrix estimation |
| `mean()`, `std()` | Feature statistics |
| `interp1()` | Trajectory template resampling |
| `eye()`, `zeros()`, `ones()` | Matrix construction |
| `norm()`, `dot()` | Vector operations |
| `sum()`, `sqrt()`, `sort()` | Basic math |
| `linspace()`, `round()`, `floor()` | Index computation |
| `fprintf()` | Debug output |

No Signal Processing Toolbox, Statistics Toolbox, or any other toolbox is required.

## File Structure

```
BMI/
├── README.md
├── MATLAB files/
│   ├── run_test.m                         # One-click test script
│   ├── testFunction_for_students_MTb.m    # Competition test harness
│   ├── monkeydata_training.mat            # Raw training data
│   └── myTeam/
│       ├── positionEstimatorTraining.m    # Training function (v4)
│       └── positionEstimator.m            # Decoding function (v4)
└── tasks/
    └── lessons.md
```

## How to Run

1. Open MATLAB
2. Navigate to the `MATLAB files/` directory
3. Run:
   ```matlab
   run_test
   ```
   This trains the model and evaluates RMSE on the test split automatically.

## State Vector

```
z = [x; y; vx; vy]    (4 x 1)

x, y   — hand position (mm)
vx, vy — hand velocity (mm/bin)
```

## Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `bin_width` | 20 ms | Spike binning resolution |
| `spike_window` | 100 ms | Causal firing rate window |
| `alpha` | 0.5 | Exponential smoothing factor |
| `vel_damp` | 0.85 | Velocity damping in state transition |
| `n_pca` | 10 | PCA components retained |
| `n_traj_bins` | 50 | Trajectory template resolution |
| `lambda` | 1e-6 | Tikhonov regularisation |
| `Q scale` | 1.5x | Observation noise inflation (first 5 steps) |
| `W scale` | 0.3x | Process noise reduction (late trial) |
| `vel_lock threshold` | 0.5 | Velocity norm for endpoint locking |
