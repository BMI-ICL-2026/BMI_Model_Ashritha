# BMI Neural Decoder — kNN + LDA + Soft Templates

A neural decoder for the Brain-Machine Interface (BMI) competition that predicts hand position from monkey motor cortex spike trains. Achieves **~9.93 mm RMSE** on the competition test harness.

## Results

| Metric | Value |
|--------|-------|
| **Final RMSE** | **~9.93 mm** (seed 2015) |
| Neurons | 98 |
| Reaching angles | 8 |
| Training trials | 50 per angle |
| Test trials | ~50 per angle |
| Decoding starts at | t = 320 ms |
| Decode step size | 20 ms |
| Runtime | ~30 seconds |

### RMSE Progression

| Version | Approach | RMSE (mm) |
|---------|----------|-----------|
| v0 | Population Vector Algorithm (PVA) | 57.93 |
| v1 | Kalman Filter (98-dim observations) | 46.18 |
| v2 | + Per-angle models, velocity damping, 100 ms spike window | 29.61 |
| v3 | + Endpoint velocity suppression, trajectory blending | 29.23 |
| v4 | + PCA subspace (10-dim), nearest centroid classifier, confidence blending | ~18.30 |
| v5 | Kalman + LDA + template blending (various Kalman architectures) | ~11.30 |
| **v6** | **kNN position estimator + LDA classifier + soft templates** | **~9.93** |

## Architecture

### Flowchart

```
                        TRAINING PIPELINE
                        =================

  training_data(n,k).spikes                training_data(n,k).spikes
       (98 x T binary)                         (98 x T binary)
              |                                       |
              v                                       v
  +-------------------------+           +---------------------------+
  |  EMA Feature Extraction |           |  Spike Count Extraction   |
  |  4 windows (80ms each)  |           |  80ms causal window       |
  |  alpha = 0.92           |           |  at each 20ms decode step |
  |  over first 320ms       |           |  from t=320 to trial end  |
  +------------+------------+           +-------------+-------------+
               |                                      |
               v                                      v
  +-------------------------+           +---------------------------+
  |  Neuron Selection       |           |  PCA via SVD              |
  |  Top 55 by directional  |           |  98-dim --> 25-dim        |
  |  modulation score       |           |  PC = U(:, 1:25)          |
  +------------+------------+           +-------------+-------------+
               |                                      |
               v                                      v
  +-------------------------+           +---------------------------+
  |  Z-score + LDA          |           |  kNN Database             |
  |  55 neurons x 4 windows |           |  Per angle: store all     |
  |  = 220 features         |           |  (PCA_feature, position)  |
  |  Shrinkage covariance   |           |  pairs from training      |
  |  --> 8 class means      |           |  ~1250 points per angle   |
  |  --> Sw_inv for scoring  |           +---------------------------+
  +-------------------------+
               |
               +-----> Also: Mean trajectory templates (50 pts per angle)
                       Resampled from t=320 to trial end



                       DECODING PIPELINE
                       =================

  test_data.spikes(:, 1:t)
              |
              +------------------+------------------+
              |                  |                  |
              v                  v                  v
  (first call only)     +---------------+   +---------------+
  +----------------+    | Spike Count   |   | Spike Count   |
  | EMA Features   |    | [t-80, t]     |   | [t-80, t]     |
  | (first 320ms)  |    | 98-dim        |   | 98-dim        |
  +-------+--------+    +-------+-------+   +-------+-------+
          |                     |                   |
          v                     v                   v
  +----------------+    +---------------+   +---------------+
  | LDA Classifier |    | PCA Project   |   | Soft Template |
  | Mahalanobis    |    | 98 --> 25-dim |   | Softmax wts   |
  | distance in    |    +-------+-------+   | x 8 angles    |
  | 220-dim space  |            |           | + offset      |
  | --> best angle |            v           | correction    |
  | --> softmax    |    +---------------+   +-------+-------+
  |    weights     |    | kNN Lookup    |           |
  +-------+--------+    | 15 nearest    |           |
          |             | neighbors     |           |
          |             | inv-distance  |           |
          |             | weighted avg  |           |
          |             +-------+-------+           |
          |                     |                   |
          v                     v                   v
  +-----------------------------------------------------+
  |              BLEND: 55% kNN + 45% Soft Template      |
  |                                                       |
  |  kNN provides trial-adaptive position estimate        |
  |  Template provides smooth baseline + robustness       |
  |  to classification errors via softmax weighting       |
  +---------------------------+---------------------------+
                              |
                              v
                         [x, y] output
```

## Key Design Decisions

- **kNN Position Estimator**: Instead of a Kalman filter, we find training trials with the most similar neural activity (in PCA space) and use their hand positions. This is non-parametric — no state-space model assumptions, no temporal drift, and naturally handles speed/shape variation across trials. 15 nearest neighbors with inverse-distance weighting.

- **LDA Classifier with Shrinkage**: At t=320ms, EMA-smoothed firing rates across 4 time windows (80ms each) form a 220-dim feature vector (55 selected neurons x 4 windows). Classified via Mahalanobis distance with shrinkage within-class covariance (30% toward diagonal).

- **Neuron Selection**: Top 55 of 98 neurons ranked by directional modulation (variance of mean rate across angles). Low-rate neurons (< 0.5 Hz mean) suppressed. Reduces overfitting in the classifier.

- **Soft Template Blending**: Instead of committing to a single classified angle, softmax weights distribute probability across all 8 angles. The template position is a weighted average, so misclassification causes a blended (not catastrophically wrong) estimate.

- **Template Offset Correction**: Each test trial's start position is compared to each angle's template start, and the offset is added throughout. Aligns templates to the actual trial geometry.

- **PCA Dimensionality Reduction**: SVD on centred spike counts reduces 98 neurons to 25 principal components for the kNN feature space. Reduces noise and speeds up distance computation.

## Toolboxes Used

**None.** The entire decoder uses only base MATLAB functions:

| Function | Purpose |
|----------|---------|
| `svd(X, 'econ')` | PCA via singular value decomposition |
| `pinv()` | Pseudoinverse for LDA covariance |
| `mean()`, `std()`, `var()` | Feature statistics |
| `interp1()` | Trajectory template resampling |
| `sort()` | kNN neighbor selection |
| `exp()`, `log()` | Softmax computation |
| `sum()`, `zeros()`, `eye()` | Matrix/vector operations |
| `linspace()`, `round()`, `min()`, `max()` | Index computation |

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
│       ├── positionEstimatorTraining.m    # Training function (v6)
│       └── positionEstimator.m            # Decoding function (v6)
└── Intitial docs/
    └── CompetitionDocument(1) (1).pdf     # Competition rules
```

## How to Run

1. Open MATLAB
2. Navigate to the `MATLAB files/` directory
3. Run:
   ```matlab
   run_test
   ```
   This trains the model and evaluates RMSE on the test split automatically.

## Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `bin_width` | 20 ms | Decode step size |
| `obs_window` | 80 ms | Spike count window for kNN features |
| `n_pca` | 25 | PCA components for kNN feature space |
| `k_neighbors` | 15 | Number of nearest neighbors |
| `knn_weight` | 0.55 | kNN blend weight (vs 0.45 template) |
| `ema_alpha` | 0.92 | EMA smoothing for classifier features |
| `n_keep` | 55 | Neurons selected for classifier |
| `shrinkage` | 0.30 | LDA covariance shrinkage toward diagonal |
| `n_traj_bins` | 50 | Trajectory template resolution |
| `class_T` | 320 ms | Spike data used for classification |
| `win_edges` | [1-80, 81-160, 161-240, 241-320] | Classifier time windows |
