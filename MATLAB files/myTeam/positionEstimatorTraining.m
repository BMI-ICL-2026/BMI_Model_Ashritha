function modelParameters = positionEstimatorTraining(training_data)
% Train a Kalman Filter decoder with PCA observation subspace (v4).
%
%   State:       z = [x; y; vx; vy]                     (4x1)
%   Observation: obs = PC' * (smooth_rates - mean_fr)    (10x1)
%
%   Models:  z_t   = A * z_{t-1} + w,   w ~ N(0, W)     4x4
%            obs_t = C * z_t     + q,   q ~ N(0, Q)    10x4

    n_neurons    = 98;
    n_angles     = 8;
    n_trials     = size(training_data, 1);
    bin_width    = 20;
    alpha        = 0.5;
    spike_window = 100;
    vel_damp     = 0.85;
    n_traj_bins  = 50;     % template resolution (was 30)
    n_pca        = 10;     % PCA components to keep
    lambda       = 1e-6;   % Tikhonov regularisation

    fprintf('Training Kalman Filter decoder (v4 — PCA + templates)...\n');

    %% ==== 1. Extract states and smoothed rates from every trial ====
    max_per_angle = n_trials * 50;

    Za  = cell(1, n_angles);   Ra  = cell(1, n_angles);
    Zca = cell(1, n_angles);   Zpa = cell(1, n_angles);
    cnt   = zeros(1, n_angles);
    cnt_t = zeros(1, n_angles);

    for k = 1:n_angles
        Za{k}  = zeros(4, max_per_angle);
        Ra{k}  = zeros(n_neurons, max_per_angle);
        Zca{k} = zeros(4, max_per_angle);
        Zpa{k} = zeros(4, max_per_angle);
    end

    for k = 1:n_angles
        for tr = 1:n_trials
            spk = training_data(tr, k).spikes;
            pos = training_data(tr, k).handPos(1:2, :);
            T   = size(spk, 2);
            n_bins = floor(T / bin_width);
            if n_bins < 3, continue; end

            % Firing rate: 100ms causal window + exponential smoothing
            rates = zeros(n_neurons, n_bins);
            bin_centers = zeros(1, n_bins);
            for b = 1:n_bins
                be = b * bin_width;
                ws = max(1, be - spike_window + 1);
                window_s = (be - ws + 1) / 1000;
                bin_centers(b) = round(((b-1)*bin_width + 1 + be) / 2);
                raw_rate = sum(spk(:, ws:be), 2) / window_s;
                if b == 1
                    rates(:, b) = raw_rate;
                else
                    rates(:, b) = alpha * raw_rate + (1 - alpha) * rates(:, b-1);
                end
            end

            % Position at bin centers
            pos_bins = zeros(2, n_bins);
            for b = 1:n_bins
                tc = min(bin_centers(b), size(pos, 2));
                pos_bins(:, b) = pos(:, tc);
            end

            % Velocity via central difference (mm/bin)
            vel_bins = zeros(2, n_bins);
            vel_bins(:, 1) = pos_bins(:, 2) - pos_bins(:, 1);
            for b = 2:n_bins - 1
                vel_bins(:, b) = (pos_bins(:, b+1) - pos_bins(:, b-1)) / 2;
            end
            vel_bins(:, n_bins) = pos_bins(:, n_bins) - pos_bins(:, n_bins-1);

            states = [pos_bins; vel_bins];

            idx = cnt(k) + (1:n_bins);
            Za{k}(:, idx)  = states;
            Ra{k}(:, idx)  = rates;
            cnt(k) = cnt(k) + n_bins;

            n_pairs = n_bins - 1;
            idx_t = cnt_t(k) + (1:n_pairs);
            Zca{k}(:, idx_t) = states(:, 2:end);
            Zpa{k}(:, idx_t) = states(:, 1:end-1);
            cnt_t(k) = cnt_t(k) + n_pairs;
        end
        fprintf('  Angle %d/%d: %d samples\n', k, n_angles, cnt(k));
    end

    % Trim
    for k = 1:n_angles
        Za{k}  = Za{k}(:, 1:cnt(k));
        Ra{k}  = Ra{k}(:, 1:cnt(k));
        Zca{k} = Zca{k}(:, 1:cnt_t(k));
        Zpa{k} = Zpa{k}(:, 1:cnt_t(k));
    end

    Z_all  = [Za{:}];
    R_all  = [Ra{:}];
    Z_curr = [Zca{:}];
    Z_prev = [Zpa{:}];
    fprintf('  Combined: %d samples, %d transitions\n', size(Z_all,2), size(Z_curr,2));

    %% ==== 2. PCA on firing rates (svd, no toolbox) ====
    mean_fr    = mean(R_all, 2);            % 98x1  centering vector
    R_centered = R_all - mean_fr;           % 98 x N
    [U, ~, ~]  = svd(R_centered, 'econ');   % U: 98x98 (econ)
    PC         = U(:, 1:n_pca);             % 98 x 10

    % Project all rates into 10-dim subspace
    R_pca = PC' * R_centered;               % 10 x N

    % Per-angle projections
    Ra_pca = cell(1, n_angles);
    for k = 1:n_angles
        Ra_pca{k} = PC' * (Ra{k} - mean_fr);  % 10 x N_k
    end
    fprintf('  PCA: kept %d of %d components\n', n_pca, n_neurons);

    %% ==== 3. Fit combined observation model in PCA space ====
    C = (R_pca * Z_all') / (Z_all * Z_all' + lambda * eye(4));   % 10 x 4
    Q = cov((R_pca - C * Z_all)');                                 % 10 x 10
    Q = Q + 0.01 * mean(diag(Q)) * eye(n_pca);

    %% ==== 4. Fit combined state transition ====
    A = (Z_curr * Z_prev') / (Z_prev * Z_prev' + lambda * eye(4));
    A(3, 3) = vel_damp;
    A(4, 4) = vel_damp;
    W = cov((Z_curr - A * Z_prev)');
    W = W + 0.001 * mean(diag(W)) * eye(4);

    fprintf('  Combined model fitted (C: %dx%d, Q: %dx%d)\n', ...
        size(C,1), size(C,2), size(Q,1), size(Q,2));

    %% ==== 5. Per-angle Kalman models in PCA space ====
    for k = 1:n_angles
        Ck = (Ra_pca{k} * Za{k}') / (Za{k} * Za{k}' + lambda * eye(4));  % 10x4
        Qk = cov((Ra_pca{k} - Ck * Za{k})');                               % 10x10
        Qk = Qk + 0.01 * mean(diag(Qk)) * eye(n_pca);

        Ak = (Zca{k} * Zpa{k}') / (Zpa{k} * Zpa{k}' + lambda * eye(4));
        Ak(3, 3) = vel_damp;
        Ak(4, 4) = vel_damp;
        Wk = cov((Zca{k} - Ak * Zpa{k})');
        Wk = Wk + 0.001 * mean(diag(Wk)) * eye(4);

        perAngle(k).A = Ak;
        perAngle(k).C = Ck;   % 10x4
        perAngle(k).Q = Qk;   % 10x10
        perAngle(k).W = Wk;
    end
    fprintf('  Per-angle models fitted (10-dim obs)\n');

    %% ==== 6. Angle classifier — nearest centroid on 320ms spike features ====
    n_class_bins = floor(320 / bin_width);           % 16
    feat_dim     = n_neurons * n_class_bins;          % 1568
    n_total      = n_trials * n_angles;

    all_feat   = zeros(n_total, feat_dim);
    all_labels = zeros(n_total, 1);
    idx = 0;

    for k = 1:n_angles
        for tr = 1:n_trials
            spk = training_data(tr, k).spikes;
            T   = size(spk, 2);
            t_use = min(320, T);
            n_bu  = floor(t_use / bin_width);

            feat = zeros(n_neurons, n_class_bins);
            for b = 1:n_bu
                bs = (b-1)*bin_width + 1;
                be = b*bin_width;
                feat(:, b) = sum(spk(:, bs:be), 2);
            end

            idx = idx + 1;
            all_feat(idx, :)  = feat(:)';
            all_labels(idx)   = k;
        end
    end

    % Manual z-score (no zscore() toolbox function)
    feat_mean = mean(all_feat, 1);             % 1 x 1568
    feat_std  = std(all_feat, 0, 1);           % 1 x 1568
    feat_std(feat_std < 1e-10) = 1;            % prevent div-by-zero
    all_feat_z = (all_feat - feat_mean) ./ feat_std;

    % Per-angle centroids
    centroids = zeros(n_angles, feat_dim);
    for k = 1:n_angles
        centroids(k, :) = mean(all_feat_z(all_labels == k, :), 1);
    end

    % Classification accuracy on training data
    correct = 0;
    for i = 1:n_total
        dists = zeros(1, n_angles);
        for k = 1:n_angles
            d = all_feat_z(i, :) - centroids(k, :);
            dists(k) = sqrt(sum(d .* d));
        end
        [~, pred] = min(dists);
        if pred == all_labels(i)
            correct = correct + 1;
        end
    end
    fprintf('  Classifier accuracy on training data: %.1f%% (%d/%d)\n', ...
        correct / n_total * 100, correct, n_total);

    %% ==== 7. Mean trajectory templates (50 pts, decode window only) ====
    meanTraj = zeros(2, n_traj_bins, n_angles);
    mean_T   = zeros(1, n_angles);

    for k = 1:n_angles
        traj_sum = zeros(2, n_traj_bins);
        T_sum    = 0;
        n_valid  = 0;
        for tr = 1:n_trials
            pos = training_data(tr, k).handPos(1:2, :);
            T   = size(pos, 2);
            T_sum = T_sum + T;
            ds = min(320, T);
            if T - ds < bin_width, continue; end
            % Resample from t=320 to t=end into n_traj_bins points
            orig_t = ds:T;
            new_t  = linspace(ds, T, n_traj_bins);
            traj_sum(1, :) = traj_sum(1, :) + interp1(orig_t, pos(1, ds:T), new_t);
            traj_sum(2, :) = traj_sum(2, :) + interp1(orig_t, pos(2, ds:T), new_t);
            n_valid = n_valid + 1;
        end
        if n_valid > 0
            meanTraj(:, :, k) = traj_sum / n_valid;
        end
        mean_T(k) = T_sum / n_trials;
    end
    fprintf('  Mean trajectories: %d pts (decode window), mean T: %.0f-%.0f ms\n', ...
        n_traj_bins, min(mean_T), max(mean_T));

    %% ==== 8. Mean velocity at t=320ms per angle ====
    mean_vel_init = zeros(2, n_angles);
    for k = 1:n_angles
        vel_sum = zeros(2, 1);
        nv = 0;
        for tr = 1:n_trials
            pos = training_data(tr, k).handPos(1:2, :);
            T   = size(pos, 2);
            if T < 340, continue; end
            vel_sum = vel_sum + (pos(:, 340) - pos(:, 300)) / 2;
            nv = nv + 1;
        end
        if nv > 0
            mean_vel_init(:, k) = vel_sum / nv;
        end
    end

    %% ==== 9. Initial state covariance ====
    P0 = cov(Z_all') * 0.1;

    %% ==== Store model parameters ====
    % Kalman model (combined)
    modelParameters.A  = A;           % 4x4
    modelParameters.C  = C;           % 10x4
    modelParameters.Q  = Q;           % 10x10
    modelParameters.W  = W;           % 4x4
    modelParameters.P0 = P0;          % 4x4
    modelParameters.perAngle = perAngle;

    % PCA projection
    modelParameters.PC      = PC;       % 98x10
    modelParameters.mean_fr = mean_fr;  % 98x1
    modelParameters.n_pca   = n_pca;

    % Angle classifier
    modelParameters.centroids    = centroids;     % 8x1568
    modelParameters.feat_mean    = feat_mean;     % 1x1568
    modelParameters.feat_std     = feat_std;      % 1x1568
    modelParameters.n_class_bins = n_class_bins;  % 16

    % Trajectory templates
    modelParameters.meanTraj     = meanTraj;      % 2x50x8
    modelParameters.mean_T       = mean_T;        % 1x8
    modelParameters.meanTrialLength = mean_T;
    modelParameters.mean_vel_init = mean_vel_init; % 2x8
    modelParameters.n_traj_bins  = n_traj_bins;

    % Spike processing constants
    modelParameters.bin_width    = bin_width;
    modelParameters.alpha        = alpha;
    modelParameters.spike_window = spike_window;

    % Runtime state (initialised per trial in positionEstimator)
    modelParameters.kalman_z      = [];
    modelParameters.kalman_P      = [];
    modelParameters.smooth_rates  = [];
    modelParameters.current_angle = 0;
    modelParameters.classify_confidence = 0;
    modelParameters.active_A = A;
    modelParameters.active_C = C;
    modelParameters.active_Q = Q;
    modelParameters.active_W = W;
    modelParameters.debug_count   = 0;
    modelParameters.step_count    = 0;
    modelParameters.low_vel_count = 0;
    modelParameters.vel_locked    = false;

    fprintf('Training complete.\n');
end
