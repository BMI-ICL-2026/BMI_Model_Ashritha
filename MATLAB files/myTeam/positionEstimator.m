function [x, y, newModelParameters] = positionEstimator(test_data, modelParameters)
% Decode hand position using a Kalman Filter with PCA subspace (v4).
%
%   [x, y, newModelParameters] = positionEstimator(test_data, modelParameters)
%
%   v4 changes:
%     - PCA subspace: 98-dim rates -> 10-dim observations
%     - Nearest centroid angle classifier (z-scored 320ms spike features)
%     - Confidence-based template blending
%     - Rich trajectory templates (50 pts, decode window only)

    newModelParameters = modelParameters;

    bin_width    = modelParameters.bin_width;       % 20 ms
    alph         = modelParameters.alpha;           % 0.5
    spike_window = modelParameters.spike_window;    % 100 ms
    n_neurons    = 98;
    t            = size(test_data.spikes, 2);       % current time (ms)
    is_first     = isempty(test_data.decodedHandPos);

    %% ---- 1. Compute smoothed firing rates (100ms window) ----
    if is_first
        n_bins = floor(t / bin_width);
        sr = zeros(n_neurons, 1);
        for b = 1:n_bins
            be = b * bin_width;
            ws = max(1, be - spike_window + 1);
            window_s = (be - ws + 1) / 1000;
            raw_rate = sum(test_data.spikes(:, ws:be), 2) / window_s;
            if b == 1
                sr = raw_rate;
            else
                sr = alph * raw_rate + (1 - alph) * sr;
            end
        end
    else
        ws = max(1, t - spike_window + 1);
        window_s = (t - ws + 1) / 1000;
        raw_rate = sum(test_data.spikes(:, ws:t), 2) / window_s;
        sr = alph * raw_rate + (1 - alph) * modelParameters.smooth_rates;
    end
    newModelParameters.smooth_rates = sr;

    %% ---- 2. Initialise or restore Kalman state ----
    if is_first
        % --- Classify angle via nearest centroid on z-scored spike features ---
        n_class_bins = modelParameters.n_class_bins;   % 16
        spk   = test_data.spikes;
        T_spk = size(spk, 2);
        t_use = min(320, T_spk);
        n_bu  = floor(t_use / bin_width);

        feat = zeros(n_neurons, n_class_bins);
        for b = 1:n_bu
            bs = (b-1)*bin_width + 1;
            be = b*bin_width;
            feat(:, b) = sum(spk(:, bs:be), 2);
        end
        feat_vec = feat(:)';                            % 1 x 1568

        % Z-score using training statistics
        feat_z = (feat_vec - modelParameters.feat_mean) ./ modelParameters.feat_std;

        % Find nearest centroid
        dists = zeros(1, 8);
        for k = 1:8
            d = feat_z - modelParameters.centroids(k, :);
            dists(k) = sqrt(sum(d .* d));
        end
        [min_dist, best_k] = min(dists);

        % Confidence: relative gap between best and second-best
        sorted_dists = sort(dists);
        if sorted_dists(1) > 0
            confidence = 1 - sorted_dists(1) / sorted_dists(2);
        else
            confidence = 1;
        end
        confidence = max(0, min(1, confidence));

        newModelParameters.current_angle        = best_k;
        newModelParameters.classify_confidence   = confidence;

        % --- Debug: print classified angle for first 5 trials ---
        dc = modelParameters.debug_count + 1;
        newModelParameters.debug_count = dc;
        if dc <= 5
            fprintf('  [debug] Trial %d classified as angle %d (conf=%.3f, dist=%.1f)\n', ...
                dc, best_k, confidence, min_dist);
        end

        % --- Use per-angle model ---
        newModelParameters.active_A = modelParameters.perAngle(best_k).A;
        newModelParameters.active_C = modelParameters.perAngle(best_k).C;   % 10x4
        newModelParameters.active_Q = modelParameters.perAngle(best_k).Q;   % 10x10
        newModelParameters.active_W = modelParameters.perAngle(best_k).W;

        % --- Init state: start position + mean velocity at t=320 ---
        init_vel = modelParameters.mean_vel_init(:, best_k);
        kalman_z = [test_data.startHandPos(1:2); init_vel];
        kalman_P = modelParameters.P0;

        % --- Init runtime counters ---
        step_count    = 0;
        low_vel_count = 0;
        vel_locked    = false;
    else
        kalman_z      = modelParameters.kalman_z;
        kalman_P      = modelParameters.kalman_P;
        step_count    = modelParameters.step_count;
        low_vel_count = modelParameters.low_vel_count;
        vel_locked    = modelParameters.vel_locked;
    end

    step_count = step_count + 1;

    %% ---- 3. Compute trial progress ----
    best_k         = newModelParameters.current_angle;
    confidence     = newModelParameters.classify_confidence;
    expected_steps = (modelParameters.meanTrialLength(best_k) - 320) / bin_width;
    progress       = step_count / expected_steps;       % 0..1+
    late_trial     = progress > 0.7;

    %% ---- 4. Get active model with temporal modifications ----
    A_act = newModelParameters.active_A;
    C_act = newModelParameters.active_C;    % 10x4
    Q_act = newModelParameters.active_Q;    % 10x10
    W_act = newModelParameters.active_W;

    n_pca = modelParameters.n_pca;          % 10

    % Early steps: inflate Q to trust state model over observations
    if step_count <= 5
        Q_eff = Q_act * 1.5;
    else
        Q_eff = Q_act;
    end

    % Late trial: shrink W so filter trusts deceleration model
    if late_trial
        W_eff = W_act * 0.3;
    else
        W_eff = W_act;
    end

    %% ---- 5. PCA projection ----
    obs_pca = modelParameters.PC' * (sr - modelParameters.mean_fr);   % 10x1

    %% ---- 6. Kalman predict ----
    z_pred = A_act * kalman_z;                          % 4x1
    P_pred = A_act * kalman_P * A_act' + W_eff;         % 4x4

    %% ---- 7. Kalman update (in PCA space) ----
    innovation = obs_pca - C_act * z_pred;              % 10x1
    S = C_act * P_pred * C_act' + Q_eff;                % 10x10
    K = P_pred * C_act' / S;                             % 4x10
    kalman_z = z_pred + K * innovation;                 % 4x1
    kalman_P = (eye(4) - K * C_act) * P_pred;           % 4x4

    %% ---- 8. Late-trial: squeeze velocity uncertainty in P ----
    if late_trial
        kalman_P(3:4, :) = 0.5 * kalman_P(3:4, :);
        kalman_P(:, 3:4) = 0.5 * kalman_P(:, 3:4);
    end

    %% ---- 9. Velocity lock ----
    if vel_locked
        kalman_z(3:4) = 0;
    else
        vel_norm = norm(kalman_z(3:4));
        if vel_norm < 0.5
            low_vel_count = low_vel_count + 1;
            if low_vel_count >= 3
                vel_locked = true;
                kalman_z(3:4) = 0;
            end
        else
            low_vel_count = 0;
        end
    end

    %% ---- 10. Confidence-based template blending (position only) ----
    mean_T_k = modelParameters.mean_T(best_k);
    n_tb     = modelParameters.n_traj_bins;             % 50

    % Progress within decode window: (t - 320) / (mean_T - 320)
    decode_progress = max(0, min(1, (t - 320) / max(1, mean_T_k - 320)));
    traj_bin = max(1, min(n_tb, round(decode_progress * n_tb)));

    mean_pos = modelParameters.meanTraj(:, traj_bin, best_k);

    % Confidence-based blending weights
    if confidence > 0.6
        w_template = 0.6;
    elseif confidence > 0.3
        w_template = 0.4;
    else
        w_template = 0.2;
    end
    w_kalman = 1 - w_template;

    kalman_z(1:2) = w_kalman * kalman_z(1:2) + w_template * mean_pos;

    %% ---- 11. Clamp ----
    kalman_z(3) = max(-5, min(5, kalman_z(3)));
    kalman_z(4) = max(-5, min(5, kalman_z(4)));
    kalman_z(1) = max(-200, min(200, kalman_z(1)));
    kalman_z(2) = max(-200, min(200, kalman_z(2)));

    %% ---- 12. Output ----
    x = kalman_z(1);
    y = kalman_z(2);

    % Persist all state for next call
    newModelParameters.kalman_z      = kalman_z;
    newModelParameters.kalman_P      = kalman_P;
    newModelParameters.step_count    = step_count;
    newModelParameters.low_vel_count = low_vel_count;
    newModelParameters.vel_locked    = vel_locked;
end
