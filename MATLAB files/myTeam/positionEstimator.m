function [x, y, newModelParameters] = positionEstimator(test_data, modelParameters)
% Decode hand position using a Kalman Filter (v2).
%
%   [x, y, newModelParameters] = positionEstimator(test_data, modelParameters)
%
%   Improvements over v1:
%     - 100ms causal spike window + alpha=0.5 smoothing
%     - Cosine similarity angle classification -> per-angle model
%     - Velocity damping (A(3,3)=A(4,4)=0.85 baked into A)
%     - Mean velocity init at t=320
%     - 80/20 mean-trajectory blending to prevent overshoot

    newModelParameters = modelParameters;

    bin_width    = modelParameters.bin_width;       % 20 ms
    alph         = modelParameters.alpha;           % 0.5
    spike_window = modelParameters.spike_window;    % 100 ms
    n_neurons    = 98;
    t            = size(test_data.spikes, 2);       % current time (ms)
    is_first     = isempty(test_data.decodedHandPos);

    %% ---- 1. Compute smoothed firing rates (100ms window) ----
    if is_first
        % First call: process ALL 20ms bins from trial start
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
        % Subsequent call: single 100ms window ending at t
        ws = max(1, t - spike_window + 1);
        window_s = (t - ws + 1) / 1000;
        raw_rate = sum(test_data.spikes(:, ws:t), 2) / window_s;
        sr = alph * raw_rate + (1 - alph) * modelParameters.smooth_rates;
    end
    newModelParameters.smooth_rates = sr;

    %% ---- 2. Initialise or restore Kalman state ----
    if is_first
        % --- Classify angle via cosine similarity ---
        mean_rates = modelParameters.mean_rates;   % 98 x 8
        cos_sims   = zeros(1, 8);
        sr_norm    = norm(sr);
        for k = 1:8
            mr_norm = norm(mean_rates(:, k));
            if sr_norm > 0 && mr_norm > 0
                cos_sims(k) = dot(sr, mean_rates(:, k)) / (sr_norm * mr_norm);
            end
        end
        [~, best_k] = max(cos_sims);
        newModelParameters.current_angle = best_k;

        % --- Debug: print classified angle for first 5 trials ---
        dc = modelParameters.debug_count + 1;
        newModelParameters.debug_count = dc;
        if dc <= 5
            fprintf('  [debug] Trial %d classified as angle %d (cos=%.3f)\n', ...
                dc, best_k, cos_sims(best_k));
        end

        % --- Always use per-angle model ---
        newModelParameters.active_A = modelParameters.perAngle(best_k).A;
        newModelParameters.active_C = modelParameters.perAngle(best_k).C;
        newModelParameters.active_Q = modelParameters.perAngle(best_k).Q;
        newModelParameters.active_W = modelParameters.perAngle(best_k).W;

        % --- Init state: start position + mean velocity at t=320 ---
        init_vel = modelParameters.mean_vel_init(:, best_k);
        kalman_z = [test_data.startHandPos(1:2); init_vel];
        kalman_P = modelParameters.P0;
    else
        kalman_z = modelParameters.kalman_z;
        kalman_P = modelParameters.kalman_P;
    end

    % Active model for this trial (set on first call, persisted after)
    A_act = newModelParameters.active_A;
    C_act = newModelParameters.active_C;
    Q_act = newModelParameters.active_Q;
    W_act = newModelParameters.active_W;

    %% ---- 3. Kalman predict ----
    z_pred = A_act * kalman_z;                        % 4x1
    P_pred = A_act * kalman_P * A_act' + W_act;       % 4x4

    %% ---- 4. Kalman update ----
    innovation = sr - C_act * z_pred;                  % 98x1
    S = C_act * P_pred * C_act' + Q_act;               % 98x98
    K = P_pred * C_act' / S;                            % 4x98 (mrdivide)
    kalman_z = z_pred + K * innovation;                % 4x1
    kalman_P = (eye(4) - K * C_act) * P_pred;          % 4x4

    %% ---- 5. Blend with mean trajectory prior (80% Kalman / 20% mean) ----
    best_k     = newModelParameters.current_angle;
    mean_T_k   = modelParameters.mean_T(best_k);
    frac       = max(0, min(1, t / mean_T_k));                   % progress [0, 1]
    n_tb       = modelParameters.n_traj_bins;
    traj_bin   = max(1, min(n_tb, round(frac * n_tb)));          % index 1..30
    mean_pos   = modelParameters.mean_traj(:, traj_bin, best_k); % 2x1
    kalman_z(1:2) = 0.8 * kalman_z(1:2) + 0.2 * mean_pos;

    %% ---- 6. Clamp to prevent drift ----
    kalman_z(3) = max(-5, min(5, kalman_z(3)));      % vx (mm/bin)
    kalman_z(4) = max(-5, min(5, kalman_z(4)));      % vy
    kalman_z(1) = max(-200, min(200, kalman_z(1)));  % x  (mm)
    kalman_z(2) = max(-200, min(200, kalman_z(2)));  % y

    %% ---- 7. Output ----
    x = kalman_z(1);
    y = kalman_z(2);

    % Persist Kalman state for next call
    newModelParameters.kalman_z = kalman_z;
    newModelParameters.kalman_P = kalman_P;
end
