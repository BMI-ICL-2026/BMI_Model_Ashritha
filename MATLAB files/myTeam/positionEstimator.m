function [x, y, newModelParameters] = positionEstimator(test_data, modelParameters)
% Decode hand position using a Kalman Filter (v3).
%
%   [x, y, newModelParameters] = positionEstimator(test_data, modelParameters)
%
%   v3 improvements:
%     - Late-trial velocity suppression (W*0.3, P squeeze, vel lock)
%     - Tighter mean-trajectory blending (30% position only)
%     - Early-step Q scaling (1.5x for first 5 steps)
%     - Per-angle expected trial length for progress tracking

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
        % --- Classify angle via cosine similarity ---
        mean_rates = modelParameters.mean_rates;
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
    expected_steps = (modelParameters.meanTrialLength(best_k) - 320) / bin_width;
    progress       = step_count / expected_steps;      % 0..1+
    late_trial     = progress > 0.7;

    %% ---- 4. Get active model with temporal modifications ----
    A_act = newModelParameters.active_A;
    C_act = newModelParameters.active_C;
    Q_act = newModelParameters.active_Q;
    W_act = newModelParameters.active_W;

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

    %% ---- 5. Kalman predict ----
    z_pred = A_act * kalman_z;                        % 4x1
    P_pred = A_act * kalman_P * A_act' + W_eff;       % 4x4

    %% ---- 6. Kalman update ----
    innovation = sr - C_act * z_pred;                  % 98x1
    S = C_act * P_pred * C_act' + Q_eff;               % 98x98
    K = P_pred * C_act' / S;                            % 4x98
    kalman_z = z_pred + K * innovation;                % 4x1
    kalman_P = (eye(4) - K * C_act) * P_pred;          % 4x4

    %% ---- 7. Late-trial: squeeze velocity uncertainty in P ----
    if late_trial
        kalman_P(3:4, :) = 0.5 * kalman_P(3:4, :);
        kalman_P(:, 3:4) = 0.5 * kalman_P(:, 3:4);
    end

    %% ---- 8. Velocity lock ----
    if vel_locked
        % Already locked — force zero velocity
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

    %% ---- 9. Mean trajectory blending (70/30, position only) ----
    mean_T_k  = modelParameters.mean_T(best_k);
    frac      = max(0, min(1, t / mean_T_k));
    n_tb      = modelParameters.n_traj_bins;
    traj_bin  = max(1, min(n_tb, round(frac * n_tb)));
    mean_pos  = modelParameters.mean_traj(:, traj_bin, best_k);
    kalman_z(1:2) = 0.7 * kalman_z(1:2) + 0.3 * mean_pos;

    %% ---- 10. Clamp ----
    kalman_z(3) = max(-5, min(5, kalman_z(3)));
    kalman_z(4) = max(-5, min(5, kalman_z(4)));
    kalman_z(1) = max(-200, min(200, kalman_z(1)));
    kalman_z(2) = max(-200, min(200, kalman_z(2)));

    %% ---- 11. Output ----
    x = kalman_z(1);
    y = kalman_z(2);

    % Persist all state for next call
    newModelParameters.kalman_z      = kalman_z;
    newModelParameters.kalman_P      = kalman_P;
    newModelParameters.step_count    = step_count;
    newModelParameters.low_vel_count = low_vel_count;
    newModelParameters.vel_locked    = vel_locked;
end
