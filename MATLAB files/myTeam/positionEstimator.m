function [x, y, newModelParameters] = positionEstimator(test_data, modelParameters)
% Decode hand position using a Kalman Filter.
%
%   [x, y, newModelParameters] = positionEstimator(test_data, modelParameters)
%
%   Returns 3 outputs so the test harness persists Kalman state between calls.
%   On each new trial (decodedHandPos is empty), state is re-initialised.

    newModelParameters = modelParameters;

    bin_width = modelParameters.bin_width;    % 20 ms
    alph      = modelParameters.alpha;        % 0.7
    n_neurons = 98;
    t         = size(test_data.spikes, 2);    % current time in ms
    is_first  = isempty(test_data.decodedHandPos);

    %% ---- 1. Compute smoothed firing rates ----
    if is_first
        % First call for this trial: process ALL bins from trial start
        n_bins = floor(t / bin_width);
        sr = zeros(n_neurons, 1);
        for b = 1:n_bins
            bs = (b - 1) * bin_width + 1;
            be = b * bin_width;
            raw_rate = sum(test_data.spikes(:, bs:be), 2) / (bin_width / 1000); % Hz
            if b == 1
                sr = raw_rate;
            else
                sr = alph * raw_rate + (1 - alph) * sr;
            end
        end
    else
        % Subsequent call: update with just the new 20ms bin
        new_spikes = sum(test_data.spikes(:, t - bin_width + 1 : t), 2);
        raw_rate   = new_spikes / (bin_width / 1000);  % Hz
        sr = alph * raw_rate + (1 - alph) * modelParameters.smooth_rates;
    end
    newModelParameters.smooth_rates = sr;

    %% ---- 2. Initialise or restore Kalman state ----
    if is_first
        % --- Classify reaching angle via correlation with training means ---
        mean_rates = modelParameters.mean_rates;   % 98 x 8
        corrs = zeros(1, 8);
        for k = 1:8
            c = corrcoef(sr, mean_rates(:, k));
            if ~isnan(c(1, 2))
                corrs(k) = c(1, 2);
            end
        end
        [max_corr, best_k] = max(corrs);
        newModelParameters.current_angle = best_k;

        % --- Select per-angle model if confident, else combined ---
        if max_corr > 0.3
            newModelParameters.active_A = modelParameters.perAngle(best_k).A;
            newModelParameters.active_C = modelParameters.perAngle(best_k).C;
            newModelParameters.active_Q = modelParameters.perAngle(best_k).Q;
            newModelParameters.active_W = modelParameters.perAngle(best_k).W;
        else
            newModelParameters.active_A = modelParameters.A;
            newModelParameters.active_C = modelParameters.C;
            newModelParameters.active_Q = modelParameters.Q;
            newModelParameters.active_W = modelParameters.W;
        end

        % --- Initial state: at start position, zero velocity ---
        kalman_z = [test_data.startHandPos(1:2); 0; 0];
        kalman_P = modelParameters.P0;
    else
        kalman_z = modelParameters.kalman_z;
        kalman_P = modelParameters.kalman_P;
    end

    % Active model matrices for this trial
    A_act = newModelParameters.active_A;
    C_act = newModelParameters.active_C;
    Q_act = newModelParameters.active_Q;
    W_act = newModelParameters.active_W;

    %% ---- 3. Kalman predict ----
    z_pred = A_act * kalman_z;                          % 4x1
    P_pred = A_act * kalman_P * A_act' + W_act;         % 4x4

    %% ---- 4. Kalman update ----
    innovation = sr - C_act * z_pred;                    % 98x1
    S = C_act * P_pred * C_act' + Q_act;                 % 98x98
    K = P_pred * C_act' / S;                              % 4x98  (mrdivide)
    kalman_z = z_pred + K * innovation;                  % 4x1
    kalman_P = (eye(4) - K * C_act) * P_pred;            % 4x4

    %% ---- 5. Clamp to prevent drift ----
    kalman_z(3) = max(-5, min(5, kalman_z(3)));   % vx  (mm/bin)
    kalman_z(4) = max(-5, min(5, kalman_z(4)));   % vy
    kalman_z(1) = max(-200, min(200, kalman_z(1)));  % x  (mm)
    kalman_z(2) = max(-200, min(200, kalman_z(2)));  % y

    %% ---- 6. Output ----
    x = kalman_z(1);
    y = kalman_z(2);

    % Persist Kalman state for next call
    newModelParameters.kalman_z = kalman_z;
    newModelParameters.kalman_P = kalman_P;
end
