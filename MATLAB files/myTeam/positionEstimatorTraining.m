function modelParameters = positionEstimatorTraining(training_data)
% Train a Kalman Filter decoder for hand position (v2).
%
%   State:       z = [x; y; vx; vy]       (4x1)
%   Observation: r = smoothed firing rates (98x1, Hz)
%
%   Models:  z_t = A*z_{t-1} + w,   w ~ N(0, W)
%            r_t = C*z_t     + q,   q ~ N(0, Q)
%
%   Improvements over v1:
%     - 100ms causal spike window (not 20ms bin)
%     - alpha=0.5 exponential smoothing
%     - Low-firing neuron pruning (< 1 Hz)
%     - Velocity damping A(3,3)=A(4,4)=0.85
%     - Mean trajectory prior per angle (30 resampled bins)
%     - Mean velocity at t=320 for initialisation
%     - Cosine similarity angle classification

    n_neurons    = 98;
    n_angles     = 8;
    n_trials     = size(training_data, 1);
    bin_width    = 20;     % ms per time step
    alpha        = 0.5;    % exponential smoothing weight for new data
    spike_window = 100;    % ms causal window for firing rate
    vel_damp     = 0.85;   % velocity decay per step
    n_traj_bins  = 30;     % bins for mean trajectory resampling

    fprintf('Training Kalman Filter decoder (v2)...\n');

    %% ---- 1. Extract states and observations from every trial ----
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
            spk = training_data(tr, k).spikes;          % 98 x T
            pos = training_data(tr, k).handPos(1:2, :);  % 2 x T
            T   = size(spk, 2);
            n_bins = floor(T / bin_width);
            if n_bins < 3, continue; end

            % -- Firing rate: 100ms causal window + exponential smoothing --
            rates = zeros(n_neurons, n_bins);
            bin_centers = zeros(1, n_bins);
            for b = 1:n_bins
                be = b * bin_width;                           % bin end (ms)
                ws = max(1, be - spike_window + 1);           % window start
                window_s = (be - ws + 1) / 1000;             % window duration (s)
                bin_centers(b) = round(((b-1)*bin_width + 1 + be) / 2);
                raw_rate = sum(spk(:, ws:be), 2) / window_s; % Hz
                if b == 1
                    rates(:, b) = raw_rate;
                else
                    rates(:, b) = alpha * raw_rate + (1 - alpha) * rates(:, b-1);
                end
            end

            % -- Position at each bin center --
            pos_bins = zeros(2, n_bins);
            for b = 1:n_bins
                tc = min(bin_centers(b), size(pos, 2));
                pos_bins(:, b) = pos(:, tc);
            end

            % -- Velocity via central difference (mm/bin) --
            vel_bins = zeros(2, n_bins);
            vel_bins(:, 1) = pos_bins(:, 2) - pos_bins(:, 1);
            for b = 2:n_bins - 1
                vel_bins(:, b) = (pos_bins(:, b+1) - pos_bins(:, b-1)) / 2;
            end
            vel_bins(:, n_bins) = pos_bins(:, n_bins) - pos_bins(:, n_bins-1);

            states = [pos_bins; vel_bins];   % 4 x n_bins

            % -- Store per-angle --
            idx = cnt(k) + (1:n_bins);
            Za{k}(:, idx) = states;
            Ra{k}(:, idx) = rates;
            cnt(k) = cnt(k) + n_bins;

            n_pairs = n_bins - 1;
            idx_t = cnt_t(k) + (1:n_pairs);
            Zca{k}(:, idx_t) = states(:, 2:end);
            Zpa{k}(:, idx_t) = states(:, 1:end-1);
            cnt_t(k) = cnt_t(k) + n_pairs;
        end
        fprintf('  Angle %d/%d: %d samples, %d transitions\n', ...
            k, n_angles, cnt(k), cnt_t(k));
    end

    % Trim to actual size
    for k = 1:n_angles
        Za{k}  = Za{k}(:, 1:cnt(k));
        Ra{k}  = Ra{k}(:, 1:cnt(k));
        Zca{k} = Zca{k}(:, 1:cnt_t(k));
        Zpa{k} = Zpa{k}(:, 1:cnt_t(k));
    end

    % Combine across all angles
    Z_all  = [Za{:}];      % 4  x N_total
    R_all  = [Ra{:}];      % 98 x N_total
    Z_curr = [Zca{:}];     % 4  x M_total
    Z_prev = [Zpa{:}];     % 4  x M_total
    fprintf('  Combined: %d samples, %d transitions\n', ...
        size(Z_all, 2), size(Z_curr, 2));

    %% ---- 2. Identify low-firing neurons ----
    mean_fr_all = mean(R_all, 2);             % 98 x 1
    low_fr = mean_fr_all < 1;                 % logical mask
    fprintf('  Pruning %d neurons with mean rate < 1 Hz\n', sum(low_fr));

    %% ---- 3. Fit combined observation model ----
    lambda = 1e-6;

    C = (R_all * Z_all') / (Z_all * Z_all' + lambda * eye(4));  % 98 x 4
    C(low_fr, :) = 0;                                            % zero out noisy neurons
    Q = cov((R_all - C * Z_all)');                                % 98 x 98 recomputed
    Q = Q + 0.01 * mean(diag(Q)) * eye(n_neurons);

    %% ---- 4. Fit combined state transition with velocity damping ----
    A = (Z_curr * Z_prev') / (Z_prev * Z_prev' + lambda * eye(4));  % 4 x 4
    A(3, 3) = vel_damp;   % force velocity decay
    A(4, 4) = vel_damp;
    W = cov((Z_curr - A * Z_prev)');   % recompute W with damped A
    W = W + 0.001 * mean(diag(W)) * eye(4);

    fprintf('  Combined model fitted (vel damp = %.2f)\n', vel_damp);

    %% ---- 5. Fit per-angle models ----
    for k = 1:n_angles
        % Observation model
        Ck = (Ra{k} * Za{k}') / (Za{k} * Za{k}' + lambda * eye(4));
        Ck(low_fr, :) = 0;
        Qk = cov((Ra{k} - Ck * Za{k})');
        Qk = Qk + 0.01 * mean(diag(Qk)) * eye(n_neurons);

        % State transition with damping
        Ak = (Zca{k} * Zpa{k}') / (Zpa{k} * Zpa{k}' + lambda * eye(4));
        Ak(3, 3) = vel_damp;
        Ak(4, 4) = vel_damp;
        Wk = cov((Zca{k} - Ak * Zpa{k})');
        Wk = Wk + 0.001 * mean(diag(Wk)) * eye(4);

        perAngle(k).A = Ak;
        perAngle(k).C = Ck;
        perAngle(k).Q = Qk;
        perAngle(k).W = Wk;
    end
    fprintf('  Per-angle models fitted\n');

    %% ---- 6. Mean trajectory per angle (resampled to 30 bins) ----
    mean_traj = zeros(2, n_traj_bins, n_angles);
    mean_T    = zeros(1, n_angles);

    for k = 1:n_angles
        traj_sum = zeros(2, n_traj_bins);
        T_sum    = 0;
        for tr = 1:n_trials
            pos = training_data(tr, k).handPos(1:2, :);
            T   = size(pos, 2);
            T_sum = T_sum + T;
            % Resample to n_traj_bins points via linear interpolation
            orig_t = 1:T;
            new_t  = linspace(1, T, n_traj_bins);
            traj_sum(1, :) = traj_sum(1, :) + interp1(orig_t, pos(1, :), new_t);
            traj_sum(2, :) = traj_sum(2, :) + interp1(orig_t, pos(2, :), new_t);
        end
        mean_traj(:, :, k) = traj_sum / n_trials;
        mean_T(k) = T_sum / n_trials;
    end
    fprintf('  Mean trajectories computed (%d bins, mean T: %.0f-%.0f ms)\n', ...
        n_traj_bins, min(mean_T), max(mean_T));

    %% ---- 7. Mean velocity at t=320ms per angle (for initialisation) ----
    mean_vel_init = zeros(2, n_angles);
    for k = 1:n_angles
        vel_sum = zeros(2, 1);
        n_valid = 0;
        for tr = 1:n_trials
            pos = training_data(tr, k).handPos(1:2, :);
            T   = size(pos, 2);
            if T < 340, continue; end
            % Central difference at t=320 over 40ms span -> mm/bin
            vel_sum = vel_sum + (pos(:, 340) - pos(:, 300)) / 2;
            n_valid = n_valid + 1;
        end
        if n_valid > 0
            mean_vel_init(:, k) = vel_sum / n_valid;
        end
    end
    fprintf('  Mean initial velocities computed\n');

    %% ---- 8. Initial state covariance ----
    P0 = cov(Z_all') * 0.1;

    %% ---- 9. Mean firing rate per angle (for cosine similarity) ----
    mean_rates = zeros(n_neurons, n_angles);
    for k = 1:n_angles
        mean_rates(:, k) = mean(Ra{k}, 2);
    end

    %% ---- Store model parameters ----
    modelParameters.A  = A;
    modelParameters.C  = C;
    modelParameters.Q  = Q;
    modelParameters.W  = W;
    modelParameters.P0 = P0;
    modelParameters.perAngle      = perAngle;       % 1x8 struct
    modelParameters.mean_rates    = mean_rates;      % 98 x 8
    modelParameters.mean_traj     = mean_traj;       % 2 x 30 x 8
    modelParameters.mean_T        = mean_T;          % 1 x 8
    modelParameters.mean_vel_init = mean_vel_init;   % 2 x 8
    modelParameters.n_traj_bins   = n_traj_bins;
    modelParameters.bin_width     = bin_width;
    modelParameters.alpha         = alpha;
    modelParameters.spike_window  = spike_window;

    % Kalman runtime state (set on first prediction call per trial)
    modelParameters.kalman_z      = [];
    modelParameters.kalman_P      = [];
    modelParameters.smooth_rates  = [];
    modelParameters.current_angle = 0;
    modelParameters.active_A = A;
    modelParameters.active_C = C;
    modelParameters.active_Q = Q;
    modelParameters.active_W = W;
    modelParameters.debug_count = 0;   % for printing first 5 classifications

    fprintf('Training complete.\n');
end
