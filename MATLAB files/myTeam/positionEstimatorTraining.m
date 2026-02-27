function modelParameters = positionEstimatorTraining(training_data)
% Train a Kalman Filter decoder for hand position.
%
%   State:       z = [x; y; vx; vy]       (4x1)
%   Observation: r = smoothed firing rates (98x1, Hz)
%
%   Models:  z_t = A*z_{t-1} + w,   w ~ N(0, W)
%            r_t = C*z_t     + q,   q ~ N(0, Q)

    n_neurons = 98;
    n_angles  = 8;
    n_trials  = size(training_data, 1);
    bin_width = 20;    % ms per bin
    alpha     = 0.7;   % exponential smoothing weight for new data

    fprintf('Training Kalman Filter decoder...\n');

    %% ---- 1. Extract states and observations from every trial ----
    % Pre-allocate per-angle storage
    max_per_angle = n_trials * 50;  % generous upper bound per angle

    % Za{k}: 4 x N_k   states for angle k
    % Ra{k}: 98 x N_k   smoothed rates for angle k
    % Zca/Zpa{k}: consecutive state pairs for transition model
    Za  = cell(1, n_angles);   Ra  = cell(1, n_angles);
    Zca = cell(1, n_angles);   Zpa = cell(1, n_angles);
    cnt   = zeros(1, n_angles);   % sample count per angle
    cnt_t = zeros(1, n_angles);   % transition pair count per angle

    for k = 1:n_angles
        Za{k}  = zeros(4, max_per_angle);
        Ra{k}  = zeros(n_neurons, max_per_angle);
        Zca{k} = zeros(4, max_per_angle);
        Zpa{k} = zeros(4, max_per_angle);
    end

    for k = 1:n_angles
        for tr = 1:n_trials
            spk = training_data(tr, k).spikes;   % 98 x T
            pos = training_data(tr, k).handPos(1:2, :);  % 2 x T
            T   = size(spk, 2);
            n_bins = floor(T / bin_width);
            if n_bins < 3, continue; end

            % -- Bin spikes and apply causal exponential smoothing --
            rates = zeros(n_neurons, n_bins);
            bin_centers = zeros(1, n_bins);
            for b = 1:n_bins
                bs = (b - 1) * bin_width + 1;
                be = b * bin_width;
                bin_centers(b) = round((bs + be) / 2);
                raw_rate = sum(spk(:, bs:be), 2) / (bin_width / 1000);  % Hz
                if b == 1
                    rates(:, b) = raw_rate;
                else
                    rates(:, b) = alpha * raw_rate + (1 - alpha) * rates(:, b - 1);
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
            vel_bins(:, 1) = pos_bins(:, 2) - pos_bins(:, 1);         % forward
            for b = 2:n_bins - 1
                vel_bins(:, b) = (pos_bins(:, b+1) - pos_bins(:, b-1)) / 2;
            end
            vel_bins(:, n_bins) = pos_bins(:, n_bins) - pos_bins(:, n_bins - 1); % backward

            % -- State vector --
            states = [pos_bins; vel_bins];  % 4 x n_bins

            % -- Store samples --
            idx = cnt(k) + (1:n_bins);
            Za{k}(:, idx) = states;
            Ra{k}(:, idx) = rates;
            cnt(k) = cnt(k) + n_bins;

            % -- Store consecutive pairs for transition model --
            n_pairs = n_bins - 1;
            idx_t = cnt_t(k) + (1:n_pairs);
            Zca{k}(:, idx_t) = states(:, 2:end);
            Zpa{k}(:, idx_t) = states(:, 1:end - 1);
            cnt_t(k) = cnt_t(k) + n_pairs;
        end
        fprintf('  Angle %d/%d: %d samples, %d transition pairs\n', ...
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
    Z_all  = [Za{:}];     % 4  x N_total
    R_all  = [Ra{:}];     % 98 x N_total
    Z_curr = [Zca{:}];    % 4  x M_total
    Z_prev = [Zpa{:}];    % 4  x M_total

    N_total = size(Z_all, 2);
    M_total = size(Z_curr, 2);
    fprintf('  Combined: %d total samples, %d transition pairs\n', N_total, M_total);

    %% ---- 4. Fit combined observation model: r = C*z + noise ----
    lambda = 1e-6;  % Tikhonov regularisation
    C = (R_all * Z_all') / (Z_all * Z_all' + lambda * eye(4));   % 98 x 4

    % Observation noise covariance
    Q = cov((R_all - C * Z_all)');                                 % 98 x 98
    Q = Q + 0.01 * mean(diag(Q)) * eye(n_neurons);                % regularise

    fprintf('  Combined observation model fitted (C: %dx%d)\n', size(C));

    %% ---- 5. Fit combined state transition: z_t = A*z_{t-1} + noise ----
    A = (Z_curr * Z_prev') / (Z_prev * Z_prev' + lambda * eye(4)); % 4 x 4

    W = cov((Z_curr - A * Z_prev)');                                % 4 x 4
    W = W + 0.001 * mean(diag(W)) * eye(4);                        % regularise

    fprintf('  Combined transition model fitted (A: %dx%d)\n', size(A));

    %% ---- 6. Per-angle models (same procedure, angle-specific data) ----
    for k = 1:n_angles
        Ck = (Ra{k} * Za{k}') / (Za{k} * Za{k}' + lambda * eye(4));
        Qk = cov((Ra{k} - Ck * Za{k})');
        Qk = Qk + 0.01 * mean(diag(Qk)) * eye(n_neurons);

        Ak = (Zca{k} * Zpa{k}') / (Zpa{k} * Zpa{k}' + lambda * eye(4));
        Wk = cov((Zca{k} - Ak * Zpa{k})');
        Wk = Wk + 0.001 * mean(diag(Wk)) * eye(4);

        perAngle(k).A = Ak;
        perAngle(k).C = Ck;
        perAngle(k).Q = Qk;
        perAngle(k).W = Wk;
    end
    fprintf('  Per-angle models fitted\n');

    %% ---- 7. Initial state covariance ----
    P0 = cov(Z_all') * 0.1;   % 4 x 4, scaled down for tighter init

    %% ---- 8. Mean firing rate per angle (for angle classification) ----
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
    modelParameters.perAngle   = perAngle;
    modelParameters.mean_rates = mean_rates;
    modelParameters.bin_width  = bin_width;
    modelParameters.alpha      = alpha;

    % Kalman runtime state (initialised on first prediction call)
    modelParameters.kalman_z     = [];
    modelParameters.kalman_P     = [];
    modelParameters.smooth_rates = [];
    modelParameters.current_angle = 0;
    % Active model matrices (set per-trial in positionEstimator)
    modelParameters.active_A = A;
    modelParameters.active_C = C;
    modelParameters.active_Q = Q;
    modelParameters.active_W = W;

    fprintf('Training complete.\n');
end
