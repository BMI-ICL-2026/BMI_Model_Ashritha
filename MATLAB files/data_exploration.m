%% data_exploration.m
% Exploratory analysis of monkey BMI neural recording data.
% Produces population rasters, single-neuron rasters, PSTHs,
% hand trajectories, and tuning curves.

clear; close all; clc;

%% 1. Load data
load('MATLAB files/monkeydata_training.mat', 'trial');
% trial: 100 trials x 8 reaching angles
% Fields: trialId, spikes (98 neurons x time_ms), handPos (3 x time_ms)

angles_deg = [30, 70, 110, 150, 190, 230, 310, 350];
angles_rad = angles_deg * pi / 180;
n_angles = length(angles_rad);
n_trials = size(trial, 1);
n_neurons = size(trial(1,1).spikes, 1);

fprintf('Data loaded: %d trials x %d angles, %d neurons\n', ...
    n_trials, n_angles, n_neurons);

%% 2. Population raster plot for trial(1,1)
spikes = trial(1,1).spikes;  % 98 x T
[neuron_idx, time_idx] = find(spikes);

fig1 = figure('Position', [100 100 900 500]);
plot(time_idx, neuron_idx, 'k.', 'MarkerSize', 2);
xlabel('Time (ms)');
ylabel('Neuron');
title(sprintf('Population raster — Trial 1, Angle %.0f°', angles_deg(1)));
ylim([0.5, n_neurons + 0.5]);
xlim([0, size(spikes, 2)]);
set(gca, 'TickDir', 'out', 'Box', 'off');

%% 3. Raster for neuron 1 across all 100 trials of angle 1
fig2 = figure('Position', [100 100 900 600]);
cmap = parula(n_trials);

for tr = 1:n_trials
    spike_times = find(trial(tr, 1).spikes(1, :));
    if ~isempty(spike_times)
        y_vals = tr * ones(size(spike_times));
        scatter(spike_times, y_vals, 4, cmap(tr, :), 'filled');
        hold on;
    end
end

xlabel('Time (ms)');
ylabel('Trial');
title(sprintf('Neuron 1 raster — Angle %.0f°, all %d trials', ...
    angles_deg(1), n_trials));
ylim([0.5, n_trials + 0.5]);
set(gca, 'TickDir', 'out', 'Box', 'off');
colormap(parula);
cb = colorbar;
cb.Label.String = 'Trial number';
caxis([1, n_trials]);

%% 4. PSTH for neuron 1, angle 1
bin_width = 20;  % ms

% Find the maximum trial length across all trials for angle 1
max_T = 0;
for tr = 1:n_trials
    T = size(trial(tr, 1).spikes, 2);
    if T > max_T
        max_T = T;
    end
end

bin_edges = 0:bin_width:max_T;
n_bins = length(bin_edges) - 1;
bin_centres = bin_edges(1:end-1) + bin_width / 2;

% Accumulate spike counts across trials
spike_counts = zeros(n_trials, n_bins);
for tr = 1:n_trials
    spike_train = trial(tr, 1).spikes(1, :);
    T = length(spike_train);
    for b = 1:n_bins
        t_start = bin_edges(b) + 1;
        t_end = min(bin_edges(b + 1), T);
        if t_start <= T
            spike_counts(tr, b) = sum(spike_train(t_start:t_end));
        end
    end
end

% Convert to firing rate (spikes/s)
firing_rates = (spike_counts / bin_width) * 1000;  % bin_width in ms -> *1000 for Hz
mean_rate = mean(firing_rates, 1);

% Gaussian smoothing kernel (sigma = 1 bin)
sigma_bins = 1.5;
kernel_width = ceil(3 * sigma_bins) * 2 + 1;
x_kern = -(kernel_width - 1)/2 : (kernel_width - 1)/2;
gauss_kernel = exp(-x_kern.^2 / (2 * sigma_bins^2));
gauss_kernel = gauss_kernel / sum(gauss_kernel);

smoothed_rate = conv(mean_rate, gauss_kernel, 'same');

fig3 = figure('Position', [100 100 800 400]);
bar(bin_centres, mean_rate, 1, 'FaceColor', [0.7 0.7 0.7], ...
    'EdgeColor', 'none', 'FaceAlpha', 0.5);
hold on;
plot(bin_centres, smoothed_rate, 'r-', 'LineWidth', 2);
xlabel('Time (ms)');
ylabel('Firing rate (Hz)');
title(sprintf('PSTH — Neuron 1, Angle %.0f° (bin = %d ms)', ...
    angles_deg(1), bin_width));
legend('Raw binned rate', 'Gaussian smoothed', 'Location', 'best');
set(gca, 'TickDir', 'out', 'Box', 'off');
xlim([0, max_T]);

%% 5. Hand trajectories for all 8 angles (trial 1 of each)
fig4 = figure('Position', [100 100 700 700]);
colours = lines(n_angles);

for a = 1:n_angles
    pos = trial(1, a).handPos;  % 3 x T
    x = pos(1, :);
    y = pos(2, :);
    plot(x, y, '-', 'Color', colours(a, :), 'LineWidth', 1.5);
    hold on;
    % Mark start position
    plot(x(1), y(1), 'o', 'Color', colours(a, :), ...
        'MarkerFaceColor', colours(a, :), 'MarkerSize', 6);
    % Mark end position
    plot(x(end), y(end), 's', 'Color', colours(a, :), ...
        'MarkerFaceColor', colours(a, :), 'MarkerSize', 8);
end

xlabel('X position (mm)');
ylabel('Y position (mm)');
title('Hand trajectories — Trial 1, all angles');
legend_labels = arrayfun(@(d) sprintf('%d°', d), angles_deg, ...
    'UniformOutput', false);
legend(legend_labels{:}, 'Location', 'bestoutside');
axis equal;
set(gca, 'TickDir', 'out', 'Box', 'off');
grid on;

%% 6. Tuning curves for neurons 1, 10, 20, 50
neurons_of_interest = [1, 10, 20, 50];

% Compute mean firing rate per neuron, per angle, per trial
% (total spikes / trial duration in seconds)
mean_fr = zeros(n_neurons, n_angles, n_trials);

for a = 1:n_angles
    for tr = 1:n_trials
        spk = trial(tr, a).spikes;  % 98 x T
        T = size(spk, 2);
        duration_s = T / 1000;
        mean_fr(:, a, tr) = sum(spk, 2) / duration_s;  % Hz
    end
end

fig5 = figure('Position', [100 100 900 600]);
colours_tc = lines(length(neurons_of_interest));

for i = 1:length(neurons_of_interest)
    n_id = neurons_of_interest(i);

    % Mean and std across trials for each angle
    fr_across_trials = squeeze(mean_fr(n_id, :, :));  % n_angles x n_trials
    mu = mean(fr_across_trials, 2);   % n_angles x 1
    sd = std(fr_across_trials, 0, 2); % n_angles x 1

    errorbar(angles_deg, mu, sd, '-o', ...
        'Color', colours_tc(i, :), ...
        'LineWidth', 1.5, ...
        'MarkerFaceColor', colours_tc(i, :), ...
        'MarkerSize', 6, ...
        'CapSize', 8);
    hold on;
end

xlabel('Reaching angle (degrees)');
ylabel('Mean firing rate (Hz)');
title('Tuning curves');
legend(arrayfun(@(n) sprintf('Neuron %d', n), neurons_of_interest, ...
    'UniformOutput', false), 'Location', 'best');
set(gca, 'XTick', angles_deg, 'TickDir', 'out', 'Box', 'off');
xlim([angles_deg(1) - 15, angles_deg(end) + 15]);
grid on;

%% 7. Save all figures as PNG
saveas(fig1, 'population_raster.png');
saveas(fig2, 'neuron1_raster.png');
saveas(fig3, 'neuron1_psth.png');
saveas(fig4, 'hand_trajectories.png');
saveas(fig5, 'tuning_curves.png');

fprintf('All figures saved as PNG.\n');
