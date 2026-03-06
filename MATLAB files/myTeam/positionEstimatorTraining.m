function modelParameters = positionEstimatorTraining(training_data)
% Train: LDA classifier + kNN position estimator + trajectory templates.

n_neurons = size(training_data(1,1).spikes, 1);
n_angles  = size(training_data, 2);
n_trials  = size(training_data, 1);
bin_width = 20;  obs_window = 80;  first_decode = 320;  n_pca = 25;

%% ==== 1. LDA Classifier (4-window EMA, top 55 neurons) ====
ema_alpha = 0.92;
win_edges = [1 80; 81 160; 161 240; 241 320];
n_wins    = size(win_edges, 1);
n_samples = n_trials * n_angles;

X_raw = zeros(n_neurons * n_wins, n_samples);
y_lab = zeros(1, n_samples);
for k = 1:n_angles
    for tr = 1:n_trials
        j = (k-1)*n_trials + tr;
        spk = training_data(tr,k).spikes;
        X_raw(:,j) = ema_features(spk(:,1:min(320,size(spk,2))), ema_alpha, win_edges, n_neurons);
        y_lab(j) = k;
    end
end

% Select top 55 neurons by directional modulation
scores_n = zeros(n_neurons, 1);
for i = 1:n_neurons
    ri = (0:n_wins-1)*n_neurons + i;
    pam = zeros(n_wins, n_angles);
    for k = 1:n_angles, pam(:,k) = mean(X_raw(ri, y_lab==k), 2); end
    tmp = X_raw(ri,:);
    if mean(tmp(:)) > 0.5, scores_n(i) = mean(var(pam,0,2)); end
end
[~, ord] = sort(scores_n, 'descend');
n_keep = min(55, n_neurons);
sel_neu = sort(ord(1:n_keep));

sel_r = zeros(1, n_keep*n_wins);
for j = 1:n_keep
    sel_r((j-1)*n_wins+(1:n_wins)) = (0:n_wins-1)*n_neurons + sel_neu(j);
end
Xs = X_raw(sel_r,:);

% Z-score + LDA
fmu = mean(Xs,2);  fstd = std(Xs,0,2);  fstd(fstd<1e-6) = 1;
Xz = (Xs - fmu) ./ fstd;
nf = size(Xz,1);

cmu = zeros(nf, n_angles);  cpr = zeros(1, n_angles);
for k = 1:n_angles
    Xk = Xz(:, y_lab==k);
    cmu(:,k) = mean(Xk,2);
    cpr(k) = size(Xk,2)/n_samples;
end

cen = zeros(size(Xz));
for j = 1:n_samples, cen(:,j) = Xz(:,j) - cmu(:, y_lab(j)); end
Sw = (cen*cen') / max(1, n_samples-n_angles);
Sw = 0.70*Sw + 0.30*diag(diag(Sw)) + 1e-3*trace(Sw)/nf*eye(nf);
Sw_inv = pinv(Sw);

%% ==== 2. PCA + kNN database (single pass over data) ====
% First pass: collect all spike counts for PCA
max_total = n_trials * n_angles * 80;
all_obs = zeros(n_neurons, max_total);
obs_cnt = 0;

% Count samples per angle for preallocation
samp_per_angle = zeros(1, n_angles);
for k = 1:n_angles
    for tr = 1:n_trials
        T = size(training_data(tr,k).spikes, 2);
        samp_per_angle(k) = samp_per_angle(k) + numel(first_decode:bin_width:T);
    end
end

for k = 1:n_angles
    for tr = 1:n_trials
        spk = training_data(tr,k).spikes;
        T = size(spk,2);
        for tt = first_decode:bin_width:T
            ws = max(1, tt - obs_window + 1);
            obs_cnt = obs_cnt + 1;
            all_obs(:, obs_cnt) = sum(spk(:,ws:tt), 2);
        end
    end
end
all_obs = all_obs(:, 1:obs_cnt);

mean_obs = mean(all_obs, 2);
[U,~,~] = svd(all_obs - mean_obs, 'econ');
PC = U(:, 1:n_pca);
pc_off = PC' * mean_obs;

% Second pass: build kNN database in PCA space
knn_feat = cell(1, n_angles);
knn_pos  = cell(1, n_angles);
for k = 1:n_angles
    nk = samp_per_angle(k);
    F = zeros(n_pca, nk);
    P = zeros(2, nk);
    idx = 0;
    for tr = 1:n_trials
        spk  = training_data(tr,k).spikes;
        hand = training_data(tr,k).handPos(1:2,:);
        T = size(spk,2);
        for tt = first_decode:bin_width:T
            ws = max(1, tt - obs_window + 1);
            idx = idx + 1;
            F(:,idx) = PC' * sum(spk(:,ws:tt), 2) - pc_off;
            P(:,idx) = hand(:,tt);
        end
    end
    knn_feat{k} = F;
    knn_pos{k}  = P;
end

%% ==== 3. Trajectory templates ====
n_tb = 50;
meanTraj = zeros(2, n_tb, n_angles);
mean_T = zeros(1, n_angles);
for k = 1:n_angles
    ts = zeros(2, n_tb);  Tsum = 0;  nv = 0;
    for tr = 1:n_trials
        pos = training_data(tr,k).handPos(1:2,:);
        T = size(pos,2);  Tsum = Tsum + T;
        ds = min(first_decode, T);
        if T - ds < bin_width, continue; end
        nt = linspace(ds, T, n_tb);
        ts(1,:) = ts(1,:) + interp1(ds:T, pos(1,ds:T), nt);
        ts(2,:) = ts(2,:) + interp1(ds:T, pos(2,ds:T), nt);
        nv = nv + 1;
    end
    if nv > 0, meanTraj(:,:,k) = ts / nv; end
    mean_T(k) = Tsum / n_trials;
end

%% ==== Store ====
modelParameters.clf_cmu = cmu;          % nf x 8
modelParameters.clf_Sinv = Sw_inv;      % nf x nf
modelParameters.clf_prior = cpr;        % 1 x 8
modelParameters.clf_fmu = fmu;          % nf x 1
modelParameters.clf_fstd = fstd;        % nf x 1
modelParameters.clf_sel = sel_neu;      % 1 x n_keep
modelParameters.clf_we = win_edges;     % n_wins x 2
modelParameters.clf_alpha = ema_alpha;
modelParameters.clf_nwins = n_wins;

modelParameters.PC = PC;
modelParameters.pc_off = pc_off;
modelParameters.knn_feat = knn_feat;
modelParameters.knn_pos  = knn_pos;

modelParameters.meanTraj = meanTraj;
modelParameters.mean_T   = mean_T;
modelParameters.n_tb     = n_tb;

modelParameters.bw = bin_width;
modelParameters.ow = obs_window;
modelParameters.fd = first_decode;
modelParameters.na = n_angles;
modelParameters.nn = n_neurons;

% Runtime (set on first decode call)
modelParameters.angle = 0;
modelParameters.sw = zeros(1, n_angles);
modelParameters.toff = zeros(2, n_angles);
end


function feat = ema_features(spikes, alpha, we, nn)
    T = size(spikes,2);  nw = size(we,1);
    feat = zeros(nn*nw, 1);
    if T == 0, return; end
    r = zeros(nn, T);  r(:,1) = spikes(:,1);
    for t = 2:T, r(:,t) = alpha*r(:,t-1) + (1-alpha)*spikes(:,t); end
    for w = 1:nw
        a = we(w,1); b = min(we(w,2), T);
        if a > T, m = zeros(nn,1); else, m = mean(r(:,a:b),2)*1000; end
        feat((w-1)*nn+(1:nn)) = m;
    end
end
