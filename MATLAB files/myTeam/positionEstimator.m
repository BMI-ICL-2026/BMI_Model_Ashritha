function [x, y, newModelParameters] = positionEstimator(test_data, modelParameters)
% Decode: LDA classify -> kNN position + soft template blend.

    newModelParameters = modelParameters;
    nn = modelParameters.nn;
    na = modelParameters.na;
    t  = size(test_data.spikes, 2);

    %% ---- First call: classify angle ----
    if isempty(test_data.decodedHandPos)
        spk = test_data.spikes(:, 1:min(320,t));
        feat = ema_features(spk, modelParameters.clf_alpha, ...
            modelParameters.clf_we, nn);

        nk = numel(modelParameters.clf_sel);
        nw = modelParameters.clf_nwins;
        sr = zeros(1, nk*nw);
        for j = 1:nk
            u = modelParameters.clf_sel(j);
            sr((j-1)*nw+(1:nw)) = (0:nw-1)*nn + u;
        end
        fz = (feat(sr) - modelParameters.clf_fmu) ./ modelParameters.clf_fstd;

        % LDA scores + softmax
        scores = zeros(1, na);
        for k = 1:na
            d = fz - modelParameters.clf_cmu(:,k);
            scores(k) = -0.5*(d'*modelParameters.clf_Sinv*d) + log(modelParameters.clf_prior(k)+eps);
        end
        [~, best_k] = max(scores);
        sw = exp(scores - max(scores));
        sw = sw / sum(sw);

        off = zeros(2, na);
        for k = 1:na
            off(:,k) = test_data.startHandPos(1:2) - modelParameters.meanTraj(:,1,k);
        end

        newModelParameters.angle = best_k;
        newModelParameters.sw    = sw;
        newModelParameters.toff  = off;
    end

    %% ---- kNN position estimate ----
    best_k = newModelParameters.angle;
    ws = max(1, t - modelParameters.ow + 1);
    obs = modelParameters.PC' * sum(test_data.spikes(:,ws:t), 2) - modelParameters.pc_off;

    F = modelParameters.knn_feat{best_k};
    dists = sum((F - obs).^2, 1);
    K = min(15, size(F,2));
    [sd, si] = sort(dists);
    w = 1 ./ (sd(1:K) + 1e-6);
    w = w / sum(w);
    knn_pos = modelParameters.knn_pos{best_k}(:, si(1:K)) * w';

    %% ---- Soft template position ----
    ntb = modelParameters.n_tb;
    sw  = newModelParameters.sw;
    off = newModelParameters.toff;
    tpl = zeros(2,1);
    for k = 1:na
        dp = max(0, min(1, (t-320)/max(1, modelParameters.mean_T(k)-320)));
        tb = max(1, min(ntb, round(dp*ntb)));
        tpl = tpl + sw(k) * (modelParameters.meanTraj(:,tb,k) + off(:,k));
    end

    %% ---- Blend + output ----
    pos = 0.55 * knn_pos + 0.45 * tpl;
    x = pos(1);
    y = pos(2);
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
