%% run_test.m
% One-click test. Run from the 'MATLAB files' folder.

clear; close all; clc;

% Create monkeydata0.mat from training data if it doesn't exist
if ~exist('monkeydata0.mat', 'file')
    fprintf('Creating monkeydata0.mat from monkeydata_training.mat...\n');
    load('monkeydata_training.mat', 'trial');
    save('monkeydata0.mat', 'trial');
end

% Run the competition test harness
RMSE = testFunction_for_students_MTb('myTeam');
fprintf('\n=== Final RMSE: %.2f mm ===\n', RMSE);
