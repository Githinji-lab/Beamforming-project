%% ========================================================================
%  MATLAB Simulation: ML-Based Beamforming in 5G Smart Antenna Network
%  Compatible with MATLAB R2018a+
%  Uses Communications System Toolbox and custom channel modeling
% =========================================================================

clear all; close all; clc;

%% ========================================================================
%  SECTION 1: PARAMETER INITIALIZATION
% =========================================================================

% Antenna Array Configuration
N_tx = 8;                    % Number of transmit antennas (ULA)
lambda = 1;                  % Normalized wavelength
d = 0.5 * lambda;            % Element spacing (0.5λ)

% User Configuration
K = 3;                       % Number of users
N_rx = 1;                    % Single antenna per user

% Simulation Parameters
N_iterations = 500;          % Number of Monte Carlo iterations
SNR_dB_range = 0:5:20;       % SNR range for evaluation (dB)
N_SNR = length(SNR_dB_range);

% Channel Model Parameters
N_paths = 6;                 % Number of multipath components
angular_spread = 15;         % Angular spread in degrees
noise_power = 1;             % Normalized noise power

% ML Dataset Storage
dataset_H = cell(N_iterations, 1);      % Channel matrices
dataset_angles = zeros(N_iterations, K); % User angles
dataset_W_MRT = cell(N_iterations, 1);   % MRT weights
dataset_W_ZF = cell(N_iterations, 1);    % ZF weights
dataset_W_MMSE = cell(N_iterations, N_SNR); % MMSE weights (SNR-dependent)
dataset_SINR_MRT = zeros(N_iterations, K, N_SNR);
dataset_SINR_ZF = zeros(N_iterations, K, N_SNR);
dataset_SINR_MMSE = zeros(N_iterations, K, N_SNR);
dataset_SNR = zeros(N_iterations, N_SNR);

fprintf('=== 5G MIMO Beamforming Simulation ===\n');
fprintf('Antenna Elements: %d (ULA, %.2fλ spacing)\n', N_tx, d/lambda);
fprintf('Number of Users: %d\n', K);
fprintf('Monte Carlo Iterations: %d\n', N_iterations);
fprintf('SNR Range: %d to %d dB\n\n', min(SNR_dB_range), max(SNR_dB_range));

%% ========================================================================
%  SECTION 2: MONTE CARLO SIMULATION LOOP
% =========================================================================

% Performance metrics storage
sum_capacity_MRT = zeros(N_SNR, 1);
sum_capacity_ZF = zeros(N_SNR, 1);
sum_capacity_MMSE = zeros(N_SNR, 1);

fprintf('Running Monte Carlo simulations...\n');
tic;

for iter = 1:N_iterations
    
    if mod(iter, 100) == 0
        fprintf('  Iteration %d/%d\n', iter, N_iterations);
    end
    
    % Generate random user angles (azimuth)
    user_angles = (rand(K, 1) - 0.5) * 120; % -60° to +60°
    dataset_angles(iter, :) = user_angles;
    
    % Generate 5G channel with multipath (Rayleigh fading)
    H = generate_5G_channel(N_tx, K, user_angles, N_paths, angular_spread, d, lambda);
    dataset_H{iter} = H;
    
    % Normalize channel for consistent power
    H_normalized = H / sqrt(trace(H * H') / (N_tx * K));
    
    % Loop over SNR values
    for snr_idx = 1:N_SNR
        SNR_dB = SNR_dB_range(snr_idx);
        SNR_linear = 10^(SNR_dB/10);
        dataset_SNR(iter, snr_idx) = SNR_dB;
        
        % ===== Maximum Ratio Transmission (MRT) =====
        W_MRT = beamforming_MRT(H_normalized, N_tx);
        if iter == 1 && snr_idx == 1
            dataset_W_MRT{iter} = W_MRT;
        end
        
        [SINR_MRT, capacity_MRT] = compute_performance(H_normalized, W_MRT, SNR_linear, noise_power);
        dataset_SINR_MRT(iter, :, snr_idx) = SINR_MRT;
        sum_capacity_MRT(snr_idx) = sum_capacity_MRT(snr_idx) + capacity_MRT;
        
        % ===== Zero Forcing (ZF) =====
        W_ZF = beamforming_ZF(H_normalized, N_tx, K);
        if iter == 1 && snr_idx == 1
            dataset_W_ZF{iter} = W_ZF;
        end
        
        [SINR_ZF, capacity_ZF] = compute_performance(H_normalized, W_ZF, SNR_linear, noise_power);
        dataset_SINR_ZF(iter, :, snr_idx) = SINR_ZF;
        sum_capacity_ZF(snr_idx) = sum_capacity_ZF(snr_idx) + capacity_ZF;
        
        % ===== Minimum Mean Square Error (MMSE) =====
        W_MMSE = beamforming_MMSE(H_normalized, N_tx, K, SNR_linear);
        dataset_W_MMSE{iter, snr_idx} = W_MMSE;
        
        [SINR_MMSE, capacity_MMSE] = compute_performance(H_normalized, W_MMSE, SNR_linear, noise_power);
        dataset_SINR_MMSE(iter, :, snr_idx) = SINR_MMSE;
        sum_capacity_MMSE(snr_idx) = sum_capacity_MMSE(snr_idx) + capacity_MMSE;
    end
end

elapsed_time = toc;
fprintf('Simulation completed in %.2f seconds.\n\n', elapsed_time);

% Average performance metrics
sum_capacity_MRT = sum_capacity_MRT / N_iterations;
sum_capacity_ZF = sum_capacity_ZF / N_iterations;
sum_capacity_MMSE = sum_capacity_MMSE / N_iterations;

avg_SINR_MRT = squeeze(mean(dataset_SINR_MRT, 1));
avg_SINR_ZF = squeeze(mean(dataset_SINR_ZF, 1));
avg_SINR_MMSE = squeeze(mean(dataset_SINR_MMSE, 1));

%% ========================================================================
%  SECTION 3: PERFORMANCE VISUALIZATION
% =========================================================================

fprintf('=== Generating Performance Plots ===\n');

% Plot 1: Sum Capacity vs SNR
figure('Position', [100, 100, 1200, 400]);

subplot(1, 3, 1);
plot(SNR_dB_range, sum_capacity_MRT, 'o-', 'LineWidth', 2, 'MarkerSize', 8); hold on;
plot(SNR_dB_range, sum_capacity_ZF, 's-', 'LineWidth', 2, 'MarkerSize', 8);
plot(SNR_dB_range, sum_capacity_MMSE, '^-', 'LineWidth', 2, 'MarkerSize', 8);
grid on;
xlabel('SNR (dB)', 'FontSize', 12);
ylabel('Sum Capacity (bps/Hz)', 'FontSize', 12);
title('Sum Capacity Comparison', 'FontSize', 13, 'FontWeight', 'bold');
legend('MRT', 'ZF', 'MMSE', 'Location', 'northwest');
set(gca, 'FontSize', 11);

% Plot 2: Average SINR per User
subplot(1, 3, 2);
colors = lines(K);
for k = 1:K
    plot(SNR_dB_range, avg_SINR_MRT(k, :), 'o-', 'LineWidth', 1.5, ...
         'Color', colors(k, :), 'MarkerSize', 6); hold on;
    plot(SNR_dB_range, avg_SINR_ZF(k, :), 's--', 'LineWidth', 1.5, ...
         'Color', colors(k, :), 'MarkerSize', 6);
    plot(SNR_dB_range, avg_SINR_MMSE(k, :), '^:', 'LineWidth', 1.5, ...
         'Color', colors(k, :), 'MarkerSize', 6);
end
grid on;
xlabel('SNR (dB)', 'FontSize', 12);
ylabel('Average SINR (dB)', 'FontSize', 12);
title('Per-User SINR Comparison', 'FontSize', 13, 'FontWeight', 'bold');
legend_labels = cell(1, K*3);
for k = 1:K
    legend_labels{3*(k-1)+1} = sprintf('User %d (MRT)', k);
    legend_labels{3*(k-1)+2} = sprintf('User %d (ZF)', k);
    legend_labels{3*(k-1)+3} = sprintf('User %d (MMSE)', k);
end
legend(legend_labels, 'Location', 'best', 'FontSize', 8);
set(gca, 'FontSize', 11);

% Plot 3: Beam Patterns (Sample Iteration)
subplot(1, 3, 3);
sample_iter = 1;
sample_snr_idx = find(SNR_dB_range == 10); % 10 dB SNR
if isempty(sample_snr_idx), sample_snr_idx = ceil(N_SNR/2); end

H_sample = dataset_H{sample_iter};
angles_sample = dataset_angles(sample_iter, :);
W_sample = dataset_W_MMSE{sample_iter, sample_snr_idx};

theta_range = -90:1:90;
beam_pattern = zeros(length(theta_range), K);

for idx = 1:length(theta_range)
    a_theta = array_response(N_tx, theta_range(idx), d, lambda);
    for k = 1:K
        beam_pattern(idx, k) = abs(a_theta' * W_sample(:, k))^2;
    end
end

% Normalize and convert to dB
beam_pattern_dB = 10*log10(beam_pattern ./ max(beam_pattern(:)));

for k = 1:K
    plot(theta_range, beam_pattern_dB(:, k), 'LineWidth', 2); hold on;
end

% Mark user locations
for k = 1:K
    xline(angles_sample(k), '--', sprintf('U%d', k), 'LineWidth', 1.5, ...
          'LabelHorizontalAlignment', 'center', 'FontSize', 9);
end

grid on;
xlabel('Azimuth Angle (degrees)', 'FontSize', 12);
ylabel('Normalized Gain (dB)', 'FontSize', 12);
title('Beam Patterns (MMSE, Sample)', 'FontSize', 13, 'FontWeight', 'bold');
legend(arrayfun(@(k) sprintf('User %d', k), 1:K, 'UniformOutput', false), ...
       'Location', 'best');
ylim([-40, 0]);
set(gca, 'FontSize', 11);

%% ========================================================================
%  SECTION 4: PERFORMANCE SUMMARY TABLE
% =========================================================================

fprintf('\n=== Performance Summary at SNR = 10 dB ===\n');
snr_10dB_idx = find(SNR_dB_range == 10);
if isempty(snr_10dB_idx), snr_10dB_idx = ceil(N_SNR/2); end

fprintf('┌──────────────┬─────────────────┬─────────────────┬─────────────────┐\n');
fprintf('│   Method     │  Sum Capacity   │  Avg SINR (dB)  │  Min SINR (dB)  │\n');
fprintf('│              │    (bps/Hz)     │                 │                 │\n');
fprintf('├──────────────┼─────────────────┼─────────────────┼─────────────────┤\n');

sinr_mrt_10 = avg_SINR_MRT(:, snr_10dB_idx);
sinr_zf_10 = avg_SINR_ZF(:, snr_10dB_idx);
sinr_mmse_10 = avg_SINR_MMSE(:, snr_10dB_idx);

fprintf('│     MRT      │     %6.2f      │      %6.2f     │      %6.2f     │\n', ...
    sum_capacity_MRT(snr_10dB_idx), mean(sinr_mrt_10), min(sinr_mrt_10));
fprintf('│     ZF       │     %6.2f      │      %6.2f     │      %6.2f     │\n', ...
    sum_capacity_ZF(snr_10dB_idx), mean(sinr_zf_10), min(sinr_zf_10));
fprintf('│    MMSE      │     %6.2f      │      %6.2f     │      %6.2f     │\n', ...
    sum_capacity_MMSE(snr_10dB_idx), mean(sinr_mmse_10), min(sinr_mmse_10));
fprintf('└──────────────┴─────────────────┴─────────────────┴─────────────────┘\n\n');

%% ========================================================================
%  SECTION 5: DATASET EXPORT FOR ML TRAINING
% =========================================================================

fprintf('=== Exporting ML Training Dataset ===\n');

% Prepare structured dataset
ML_Dataset.Description = '5G MIMO Beamforming Dataset for Machine Learning';
ML_Dataset.Parameters.N_tx = N_tx;
ML_Dataset.Parameters.K = K;
ML_Dataset.Parameters.N_iterations = N_iterations;
ML_Dataset.Parameters.SNR_dB_range = SNR_dB_range;
ML_Dataset.Parameters.N_paths = N_paths;

% Features (inputs for ML model)
ML_Dataset.Features.H = dataset_H;                  % Channel matrices
ML_Dataset.Features.user_angles = dataset_angles;   % User angles
ML_Dataset.Features.SNR = dataset_SNR;              % SNR values

% Labels (targets for ML model)
ML_Dataset.Labels.W_MRT = dataset_W_MRT;            % MRT weights
ML_Dataset.Labels.W_ZF = dataset_W_ZF;              % ZF weights
ML_Dataset.Labels.W_MMSE = dataset_W_MMSE;          % MMSE weights (best baseline)

% Performance metrics
ML_Dataset.Performance.SINR_MRT = dataset_SINR_MRT;
ML_Dataset.Performance.SINR_ZF = dataset_SINR_ZF;
ML_Dataset.Performance.SINR_MMSE = dataset_SINR_MMSE;

% Find optimal beamformer index for each sample
[~, optimal_method_idx] = max(cat(3, ...
    squeeze(mean(dataset_SINR_MRT, 2)), ...
    squeeze(mean(dataset_SINR_ZF, 2)), ...
    squeeze(mean(dataset_SINR_MMSE, 2))), [], 3);

ML_Dataset.Labels.optimal_method = optimal_method_idx; % 1=MRT, 2=ZF, 3=MMSE

% Save to MAT file
save('beamforming_dataset.mat', 'ML_Dataset', '-v7.3');
fprintf('Dataset saved to: beamforming_dataset.mat\n');
fprintf('  - Channel matrices (H): %d samples\n', N_iterations);
fprintf('  - User angles: %d samples\n', N_iterations);
fprintf('  - Beamforming weights (MRT, ZF, MMSE): %d samples\n', N_iterations);
fprintf('  - Performance metrics: %d iterations x %d SNR points\n\n', N_iterations, N_SNR);

fprintf('=== Simulation Complete ===\n');
fprintf('Total execution time: %.2f seconds\n', elapsed_time);

%% ========================================================================
%  SUPPORTING FUNCTIONS
% =========================================================================

% Function: Generate 5G Channel with Multipath (Rayleigh Fading)
function H = generate_5G_channel(N_tx, K, user_angles, N_paths, angular_spread, d, lambda)
    H = zeros(K, N_tx);
    
    for k = 1:K
        % Main path (LOS-like component)
        main_angle = user_angles(k);
        a_main = array_response(N_tx, main_angle, d, lambda);
        
        % Multipath components with angular spread
        path_angles = main_angle + angular_spread * (rand(N_paths, 1) - 0.5);
        path_gains = (randn(N_paths, 1) + 1i*randn(N_paths, 1)) / sqrt(2 * N_paths);
        
        % Combine all paths
        h_k = zeros(N_tx, 1);
        for p = 1:N_paths
            a_p = array_response(N_tx, path_angles(p), d, lambda);
            h_k = h_k + path_gains(p) * a_p;
        end
        
        H(k, :) = h_k.';
    end
    
    % Normalize channel power
    H = H / sqrt(mean(abs(H(:)).^2)) * sqrt(N_tx);
end

% Function: Array Response Vector (Uniform Linear Array)
function a = array_response(N, theta_deg, d, lambda)
    theta_rad = deg2rad(theta_deg);
    n = (0:N-1).';
    a = exp(1i * 2 * pi * d / lambda * n * sin(theta_rad)) / sqrt(N);
end

% Function: MRT Beamforming
function W = beamforming_MRT(H, N_tx)
    % W = H^H (matched filter)
    W = H';
    % Normalize each column to unit power
    for k = 1:size(W, 2)
        W(:, k) = W(:, k) / norm(W(:, k)) * sqrt(N_tx);
    end
end

% Function: Zero Forcing Beamforming
function W = beamforming_ZF(H, N_tx, K)
    % W = H^H * (H * H^H)^{-1}
    if rank(H) == K
        W = H' / (H * H');
    else
        % Regularized pseudo-inverse if rank deficient
        W = H' / (H * H' + 1e-6 * eye(K));
    end
    % Normalize each column
    for k = 1:size(W, 2)
        W(:, k) = W(:, k) / norm(W(:, k)) * sqrt(N_tx);
    end
end

% Function: MMSE Beamforming
function W = beamforming_MMSE(H, N_tx, K, SNR_linear)
    % W = H^H * (H * H^H + (K/SNR) * I)^{-1}
    W = H' / (H * H' + (K / SNR_linear) * eye(K));
    % Normalize each column
    for k = 1:size(W, 2)
        W(:, k) = W(:, k) / norm(W(:, k)) * sqrt(N_tx);
    end
end

% Function: Compute SINR and Sum Capacity
function [SINR_dB, sum_capacity] = compute_performance(H, W, SNR_linear, noise_power)
    K = size(H, 1);
    P_total = trace(W * W') / size(W, 2); % Average power per user
    
    SINR = zeros(K, 1);
    for k = 1:K
        % Desired signal power
        signal_power = abs(H(k, :) * W(:, k))^2;
        
        % Interference power from other users
        interference_power = 0;
        for j = 1:K
            if j ~= k
                interference_power = interference_power + abs(H(k, :) * W(:, j))^2;
            end
        end
        
        % SINR calculation
        SINR(k) = (SNR_linear * signal_power) / (SNR_linear * interference_power + noise_power);
    end
    
    SINR_dB = 10 * log10(SINR);
    
    % Sum capacity (Shannon formula)
    sum_capacity = sum(log2(1 + SINR));
end

