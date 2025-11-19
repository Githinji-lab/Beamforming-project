%% ========================================================================
%  MATLAB Simulation: Adaptive Beamforming Dataset Generator - Variant 2
%  Focus: Uniform Planar Array (UPA) with User Mobility Scenarios
%  Generates different dataset for ML training diversity
% =========================================================================

clear all; close all; clc;

%% ========================================================================
%  SECTION 1: PARAMETER INITIALIZATION (VARIANT 2 CONFIGURATION)
% =========================================================================

% Antenna Array Configuration - UNIFORM PLANAR ARRAY (UPA)
N_h = 4;                     % Horizontal elements
N_v = 4;                     % Vertical elements
N_tx = N_h * N_v;            % Total antennas (16)
lambda = 1;                  % Normalized wavelength
d_h = 0.5 * lambda;          % Horizontal spacing
d_v = 0.5 * lambda;          % Vertical spacing

% User Configuration - MORE USERS
K = 5;                       % Number of users (increased from 3)
N_rx = 1;                    % Single antenna per user

% Simulation Parameters
N_iterations = 600;          % Increased iterations for diversity
SNR_dB_range = -5:5:25;      % Extended SNR range
N_SNR = length(SNR_dB_range);

% Channel Model - URBAN MACRO SCENARIO
N_paths = 10;                % More multipath (urban environment)
angular_spread_az = 20;      % Azimuth angular spread (degrees)
angular_spread_el = 10;      % Elevation angular spread (degrees)
doppler_max = 50;            % Maximum Doppler shift (Hz) - mobility
noise_power = 1;

% User Mobility Parameters (NEW)
user_velocity = 3;           % Average velocity (m/s) - walking speed
scenario_types = {'static', 'slow_mobile', 'fast_mobile', 'clustered'};

% ML Dataset Storage
dataset_config = struct();
dataset_config.variant = 2;
dataset_config.array_type = 'UPA';
dataset_config.mobility = 'enabled';

dataset_H = cell(N_iterations, 1);
dataset_angles_az = zeros(N_iterations, K);
dataset_angles_el = zeros(N_iterations, K);
dataset_scenario = cell(N_iterations, 1);
dataset_doppler = zeros(N_iterations, K);
dataset_W_MRT = cell(N_iterations, 1);
dataset_W_ZF = cell(N_iterations, 1);
dataset_W_MMSE = cell(N_iterations, N_SNR);
dataset_W_RZF = cell(N_iterations, N_SNR);  % Regularized ZF (NEW)
dataset_SINR_MRT = zeros(N_iterations, K, N_SNR);
dataset_SINR_ZF = zeros(N_iterations, K, N_SNR);
dataset_SINR_MMSE = zeros(N_iterations, K, N_SNR);
dataset_SINR_RZF = zeros(N_iterations, K, N_SNR);
dataset_SNR = zeros(N_iterations, N_SNR);

fprintf('=== 5G Adaptive Beamforming - Dataset Variant 2 ===\n');
fprintf('Array Configuration: %dx%d UPA (%d elements)\n', N_h, N_v, N_tx);
fprintf('Element Spacing: %.2fλ (H) x %.2fλ (V)\n', d_h/lambda, d_v/lambda);
fprintf('Number of Users: %d\n', K);
fprintf('Scenario: Urban Macro with User Mobility\n');
fprintf('Monte Carlo Iterations: %d\n', N_iterations);
fprintf('SNR Range: %d to %d dB\n\n', min(SNR_dB_range), max(SNR_dB_range));

%% ========================================================================
%  SECTION 2: MONTE CARLO SIMULATION WITH MOBILITY
% =========================================================================

sum_capacity_MRT = zeros(N_SNR, 1);
sum_capacity_ZF = zeros(N_SNR, 1);
sum_capacity_MMSE = zeros(N_SNR, 1);
sum_capacity_RZF = zeros(N_SNR, 1);

fprintf('Running Monte Carlo simulations with mobility scenarios...\n');
tic;

for iter = 1:N_iterations
    
    if mod(iter, 100) == 0
        fprintf('  Iteration %d/%d\n', iter, N_iterations);
    end
    
    % Randomly select scenario type
    scenario_idx = randi(length(scenario_types));
    scenario = scenario_types{scenario_idx};
    dataset_scenario{iter} = scenario;
    
    % Generate user positions based on scenario
    switch scenario
        case 'static'
            % Wide angular distribution, no mobility
            user_angles_az = (rand(K, 1) - 0.5) * 180;  % -90° to +90°
            user_angles_el = (rand(K, 1) - 0.5) * 60;   % -30° to +30°
            doppler_shifts = zeros(K, 1);
            
        case 'slow_mobile'
            % Moderate spread, low Doppler
            user_angles_az = (rand(K, 1) - 0.5) * 120;
            user_angles_el = (rand(K, 1) - 0.5) * 40;
            doppler_shifts = doppler_max * 0.3 * rand(K, 1);
            
        case 'fast_mobile'
            % Wide spread, high Doppler
            user_angles_az = (rand(K, 1) - 0.5) * 180;
            user_angles_el = (rand(K, 1) - 0.5) * 60;
            doppler_shifts = doppler_max * (0.5 + 0.5 * rand(K, 1));
            
        case 'clustered'
            % Users in 2-3 angular clusters
            n_clusters = randi([2, 3]);
            cluster_centers_az = (rand(n_clusters, 1) - 0.5) * 150;
            cluster_centers_el = (rand(n_clusters, 1) - 0.5) * 40;
            user_angles_az = zeros(K, 1);
            user_angles_el = zeros(K, 1);
            doppler_shifts = doppler_max * 0.4 * rand(K, 1);
            
            for k = 1:K
                cluster_id = randi(n_clusters);
                user_angles_az(k) = cluster_centers_az(cluster_id) + randn() * 10;
                user_angles_el(k) = cluster_centers_el(cluster_id) + randn() * 5;
            end
    end
    
    dataset_angles_az(iter, :) = user_angles_az;
    dataset_angles_el(iter, :) = user_angles_el;
    dataset_doppler(iter, :) = doppler_shifts;
    
    % Generate 3D channel (UPA with elevation)
    H = generate_3D_channel_UPA(N_h, N_v, K, user_angles_az, user_angles_el, ...
                                 N_paths, angular_spread_az, angular_spread_el, ...
                                 d_h, d_v, lambda, doppler_shifts);
    dataset_H{iter} = H;
    
    % Normalize channel
    H_normalized = H / sqrt(trace(H * H') / (N_tx * K));
    
    % Loop over SNR values
    for snr_idx = 1:N_SNR
        SNR_dB = SNR_dB_range(snr_idx);
        SNR_linear = 10^(SNR_dB/10);
        dataset_SNR(iter, snr_idx) = SNR_dB;
        
        % ===== MRT Beamforming =====
        W_MRT = beamforming_MRT(H_normalized, N_tx);
        if iter == 1 && snr_idx == 1
            dataset_W_MRT{iter} = W_MRT;
        end
        [SINR_MRT, capacity_MRT] = compute_performance(H_normalized, W_MRT, SNR_linear, noise_power);
        dataset_SINR_MRT(iter, :, snr_idx) = SINR_MRT;
        sum_capacity_MRT(snr_idx) = sum_capacity_MRT(snr_idx) + capacity_MRT;
        
        % ===== Zero Forcing =====
        W_ZF = beamforming_ZF(H_normalized, N_tx, K);
        if iter == 1 && snr_idx == 1
            dataset_W_ZF{iter} = W_ZF;
        end
        [SINR_ZF, capacity_ZF] = compute_performance(H_normalized, W_ZF, SNR_linear, noise_power);
        dataset_SINR_ZF(iter, :, snr_idx) = SINR_ZF;
        sum_capacity_ZF(snr_idx) = sum_capacity_ZF(snr_idx) + capacity_ZF;
        
        % ===== MMSE Beamforming =====
        W_MMSE = beamforming_MMSE(H_normalized, N_tx, K, SNR_linear);
        dataset_W_MMSE{iter, snr_idx} = W_MMSE;
        [SINR_MMSE, capacity_MMSE] = compute_performance(H_normalized, W_MMSE, SNR_linear, noise_power);
        dataset_SINR_MMSE(iter, :, snr_idx) = SINR_MMSE;
        sum_capacity_MMSE(snr_idx) = sum_capacity_MMSE(snr_idx) + capacity_MMSE;
        
        % ===== Regularized ZF (NEW METHOD) =====
        alpha_reg = 0.1;  % Regularization parameter
        W_RZF = beamforming_RZF(H_normalized, N_tx, K, alpha_reg);
        dataset_W_RZF{iter, snr_idx} = W_RZF;
        [SINR_RZF, capacity_RZF] = compute_performance(H_normalized, W_RZF, SNR_linear, noise_power);
        dataset_SINR_RZF(iter, :, snr_idx) = SINR_RZF;
        sum_capacity_RZF(snr_idx) = sum_capacity_RZF(snr_idx) + capacity_RZF;
    end
end

elapsed_time = toc;
fprintf('Simulation completed in %.2f seconds.\n\n', elapsed_time);

% Average performance
sum_capacity_MRT = sum_capacity_MRT / N_iterations;
sum_capacity_ZF = sum_capacity_ZF / N_iterations;
sum_capacity_MMSE = sum_capacity_MMSE / N_iterations;
sum_capacity_RZF = sum_capacity_RZF / N_iterations;

avg_SINR_MRT = squeeze(mean(dataset_SINR_MRT, 1));
avg_SINR_ZF = squeeze(mean(dataset_SINR_ZF, 1));
avg_SINR_MMSE = squeeze(mean(dataset_SINR_MMSE, 1));
avg_SINR_RZF = squeeze(mean(dataset_SINR_RZF, 1));

%% ========================================================================
%  SECTION 3: PERFORMANCE VISUALIZATION
% =========================================================================

fprintf('=== Generating Performance Plots ===\n');

figure('Position', [100, 100, 1400, 900]);

% Plot 1: Sum Capacity Comparison
subplot(2, 3, 1);
plot(SNR_dB_range, sum_capacity_MRT, 'o-', 'LineWidth', 2, 'MarkerSize', 8); hold on;
plot(SNR_dB_range, sum_capacity_ZF, 's-', 'LineWidth', 2, 'MarkerSize', 8);
plot(SNR_dB_range, sum_capacity_MMSE, '^-', 'LineWidth', 2, 'MarkerSize', 8);
plot(SNR_dB_range, sum_capacity_RZF, 'd-', 'LineWidth', 2, 'MarkerSize', 8);
grid on;
xlabel('SNR (dB)', 'FontSize', 11);
ylabel('Sum Capacity (bps/Hz)', 'FontSize', 11);
title('Sum Capacity vs SNR (UPA)', 'FontSize', 12, 'FontWeight', 'bold');
legend('MRT', 'ZF', 'MMSE', 'RZF', 'Location', 'northwest');
set(gca, 'FontSize', 10);

% Plot 2: Average SINR Comparison
subplot(2, 3, 2);
plot(SNR_dB_range, mean(avg_SINR_MRT, 1), 'o-', 'LineWidth', 2, 'MarkerSize', 8); hold on;
plot(SNR_dB_range, mean(avg_SINR_ZF, 1), 's-', 'LineWidth', 2, 'MarkerSize', 8);
plot(SNR_dB_range, mean(avg_SINR_MMSE, 1), '^-', 'LineWidth', 2, 'MarkerSize', 8);
plot(SNR_dB_range, mean(avg_SINR_RZF, 1), 'd-', 'LineWidth', 2, 'MarkerSize', 8);
grid on;
xlabel('SNR (dB)', 'FontSize', 11);
ylabel('Average SINR (dB)', 'FontSize', 11);
title('Mean SINR across All Users', 'FontSize', 12, 'FontWeight', 'bold');
legend('MRT', 'ZF', 'MMSE', 'RZF', 'Location', 'northwest');
set(gca, 'FontSize', 10);

% Plot 3: 3D Beam Pattern (MMSE)
subplot(2, 3, 3);
sample_iter = find(strcmp(dataset_scenario, 'clustered'), 1);
if isempty(sample_iter), sample_iter = 1; end
sample_snr_idx = find(SNR_dB_range == 10);
if isempty(sample_snr_idx), sample_snr_idx = ceil(N_SNR/2); end

H_sample = dataset_H{sample_iter};
W_sample = dataset_W_MMSE{sample_iter, sample_snr_idx};
angles_az_sample = dataset_angles_az(sample_iter, :);
angles_el_sample = dataset_angles_el(sample_iter, :);

theta_range = -90:5:90;
phi_range = -30:5:30;
[THETA, PHI] = meshgrid(theta_range, phi_range);
beam_gain = zeros(size(THETA));

for i = 1:numel(THETA)
    a_3d = array_response_UPA(N_h, N_v, THETA(i), PHI(i), d_h, d_v, lambda);
    beam_gain(i) = abs(a_3d' * W_sample(:, 1))^2;
end

beam_gain_dB = 10*log10(beam_gain / max(beam_gain(:)));
surf(THETA, PHI, beam_gain_dB, 'EdgeColor', 'none');
hold on;
plot3(angles_az_sample, angles_el_sample, zeros(1, K), 'r*', 'MarkerSize', 12, 'LineWidth', 2);
xlabel('Azimuth (°)', 'FontSize', 10);
ylabel('Elevation (°)', 'FontSize', 10);
zlabel('Gain (dB)', 'FontSize', 10);
title('3D Beam Pattern (MMSE)', 'FontSize', 12, 'FontWeight', 'bold');
colorbar;
view(45, 30);
set(gca, 'FontSize', 10);

% Plot 4: Scenario Distribution
subplot(2, 3, 4);
scenario_counts = zeros(1, length(scenario_types));
for i = 1:length(scenario_types)
    scenario_counts(i) = sum(strcmp(dataset_scenario, scenario_types{i}));
end
bar(scenario_counts);
set(gca, 'XTickLabel', scenario_types);
ylabel('Number of Samples', 'FontSize', 11);
title('Mobility Scenario Distribution', 'FontSize', 12, 'FontWeight', 'bold');
grid on;
set(gca, 'FontSize', 10);

% Plot 5: Doppler Distribution
subplot(2, 3, 5);
histogram(dataset_doppler(:), 30, 'FaceColor', [0.3, 0.6, 0.9]);
xlabel('Doppler Shift (Hz)', 'FontSize', 11);
ylabel('Frequency', 'FontSize', 11);
title('Doppler Shift Distribution', 'FontSize', 12, 'FontWeight', 'bold');
grid on;
set(gca, 'FontSize', 10);

% Plot 6: User Angular Distribution
subplot(2, 3, 6);
scatter(dataset_angles_az(:), dataset_angles_el(:), 20, 'filled', 'MarkerFaceAlpha', 0.3);
xlabel('Azimuth Angle (°)', 'FontSize', 11);
ylabel('Elevation Angle (°)', 'FontSize', 11);
title('User Angular Distribution', 'FontSize', 12, 'FontWeight', 'bold');
grid on;
axis equal;
xlim([-100, 100]);
ylim([-40, 40]);
set(gca, 'FontSize', 10);

%% ========================================================================
%  SECTION 4: PERFORMANCE SUMMARY
% =========================================================================

fprintf('\n=== Performance Summary at SNR = 10 dB ===\n');
snr_10dB_idx = find(SNR_dB_range == 10);
if isempty(snr_10dB_idx), snr_10dB_idx = find(SNR_dB_range >= 10, 1); end

fprintf('┌──────────────┬─────────────────┬─────────────────┬─────────────────┐\n');
fprintf('│   Method     │  Sum Capacity   │  Avg SINR (dB)  │  Min SINR (dB)  │\n');
fprintf('├──────────────┼─────────────────┼─────────────────┼─────────────────┤\n');

sinr_mrt = avg_SINR_MRT(:, snr_10dB_idx);
sinr_zf = avg_SINR_ZF(:, snr_10dB_idx);
sinr_mmse = avg_SINR_MMSE(:, snr_10dB_idx);
sinr_rzf = avg_SINR_RZF(:, snr_10dB_idx);

fprintf('│     MRT      │     %6.2f      │      %6.2f     │      %6.2f     │\n', ...
    sum_capacity_MRT(snr_10dB_idx), mean(sinr_mrt), min(sinr_mrt));
fprintf('│     ZF       │     %6.2f      │      %6.2f     │      %6.2f     │\n', ...
    sum_capacity_ZF(snr_10dB_idx), mean(sinr_zf), min(sinr_zf));
fprintf('│    MMSE      │     %6.2f      │      %6.2f     │      %6.2f     │\n', ...
    sum_capacity_MMSE(snr_10dB_idx), mean(sinr_mmse), min(sinr_mmse));
fprintf('│     RZF      │     %6.2f      │      %6.2f     │      %6.2f     │\n', ...
    sum_capacity_RZF(snr_10dB_idx), mean(sinr_rzf), min(sinr_rzf));
fprintf('└──────────────┴─────────────────┴─────────────────┴─────────────────┘\n\n');

%% ========================================================================
%  SECTION 5: ML DATASET EXPORT (VARIANT 2)
% =========================================================================

fprintf('=== Exporting ML Training Dataset (Variant 2) ===\n');

ML_Dataset_V2.Description = 'Adaptive Beamforming Dataset - UPA with Mobility';
ML_Dataset_V2.Variant = 2;
ML_Dataset_V2.Parameters.array_type = 'UPA';
ML_Dataset_V2.Parameters.N_h = N_h;
ML_Dataset_V2.Parameters.N_v = N_v;
ML_Dataset_V2.Parameters.N_tx = N_tx;
ML_Dataset_V2.Parameters.K = K;
ML_Dataset_V2.Parameters.N_iterations = N_iterations;
ML_Dataset_V2.Parameters.SNR_dB_range = SNR_dB_range;
ML_Dataset_V2.Parameters.N_paths = N_paths;
ML_Dataset_V2.Parameters.mobility_enabled = true;

ML_Dataset_V2.Features.H = dataset_H;
ML_Dataset_V2.Features.angles_azimuth = dataset_angles_az;
ML_Dataset_V2.Features.angles_elevation = dataset_angles_el;
ML_Dataset_V2.Features.doppler_shifts = dataset_doppler;
ML_Dataset_V2.Features.scenario_type = dataset_scenario;
ML_Dataset_V2.Features.SNR = dataset_SNR;

ML_Dataset_V2.Labels.W_MRT = dataset_W_MRT;
ML_Dataset_V2.Labels.W_ZF = dataset_W_ZF;
ML_Dataset_V2.Labels.W_MMSE = dataset_W_MMSE;
ML_Dataset_V2.Labels.W_RZF = dataset_W_RZF;

ML_Dataset_V2.Performance.SINR_MRT = dataset_SINR_MRT;
ML_Dataset_V2.Performance.SINR_ZF = dataset_SINR_ZF;
ML_Dataset_V2.Performance.SINR_MMSE = dataset_SINR_MMSE;
ML_Dataset_V2.Performance.SINR_RZF = dataset_SINR_RZF;

% Optimal method selection
[~, optimal_idx] = max(cat(3, ...
    squeeze(mean(dataset_SINR_MRT, 2)), ...
    squeeze(mean(dataset_SINR_ZF, 2)), ...
    squeeze(mean(dataset_SINR_MMSE, 2)), ...
    squeeze(mean(dataset_SINR_RZF, 2))), [], 3);

ML_Dataset_V2.Labels.optimal_method = optimal_idx;

save('beamforming_dataset_v2.mat', 'ML_Dataset_V2', '-v7.3');

fprintf('Dataset saved to: beamforming_dataset_v2.mat\n');
fprintf('  - Array Type: %dx%d UPA (%d elements)\n', N_h, N_v, N_tx);
fprintf('  - Channel matrices: %d samples\n', N_iterations);
fprintf('  - User angles (Az/El): %d samples\n', N_iterations);
fprintf('  - Mobility scenarios: 4 types\n');
fprintf('  - Doppler shifts: included\n');
fprintf('  - Beamforming methods: 4 (MRT, ZF, MMSE, RZF)\n\n');

fprintf('=== Variant 2 Simulation Complete ===\n');
fprintf('Total execution time: %.2f seconds\n', elapsed_time);

%% ========================================================================
%  SUPPORTING FUNCTIONS
% =========================================================================

function H = generate_3D_channel_UPA(N_h, N_v, K, angles_az, angles_el, ...
                                     N_paths, spread_az, spread_el, ...
                                     d_h, d_v, lambda, doppler)
    N_tx = N_h * N_v;
    H = zeros(K, N_tx);
    
    for k = 1:K
        main_az = angles_az(k);
        main_el = angles_el(k);
        
        path_angles_az = main_az + spread_az * (rand(N_paths, 1) - 0.5);
        path_angles_el = main_el + spread_el * (rand(N_paths, 1) - 0.5);
        
        % Path gains with Doppler effect
        doppler_phase = 2 * pi * doppler(k) * rand(N_paths, 1);
        path_gains = (randn(N_paths, 1) + 1i*randn(N_paths, 1)) / sqrt(2 * N_paths);
        path_gains = path_gains .* exp(1i * doppler_phase);
        
        h_k = zeros(N_tx, 1);
        for p = 1:N_paths
            a_p = array_response_UPA(N_h, N_v, path_angles_az(p), ...
                                     path_angles_el(p), d_h, d_v, lambda);
            h_k = h_k + path_gains(p) * a_p;
        end
        
        H(k, :) = h_k.';
    end
    
    H = H / sqrt(mean(abs(H(:)).^2)) * sqrt(N_tx);
end

function a = array_response_UPA(N_h, N_v, theta_deg, phi_deg, d_h, d_v, lambda)
    theta_rad = deg2rad(theta_deg);
    phi_rad = deg2rad(phi_deg);
    
    a = zeros(N_h * N_v, 1);
    idx = 1;
    for n_v = 0:N_v-1
        for n_h = 0:N_h-1
            phase = 2*pi/lambda * (n_h*d_h*sin(theta_rad)*cos(phi_rad) + ...
                                   n_v*d_v*sin(phi_rad));
            a(idx) = exp(1i * phase);
            idx = idx + 1;
        end
    end
    a = a / sqrt(N_h * N_v);
end

function W = beamforming_MRT(H, N_tx)
    W = H';
    for k = 1:size(W, 2)
        W(:, k) = W(:, k) / norm(W(:, k)) * sqrt(N_tx);
    end
end

function W = beamforming_ZF(H, N_tx, K)
    if rank(H) == K
        W = H' / (H * H');
    else
        W = H' / (H * H' + 1e-6 * eye(K));
    end
    for k = 1:size(W, 2)
        W(:, k) = W(:, k) / norm(W(:, k)) * sqrt(N_tx);
    end
end

function W = beamforming_MMSE(H, N_tx, K, SNR_linear)
    W = H' / (H * H' + (K / SNR_linear) * eye(K));
    for k = 1:size(W, 2)
        W(:, k) = W(:, k) / norm(W(:, k)) * sqrt(N_tx);
    end
end

function W = beamforming_RZF(H, N_tx, K, alpha)
    % Regularized Zero Forcing
    W = H' / (H * H' + alpha * eye(K));
    for k = 1:size(W, 2)
        W(:, k) = W(:, k) / norm(W(:, k)) * sqrt(N_tx);
    end
end

function [SINR_dB, sum_capacity] = compute_performance(H, W, SNR_linear, noise_power)
    K = size(H, 1);
    SINR = zeros(K, 1);
    
    for k = 1:K
        signal_power = abs(H(k, :) * W(:, k))^2;
        interference_power = 0;
        for j = 1:K
            if j ~= k
                interference_power = interference_power + abs(H(k, :) * W(:, j))^2;
            end
        end
        SINR(k) = (SNR_linear * signal_power) / (SNR_linear * interference_power + noise_power);
    end
    
    SINR_dB = 10 * log10(SINR);
    sum_capacity = sum(log2(1 + SINR));
end
