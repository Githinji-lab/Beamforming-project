%% ========================================================================
%  MATLAB Simulation: Adaptive Beamforming Dataset Generator - Variant 3
%  Focus: MICROSTRIP PATCH ANTENNA with Realistic 5G Effects
%  Includes: Element pattern, mutual coupling, polarization, substrate
% =========================================================================

clear all; close all; clc;

%% ========================================================================
%  SECTION 1: PARAMETER INITIALIZATION (MICROSTRIP PATCH CONFIGURATION)
% =========================================================================

% Microstrip Patch Antenna Specifications
fc = 3.5e9;                  % 5G Sub-6 GHz frequency (3.5 GHz)
c = 3e8;                     % Speed of light (m/s)
lambda = c / fc;             % Wavelength (0.0857 m = 8.57 cm)

% Substrate Parameters (FR-4 or Rogers RO4003C)
epsilon_r = 3.55;            % Relative permittivity (FR-4: 4.4, Rogers: 3.55)
tan_delta = 0.0027;          % Loss tangent
substrate_height = 1.524e-3; % Substrate thickness (mm)

% Array Configuration - LINEAR ARRAY (Realistic Base Station)
N_tx = 8;                    % Number of patch elements
d = 0.5 * lambda;            % Element spacing (half-wavelength)

% Microstrip Patch Pattern Parameters
half_power_beamwidth = 65;   % HPBW in degrees (typical for patch)
exponent_n = 1.5;            % Directivity factor (cos^n pattern)
front_to_back_ratio = 20;    % dB (typical for patch antenna)

% User Configuration
K = 4;                       % Number of users
N_rx = 1;                    % Single antenna per user

% Simulation Parameters
N_iterations = 600;          % Monte Carlo iterations
SNR_dB_range = 0:5:25;       % SNR range for evaluation
N_SNR = length(SNR_dB_range);

% Channel Model - Urban Micro (5G NR)
N_paths = 8;                 % Multipath components
angular_spread = 15;         % Angular spread (degrees)
rician_K = 5;                % Rician K-factor (dB) - partial LOS
noise_power = 1;

% Mutual Coupling Parameters
coupling_enabled = true;     % Enable mutual coupling effects
coupling_strength = -15;     % Coupling coefficient (dB) at 0.5λ

% Polarization
polarization = 'vertical';   % Linear vertical polarization
cross_pol_discrimination = 20; % XPD in dB

% ML Dataset Storage
dataset_config = struct();
dataset_config.variant = 3;
dataset_config.antenna_type = 'Microstrip Patch';
dataset_config.frequency = fc / 1e9;
dataset_config.substrate = sprintf('εr=%.2f, tanδ=%.4f', epsilon_r, tan_delta);

dataset_H = cell(N_iterations, 1);
dataset_H_ideal = cell(N_iterations, 1);  % For comparison
dataset_angles = zeros(N_iterations, K);
dataset_distances = zeros(N_iterations, K);
dataset_element_gains = cell(N_iterations, K);
dataset_coupling_matrix = cell(N_iterations, 1);
dataset_W_MRT = cell(N_iterations, 1);
dataset_W_ZF = cell(N_iterations, 1);
dataset_W_MMSE = cell(N_iterations, N_SNR);
dataset_W_SLNR = cell(N_iterations, N_SNR);  % Signal-to-Leakage-Noise Ratio
dataset_SINR_MRT = zeros(N_iterations, K, N_SNR);
dataset_SINR_ZF = zeros(N_iterations, K, N_SNR);
dataset_SINR_MMSE = zeros(N_iterations, K, N_SNR);
dataset_SINR_SLNR = zeros(N_iterations, K, N_SNR);
dataset_SNR = zeros(N_iterations, N_SNR);

fprintf('=== 5G Adaptive Beamforming - Dataset Variant 3 ===\n');
fprintf('Antenna Type: Microstrip Patch Array\n');
fprintf('Frequency: %.2f GHz (λ = %.2f cm)\n', fc/1e9, lambda*100);
fprintf('Substrate: εr = %.2f, tan δ = %.4f, h = %.3f mm\n', ...
        epsilon_r, tan_delta, substrate_height*1000);
fprintf('Array: %d elements, %.2fλ spacing\n', N_tx, d/lambda);
fprintf('HPBW: %d°, F/B Ratio: %d dB\n', half_power_beamwidth, front_to_back_ratio);
fprintf('Mutual Coupling: %s (%.1f dB)\n', ...
        string(coupling_enabled), coupling_strength);
fprintf('Number of Users: %d\n', K);
fprintf('Monte Carlo Iterations: %d\n\n', N_iterations);

%% ========================================================================
%  SECTION 2: MONTE CARLO SIMULATION WITH MICROSTRIP EFFECTS
% =========================================================================

sum_capacity_MRT = zeros(N_SNR, 1);
sum_capacity_ZF = zeros(N_SNR, 1);
sum_capacity_MMSE = zeros(N_SNR, 1);
sum_capacity_SLNR = zeros(N_SNR, 1);

% Performance degradation tracking
pattern_loss_avg = 0;
coupling_loss_avg = 0;

fprintf('Running Monte Carlo simulations with microstrip patch effects...\n');
tic;

for iter = 1:N_iterations
    
    if mod(iter, 100) == 0
        fprintf('  Iteration %d/%d\n', iter, N_iterations);
    end
    
    % Generate user angles (limited scan range for patches)
    % Microstrip patches work best within ±60° from broadside
    user_angles = (rand(K, 1) - 0.5) * 100; % -50° to +50° (realistic coverage)
    dataset_angles(iter, :) = user_angles;
    
    % User distances (for path loss calculation)
    user_distances = 50 + rand(K, 1) * 200; % 50-250 meters
    dataset_distances(iter, :) = user_distances;
    
    % Generate mutual coupling matrix (frequency and spacing dependent)
    C = generate_coupling_matrix(N_tx, d, lambda, coupling_enabled, coupling_strength);
    dataset_coupling_matrix{iter} = C;
    
    % Generate ideal channel (without antenna effects)
    H_ideal = generate_channel_with_rician(N_tx, K, user_angles, N_paths, ...
                                           angular_spread, d, lambda, rician_K);
    dataset_H_ideal{iter} = H_ideal;
    
    % Apply microstrip patch antenna effects
    [H_patch, element_gains] = apply_microstrip_effects(H_ideal, user_angles, ...
                                N_tx, K, C, exponent_n, front_to_back_ratio, ...
                                half_power_beamwidth);
    dataset_H{iter} = H_patch;
    dataset_element_gains{iter} = element_gains;
    
    % Calculate performance degradation due to antenna effects
    pattern_loss = 10*log10(norm(H_patch,'fro')^2 / norm(H_ideal,'fro')^2);
    pattern_loss_avg = pattern_loss_avg + pattern_loss;
    
    % Normalize channel for fair comparison
    H_normalized = H_patch / sqrt(trace(H_patch * H_patch') / (N_tx * K));
    
    % Add substrate losses (dielectric loss)
    substrate_loss_dB = 0.1 * sqrt(fc/1e9); % Approximate loss model
    substrate_loss_linear = 10^(-substrate_loss_dB/20);
    H_normalized = H_normalized * substrate_loss_linear;
    
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
        
        % ===== MMSE Beamforming =====
        W_MMSE = beamforming_MMSE(H_normalized, N_tx, K, SNR_linear);
        dataset_W_MMSE{iter, snr_idx} = W_MMSE;
        [SINR_MMSE, capacity_MMSE] = compute_performance(H_normalized, W_MMSE, SNR_linear, noise_power);
        dataset_SINR_MMSE(iter, :, snr_idx) = SINR_MMSE;
        sum_capacity_MMSE(snr_idx) = sum_capacity_MMSE(snr_idx) + capacity_MMSE;
        
        % ===== SLNR Beamforming (NEW - Good for Microstrip) =====
        W_SLNR = beamforming_SLNR(H_normalized, N_tx, K, SNR_linear);
        dataset_W_SLNR{iter, snr_idx} = W_SLNR;
        [SINR_SLNR, capacity_SLNR] = compute_performance(H_normalized, W_SLNR, SNR_linear, noise_power);
        dataset_SINR_SLNR(iter, :, snr_idx) = SINR_SLNR;
        sum_capacity_SLNR(snr_idx) = sum_capacity_SLNR(snr_idx) + capacity_SLNR;
    end
end

elapsed_time = toc;
pattern_loss_avg = pattern_loss_avg / N_iterations;

fprintf('Simulation completed in %.2f seconds.\n', elapsed_time);
fprintf('Average pattern loss: %.2f dB\n\n', pattern_loss_avg);

% Average performance metrics
sum_capacity_MRT = sum_capacity_MRT / N_iterations;
sum_capacity_ZF = sum_capacity_ZF / N_iterations;
sum_capacity_MMSE = sum_capacity_MMSE / N_iterations;
sum_capacity_SLNR = sum_capacity_SLNR / N_iterations;

avg_SINR_MRT = squeeze(mean(dataset_SINR_MRT, 1));
avg_SINR_ZF = squeeze(mean(dataset_SINR_ZF, 1));
avg_SINR_MMSE = squeeze(mean(dataset_SINR_MMSE, 1));
avg_SINR_SLNR = squeeze(mean(dataset_SINR_SLNR, 1));

%% ========================================================================
%  SECTION 3: MICROSTRIP-SPECIFIC VISUALIZATIONS
% =========================================================================

fprintf('=== Generating Microstrip Performance Plots ===\n');

figure('Position', [50, 50, 1500, 900]);

% Plot 1: Sum Capacity Comparison
subplot(2, 3, 1);
plot(SNR_dB_range, sum_capacity_MRT, 'o-', 'LineWidth', 2.5, 'MarkerSize', 8); hold on;
plot(SNR_dB_range, sum_capacity_ZF, 's-', 'LineWidth', 2.5, 'MarkerSize', 8);
plot(SNR_dB_range, sum_capacity_MMSE, '^-', 'LineWidth', 2.5, 'MarkerSize', 8);
plot(SNR_dB_range, sum_capacity_SLNR, 'd-', 'LineWidth', 2.5, 'MarkerSize', 8);
grid on;
xlabel('SNR (dB)', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Sum Capacity (bps/Hz)', 'FontSize', 12, 'FontWeight', 'bold');
title('Microstrip Patch: Sum Capacity', 'FontSize', 13, 'FontWeight', 'bold');
legend('MRT', 'ZF', 'MMSE', 'SLNR', 'Location', 'northwest', 'FontSize', 10);
set(gca, 'FontSize', 11);

% Plot 2: Microstrip Element Pattern
subplot(2, 3, 2);
theta_plot = -180:1:180;
element_pattern = microstrip_element_pattern(theta_plot, exponent_n, front_to_back_ratio);
element_pattern_dB = 10*log10(element_pattern / max(element_pattern));
plot(theta_plot, element_pattern_dB, 'LineWidth', 2.5, 'Color', [0.85, 0.33, 0.1]);
hold on;
yline(-3, '--k', 'HPBW', 'LineWidth', 1.5);
xline(-half_power_beamwidth/2, ':r', 'LineWidth', 1.5);
xline(half_power_beamwidth/2, ':r', 'LineWidth', 1.5);
grid on;
xlabel('Angle (degrees)', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Normalized Gain (dB)', 'FontSize', 12, 'FontWeight', 'bold');
title('Microstrip Patch Element Pattern', 'FontSize', 13, 'FontWeight', 'bold');
xlim([-180, 180]);
ylim([-40, 0]);
set(gca, 'FontSize', 11);

% Plot 3: Coupling Matrix Visualization
subplot(2, 3, 3);
C_sample = dataset_coupling_matrix{1};
imagesc(abs(C_sample));
colorbar;
xlabel('Antenna Element', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Antenna Element', 'FontSize', 12, 'FontWeight', 'bold');
title('Mutual Coupling Matrix |C|', 'FontSize', 13, 'FontWeight', 'bold');
set(gca, 'FontSize', 11);
axis square;

% Plot 4: Array Pattern with Microstrip Effects
subplot(2, 3, 4);
sample_iter = randi(N_iterations);
H_sample = dataset_H{sample_iter};
W_sample = dataset_W_MMSE{sample_iter, find(SNR_dB_range==15)};
angles_sample = dataset_angles(sample_iter, :);

theta_range = -90:0.5:90;
array_pattern = zeros(length(theta_range), K);

for idx = 1:length(theta_range)
    % Array response with element pattern
    a_theta = array_response(N_tx, theta_range(idx), d, lambda);
    g_elem = microstrip_element_pattern(theta_range(idx), exponent_n, front_to_back_ratio);
    a_theta = a_theta * sqrt(g_elem);
    
    for k = 1:K
        array_pattern(idx, k) = abs(a_theta' * W_sample(:, k))^2;
    end
end

array_pattern_dB = 10*log10(array_pattern ./ max(array_pattern(:)));

for k = 1:K
    plot(theta_range, array_pattern_dB(:, k), 'LineWidth', 2.5); hold on;
end

for k = 1:K
    xline(angles_sample(k), '--', sprintf('U%d', k), 'LineWidth', 1.8, ...
          'LabelHorizontalAlignment', 'center', 'FontSize', 10);
end

grid on;
xlabel('Azimuth Angle (°)', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Normalized Gain (dB)', 'FontSize', 12, 'FontWeight', 'bold');
title('Array Beam Pattern (MMSE)', 'FontSize', 13, 'FontWeight', 'bold');
ylim([-40, 0]);
legend(arrayfun(@(k) sprintf('User %d', k), 1:K, 'UniformOutput', false), ...
       'Location', 'best', 'FontSize', 9);
set(gca, 'FontSize', 11);

% Plot 5: SINR vs Angle (Microstrip Sensitivity)
subplot(2, 3, 5);
angle_bins = -50:10:50;
sinr_vs_angle = zeros(length(angle_bins)-1, 1);

for i = 1:length(angle_bins)-1
    angle_mask = (dataset_angles >= angle_bins(i)) & (dataset_angles < angle_bins(i+1));
    sinr_in_bin = dataset_SINR_MMSE(angle_mask, :, find(SNR_dB_range==15));
    sinr_vs_angle(i) = mean(sinr_in_bin(:));
end

bar(angle_bins(1:end-1) + 5, sinr_vs_angle, 'FaceColor', [0.3, 0.6, 0.9]);
grid on;
xlabel('User Angle (degrees)', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Average SINR (dB)', 'FontSize', 12, 'FontWeight', 'bold');
title('SINR vs Angle (15 dB SNR)', 'FontSize', 13, 'FontWeight', 'bold');
set(gca, 'FontSize', 11);

% Plot 6: Performance Comparison Table
subplot(2, 3, 6);
axis off;
snr_ref = find(SNR_dB_range == 15);

table_data = {
    'Method', 'Capacity', 'Avg SINR', 'Min SINR';
    'MRT', sprintf('%.2f', sum_capacity_MRT(snr_ref)), ...
           sprintf('%.2f', mean(avg_SINR_MRT(:, snr_ref))), ...
           sprintf('%.2f', min(avg_SINR_MRT(:, snr_ref)));
    'ZF', sprintf('%.2f', sum_capacity_ZF(snr_ref)), ...
          sprintf('%.2f', mean(avg_SINR_ZF(:, snr_ref))), ...
          sprintf('%.2f', min(avg_SINR_ZF(:, snr_ref)));
    'MMSE', sprintf('%.2f', sum_capacity_MMSE(snr_ref)), ...
            sprintf('%.2f', mean(avg_SINR_MMSE(:, snr_ref))), ...
            sprintf('%.2f', min(avg_SINR_MMSE(:, snr_ref)));
    'SLNR', sprintf('%.2f', sum_capacity_SLNR(snr_ref)), ...
            sprintf('%.2f', mean(avg_SINR_SLNR(:, snr_ref))), ...
            sprintf('%.2f', min(avg_SINR_SLNR(:, snr_ref)))
};

text(0.1, 0.9, 'Performance at 15 dB SNR', 'FontSize', 14, 'FontWeight', 'bold');
text(0.1, 0.75, sprintf('Frequency: %.2f GHz', fc/1e9), 'FontSize', 11);
text(0.1, 0.65, sprintf('Substrate: εr=%.2f', epsilon_r), 'FontSize', 11);
text(0.1, 0.55, sprintf('Pattern Loss: %.2f dB', pattern_loss_avg), 'FontSize', 11);

for i = 1:size(table_data, 1)
    for j = 1:size(table_data, 2)
        if i == 1
            text(0.05 + (j-1)*0.25, 0.40, table_data{i,j}, ...
                 'FontSize', 10, 'FontWeight', 'bold');
        else
            text(0.05 + (j-1)*0.25, 0.40 - i*0.08, table_data{i,j}, 'FontSize', 10);
        end
    end
end

%% ========================================================================
%  SECTION 4: ML DATASET EXPORT (VARIANT 3 - MICROSTRIP)
% =========================================================================

fprintf('\n=== Exporting ML Training Dataset (Variant 3 - Microstrip) ===\n');

ML_Dataset_V3.Description = 'Adaptive Beamforming - Microstrip Patch Antenna';
ML_Dataset_V3.Variant = 3;
ML_Dataset_V3.Antenna.type = 'Microstrip Patch';
ML_Dataset_V3.Antenna.frequency_GHz = fc / 1e9;
ML_Dataset_V3.Antenna.wavelength_m = lambda;
ML_Dataset_V3.Antenna.substrate_er = epsilon_r;
ML_Dataset_V3.Antenna.substrate_tand = tan_delta;
ML_Dataset_V3.Antenna.HPBW_deg = half_power_beamwidth;
ML_Dataset_V3.Antenna.front_to_back_dB = front_to_back_ratio;
ML_Dataset_V3.Antenna.coupling_enabled = coupling_enabled;

ML_Dataset_V3.Parameters.N_tx = N_tx;
ML_Dataset_V3.Parameters.K = K;
ML_Dataset_V3.Parameters.element_spacing = d / lambda;
ML_Dataset_V3.Parameters.N_iterations = N_iterations;
ML_Dataset_V3.Parameters.SNR_dB_range = SNR_dB_range;

ML_Dataset_V3.Features.H = dataset_H;
ML_Dataset_V3.Features.H_ideal = dataset_H_ideal;  % For comparison
ML_Dataset_V3.Features.user_angles = dataset_angles;
ML_Dataset_V3.Features.user_distances = dataset_distances;
ML_Dataset_V3.Features.element_gains = dataset_element_gains;
ML_Dataset_V3.Features.coupling_matrix = dataset_coupling_matrix;
ML_Dataset_V3.Features.SNR = dataset_SNR;

ML_Dataset_V3.Labels.W_MRT = dataset_W_MRT;
ML_Dataset_V3.Labels.W_ZF = dataset_W_ZF;
ML_Dataset_V3.Labels.W_MMSE = dataset_W_MMSE;
ML_Dataset_V3.Labels.W_SLNR = dataset_W_SLNR;

ML_Dataset_V3.Performance.SINR_MRT = dataset_SINR_MRT;
ML_Dataset_V3.Performance.SINR_ZF = dataset_SINR_ZF;
ML_Dataset_V3.Performance.SINR_MMSE = dataset_SINR_MMSE;
ML_Dataset_V3.Performance.SINR_SLNR = dataset_SINR_SLNR;

% Optimal method
[~, optimal_idx] = max(cat(3, ...
    squeeze(mean(dataset_SINR_MRT, 2)), ...
    squeeze(mean(dataset_SINR_ZF, 2)), ...
    squeeze(mean(dataset_SINR_MMSE, 2)), ...
    squeeze(mean(dataset_SINR_SLNR, 2))), [], 3);

ML_Dataset_V3.Labels.optimal_method = optimal_idx;
ML_Dataset_V3.Performance.pattern_loss_dB = pattern_loss_avg;

save('beamforming_dataset_v3.mat', 'ML_Dataset_V3', '-v7.3');

fprintf('Dataset saved to: beamforming_dataset_v3.mat\n');
fprintf('  - Antenna: Microstrip Patch @ %.2f GHz\n', fc/1e9);
fprintf('  - Channel matrices: %d samples\n', N_iterations);
fprintf('  - Element patterns: included\n');
fprintf('  - Mutual coupling: included\n');
fprintf('  - Substrate effects: included\n');
fprintf('  - Beamforming methods: 4 (MRT, ZF, MMSE, SLNR)\n\n');

fprintf('=== Variant 3 (Microstrip) Simulation Complete ===\n');
fprintf('Total execution time: %.2f seconds\n', elapsed_time);
fprintf('Average pattern loss due to microstrip: %.2f dB\n', pattern_loss_avg);

%% ========================================================================
%  SUPPORTING FUNCTIONS - MICROSTRIP SPECIFIC
% =========================================================================

% Microstrip Element Pattern (cos^n model with front-to-back ratio)
function g = microstrip_element_pattern(theta_deg, n, fb_ratio_dB)
    theta_rad = deg2rad(theta_deg);
    
    % Front hemisphere (cosine pattern)
    g_front = max(cos(theta_rad), 0).^n;
    
    % Back hemisphere (attenuated)
    fb_ratio_linear = 10^(-fb_ratio_dB/10);
    g_back = fb_ratio_linear * max(cos(theta_rad + pi), 0).^n;
    
    % Combine
    g = g_front + g_back;
    
    % Normalize
    g = g / max(g(:));
end

% Generate Mutual Coupling Matrix
function C = generate_coupling_matrix(N, d, lambda, enabled, coupling_dB)
    C = eye(N);
    
    if ~enabled
        return;
    end
    
    coupling_linear = 10^(coupling_dB/20);
    
    for i = 1:N
        for j = 1:N
            if i ~= j
                % Distance-dependent coupling
                dist = abs(i - j) * d / lambda;
                % Coupling decreases with distance
                C(i, j) = coupling_linear / dist;
            end
        end
    end
end

% Generate Channel with Rician Fading (Partial LOS)
function H = generate_channel_with_rician(N_tx, K, angles, N_paths, spread, d, lambda, K_dB)
    H = zeros(K, N_tx);
    K_linear = 10^(K_dB/10);
    
    for k = 1:K
        % LOS component
        a_los = array_response(N_tx, angles(k), d, lambda);
        h_los = sqrt(K_linear / (K_linear + 1)) * a_los;
        
        % NLOS components (Rayleigh)
        h_nlos = zeros(N_tx, 1);
        path_angles = angles(k) + spread * (rand(N_paths, 1) - 0.5);
        path_gains = (randn(N_paths, 1) + 1i*randn(N_paths, 1)) / sqrt(2 * N_paths);
        
        for p = 1:N_paths
            a_p = array_response(N_tx, path_angles(p), d, lambda);
            h_nlos = h_nlos + path_gains(p) * a_p;
        end
        h_nlos = sqrt(1 / (K_linear + 1)) * h_nlos;
        
        H(k, :) = (h_los + h_nlos).';
    end
    
    H = H / sqrt(mean(abs(H(:)).^2)) * sqrt(N_tx);
end

% Apply Microstrip Antenna Effects to Channel
function [H_patch, element_gains] = apply_microstrip_effects(H_ideal, angles, ...
                                    N_tx, K, C, n, fb_ratio, hpbw)
    element_gains = zeros(K, N_tx);
    
    % Apply element pattern to each user
    for k = 1:K
        for m = 1:N_tx
            % Angle from each element to user k
            % (In ULA, all elements see same angle)
            g = microstrip_element_pattern(angles(k), n, fb_ratio);
            element_gains(k, m) = sqrt(g);
        end
    end
    
    % Apply element gains
    H_patch = H_ideal;
    for k = 1:K
        H_patch(k, :) = H_patch(k, :) .* element_gains(k, :);
    end
    
    % Apply mutual coupling
    H_patch = H_patch * C;
end

% Array Response Vector
function a = array_response(N, theta_deg, d, lambda)
    theta_rad = deg2rad(theta_deg);
    n = (0:N-1).';
    a = exp(1i * 2 * pi * d / lambda * n * sin(theta_rad)) / sqrt(N);
end

% Beamforming Algorithms
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
    for k = 1:size
