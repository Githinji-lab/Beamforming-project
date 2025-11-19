%% ========================================================================
%  MATLAB Simulation: ML-Ready Beamforming Dataset Generator - Variant 4
%  BASE: V3 (Microstrip Patch, Rician) + REFINEMENTS: Discrete Actions + Sequential States (s -> s')
%  Output: Dataset structured for Q-Learning and Clustering
% =========================================================================

clear all; close all; clc;

%% ========================================================================
%  SECTION 1: PARAMETER INITIALIZATION
% =========================================================================

% Microstrip Patch Antenna Specifications (from V3)
fc = 3.5e9;                    % 5G Sub-6 GHz frequency
c = 3e8;                       % Speed of light
lambda = c / fc;               % Wavelength

% Array Configuration - ULA
N_tx = 8;                      % Number of patch elements
d = 0.5 * lambda;              % Element spacing

% User Configuration
K = 4;                         % Number of users

% Simulation Parameters
N_iterations = 6000;           % HIGH ITERATIONS for ML training
SNR_dB_range = 10;             % Fixed SNR for RL training (e.g., 10 dB)
N_SNR = length(SNR_dB_range);

% Channel Model (from V3)
N_paths = 8;                   % Multipath components
angular_spread = 15;           % Angular spread (degrees)
rician_K = 5;                  % Rician K-factor (dB)
noise_power = 1;
coupling_enabled = true;
coupling_strength = -15;       % Mutual coupling (dB)

% Microstrip Pattern Parameters (from V3)
exponent_n = 1.5;
front_to_back_ratio = 20;
half_power_beamwidth = 65;

% === V4 REFINEMENT 1A: DEFINE CODEBOOK (Action Space) ===
N_beams = 32; % Total number of discrete actions (A)
Codebook = zeros(N_tx, N_beams); 
for a = 1:N_beams
    for n = 1:N_tx
        Codebook(n, a) = (1/sqrt(N_tx)) * exp(1i * 2*pi * (n - 1) * (a - 1) / N_beams);
    end
end

% ML Dataset Storage (Sequential Structure)
dataset_H_current = cell(N_iterations, 1);
dataset_H_next = cell(N_iterations, 1);
dataset_angles_current = zeros(N_iterations, K);
dataset_angles_next = zeros(N_iterations, K);
dataset_SINR_current = zeros(N_iterations, K);
dataset_SINR_next = zeros(N_iterations, K);
dataset_A_Best_Index = zeros(N_iterations, K); % Discrete action label (1 to 32)

fprintf('=== V4: ML-Ready Simulation Start ===\n');
fprintf('RL Dataset Size: %d Episodes\n', N_iterations);
fprintf('Action Space: %d Discrete Beams\n', N_beams);
fprintf('Channel Model: Microstrip Patch + Rician Fading\n\n');

%% ========================================================================
%  SECTION 2: MONTE CARLO SIMULATION WITH SEQUENTIAL STATES (s -> s')
% =========================================================================

% Performance degradation tracking (from V3)
substrate_loss_dB = 0.1 * sqrt(fc/1e9); 
substrate_loss_linear = 10^(-substrate_loss_dB/20);

% Initialize first state (t=0)
user_angles_prev = (rand(K, 1) - 0.5) * 100;
H_prev = generate_channel_with_rician_V4(N_tx, K, user_angles_prev, N_paths, angular_spread, d, lambda, rician_K);

tic;

for iter = 1:N_iterations
    
    if mod(iter, 500) == 0
        fprintf('  Iteration %d/%d\n', iter, N_iterations);
    end
    
    % 1. DEFINE CURRENT STATE (s_t)
    H_t_ideal = H_prev;
    Angles_t = user_angles_prev;
    
    % 2. SIMULATE CHANNEL EVOLUTION (to get s_{t+1})
    
    % Small angle perturbation (Simulating movement during one time step)
    % Perturb the angle slightly based on a random walk (Max change 0.5 deg)
    angle_perturbation = 1.0 * (rand(K, 1) - 0.5); 
    user_angles_next = Angles_t + angle_perturbation; 
    user_angles_next = max(-50, min(50, user_angles_next)); % Keep in range
    
    % Generate the Next State Channel (H_{t+1})
    H_next_ideal = generate_channel_with_rician_V4(N_tx, K, user_angles_next, N_paths, angular_spread, d, lambda, rician_K); 
    
    % 3. APPLY PHYSICAL EFFECTS (to H_t and H_next)
    
    % Generate Coupling Matrix for current state (C_t)
    C_t = generate_coupling_matrix_V4(N_tx, d, lambda, coupling_enabled, coupling_strength);

    % Apply microstrip effects to current channel (H_t)
    [H_patch_t, ~] = apply_microstrip_effects_V4(H_t_ideal, Angles_t, N_tx, K, C_t, exponent_n, front_to_back_ratio, half_power_beamwidth);
    
    % Apply microstrip effects to next channel (H_{t+1})
    [H_patch_next, ~] = apply_microstrip_effects_V4(H_next_ideal, user_angles_next, N_tx, K, C_t, exponent_n, front_to_back_ratio, half_power_beamwidth);

    % Normalize and apply fixed substrate loss
    H_norm_t = (H_patch_t / sqrt(trace(H_patch_t * H_patch_t') / (N_tx * K))) * substrate_loss_linear;
    H_norm_next = (H_patch_next / sqrt(trace(H_patch_next * H_patch_next') / (N_tx * K))) * substrate_loss_linear;

    % 4. CALCULATE OPTIMAL ACTION AND REWARD (at fixed SNR)
    
    % We only use the single fixed SNR point for training
    SNR_linear = 10^(SNR_dB_range(1)/10);
    
    % Calculate optimal beam W_SLNR based on current state H_t (The optimal action)
    W_SLNR_t = beamforming_SLNR_V4(H_norm_t, N_tx, K, SNR_linear);

    % Get SINR_t (Performance of the optimal beam in the current state)
    [SINR_t, ~] = compute_performance_V4(H_norm_t, W_SLNR_t, SNR_linear, noise_power);
    
    % === V4 REFINEMENT 1B: DISCRETIZE W_SLNR ===
    A_Best_Index_t = zeros(K, 1);
    for k = 1:K
        W_k_optimal = W_SLNR_t(:, k);
        alignment_scores = abs(Codebook' * W_k_optimal);
        [~, A_Best_Index_t(k)] = max(alignment_scores); 
    end

    % Get SINR_next (Performance of the optimal beam W_SLNR_t in the NEXT state H_{t+1})
    [SINR_next, ~] = compute_performance_V4(H_norm_next, W_SLNR_t, SNR_linear, noise_power);
    
    % 5. EXPORT SEQUENTIAL DATA
    
    dataset_H_current{iter} = H_norm_t;
    dataset_H_next{iter} = H_norm_next;
    dataset_angles_current(iter, :) = Angles_t;
    dataset_angles_next(iter, :) = user_angles_next;
    dataset_SINR_current(iter, :) = SINR_t.';
    dataset_SINR_next(iter, :) = SINR_next.';
    dataset_A_Best_Index(iter, :) = A_Best_Index_t.';
    
    % 6. PREPARE FOR NEXT ITERATION
    H_prev = H_next_ideal; % Use ideal H for next loop's starting point
    user_angles_prev = user_angles_next;
end

elapsed_time = toc;
fprintf('Simulation completed in %.2f seconds.\n', elapsed_time);

%% ========================================================================
%  SECTION 3: DATASET EXPORT FOR ML TRAINING
% =========================================================================

fprintf('\n=== Exporting FINAL ML Training Dataset (V4) ===\n');

ML_Dataset_V4.Description = 'RL-Ready Dataset: Microstrip Patch ULA, SLNR Labels, Sequential States';
ML_Dataset_V4.Parameters.N_tx = N_tx;
ML_Dataset_V4.Parameters.K = K;
ML_Dataset_V4.Parameters.N_iterations = N_iterations;
ML_Dataset_V4.Parameters.Fixed_SNR_dB = SNR_dB_range(1);

% Features for RL State (s_t)
ML_Dataset_V4.Features.H_Current = dataset_H_current;
ML_Dataset_V4.Features.Angles_Current = dataset_angles_current;
ML_Dataset_V4.Features.SINR_Current = dataset_SINR_current;

% Features for Next State (s_{t+1})
ML_Dataset_V4.Features.H_Next = dataset_H_next;
ML_Dataset_V4.Features.Angles_Next = dataset_angles_next;
ML_Dataset_V4.Features.SINR_Next = dataset_SINR_next;

% Label/Action (a)
ML_Dataset_V4.Labels.A_Best_Index = dataset_A_Best_Index; % Target discrete action (1 to 32)
ML_Dataset_V4.Labels.Codebook = Codebook; 

% Save to MAT file
save('beamforming_dataset_v4_RL_ready.mat', 'ML_Dataset_V4', '-v7.3');

fprintf('Dataset saved to: beamforming_dataset_v4_RL_ready.mat\n');
fprintf('  - Ready for Python/TensorFlow Q-Learning framework.\n');
fprintf('  - H_Current and H_Next are stored for sequential training.\n');

%% ========================================================================
%  SECTION 4: SUPPORTING FUNCTIONS (Renamed to V4)
% =========================================================================

% Function: SLNR Beamforming (Completed)
function W = beamforming_SLNR_V4(H, N_tx, K, SNR_linear)
    W = zeros(N_tx, K);
    for k = 1:K
        h_k = H(k, :).';
        
        % Leakage-plus-Noise Covariance Matrix (B)
        B = (1 / SNR_linear) * eye(N_tx);
        for j = 1:K
            if j ~= k
                h_j = H(j, :).';
                B = B + h_j * h_j'; 
            end
        end
        
        % Signal Covariance Matrix (A)
        A = h_k * h_k';
        
        % Solve Generalized Eigenvalue Problem: A*w = lambda*B*w
        [V, D] = eig(A, B);
        
        % Optimal w_k is the eigenvector corresponding to the largest eigenvalue
        [~, max_idx] = max(diag(D));
        w_k = V(:, max_idx);
        
        W(:, k) = w_k / norm(w_k) * sqrt(N_tx);
    end
end

% Function: Generate Channel with Rician Fading (Partial LOS)
function H = generate_channel_with_rician_V4(N_tx, K, angles, N_paths, spread, d, lambda, K_dB)
    H = zeros(K, N_tx);
    K_linear = 10^(K_dB/10);
    
    for k = 1:K
        % LOS component
        a_los = array_response_V4(N_tx, angles(k), d, lambda);
        h_los = sqrt(K_linear / (K_linear + 1)) * a_los;
        
        % NLOS components (Rayleigh)
        h_nlos = zeros(N_tx, 1);
        path_angles = angles(k) + spread * (rand(N_paths, 1) - 0.5);
        path_gains = (randn(N_paths, 1) + 1i*randn(N_paths, 1)) / sqrt(2 * N_paths);
        
        for p = 1:N_paths
            a_p = array_response_V4(N_tx, path_angles(p), d, lambda);
            h_nlos = h_nlos + path_gains(p) * a_p;
        end
        h_nlos = sqrt(1 / (K_linear + 1)) * h_nlos;
        
        H(k, :) = (h_los + h_nlos).';
    end
    H = H / sqrt(mean(abs(H(:)).^2)) * sqrt(N_tx);
end

% Function: Apply Microstrip Antenna Effects to Channel
function [H_patch, element_gains] = apply_microstrip_effects_V4(H_ideal, angles, N_tx, K, C, n, fb_ratio, hpbw)
    % C is the coupling matrix
    element_gains = zeros(K, N_tx);
    
    % Apply element pattern to each user
    for k = 1:K
        g = microstrip_element_pattern_V4(angles(k), n, fb_ratio);
        element_gains(k, :) = sqrt(g); % All elements see same angle in ULA
    end
    
    % H_ideal (K x N_tx) .* element_gains (K x N_tx)
    H_pattern = H_ideal .* element_gains;
    
    % Apply mutual coupling (H_pattern * C)
    H_patch = H_pattern * C;
end

% Function: Microstrip Element Pattern
function g = microstrip_element_pattern_V4(theta_deg, n, fb_ratio_dB)
    theta_rad = deg2rad(theta_deg);
    g_front = max(cos(theta_rad), 0).^n;
    fb_ratio_linear = 10^(-fb_ratio_dB/10);
    g_back = fb_ratio_linear * max(cos(theta_rad + pi), 0).^n;
    g = g_front + g_back;
    g = g / max(g(:));
end

% Function: Generate Mutual Coupling Matrix
function C = generate_coupling_matrix_V4(N, d, lambda, enabled, coupling_dB)
    C = eye(N);
    if ~enabled, return; end
    
    coupling_linear = 10^(coupling_dB/20);
    
    for i = 1:N
        for j = 1:N
            if i ~= j
                dist = abs(i - j) * d / lambda;
                C(i, j) = coupling_linear / dist;
            end
        end
    end
end

% Function: Array Response Vector (ULA)
function a = array_response_V4(N, theta_deg, d, lambda)
    theta_rad = deg2rad(theta_deg);
    n = (0:N-1).';
    a = exp(1i * 2 * pi * d / lambda * n * sin(theta_rad)) / sqrt(N);
end

% Function: Compute SINR and Sum Capacity
function [SINR_dB, sum_capacity] = compute_performance_V4(H, W, SNR_linear, noise_power)
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
