% -------------------------------------------------------------------------
% MALA
% -------------------------------------------------------------------------
% Summary:
%   Runs MALA with time horizons T = 2.4, 2.4*9, 2.4*9*9 for beta = 0.4, 0.6, 0.8
%   respectively. Saves density snapshots and computes overlap with the Gibbs
%   distribution for each beta.
%
% Outputs:
%   - data/beta04MALA.mat          (density_storage for beta = 0.4)
%   - data/beta06MALA.mat          (density_storage for beta = 0.6)
%   - data/beta08MALA.mat          (density_storage for beta = 0.8)
%   - data/All_Overlap_MALA.mat    (vector All_Overlap with three overlaps)
%
% Author:  Zherui Chen
% Created: 2025-02-24
% -------------------------------------------------------------------------

%% Housekeeping
close all
clear
tic

if ~isfolder('data')
    mkdir('data');
end

%% Setup
% Constants for Müller–Brown potential
k = 0.15;
d_MB = [-200, -100, -170, 15];
a = [-1, -1, -6.5, 0.7];
b = [0, 0, 11, 0.6];
c = [-10, -10, -6.5, 0.7];
x0_MB = [1, 0, -0.5, -1];
y0_MB = [0, 0.5, 1.5, 1];

% Potential and gradient handles
U = @(x,y) U_func(x,y,a,b,c,d_MB,x0_MB,y0_MB,k);
V = @(x,y) U_func(x,y,a,b,c,d_MB,x0_MB,y0_MB,k);
gradV_x = @(x, y) dU_dx_func(x, y, a, b, c, d_MB, x0_MB, y0_MB, k);
gradV_y = @(x, y) dU_dy_func(x, y, a, b, c, d_MB, x0_MB, y0_MB, k);

% Initial Gaussian mean/covariance (sigma depends on beta below)
mu_init = [2.33; 0.54];

% Trajectory and grid parameters
each_sample_num = 1e3;
n_traj = 30*each_sample_num;   % Number of initial points
x_min = 0; x_max = 3;
y_min = 0; y_max = 3;
N = 50;                        % Grid resolution
grid_x = linspace(x_min, x_max, N+1);
grid_y = linspace(y_min, y_max, N+1);

% MALA time-stepping parameters
dt = 1e-3;
save_step = 10;                % Store ~10 snapshots over [0, T]
mkdir('data');

% Beta/T configurations
beta_list = [0.4, 0.6, 0.8];
T_list    = [2.4, 2.4*9, 2.4*9*9];

% To collect final overlaps for each beta
All_Overlap = zeros(1, numel(beta_list));

%% Main loop over beta
for idx = 1:numel(beta_list)
    beta = beta_list(idx);
    T    = T_list(idx);

    % Set sigma_init per beta (preserving the original mapping)
    if beta == 0.4
        sigma_init = 1/sqrt(2*1000);
    elseif beta == 0.6
        sigma_init = 1/sqrt(2*500);
    elseif beta == 0.8
        sigma_init = 1/sqrt(2*70);
    end
    Sigma_init = (sigma_init)^2 .* eye(2);

    % Derived timing and storage parameters
    n_iter = ceil(T/dt);
    capture_every = ceil(n_iter ./ save_step);
    save_indices = 1 : capture_every : n_iter;
    save_indices(end) = n_iter;    % Ensure final step is saved
    save_times_indices = save_indices * dt; %#ok<NASGU>
    num_steps_to_track = length(save_indices);

    % Diffusion coefficient
    sigma = sqrt(2 * dt / beta);

    % Storage for parallel batches (columns are batches; rows are time snapshots)
    saved_trajectories = cell(num_steps_to_track, ceil(n_traj/each_sample_num));

    %% MALA sampling (parallel over batches)
    parfor ij = 1 : ceil(n_traj/each_sample_num)
        % Draw initial points from N(mu_init, Sigma_init)
        X_initial = mvnrnd(mu_init, Sigma_init, each_sample_num)';  % 2 x each_sample_num
        X = X_initial;

        % Per-batch density storage
        density_storage = cell(num_steps_to_track, 1);
        density_counter = 1;

        for step = 1:n_iter
            % Current positions
            x_curr = X(1, :);
            y_curr = X(2, :);

            % Gradient at current positions
            gradV_x_curr = gradV_x(x_curr, y_curr);
            gradV_y_curr = gradV_y(x_curr, y_curr);

            % MALA proposal
            x_prop = x_curr - dt .* gradV_x_curr + sigma .* randn(1, each_sample_num);
            y_prop = y_curr - dt .* gradV_y_curr + sigma .* randn(1, each_sample_num);

            % Gradient at proposed positions
            gradV_x_prop = gradV_x(x_prop, y_prop);
            gradV_y_prop = gradV_y(x_prop, y_prop);

            % Potential difference
            V_prop = V(x_prop, y_prop);
            V_curr = V(x_curr, y_curr);
            dV = V_prop - V_curr;

            % Forward/backward Gaussian proposal corrections
            forwardDiff_x = x_prop - x_curr + dt .* gradV_x_curr;
            forwardDiff_y = y_prop - y_curr + dt .* gradV_y_curr;
            backwardDiff_x = x_curr - x_prop + dt .* gradV_x_prop;
            backwardDiff_y = y_curr - y_prop + dt .* gradV_y_prop;

            forwardNormSqr  = forwardDiff_x.^2 + forwardDiff_y.^2;
            backwardNormSqr = backwardDiff_x.^2 + backwardDiff_y.^2;

            % Log-acceptance and accept/reject
            log_alpha = -beta .* dV - (beta ./ (4 * dt)) .* (backwardNormSqr - forwardNormSqr);
            u = rand(1, each_sample_num);
            accept = log(u) < log_alpha;

            % Update accepted proposals
            X(1, accept) = x_prop(accept);
            X(2, accept) = y_prop(accept);

            % Record density at selected steps
            if ismember(step, save_indices)
                [counts, ~, ~] = histcounts2(X(1, :), X(2, :), grid_x, grid_y);
                density_storage{density_counter} = counts / each_sample_num;
                density_counter = density_counter + 1;
            end
        end

        saved_trajectories(:, ij) = density_storage;
    end

    %% Average across batches
    for i = 1:num_steps_to_track
        current_row_cells = saved_trajectories(i, :);
        matrices_stack = cat(3, current_row_cells{:});
        mean_matrix = mean(matrices_stack, 3);
        mean_trajectories{i,1} = mean_matrix; %#ok<SAGROW>
    end
    density_storage = mean_trajectories; %#ok<NASGU>

    %% Overlap with Gibbs distribution
    [X_grid,Y_grid] = meshgrid(linspace(x_min, x_max, N), linspace(y_min, y_max, N));
    V_grid = U(X_grid, Y_grid);

    gibbs_state = exp(-beta * V_grid);
    gibbs_state = gibbs_state / sum(gibbs_state(:));

    density = density_storage{end,1};
    density = density ./ sum(density(:));
    density_trans = transpose(density);

    Overlap = sqrt(density_trans(:)).' * sqrt(gibbs_state(:));
    All_Overlap(idx) = Overlap;  % Collect the three overlaps

    %% Save results for this beta
    if beta == 0.4
        save(fullfile('data','beta04MALA.mat'), 'density_storage');
    elseif beta == 0.6
        save(fullfile('data','beta06MALA.mat'), 'density_storage');
    elseif beta == 0.8
        save(fullfile('data','beta08MALA.mat'), 'density_storage');
    end

    clear saved_trajectories mean_trajectories

    fprintf('beta = %.1f\n', beta);  
end

% Save the overlap vector for all betas
save(fullfile('data','All_Overlap_MALA.mat'), 'All_Overlap');

%% Local functions
function z = U_func(x,y,a,b,c,d,x0,y0,k)
% U_FUNC  Müller–Brown potential evaluated at (x, y).
%
%   z = U_FUNC(x, y, a, b, c, d, x0, y0, k) evaluates the (scaled)
%   Müller–Brown potential with shifts (1.7, 0.5) and offsets (x0(ii), y0(ii)).
%
% Inputs:
%   x, y   - Grid coordinates.
%   a, b, c, d - Müller–Brown parameters (1x4 arrays).
%   x0, y0 - Per-term coordinate offsets (1x4 arrays).
%   k      - Global scaling factor.
%
% Output:
%   z      - Potential evaluated at (x, y).
    z = 0;
    for ii = 1:4
        xi = x - 1.7 - x0(ii);
        yi = y - 0.5 - y0(ii);
        exp_term = exp(a(ii)*xi.^2 + b(ii)*xi.*yi + c(ii)*yi.^2);
        z = z + d(ii)*exp_term;
    end
    z = k*z;
end

function du_dx = dU_dx_func(x,y,a,b,c,d,x0,y0,k)
% DU_DX_FUNC  ∂/∂x of the Müller–Brown potential.
%
%   du_dx = DU_DX_FUNC(x, y, a, b, c, d, x0, y0, k) returns the x-partial
%   derivative of the scaled Müller–Brown potential.
%
% Inputs:
%   x, y   - Grid coordinates.
%   a, b, c, d - Müller–Brown parameters (1x4 arrays).
%   x0, y0 - Per-term coordinate offsets (1x4 arrays).
%   k      - Global scaling factor.
%
% Output:
%   du_dx  - ∂U/∂x evaluated at (x, y).
    du_dx = 0;
    for ii = 1:4
        xi = x - 1.7 - x0(ii);
        yi = y - 0.5 - y0(ii);
        exp_term = exp(a(ii)*xi.^2 + b(ii)*xi.*yi + c(ii)*yi.^2);
        du_dx = du_dx + d(ii)*exp_term .* (2*a(ii)*xi + b(ii)*yi);
    end
    du_dx = k*du_dx;
end

function du_dy = dU_dy_func(x,y,a,b,c,d,x0,y0,k)
% DU_DY_FUNC  ∂/∂y of the Müller–Brown potential.
%
%   du_dy = DU_DY_FUNC(x, y, a, b, c, d, x0, y0, k) returns the y-partial
%   derivative of the scaled Müller–Brown potential.
%
% Inputs:
%   x, y   - Grid coordinates.
%   a, b, c, d - Müller–Brown parameters (1x4 arrays).
%   x0, y0 - Per-term coordinate offsets (1x4 arrays).
%   k      - Global scaling factor.
%
% Output:
%   du_dy  - ∂U/∂y evaluated at (x, y).
    du_dy = 0;
    for ii = 1:4
        xi = x - 1.7 - x0(ii);
        yi = y - 0.5 - y0(ii);
        exp_term = exp(a(ii)*xi.^2 + b(ii)*xi.*yi + c(ii)*yi.^2);
        du_dy = du_dy + d(ii)*exp_term .* (b(ii)*xi + 2*c(ii)*yi);
    end
    du_dy = k*du_dy;
end








