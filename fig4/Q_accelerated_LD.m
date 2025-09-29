% -------------------------------------------------------------------------
% QUANTUM-ACCELERATED LANGEVIN DYNAMICS
% -------------------------------------------------------------------------
% Summary:
%   Runs the pipeline for beta in {0.4, 0.6, 0.8}. Tracks overlaps with the
%   Gibbs state and saves results to the 'data' directory.
%
% Outputs:
%   - QSVT results: data/beta04QSVT.mat, data/beta06QSVT.mat, data/beta08QSVT.mat
%   - Gibbs states: data/gibbs_04.mat, data/gibbs_06.mat, data/gibbs_08.mat
%   - Overlap summary: data/All_Overlap_QSVT.mat
%
% Dependencies:
%   Folder 'QSVT_filter_construction' containing:
%     * smoothStep.m
%     * filter_total.m
%
% Author:  Zherui Chen
% Created: 2025-02-24
% -------------------------------------------------------------------------

%% Housekeeping
close all
clear
tic

addpath('QSVT_filter_construction'); % Path to QSVT filter-construction helpers
if ~isfolder('data')
    mkdir('data');
end

%% Setup
% Parameters for the scaled Müller–Brown potential
k_MB = 0.15;
d = [-200, -100, -170, 15];
a = [-1, -1, -6.5, 0.7];
b = [0, 0, 11, 0.6];
c = [-10, -10, -6.5, 0.7];
x0 = [1, 0, -0.5, -1];
y0 = [0, 0.5, 1.5, 1];

% Potential and gradient function handles (Müller–Brown)
U = @(x,y) U_func(x,y,a,b,c,d,x0,y0,k_MB);
dU_dx = @(x,y) dU_dx_func(x,y,a,b,c,d,x0,y0,k_MB);
dU_dy = @(x,y) dU_dy_func(x,y,a,b,c,d,x0,y0,k_MB);

% Spatial domain and grid
lb = 0;          % Left boundary
rb = 3;          % Right boundary
N = 50;          % Number of points per dimension (assumed even)
beta = 0.4;      % Placeholder; overwritten in the loop
deg = 1000;      % Base polynomial degree for filter construction
npts = 1290;     % Base number of quadrature/approximation points

% Build Fourier-spectral first derivative and mesh
L = rb - lb;
[Dx, x_data] = DFT_fun_Dx(lb, rb, N); % DFT-based first-derivative operator
x_data = x_data.';                     % Row vector for meshgrid compatibility
dx = x_data(1,2)-x_data(1,1);          % Grid spacing (kept for completeness)
[Y,X] = meshgrid(x_data, x_data);      % 2D grid

%% Potential (beta-independent)
% Evaluate the potential on the grid and cap large values to avoid overflow
threshold  = 30;
V = U(X,Y);
V(V > threshold) = threshold;

% Evaluate and vectorize gradients; zero-out at clipped regions for stability
vx = dU_dx(X, Y);
vx(V == threshold) = 0;
vx = vx(:);
vy = dU_dy(X, Y);
vy(V == threshold) = 0;
vy = vy(:);

% Estimate gradient magnitude bound R used in alpha normalization
grad_norm = sqrt(vx.^2 + vy.^2);
R = max(grad_norm);

% Alpha depends on beta; recomputed inside the loop (kept here to preserve structure)
alpha = 2*max(pi*N*sqrt(2/beta)/L, sqrt(beta)*R/2); 

%% Output directory & loop setup
outdir = 'data';
if ~exist(outdir,'dir'); mkdir(outdir); end % Ensure output directory exists

betas = [0.4, 0.6, 0.8];                 % Requested beta values
All_Overlap = zeros(1, numel(betas));    % Collect overlaps across betas

% Preserve base values of deg/npts to avoid compounding scale factors
base_deg  = deg;
base_npts = npts;

% =======================================================================
%                             MAIN BETA LOOP
% =======================================================================
for ib = 1:numel(betas)
    beta = betas(ib);   % Keep variable name exactly as used downstream

    %% Gibbs state
    % Build Gibbs distribution and corresponding "wavefunction" psi_gibbs
    gibbs_state = exp(-beta*V);
    gibbs_state = gibbs_state / sum(gibbs_state(:)); % Normalize to probability
    psi_gibbs = sqrt(gibbs_state);                   % Amplitude (for inner products)

    %% Initial state (warm start)
    % Prepare a localized initial distribution depending on beta
    if beta == 0.4
        initial_state = exp(-1000*((X-2.33).^2+(Y-0.54).^2)); % Localized near the small well
    elseif beta == 0.6
        initial_state = exp(-500*((X-2.33).^2+(Y-0.54).^2));  % Localized near the small well
    elseif beta == 0.8
        initial_state = exp(-70*((X-2.33).^2+(Y-0.54).^2));   % Localized near the small well (overlap ≈ 0.091)
    end
    initial_state = initial_state / sum(initial_state(:)); % Normalize to probability

    % Amplitude of the initial state
    psi_initial = sqrt(initial_state);

    %% Build Lx and Ly
    % Recompute normalization factor alpha for this beta
    alpha = 2*max(pi*N*sqrt(2/beta)/L, sqrt(beta)*R/2);

    % Assemble sparse gradient multiplication and momentum operators
    Dy = Dx; % Same derivative stencil along y after reshaping order
    Vx = spdiags(vx, 0, N*N, N*N); % diag of dU/dx over flattened grid
    Vy = spdiags(vy, 0, N*N, N*N); % diag of dU/dy over flattened grid

    % Momentum operators on the 2D tensor grid (ordering consistent with kron usage)
    px = -1i * kron(speye(N),Dx);
    py = -1i * kron(Dx,speye(N));

    % Langevin-inspired generators (non-Hermitian)
    Lx = 1/sqrt(beta) * px - 1i * sqrt(beta)/2 * Vx;
    Ly = 1/sqrt(beta) * py - 1i * sqrt(beta)/2 * Vy;

    % Stack operators; full() to ensure svd operates on a dense matrix
    A = [Lx;Ly];
    A = full(A);

    % Singular Value Decomposition and normalization by alpha
    [U_sig, Sigma, V_sig] = svd(A);
    vec_Sigma = diag(Sigma);
    normalized_singular_value = vec_Sigma./alpha;

    % Spectral gap between the smallest two normalized singular values
    % (by construction, the smallest corresponds to the target mode)
    Gap = normalized_singular_value(end-1,1); % Displayed in console for diagnostics    

    % Reset deg/npts to base before applying beta-dependent scaling
    deg  = base_deg;
    npts = base_npts;
    if beta == 0.4
        deg  = deg;
        npts = npts;
    elseif beta == 0.6
        deg  = deg*3;
        npts = npts*3;
    elseif beta == 0.8
        deg  = deg*9;
        npts = npts*9;
    end

    % Construct approximation to the filter via Chebyshev/Fourier series
    sol_cheb = filter_total(Gap, deg, npts);

    %% Filter & projection
    % Apply the spectral filter to the right singular vectors to form a projector
    filtered_normalized_singular_value = sol_cheb(normalized_singular_value');
    filtered_normalized_singular_value = filtered_normalized_singular_value';
    P_s_filter = V_sig * diag(filtered_normalized_singular_value) * V_sig';

    % Project initial amplitude psi_initial onto the filtered subspace
    psi_final = P_s_filter * psi_initial(:);
    psi_final = psi_final/norm(psi_final);
    psi_final = reshape(psi_final,N,N);

    % Convert final amplitude to probability and compute overlap with Gibbs
    probability_final = conj(psi_final).*psi_final;
    Overlap = abs(psi_gibbs(:)'* psi_final(:));

    % Record overlap for this beta
    All_Overlap(ib) = Overlap;

    %% Save outputs
    % Suffix mapping: 0.4 -> '04', 0.6 -> '06', 0.8 -> '08'
    suffix = sprintf('%02.0f', betas(ib)*10);

    % Save final probability matrix for this beta
    save(fullfile(outdir, ['beta', suffix, 'QSVT.mat']), 'probability_final', '-v7.3');

    % Save Gibbs state matrix for this beta
    save(fullfile(outdir, ['gibbs_', suffix, '.mat']), 'gibbs_state', '-v7.3');
end

% Print the summary vector of overlaps across all betas
save(fullfile(outdir, 'All_Overlap_QSVT.mat'), 'All_Overlap', '-v7.3');
disp('All_Overlap = ')
disp(All_Overlap)

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
    z = k*z; % Global scaling
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

function [D1, x_data] = DFT_fun_Dx(lb, rb, N)
% DFT_FUN_DX  First-derivative matrix via unitary DFT on [lb, rb) with periodic BCs.
%
%   [D1, x_data] = DFT_FUN_DX(lb, rb, N) constructs an N-by-N operator D1
%   approximating d/dx for periodic functions, and returns the equispaced
%   grid points x_data in [lb, rb).
%
% Inputs:
%   lb, rb - Domain boundaries.
%   N      - Number of equispaced grid points (even).
%
% Outputs:
%   D1     - N-by-N first-derivative operator (periodic).
%   x_data - N-by-1 grid points within [lb, rb).
%
% Notes:
%   Uses the unitary DFT with 1/sqrt(N) normalization and wave numbers
%   [0, 1, ..., N/2-1, -N/2, -(N/2-1), ..., -1].
    % 1) Domain and grid
    L = rb - lb;                 % Domain length
    dx = L / N;                  % Grid spacing
    x_data = linspace(lb, rb - dx, N).';  % N points; last excluded (periodicity)

    % 2) Unitary DFT matrix with 1/sqrt(N) normalization
    DFT = zeros(N, N);
    for n = 0:N-1
        for m = 0:N-1
            DFT(n+1, m+1) = exp(1i * 2*pi * m * n / N) / sqrt(N);
        end
    end

    % 3) Wave numbers for even N: [0, 1, ..., N/2-1, -N/2, -(N/2-1), ..., -1]
    wave_number = [0:(N/2 - 1), -N/2:-1];

    % 4) Differentiation multiplier in Fourier space (d/dx -> i*k), k = 2*pi*wave_number/L
    d1 = 1i * 2*pi * (wave_number / L);

    % 5) Assemble D1 = F^H * diag(d1) * F (first-derivative operator)
    %    Minus sign matches the original convention
    D1_hat = diag(d1);
    D1 = - (DFT') * D1_hat * DFT;
end

