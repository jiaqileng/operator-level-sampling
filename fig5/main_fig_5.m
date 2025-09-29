% -------------------------------------------------------------------------
% INVERSE SPECTRAL GAPS OF DIFFERENT DYNAMICS
% -------------------------------------------------------------------------
% Summary:
%   Computes (approximate) inverse spectral gaps for several dynamics
%   (LD, Quantum-accelerated LD, RELD, Quantum-accelerated RE) over a range 
%   of beta values.
%   Stores the four gap vectors under ./data for downstream Python use.
%
% Outputs (under ./data):
%   - spectral_gap.mat
%   - spectral_gap_cla.mat
%   - spectral_gap_witten_lap.mat
%   - spectral_gap_Lx_witten.mat
%  
%
% Author:  Zherui Chen
% Created: 2025-02-24
% -------------------------------------------------------------------------

%% Parameters
clear;
tic;

mkdir data

V_func        = @(x) cos(pi*x).^2 + 0.25 * x.^4;
U             = V_func;
der_V_func    = @(x) -2 * pi * cos(pi*x) .* sin(pi*x) + x.^3;
derder_V_func = @(x) -2*pi*pi*cos(2*pi*x) + 3 * x.^2;

dU_dx = der_V_func;
dU_dy = dU_dx;

lb = -2; rb =  2;
N  = 150;                 % grid size (even)
beta        = 2:0.5:10;   % range of inverse temperatures
beta_prime  = 1;
exp_clock   = 1;

if ~isfolder('data'); mkdir('data'); end

%% Grid & spectral differentiation
L = rb - lb;
[Dx, DDx, x_data] = DFT_fun_Dx(lb, rb, N);
x_data = x_data.';                        % row vector for meshgrid
[Y, X] = meshgrid(x_data, x_data);

%% Potential and its derivatives on the grid
vx      = dU_dx(X);     vx  = vx(:);
vy      = dU_dy(Y);     vy  = vy(:);
vx_vec  = dU_dx(x_data);
vxx_vec = derder_V_func(x_data);

%% Swap operator W (N^2 x N^2, sparse)
W = zeros(N^2, N^2);
for x = 1:N
    for y = 1:N
        colIndex = (y-1)*N + x;   % index of u(x,y)
        rowIndex = (x-1)*N + y;   % index of u(y,x)
        W(rowIndex, colIndex) = 1;
    end
end
W = sparse(W);

%% Operators for 2D dynamics
Vx = spdiags(vx, 0, N*N, N*N);
Vy = spdiags(vy, 0, N*N, N*N);
px = -1i * kron(speye(N), Dx);
py = -1i * kron(Dx, speye(N));
Ly = 1/sqrt(beta_prime) * py - 1i * sqrt(beta_prime)/2 * Vy;

%% Storage
spectral_gap            = zeros(size(beta));
spectral_gap_cla        = zeros(size(beta));
spectral_gap_witten_lap = zeros(size(beta));
spectral_gap_Lx_witten  = zeros(size(beta));

%% Main loop over beta
parfor ii = 1:numel(beta)
    % QSVT-inspired block operators
    s_proba_matrix          = exp(min(0, (beta(ii) - beta_prime) * (U(X) - U(Y))));
    s_proba_tilde           = spdiags(s_proba_matrix(:), 0, N*N, N*N);
    temp_ = transpose(s_proba_matrix);
    s_proba_tilde_transpose = spdiags(temp_(:), 0, N*N, N*N);
    Lx  = 1/sqrt(beta(ii)) * px - 1i * sqrt(beta(ii))/2 * Vx;
    Ls  = sqrt(exp_clock/2) * (sqrt(s_proba_tilde) - sqrt(s_proba_tilde_transpose) * W);

    A_cla = Lx'*Lx + Ly'*Ly + Ls'*Ls;
    A     = [Lx; Ly; Ls];

    k = 3;  % number of smallest singular values to compute

    % Smallest singular values (QSVT stack and normal equations)
    [~, Sigma,     ~] = svds(A,     k, 'smallest');
    [~, Sigma_cla, ~] = svds(A_cla, k, 'smallest');

    svals     = diag(Sigma);
    svals_cla = diag(Sigma_cla);

    spectral_gap(ii)     = svals(end-1)     - svals(end);
    spectral_gap_cla(ii) = svals_cla(end-1) - svals_cla(end);

    % Witten Laplacian (1D)
    witten_lap = -1/beta(ii) * DDx ...
                 + (beta(ii))/4 * diag(vx_vec) .* diag(vx_vec) ...
                 - 0.5 * diag(vxx_vec);
    witten_lap = sparse(witten_lap);
    [~, Sigma_wit, ~] = svds(witten_lap, k, 'smallest');
    svals_wit = diag(Sigma_wit);
    spectral_gap_witten_lap(ii) = svals_wit(end-1) - svals_wit(end);

    % Lx for Witten (1D)
    Lx_witten = -1i * 1/sqrt(beta(ii)) * Dx - 1i * sqrt(beta(ii))/2 * diag(vx_vec);
    Lx_witten = sparse(Lx_witten);
    [~, Sigma_Lx_w, ~] = svds(Lx_witten, k, 'smallest');
    svals_Lx_w = diag(Sigma_Lx_w);
    spectral_gap_Lx_witten(ii) = svals_Lx_w(end-1) - svals_Lx_w(end);
end

%% Save gaps (each file also stores beta)
save(fullfile('data','spectral_gap.mat'),            'spectral_gap',            'beta');
save(fullfile('data','spectral_gap_cla.mat'),        'spectral_gap_cla',        'beta');
save(fullfile('data','spectral_gap_witten_lap.mat'), 'spectral_gap_witten_lap', 'beta');
save(fullfile('data','spectral_gap_Lx_witten.mat'),  'spectral_gap_Lx_witten',  'beta');

%% Local function: unitary-DFT-based derivative operators
function [D1, D2, x_data] = DFT_fun_Dx(lb, rb, N)
% DFT_FUN_DX — First- and second-derivative via unitary DFT on [lb, rb)
%
% Syntax:
%   [D1, D2, x_data] = DFT_fun_Dx(lb, rb, N)
%
% Inputs:
%   lb, rb — Interval boundaries.
%   N      — Number of grid points (even).
%
% Outputs:
%   D1     — N-by-N first-derivative operator (periodic BCs).
%   D2     — N-by-N second-derivative operator (periodic BCs).
%   x_data — N-by-1 grid points in [lb, rb).
    L = rb - lb;
    dx = L / N;
    x_data = linspace(lb, rb - dx, N).';  % periodic grid (last point excluded)

    % Unitary DFT
    DFT = zeros(N, N);
    for n = 0:N-1
        for m = 0:N-1
            DFT(n+1, m+1) = exp(1i * 2*pi * m * n / N) / sqrt(N);
        end
    end

    % Wave numbers (even N)
    wave_number = [0:(N/2 - 1), -N/2:-1];

    % First derivative: d/dx -> i*k
    d1 = 1i * 2*pi * (wave_number / (rb - lb));
    D1 = - (DFT') * diag(d1) * DFT;

    % Second derivative: (i*k)^2 = -k^2
    d2 = - (2*pi * wave_number / (rb - lb)).^2;
    D2 =   (DFT') * diag(d2) * DFT;
end
