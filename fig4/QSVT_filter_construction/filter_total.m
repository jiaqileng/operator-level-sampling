function filter_fun = filter_total(Gap, deg, npts)
% -------------------------------------------------------------------------
% FILTER_TOTAL — Chebyshev filter approximation
% -------------------------------------------------------------------------
% Summary:
%   Builds a normalized filter approximation using Chebyshev coefficients.
%
% Syntax:
%   filter_fun = filter_total(Gap, deg, npts)
%
% Inputs:
%   Gap  — Spectral gap parameter that controls filter sharpness.
%   deg  — Polynomial degree for the Chebyshev approximation.
%   npts — Number of FFT sampling points for coefficient calculation.
%
% Output:
%   filter_fun — Function handle that evaluates the normalized filter.
%
% Author:  Zherui Chen
% Created: 2025-02-24
% -------------------------------------------------------------------------

%% Target filter definition
% Smooth-step filter centered near 0 with width tied to Gap.
a = -Gap/4;
b = -a;
delta = Gap/2;
F = @(x) smoothStep(x, a, b, delta);

%% Chebyshev coefficients (FFT-based routine, external dependency)
[coef] =  chebyshev_coeffs_fft(F, npts, deg);

%% Parity-selective expansion to full coefficient vector
coef_full = zeros(deg+1,1);
parity = 0;             % 0: use odd indices (1-based 1:2:end), 1: use even indices
if (parity == 0)
  coef_full(1:2:end) = coef(1:2:end);
else
  coef_full(2:2:end) = coef(2:2:end);
end

%% Chebyshev series evaluator on x (expects x within [-1, 1])
sol_cheb = @(x) sum(coef_full .* cos((0:numel(coef_full)-1)' * acos(x)), 1);

%% Segmented evaluation parameters (memory-friendly)
num_segments = 20;          % Number of sub-intervals in [-1, 1]
points_per_segment = 1000;  % Samples per sub-interval

%% Accumulators for concatenated samples (diagnostics/normalization)
x_total = [];
y_sol_total = [];
y_exact_total = [];

%% Piecewise evaluation of sol_cheb and exact filter F over [-1, 1]
for i = 1:num_segments
    % Current segment in [-1, 1]
    a_seg = -1 + (i-1) * (2 / num_segments);
    b_seg = -1 + i * (2 / num_segments);

    % x-grid on the current segment
    x_seg = linspace(a_seg, b_seg, points_per_segment);

    % Evaluate Chebyshev approximation and reference filter
    y_sol_seg = sol_cheb(x_seg);
    y_exact_seg = F(x_seg);

    % Avoid duplicating shared endpoints between adjacent segments
    if i < num_segments
        x_seg = x_seg(1:end-1);
        y_sol_seg = y_sol_seg(1:end-1);
        y_exact_seg = y_exact_seg(1:end-1);
    end

    % Concatenate segment data
    x_total = [x_total, x_seg];
    y_sol_total = [y_sol_total, y_sol_seg];
    y_exact_total = [y_exact_total, y_exact_seg];
end

%% Additional wide-range probe for normalization
% Reuses y_sol_seg as a variable name (structure preserved).
y_sol_seg = linspace(-Gap*5, Gap*5, 2000);
y_sol_seg = sol_cheb(y_sol_seg);
y_sol_total = [y_sol_total, y_sol_seg];

%% Compute normalization factors from assembled samples
max_y = max(y_sol_total);
min_y = min(y_sol_total); %#ok<NASGU> % Kept for completeness; not used below.

%% Normalized filter (scale by max only, per original choice)
% Alternative min-max normalization retained as a commented line.
% filter_fun = @(x) (sol_cheb(x) - min_y)./(max_y - min_y);
filter_fun = @(x) (sol_cheb(x))./(max_y);

end

