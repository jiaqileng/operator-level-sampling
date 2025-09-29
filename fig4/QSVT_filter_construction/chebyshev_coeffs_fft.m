function [coef_full]  = chebyshev_coeffs_fft(F, K, d)
% -------------------------------------------------------------------------
% CHEBYSHEV_COEFFS_FFT — Chebyshev coefficients via FFT acceleration
% -------------------------------------------------------------------------
% Summary:
%   Computes the first (d+1) Chebyshev coefficients for approximating F(x)
%   on x ∈ [-1, 1] using an FFT-based summation.
%
% Syntax:
%   coef_full = chebyshev_coeffs_fft(F, K, d)
%
% Inputs:
%   F  — Function handle, evaluable on [-1, 1].
%   K  — Positive integer; defines 2K quadrature nodes θ_l = π l / K.
%   d  — Nonnegative integer; number of coefficients minus 1 (degree).
%
% Output:
%   coef_full — (d+1)-by-1 real vector of Chebyshev coefficients c_j.
%
% Formula:
%   c_j = ((2 - δ_{j0})/(2K)) * (-1)^j * Σ_{l=0}^{2K-1} F(-cos θ_l) * e^{i j θ_l},
%   where θ_l = (π l)/K and δ_{j0} is the Kronecker delta.
%
% Notes:
%   - Uses S = conj(fft(conj(Fvals))) so that S(j+1) = Σ_l Fvals(l) e^{+i (2π j l / N)}
%     with N = 2K, matching the required +i exponent.
%   - For real-valued F, the coefficients are real; we take real(c) at the end.
%
% Author:  Zherui Chen
% Created: 2025-02-24
% -------------------------------------------------------------------------

%% 1) Set up quadrature nodes
theta = pi*(0:2*K-1)/K;   % angles: theta_l
xquad = -cos(theta);      % quadrature pts in [-1,1]

%% 2) Evaluate F at those points
Fvals = arrayfun(F, xquad);  % column vector of size (2K)

%% 3) FFT-based sum  Σ_l Fvals_l e^{+i*j*θ_l}
% Standard MATLAB fft computes: fft(g)(j) = Σ_l g(l) * exp(-2π i l j / N).
% To obtain the +i exponent, use S = conj(fft(conj(Fvals))).
% Then S(j+1) = Σ_l Fvals(l) * e^{+i (2π j l / N)} with N = 2K.
G = conj(Fvals);          % conjugate of data
S = conj(fft(G));         % yields sum_l Fvals(l)*e^{+i*2pi(j*l/N)}

%% 4) Chebyshev coefficients c_j
c = zeros(d+1,1);

% j = 0 case:
%   Desired: c0 = (1/(2K)) * S(1).
%   Implementation below sets c(1) = (1/K)*S(1) and then halves it once.
c(1) = (1/K) * S(1);

% j = 1..d:
for j = 1:d
    c(j+1) = (2/(2*K)) * ((-1)^j) * S(j+1);
end
c(1,1) = c(1,1)/2;

% Take real part if F is real-valued
coef_full = real(c);

end
