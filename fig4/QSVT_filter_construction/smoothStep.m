function y = smoothStep(x, a, b, delta)
% -------------------------------------------------------------------------
% SMOOTHSTEP — Smooth step function over [-1, 1]
% -------------------------------------------------------------------------
% Summary:
%   Computes a smooth step function that equals 0 outside [a, b] (with
%   cosine tapers of width 'delta' on both sides) and equals 1 on [a, b].
%
% Syntax:
%   y = smoothStep(x, a, b, delta)
%
% Inputs:
%   x     — Evaluation points (vector or array), must satisfy x ∈ [-1, 1].
%   a, b  — Flat-top interval endpoints with a ≤ b.
%   delta — Taper half-width; assumes -1 ≤ a - delta and b + delta ≤ 1.
%
% Output:
%   y     — Same size as x. Piecewise definition:
%             y = 0                                 for x ≤ a - delta or x ≥ b + delta
%             y = 0.5 * (1 - cos(pi*((x - (a - delta))/delta)))   for x ∈ [a - delta, a]
%             y = 1                                 for x ∈ [a, b]
%             y = 0.5 * (1 + cos(pi*((x - b)/delta)))             for x ∈ [b, b + delta]
%
% Example:
%   x = linspace(-1, 1, 500);
%   a = -0.2; b = 0.2; delta = 0.1;
%   y = smoothStep(x, a, b, delta);
%   plot(x, y); xlabel('x'); ylabel('y');
%
% Author:  Zherui Chen
% Created: 2025-02-24
% -------------------------------------------------------------------------

    % Validate that all x are in [-1, 1]
    if any(x < -1) || any(x > 1)
        error('Input x must be in the interval [-1, 1].');
    end

    % Initialize the output y with zeros
    y = zeros(size(x));

    % Left smooth transition: x in [a-delta, a]
    left_idx = (x >= (a - delta)) & (x < a);
    y(left_idx) = 0.5 * (1 - cos(pi * ((x(left_idx) - (a - delta)) / delta)));

    % Flat region: x in [a, b]
    flat_idx = (x >= a) & (x <= b);
    y(flat_idx) = 1;

    % Right smooth transition: x in (b, b+delta]
    right_idx = (x > b) & (x <= (b + delta));
    y(right_idx) = 0.5 * (1 + cos(pi * ((x(right_idx) - b) / delta)));
    
    % The remaining portions (for x < a-delta and x > b+delta) are already set to 0.
end
