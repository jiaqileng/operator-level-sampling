% -------------------------------------------------------------------------
% VISUALIZATION OF MALA, QSVT, AND GIBBS DISTRIBUTIONS
% DRAW FIG. 4 OF [Operator-Level Quantum Acceleration of Non-Logconcave
% Sampling]
% -------------------------------------------------------------------------
% Summary:
%   Loads probability distributions produced by:
%     - MALA (Metropolis-Adjusted Langevin Algorithm)
%     - QSVT (Quantum Singular Value Transformation)
%     - Gibbs (true distribution)
%
% Requirements (files under ./data/):
%   - beta04MALA.mat, beta06MALA.mat, beta08MALA.mat   (fields: density_storage)
%   - beta04QSVT.mat, beta06QSVT.mat, beta08QSVT.mat   (fields: probability_final)
%   - gibbs_04.mat,  gibbs_06.mat,  gibbs_08.mat       (fields: gibbs_state)
%   - All_Overlap_MALA.mat (field: All_Overlap, row vector length 3)
%   - All_Overlap_QSVT.mat (field: All_Overlap, row vector length 3)
%
% Author:  Zherui Chen
% Created: 2025-02-24
% -------------------------------------------------------------------------

%% Load data
% The cell array proba_MALA_QSVT will store 9 matrices in the following order:
%   1-3: MALA   (beta = 0.4, 0.6, 0.8)
%   4-6: QSVT   (beta = 0.4, 0.6, 0.8)
%   7-9: Gibbs  (beta = 0.4, 0.6, 0.8)

proba_MALA_QSVT = cell(9,1);

% MALA results (density_storage is a cell array; {end,1} holds final density)
temp = load('data/beta04MALA.mat');  proba_MALA_QSVT{1,1} = temp.density_storage{end,1};
temp = load('data/beta06MALA.mat');  proba_MALA_QSVT{2,1} = temp.density_storage{end,1};
temp = load('data/beta08MALA.mat');  proba_MALA_QSVT{3,1} = temp.density_storage{end,1};

% QSVT results (probability_final holds the final probability distribution)
temp = load('data/beta04QSVT.mat'); proba_MALA_QSVT{4,1} = temp.probability_final;
temp = load('data/beta06QSVT.mat'); proba_MALA_QSVT{5,1} = temp.probability_final;
temp = load('data/beta08QSVT.mat'); proba_MALA_QSVT{6,1} = temp.probability_final;

% Gibbs (true distribution)
temp = load('data/gibbs_04.mat');   proba_MALA_QSVT{7,1} = temp.gibbs_state;
temp = load('data/gibbs_06.mat');   proba_MALA_QSVT{8,1} = temp.gibbs_state;
temp = load('data/gibbs_08.mat');   proba_MALA_QSVT{9,1} = temp.gibbs_state;

% Grid size parameter used only for tick labeling
N = 50;

%% Recompute overlap from files
% Concatenate:
%   - MALA overlaps from data/All_Overlap_MALA.mat
%   - QSVT overlaps from data/All_Overlap_QSVT.mat
% Final shape: 6x1 (first 3 MALA, next 3 QSVT).

% Load MALA overlaps
temp = load('data/All_Overlap_MALA.mat');  % expects temp.All_Overlap (row vector of length 3)
overlap_MALA = temp.All_Overlap(:);        % ensure column vector

% Load QSVT overlaps
temp = load('data/All_Overlap_QSVT.mat');  % expects temp.All_Overlap (row vector of length 3)
overlap_QSVT = temp.All_Overlap(:);        % ensure column vector

% Concatenate into a single 6x1 vector: [MALA(3); QSVT(3)]
overlap = [overlap_MALA; overlap_QSVT];

% Optional sanity check
if numel(overlap) ~= 6
    warning('Expected overlap to have 6 elements (3 MALA + 3 QSVT). Found %d.', numel(overlap));
end

%% Figure and global visual settings
% Create figure and set fonts
figure('Position', [100, 100, 1000, 1100], 'Color', 'w');

% Set default fonts for axes and text to Times New Roman
set(groot, 'DefaultAxesFontName', 'Times New Roman');
set(groot, 'DefaultTextFontName', 'Times New Roman');

% Collect global min/max across all 9 matrices for shared color scaling
allVals = [];
for i = 1:9
    % Concatenate values (NaNs are ignored by caxis)
    allVals = [allVals; proba_MALA_QSVT{i}(:)];
end

%% Build nonlinear (gamma) colormap
% The gamma-corrected mapping emphasizes low-intensity regions.
% gammaVal < 1 boosts low values; gammaVal > 1 compresses low values.
nColors = 1000;              % number of colors sampled from base colormap
originalCMap = jet(nColors); % base colormap
gammaVal = 0.45;             % gamma value

% Normalized indices in [0, 1]
x_color = linspace(0, 1, nColors);

% Apply gamma correction
xGamma = x_color .^ gammaVal;

% Interpolate base colormap over gamma-corrected indices
newCMap = interp1(x_color, originalCMap, xGamma);

%% Plotting layout
% 3x3 tiled layout; upsample, transpose for orientation, and share color scale
tiledlayout(3, 3, 'TileSpacing', 'compact');

for i = 1:9
    % Select next tile
    ax = nexttile;

    % Extract the i-th distribution matrix
    temp_rho = proba_MALA_QSVT{i, 1};

    % Upsample for smoother visuals without changing data semantics
    factor = 10;  % upscaling factor
    temp_rho_up = imresize(transpose(temp_rho), factor, 'bicubic');

    % Render as an image
    imagesc(temp_rho_up);
    axis image tight;     % square pixels and tight axes

    % Ensure Y-axis increases upwards (matrix row 1 at bottom)
    set(ax, 'YDir', 'normal');
    axis(ax, 'equal', 'tight');

    % Shared color scaling across all subplots
    caxis(ax, [min(allVals), 0.075]);

    % Titles for MALA (1–3) and QSVT (4–6) include overlap; Gibbs (7–9) omit it
    if i == 1
        tStr = sprintf('$$\\mathrm{iter}_1 = 2400,\\quad \\mathrm{overlap} = %.4f$$', overlap(i, 1));
        title(ax, tStr, 'Interpreter', 'latex', 'FontSize', 11);
    elseif i == 2
        tStr = sprintf('$$\\mathrm{iter}_2 = 9 \\times \\mathrm{iter}_1,\\quad \\mathrm{overlap} = %.4f$$', overlap(i, 1));
        title(ax, tStr, 'Interpreter', 'latex', 'FontSize', 11);
    elseif i == 3
        tStr = sprintf('$$\\mathrm{iter}_3 = 9^2 \\times \\mathrm{iter}_1,\\quad \\mathrm{overlap} = %.4f$$', overlap(i, 1));
        title(ax, tStr, 'Interpreter', 'latex', 'FontSize', 11);
    elseif i == 4
        tStr = sprintf('$$\\mathrm{deg}_1 = 1000,\\quad \\mathrm{overlap} = %.4f$$', overlap(i, 1));
        title(ax, tStr, 'Interpreter', 'latex', 'FontSize', 11);
    elseif i == 5
        tStr = sprintf('$$\\mathrm{deg}_2 = 3 \\times \\mathrm{deg}_1,\\quad \\mathrm{overlap} = %.4f$$', overlap(i, 1));
        title(ax, tStr, 'Interpreter', 'latex', 'FontSize', 11);
    elseif i == 6
        tStr = sprintf('$$\\mathrm{deg}_3 = 3^2 \\times \\mathrm{deg}_1,\\quad \\mathrm{overlap} = %.4f$$', overlap(i, 1));
        title(ax, tStr, 'Interpreter', 'latex', 'FontSize', 11);
    end

    % Axis ticks and labels
    xticks(ax, [1, 0.33*factor*N, 0.33*2*factor*N, 0.33*3*factor*N]);
    xticklabels(ax, {'0','1','2','3'});
    yticks(ax, [1, 0.33*factor*N, 0.33*2*factor*N, 0.33*3*factor*N]);
    yticklabels(ax, {'0','1','2','3'});

    % Apply nonlinear colormap
    colormap(ax, newCMap);
end

%% Annotations
% Column headers for beta values
annotation('textbox',[0.20 0.93 0.1 0.05], 'String','\beta=0.4', ...
           'LineStyle','none','HorizontalAlignment','center','FontSize',18);
annotation('textbox',[0.47 0.93 0.1 0.05], 'String','\beta=0.6', ...
           'LineStyle','none','HorizontalAlignment','center','FontSize',18);
annotation('textbox',[0.74 0.93 0.1 0.05], 'String','\beta=0.8', ...
           'LineStyle','none','HorizontalAlignment','center','FontSize',18);

% Row labels
annotation('textbox',[0.47 0.90 0.1 0.05], 'String','MALA', ...
           'LineStyle','none','HorizontalAlignment','center','FontSize',14);
annotation('textbox',[0.315 0.62 0.4 0.05], 'String','Quantum-accelerated Langevin dynamics', ...
           'LineStyle','none','HorizontalAlignment','center','FontSize',14);
annotation('textbox',[0.32 0.33 0.4 0.05], 'String','True distribution', ...
           'LineStyle','none','HorizontalAlignment','center','FontSize',14);

% Single colorbar at the far right
cb = colorbar;
cb.Layout.Tile = 'east';










