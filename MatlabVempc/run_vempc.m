function results = run_vempc(configPath)
%RUN_VEMPC Plaintext variational MPC demo (the unencrypted baseline of VEMPC).
%
%   run_vempc()            uses MatlabVempc/config/vempc_config.json
%   run_vempc(configPath)  uses the JSON config at `configPath`
%   results = run_vempc(...) also returns a struct with trajectories/metrics
%
%   This reproduces the *unencrypted* variational MPC of the paper
%   "Variational Encrypted Model Predictive Control" (Suh et al., IEEE L-CSS
%   2026). The encrypted (CKKS) protocol of Algorithms 1-2 is provided in the
%   Python (vempc/) and Go (GoVempc/) implementations; CKKS is not available
%   natively in MATLAB, and per Fig. 1 of the paper VEMPC reproduces this
%   plaintext trajectory up to bounded encryption error.
%
%   Outputs (trajectories, timing, a figure) are written to MatlabVempc/output/.

    here = fileparts(mfilename('fullpath'));
    addpath(here);   % make the +vempc package reachable regardless of pwd

    if nargin < 1 || isempty(configPath)
        configPath = fullfile(here, 'config', 'vempc_config.json');
    end
    cfg = vempc.loadConfig(configPath);

    % Resolve simulation length with the same precedence as the Go reference:
    % use TSteps if set (> 0), otherwise fall back to T.
    steps = cfg.T;
    if isfield(cfg, 'TSteps') && cfg.TSteps > 0
        steps = cfg.TSteps;
    end
    assert(steps > 0, 'vempc:config', 'Invalid T/TSteps in config.');

    % ------------------------------------------------------------------
    % System: linearized inverted pendulum, ZOH-discretized.
    % ------------------------------------------------------------------
    [Ac, Bc] = vempc.invertedPendulum(cfg.m, cfg.l, cfg.g);
    [A, B]   = vempc.discretizeZOH(Ac, Bc, cfg.dt);
    n   = size(A, 1);
    mIn = size(B, 2);

    % Dimension contract, matching the Go reference's config validation.
    assert(numel(cfg.QDiag) == n,  'vempc:config', 'QDiag length must equal state dim n=%d.', n);
    assert(numel(cfg.RDiag) == mIn, 'vempc:config', 'RDiag length must equal input dim m=%d.', mIn);
    assert(numel(cfg.x0) == n,     'vempc:config', 'x0 length must equal state dim n=%d.', n);
    assert(cfg.QfScale > 0 && cfg.sigma0 > 0 && cfg.lambda > 0 && cfg.K > 0, ...
        'vempc:config', 'QfScale, sigma0, lambda, K must be positive.');

    Q  = diag(cfg.QDiag(:));
    R  = diag(cfg.RDiag(:));
    Qf = cfg.QfScale * Q;

    % Per-step constraints  Gx x <= hx,  Gu u <= hu  (box constraints).
    xBound = [cfg.thetaMax; cfg.omegaMax];
    Gx = [eye(n); -eye(n)];        hx = [xBound; xBound];
    Gu = [eye(mIn); -eye(mIn)];    hu = repmat(cfg.uMax, 2 * mIn, 1);

    % ------------------------------------------------------------------
    % Condensed MPC + variational reformulation.
    % ------------------------------------------------------------------
    mpc = vempc.MPCProblem(A, B, Q, R, Qf, cfg.N);
    [G, hFunc] = mpc.buildConstraintMatrices(Gx, hx, Gu, hu);
    penalty = vempc.ConstraintPenalty(G, hFunc);

    Sigma0 = (cfg.sigma0^2) * eye(mpc.Nm);
    variational = vempc.VariationalMPC(mpc, penalty, cfg.lambda, Sigma0);
    coeffs = vempc.chebyshevReLUCoeffs(cfg.chebOrder, cfg.chebBound, 0);

    x0 = cfg.x0(:);
    fprintf('VEMPC (MATLAB, plaintext variational MPC)\n');
    fprintf('  system: n=%d, m=%d, N=%d, p=%d constraints\n', n, mIn, cfg.N, penalty.p);
    fprintf('  sampling: K=%d, lambda=%.3g, sigma0=%.3g, cheb(order=%d, bound=%.3g, eta=%.3g)\n', ...
        cfg.K, cfg.lambda, cfg.sigma0, cfg.chebOrder, cfg.chebBound, cfg.chebEta);

    % ------------------------------------------------------------------
    % (Optional) standard QP-MPC reference via quadprog, if available.
    % ------------------------------------------------------------------
    hasQuadprog = exist('quadprog', 'file') == 2;
    xsStd = []; usStd = [];
    if hasQuadprog
        qpOpts = optimoptions('quadprog', 'Display', 'off');
        ctrlStd = @(x, warm) standardController(x, mpc, G, hFunc, qpOpts);
        [xsStd, usStd, ~, tStd] = vempc.simulate(x0, ctrlStd, A, B, steps);
        stdCost = vempc.trajectoryCost(xsStd, usStd, Q, R, Qf);
        [stdVx, stdVu] = vempc.maxConstraintViolation(xsStd, usStd, Gx, hx, Gu, hu);
        fprintf('\nStandard MPC (quadprog)\n');
        fprintf('  runtime: %.1f ms\n', tStd * 1000);
        fprintf('  cost:    %.6f\n', stdCost);
        fprintf('  viol:    (x=%.6f, u=%.6f)\n', stdVx, stdVu);
    else
        fprintf('\n(quadprog not found: skipping standard-MPC reference)\n');
    end

    % ------------------------------------------------------------------
    % Variational MPC closed-loop simulation.
    % ------------------------------------------------------------------
    rng(cfg.seed, 'twister');   % reproducible per-step sample seeds
    ctrlVar = @(x, warm) variationalController(x, variational, penalty, cfg, coeffs);
    [xsVar, usVar, infoVar, tVar] = vempc.simulate(x0, ctrlVar, A, B, steps);

    varCost = vempc.trajectoryCost(xsVar, usVar, Q, R, Qf);
    [varVx, varVu] = vempc.maxConstraintViolation(xsVar, usVar, Gx, hx, Gu, hu);
    iterMs = [infoVar.iterMs];
    accept = [infoVar.accept];

    fprintf('\nVariational MPC\n');
    fprintf('  runtime:      %.1f ms total (%.3f ms/step mean)\n', tVar * 1000, mean(iterMs));
    fprintf('  cost:         %.6f\n', varCost);
    fprintf('  viol:         (x=%.6f, u=%.6f)\n', varVx, varVu);
    fprintf('  feasible/step %.1f of %d (mean)\n', mean(accept), cfg.K);

    % ------------------------------------------------------------------
    % Save outputs and plot.
    % ------------------------------------------------------------------
    outDir = fullfile(here, 'output');
    if ~isfolder(outDir); mkdir(outDir); end
    writematrix(xsVar, fullfile(outDir, 'xs_variational.csv'));
    writematrix(usVar, fullfile(outDir, 'us_variational.csv'));
    if hasQuadprog
        writematrix(xsStd, fullfile(outDir, 'xs_standard.csv'));
        writematrix(usStd, fullfile(outDir, 'us_standard.csv'));
    end

    figPath = fullfile(outDir, 'variational_mpc.png');
    plotResults(cfg, xsVar, usVar, xsStd, usStd, figPath);
    fprintf('\nSaved CSV + figure to %s\n', outDir);

    if nargout > 0
        results = struct('A', A, 'B', B, 'mpc', mpc, 'variational', variational, ...
            'xsVar', xsVar, 'usVar', usVar, 'xsStd', xsStd, 'usStd', usStd, ...
            'cost', varCost, 'viol', [varVx, varVu], 'iterMs', iterMs, ...
            'accept', accept, 'elapsed', tVar);
    end
end

% ----------------------------------------------------------------------
% Controllers
% ----------------------------------------------------------------------
function [u, Useq, info] = variationalController(x, variational, penalty, cfg, coeffs)
    seed = randi(2^31 - 1);   % drawn from the (seeded) global stream
    [u, Useq, wSum, acc] = vempc.sampleVariationalControl( ...
        x, variational, penalty, cfg.K, coeffs, cfg.chebBound, cfg.chebEta, seed);
    info = struct('wSum', wSum, 'accept', acc);
end

function [u, Useq, info] = standardController(x, mpc, G, hFunc, qpOpts)
    % min 0.5 U' H U + (S' x0)' U  s.t.  G U <= h(x0)
    f = mpc.S.' * x(:);
    [Useq, ~, flag] = quadprog(mpc.H, f, G, hFunc(x), [], [], [], [], [], qpOpts);
    if flag <= 0 || isempty(Useq)
        Useq = zeros(mpc.Nm, 1);
    end
    u = Useq(1:mpc.m);
    info = struct();
end

% ----------------------------------------------------------------------
% Plotting (paper Fig. 1 style)
% ----------------------------------------------------------------------
function plotResults(cfg, xsVar, usVar, xsStd, usStd, figPath)
    tX = (0:size(xsVar, 1) - 1) * cfg.dt;
    tU = (0:size(usVar, 1) - 1) * cfg.dt;
    tEnd = max(tX(end), cfg.dt);
    hasStd = ~isempty(xsStd);
    if hasStd, sX1 = xsStd(:,1); sX2 = xsStd(:,2); sU = usStd(:,1);
    else,      sX1 = [];        sX2 = [];        sU = []; end

    fig = figure('Visible', 'off', 'Position', [100, 100, 1100, 230]);

    ax1 = subplot(1, 3, 1); hold(ax1, 'on'); grid(ax1, 'on'); box(ax1, 'on');
    hS = []; if hasStd, hS = plot(ax1, tX, sX1, '--', 'LineWidth', 1.6, 'Color', vempcColor('orange')); end
    hV = plot(ax1, tX, xsVar(:, 1), '-', 'LineWidth', 1.6, 'Color', vempcColor('blue'));
    drawBounds(ax1, tEnd, cfg.thetaMax);
    xlabel(ax1, 'Time [s]'); ylabel(ax1, '$\theta(t)$', 'Interpreter', 'latex');
    xlim(ax1, [0 tEnd]); ylim(ax1, ylimPad(cfg.thetaMax, xsVar(:,1), sX1));
    if hasStd
        legend(ax1, [hV hS], {'Variational', 'Standard'}, ...
            'Orientation', 'horizontal', 'Location', 'north', 'Box', 'off');
    end

    ax2 = subplot(1, 3, 2); hold(ax2, 'on'); grid(ax2, 'on'); box(ax2, 'on');
    if hasStd, plot(ax2, tX, sX2, '--', 'LineWidth', 1.6, 'Color', vempcColor('orange')); end
    plot(ax2, tX, xsVar(:, 2), '-', 'LineWidth', 1.6, 'Color', vempcColor('blue'));
    drawBounds(ax2, tEnd, cfg.omegaMax);
    xlabel(ax2, 'Time [s]'); ylabel(ax2, '$\dot{\theta}(t)$', 'Interpreter', 'latex');
    xlim(ax2, [0 tEnd]); ylim(ax2, ylimPad(cfg.omegaMax, xsVar(:,2), sX2));

    ax3 = subplot(1, 3, 3); hold(ax3, 'on'); grid(ax3, 'on'); box(ax3, 'on');
    if hasStd, stairs(ax3, tU, sU, '--', 'LineWidth', 1.6, 'Color', vempcColor('orange')); end
    stairs(ax3, tU, usVar(:, 1), '-', 'LineWidth', 1.6, 'Color', vempcColor('blue'));
    drawBounds(ax3, tEnd, cfg.uMax);
    xlabel(ax3, 'Time [s]'); ylabel(ax3, '$u(t)$', 'Interpreter', 'latex');
    xlim(ax3, [0 tEnd]); ylim(ax3, ylimPad(cfg.uMax, usVar(:,1), sU));

    try
        exportgraphics(fig, figPath, 'Resolution', 150);
    catch
        saveas(fig, figPath);
    end
    close(fig);
end

% Shared paper-style helpers -------------------------------------------
function c = vempcColor(name)
    switch name
        case 'blue',   c = [0 0.2 0.8];
        case 'orange', c = [0.90 0.50 0.10];
        case 'red',    c = [0.85 0.10 0.10];
        otherwise,     c = [0 0 0];
    end
end

function drawBounds(ax, tEnd, bound)
    % Red dotted constraint lines spanning the time axis (paper style).
    red = vempcColor('red');
    plot(ax, [0 tEnd], [ bound  bound], ':', 'Color', red, 'LineWidth', 1.2);
    plot(ax, [0 tEnd], [-bound -bound], ':', 'Color', red, 'LineWidth', 1.2);
end

function yl = ylimPad(bound, varargin)
    % Y-limits a little wider than the symmetric constraint +/- bound, expanded
    % further only if the data would otherwise be clipped.
    m  = 0.2;
    lo = -bound * (1 + m);
    hi =  bound * (1 + m);
    for k = 1:numel(varargin)
        d = varargin{k};
        if ~isempty(d)
            lo = min(lo, min(d(:)) - 0.05 * bound);
            hi = max(hi, max(d(:)) + 0.05 * bound);
        end
    end
    yl = [lo hi];
end
