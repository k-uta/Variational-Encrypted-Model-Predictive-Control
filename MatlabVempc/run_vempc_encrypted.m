function results = run_vempc_encrypted(configPath)
%RUN_VEMPC_ENCRYPTED MATLAB-orchestrated *encrypted* VEMPC (real CKKS via Go).
%
%   run_vempc_encrypted()           uses MatlabVempc/config/vempc_config.json
%   run_vempc_encrypted(configPath) uses the JSON config at `configPath`
%
%   This drives the genuine CKKS protocol of Algorithms 1-2 from MATLAB:
%   the client-side preparation and orchestration run in MATLAB, while the
%   homomorphic cloud computation is executed by the project's Lattigo (Go)
%   engine (GoVempc/cmd/ckks_offline + ckks_online) at 128-bit security.
%   MATLAB then overlays the encrypted trajectory against the plaintext
%   variational MPC, reproducing Fig. 1 of the paper.
%
%   Requires the Go toolchain (https://go.dev/dl). If Go is not found, the
%   function prints setup instructions and returns without error.
%
%   Outputs go to MatlabVempc/output/ (encrypted CSVs + overlay figure).

    here = fileparts(mfilename('fullpath'));
    addpath(here);
    if nargin < 1 || isempty(configPath)
        configPath = fullfile(here, 'config', 'vempc_config.json');
    end
    cfg = vempc.loadConfig(configPath);

    repoRoot   = fileparts(here);
    goVempcDir = fullfile(repoRoot, 'GoVempc');

    fprintf('VEMPC (MATLAB-orchestrated, real CKKS via Lattigo/Go)\n');
    fprintf('  config: N=%d, K=%d, nWorkers=%d, logN=%d, T=%d\n', ...
        cfg.N, cfg.K, cfg.nWorkers, cfg.logN, cfg.T);

    if ~isfolder(goVempcDir)
        error('vempc:noGoVempc', 'GoVempc directory not found at %s', goVempcDir);
    end

    goExe = vempc.findGo();
    if isempty(goExe)
        printGoSetupHelp(goVempcDir, configPath);
        if nargout > 0, results = struct('available', false); end
        return;
    end
    fprintf('  Go toolchain: %s\n\n', goExe);

    % ------------------------------------------------------------------
    % 1. Encrypted run (Lattigo / Go).
    % ------------------------------------------------------------------
    enc = vempc.runGoEncrypted(cfg, goVempcDir);

    % ------------------------------------------------------------------
    % 2. Plaintext variational reference (MATLAB) for the overlay.
    % ------------------------------------------------------------------
    fprintf('\n--- plaintext variational reference (MATLAB) ---\n');
    plain = run_vempc(configPath);

    % ------------------------------------------------------------------
    % 3. Metrics + comparison.
    % ------------------------------------------------------------------
    prob = buildProblem(cfg);
    [encCost, encVx, encVu] = metrics(prob, enc.xs, enc.us);
    encErr = trajDiff(plain.xsVar, enc.xs);

    fprintf('\nEncrypted VEMPC (CKKS / Lattigo)\n');
    if ~isnan(enc.meanMs)
        fprintf('  online time:  %.3f ms/step (mean, from Go)\n', enc.meanMs);
    end
    fprintf('  cost:         %.6f\n', encCost);
    fprintf('  viol:         (x=%.6f, u=%.6f)\n', encVx, encVu);
    fprintf('  max |x_enc - x_plain|: %.3e  (CKKS approximation error)\n', encErr);

    % ------------------------------------------------------------------
    % 4. Save + plot overlay (Fig. 1 reproduction).
    % ------------------------------------------------------------------
    outDir = fullfile(here, 'output');
    if ~isfolder(outDir), mkdir(outDir); end
    writematrix(enc.xs, fullfile(outDir, 'xs_ckks.csv'));
    writematrix(enc.us, fullfile(outDir, 'us_ckks.csv'));
    if ~isempty(enc.controlMs)
        writematrix(enc.controlMs, fullfile(outDir, 'control_iter_ms_ckks.csv'));
    end

    figPath = fullfile(outDir, 'encrypted_vs_plaintext.png');
    plotOverlay(cfg, plain.xsVar, plain.usVar, enc.xs, enc.us, figPath);
    fprintf('\nSaved encrypted CSVs + overlay figure to %s\n', outDir);

    if nargout > 0
        results = struct('available', true, 'enc', enc, 'plain', plain, ...
            'encCost', encCost, 'encViol', [encVx, encVu], 'encErr', encErr);
    end
end

% ----------------------------------------------------------------------
% Problem rebuild (mirrors run_vempc, for metric computation only)
% ----------------------------------------------------------------------
function prob = buildProblem(cfg)
    [Ac, Bc] = vempc.invertedPendulum(cfg.m, cfg.l, cfg.g);
    [A, B]   = vempc.discretizeZOH(Ac, Bc, cfg.dt);
    n = size(A, 1); mIn = size(B, 2);
    prob.Q  = diag(cfg.QDiag(:));
    prob.R  = diag(cfg.RDiag(:));
    prob.Qf = cfg.QfScale * prob.Q;
    xBound  = [cfg.thetaMax; cfg.omegaMax];
    prob.Gx = [eye(n); -eye(n)];     prob.hx = [xBound; xBound];
    prob.Gu = [eye(mIn); -eye(mIn)]; prob.hu = repmat(cfg.uMax, 2 * mIn, 1);
end

function [cost, vx, vu] = metrics(prob, xs, us)
    cost = vempc.trajectoryCost(xs, us, prob.Q, prob.R, prob.Qf);
    [vx, vu] = vempc.maxConstraintViolation(xs, us, prob.Gx, prob.hx, prob.Gu, prob.hu);
end

function d = trajDiff(xsA, xsB)
    nrow = min(size(xsA, 1), size(xsB, 1));
    d = max(max(abs(xsA(1:nrow, :) - xsB(1:nrow, :))));
end

% ----------------------------------------------------------------------
function printGoSetupHelp(goVempcDir, configPath)
    fprintf(2, ['\n[Go toolchain not found]\n', ...
        'The encrypted path drives the Lattigo CKKS engine, which needs Go.\n\n', ...
        'Setup (one time):\n', ...
        '  1. Install Go:  https://go.dev/dl  (then restart MATLAB so PATH updates)\n', ...
        '  2. Fetch deps:  cd "%s" && go mod download\n\n', ...
        'Then re-run:  run_vempc_encrypted(''%s'')\n\n', ...
        'Note: the plaintext MATLAB port (run_vempc) needs no Go and already\n', ...
        'reproduces the VEMPC control result up to CKKS approximation error.\n'], ...
        goVempcDir, configPath);
end

% ----------------------------------------------------------------------
% Overlay plot in the paper Fig. 1 style: encrypted (blue solid) vs
% unencrypted (orange dashed), red dotted constraints, no title.
function plotOverlay(cfg, xsP, usP, xsE, usE, figPath)
    tXp = (0:size(xsP, 1) - 1) * cfg.dt;
    tUp = (0:size(usP, 1) - 1) * cfg.dt;
    tXe = (0:size(xsE, 1) - 1) * cfg.dt;
    tUe = (0:size(usE, 1) - 1) * cfg.dt;
    tEnd = max([tXp(end), tXe(end), cfg.dt]);

    fig = figure('Visible', 'off', 'Position', [100, 100, 1100, 230]);

    ax1 = subplot(1, 3, 1); hold(ax1, 'on'); grid(ax1, 'on'); box(ax1, 'on');
    hU = plot(ax1, tXp, xsP(:, 1), '--', 'LineWidth', 1.6, 'Color', vempcColor('orange'));
    hE = plot(ax1, tXe, xsE(:, 1), '-',  'LineWidth', 1.6, 'Color', vempcColor('blue'));
    drawBounds(ax1, tEnd, cfg.thetaMax);
    xlabel(ax1, 'Time [s]'); ylabel(ax1, '$\theta(t)$', 'Interpreter', 'latex');
    xlim(ax1, [0 tEnd]); ylim(ax1, ylimPad(cfg.thetaMax, xsP(:,1), xsE(:,1)));
    legend(ax1, [hE hU], {'Encrypted', 'Unencrypted'}, ...
        'Orientation', 'horizontal', 'Location', 'north', 'Box', 'off');

    ax2 = subplot(1, 3, 2); hold(ax2, 'on'); grid(ax2, 'on'); box(ax2, 'on');
    plot(ax2, tXp, xsP(:, 2), '--', 'LineWidth', 1.6, 'Color', vempcColor('orange'));
    plot(ax2, tXe, xsE(:, 2), '-',  'LineWidth', 1.6, 'Color', vempcColor('blue'));
    drawBounds(ax2, tEnd, cfg.omegaMax);
    xlabel(ax2, 'Time [s]'); ylabel(ax2, '$\dot{\theta}(t)$', 'Interpreter', 'latex');
    xlim(ax2, [0 tEnd]); ylim(ax2, ylimPad(cfg.omegaMax, xsP(:,2), xsE(:,2)));

    ax3 = subplot(1, 3, 3); hold(ax3, 'on'); grid(ax3, 'on'); box(ax3, 'on');
    stairs(ax3, tUp, usP(:, 1), '--', 'LineWidth', 1.6, 'Color', vempcColor('orange'));
    stairs(ax3, tUe, usE(:, 1), '-',  'LineWidth', 1.6, 'Color', vempcColor('blue'));
    drawBounds(ax3, tEnd, cfg.uMax);
    xlabel(ax3, 'Time [s]'); ylabel(ax3, '$u(t)$', 'Interpreter', 'latex');
    xlim(ax3, [0 tEnd]); ylim(ax3, ylimPad(cfg.uMax, usP(:,1), usE(:,1)));

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
    red = vempcColor('red');
    plot(ax, [0 tEnd], [ bound  bound], ':', 'Color', red, 'LineWidth', 1.2);
    plot(ax, [0 tEnd], [-bound -bound], ':', 'Color', red, 'LineWidth', 1.2);
end

function yl = ylimPad(bound, varargin)
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
