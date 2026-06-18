function res = runGoEncrypted(cfg, goVempcDir, opts)
%RUNGOENCRYPTED Drive the real CKKS VEMPC protocol (Lattigo/Go) from MATLAB.
%   res = vempc.runGoEncrypted(cfg, goVempcDir, opts) writes the shared config,
%   then runs the Go offline (Algorithm 1) and online (Algorithm 2) binaries
%   and reads back their encrypted-control outputs. This executes the genuine
%   128-bit-secure CKKS pipeline; only the orchestration lives in MATLAB.
%
%   opts.regenerateCache (default true): re-run the offline phase. Set false to
%   reuse an existing output/ckks_cache (valid only if the config is unchanged).
%
%   Returns a struct with fields:
%       xs        (T+1 x n) encrypted closed-loop state history
%       us        (T   x m) encrypted control history
%       controlMs (T   x 1) per-step online time [ms]
%       meanMs    scalar mean online time parsed from the binary's output
%       offlineOut, onlineOut  captured stdout/stderr
    if nargin < 3, opts = struct(); end
    regen = ~isfield(opts, 'regenerateCache') || opts.regenerateCache;

    goExe = vempc.findGo();
    if isempty(goExe)
        error('vempc:noGo', ['Go toolchain not found. Install Go (https://go.dev/dl), ', ...
            'ensure `go` is on PATH, then re-run. Go + Lattigo provide the CKKS engine.']);
    end

    % 1. Write the config consumed by the Go binaries.
    vempc.writeGoConfig(cfg, goVempcDir);

    % Run all Go commands with the working directory set to GoVempc so that the
    % binaries resolve "output/ckks_config.json" and the module correctly.
    oldPwd = pwd;
    cleaner = onCleanup(@() cd(oldPwd)); %#ok<NASGU>
    cd(goVempcDir);

    res = struct();

    % 2. Ensure module dependencies (Lattigo, gonum) are available.
    runStage(goExe, 'mod download', 'go mod download');

    % 3. Offline protocol (Algorithm 1) -> output/ckks_cache/.
    cryptoReady = isfile(fullfile('output', 'ckks_cache', 'crypto', 'params.bin'));
    if regen || ~cryptoReady
        res.offlineOut = runStage(goExe, 'run ./cmd/ckks_offline', 'ckks_offline (Algorithm 1)');
    else
        res.offlineOut = '(skipped: reusing existing ckks_cache)';
        fprintf('  [offline] reusing existing output/ckks_cache\n');
    end

    % 4. Online protocol (Algorithm 2) -> output/xs_ckks.csv, us_ckks.csv, ...
    res.onlineOut = runStage(goExe, 'run ./cmd/ckks_online', 'ckks_online (Algorithm 2)');

    % 5. Read encrypted outputs.
    res.xs = readmatrix(fullfile('output', 'xs_ckks.csv'));
    res.us = readmatrix(fullfile('output', 'us_ckks.csv'));
    cmsPath = fullfile('output', 'control_iter_ms_ckks.csv');
    if isfile(cmsPath)
        res.controlMs = readmatrix(cmsPath);
    else
        res.controlMs = [];
    end
    res.meanMs = parseMeanMs(res.onlineOut);
end

% ----------------------------------------------------------------------
function out = runStage(goExe, args, label)
    fprintf('  [go] %s ...\n', label);
    cmd = sprintf('"%s" %s 2>&1', goExe, args);
    [st, out] = system(cmd);
    if st ~= 0
        error('vempc:goStage', '%s failed (exit %d):\n%s', label, st, strtrim(out));
    end
end

function v = parseMeanMs(s)
    tok = regexp(s, 'mean/std:\s*([\d.]+)\s*ms', 'tokens', 'once');
    if isempty(tok)
        v = NaN;
    else
        v = str2double(tok{1});
    end
end
