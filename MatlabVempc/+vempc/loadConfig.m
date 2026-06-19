function cfg = loadConfig(path)
%LOADCONFIG Load VEMPC parameters, falling back to the paper defaults.
%   cfg = vempc.loadConfig(path) reads a JSON config and merges it over the
%   built-in defaults below. The schema is identical to the Go reference
%   (GoVempc/internal/ckksconfig/config.go and GoVempc/output/ckks_config.json):
%   the *same JSON file* can be used by the Go, Python, and MATLAB ports.
%
%   The CKKS-specific fields (logN, logQ, logP, logDefaultScale, nWorkers) are
%   accepted for schema parity but ignored by the plaintext MATLAB port; only
%   the control/sampling parameters affect the result.
%
%   Pass no argument (or a missing path) to use the defaults only.

    % Defaults match the canonical configuration of the VEMPC paper (Section V)
    % and the documented GoVempc default in the top-level README.
    cfg = struct( ...
        'm', 0.2, 'l', 0.5, 'g', 9.81, ...
        'dt', 0.05, 'N', 10, ...
        'QDiag', [50.0, 5.0], 'RDiag', 0.1, 'QfScale', 2.0, ...
        'x0', [0.3, 0.1], ...
        'thetaMax', 0.5, 'omegaMax', 0.8, 'uMax', 1.0, ...
        'sigma0', 0.25, 'lambda', 0.1, 'K', 240, ...
        'logN', 13, 'logQ', [33, 30, 30, 30], 'logP', 35, ...
        'logDefaultScale', 30, 'nWorkers', 4, ...
        'chebOrder', 3, 'chebBound', 5.0, 'chebEta', 500.0, ...
        'T', 40, 'TSteps', 0, ...
        'seed', 0);   % matches the Go reference's rand.Seed(0)

    if nargin < 1 || isempty(path)
        return;
    end
    if ~isfile(path)
        warning('vempc:loadConfig:missingFile', ...
            'Config file "%s" not found; using built-in defaults.', path);
        return;
    end

    raw = jsondecode(fileread(path));
    fns = fieldnames(raw);
    for i = 1:numel(fns)
        cfg.(fns{i}) = raw.(fns{i});   % override defaults with provided values
    end
end
