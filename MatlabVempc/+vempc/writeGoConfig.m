function jsonPath = writeGoConfig(cfg, goVempcDir)
%WRITEGOCONFIG Write GoVempc/output/ckks_config.json from a MATLAB cfg struct.
%   jsonPath = vempc.writeGoConfig(cfg, goVempcDir) serializes the (shared)
%   VEMPC config into the exact JSON schema consumed by the Go CKKS binaries
%   (GoVempc/internal/ckksconfig). Array-valued fields (QDiag, RDiag, x0, logQ,
%   logP) are forced to JSON arrays even when length 1, since Go decodes them
%   into slices.
    outDir = fullfile(goVempcDir, 'output');
    if ~isfolder(outDir)
        mkdir(outDir);
    end
    jsonPath = fullfile(outDir, 'ckks_config.json');

    g = struct();
    g.m = cfg.m; g.l = cfg.l; g.g = cfg.g;
    g.dt = cfg.dt; g.N = round(cfg.N);
    g.QDiag = asArray(cfg.QDiag);
    g.RDiag = asArray(cfg.RDiag);
    g.QfScale = cfg.QfScale;
    g.x0 = asArray(cfg.x0);
    g.thetaMax = cfg.thetaMax; g.omegaMax = cfg.omegaMax; g.uMax = cfg.uMax;
    g.sigma0 = cfg.sigma0; g.lambda = cfg.lambda; g.K = round(cfg.K);
    g.logN = round(cfg.logN);
    g.logQ = asIntArray(cfg.logQ);
    g.logP = asIntArray(cfg.logP);
    g.logDefaultScale = round(cfg.logDefaultScale);
    g.nWorkers = round(cfg.nWorkers);
    g.T = round(cfg.T);
    if isfield(cfg, 'TSteps') && cfg.TSteps > 0
        g.TSteps = round(cfg.TSteps);
    end
    g.chebOrder = round(cfg.chebOrder);
    g.chebBound = cfg.chebBound;
    g.chebEta = cfg.chebEta;

    txt = jsonencode(g, 'PrettyPrint', true);
    fid = fopen(jsonPath, 'w');
    if fid < 0
        error('vempc:writeGoConfig', 'Cannot open %s for writing.', jsonPath);
    end
    cleaner = onCleanup(@() fclose(fid));
    fwrite(fid, txt, 'char');
end

function c = asArray(v)
    % Force a JSON array (cell of scalars) even for length-1 vectors.
    c = num2cell(double(v(:)).');
end

function c = asIntArray(v)
    c = num2cell(round(double(v(:)).'));
end
