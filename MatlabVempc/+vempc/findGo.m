function goExe = findGo()
%FINDGO Locate the Go toolchain executable, or '' if not found.
%   goExe = vempc.findGo() returns 'go' if it is on PATH, otherwise the full
%   path to go(.exe) in a common install location, or '' if Go is not found.
%   Go (with the Lattigo dependency) drives the real CKKS engine for the
%   encrypted VEMPC protocol.
    goExe = '';

    % 1. On PATH?
    [st, ~] = system('go version');
    if st == 0
        goExe = 'go';
        return;
    end

    % 2. Common install locations.
    home  = getenv('USERPROFILE'); if isempty(home), home = getenv('HOME'); end
    local = getenv('LOCALAPPDATA');
    candidates = {
        'C:\Program Files\Go\bin\go.exe'
        'C:\Go\bin\go.exe'
        fullfile(local, 'Programs', 'Go', 'bin', 'go.exe')
        fullfile(home, 'go', 'bin', 'go.exe')
        fullfile(home, 'scoop', 'apps', 'go', 'current', 'bin', 'go.exe')
        '/usr/local/go/bin/go'
        '/usr/lib/go/bin/go'
        fullfile(home, 'go', 'bin', 'go')
    };
    for i = 1:numel(candidates)
        c = candidates{i};
        if ~isempty(c) && isfile(c)
            goExe = c;
            return;
        end
    end
end
