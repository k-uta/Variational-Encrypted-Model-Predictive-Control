classdef MPCProblem
    %MPCPROBLEM Linear-Quadratic MPC in condensed (quadratic-program) form.
    %
    %   Formulates the finite-horizon LQ-MPC problem
    %
    %       min_U   x0' P x0 + x0' S U + (1/2) U' H U
    %       s.t.    G U <= h(x0)
    %
    %   where U = [u_0; u_1; ...; u_{N-1}] is the stacked control sequence.
    %   This is the MATLAB counterpart of vempc/core/mpc.py and
    %   GoVempc/core/mpc.go.
    %
    %   Construction:
    %       mpc = vempc.MPCProblem(A, B, Q, R, Qf, N)

    properties
        A           % (n x n) state transition matrix
        B           % (n x m) input matrix
        Q           % (n x n) state cost
        R           % (m x m) input cost
        Qf          % (n x n) terminal state cost
        N           % prediction horizon
        n           % state dimension
        m           % input dimension
        Nm          % stacked input dimension (N*m)
        Lambda      % (N*n x n) prediction matrix for x0
        Psi         % (N*n x N*m) prediction matrix for U
        P           % (n x n) condensed initial-state cost
        S           % (n x N*m) condensed cross term
        H           % (N*m x N*m) condensed Hessian
    end

    methods
        function obj = MPCProblem(A, B, Q, R, Qf, N)
            obj.A = A; obj.B = B; obj.Q = Q; obj.R = R; obj.Qf = Qf; obj.N = N;
            obj.n = size(A, 1);
            obj.m = size(B, 2);
            obj.Nm = N * obj.m;
            [obj.Lambda, obj.Psi] = obj.computePredictionMatrices();
            [obj.P, obj.S, obj.H] = obj.computeCostMatrices();
        end

        function [Lambda, Psi] = computePredictionMatrices(obj)
            % X = Lambda * x0 + Psi * U
            n = obj.n; m = obj.m; N = obj.N; A = obj.A; B = obj.B;

            % Precompute A^k (Apow{k+1} = A^k) to avoid repeated mpower.
            Apow = cell(N + 1, 1);
            Apow{1} = eye(n);
            for k = 2:N + 1
                Apow{k} = Apow{k - 1} * A;
            end

            Lambda = zeros(N * n, n);
            for i = 1:N
                Lambda((i - 1) * n + 1:i * n, :) = Apow{i + 1};   % A^i
            end

            Psi = zeros(N * n, N * m);
            for i = 1:N
                for j = 1:i
                    Psi((i - 1) * n + 1:i * n, (j - 1) * m + 1:j * m) = Apow{i - j + 1} * B; % A^(i-j) B
                end
            end
        end

        function [P, S, H] = computeCostMatrices(obj)
            % J0(x0,U) = x0' P x0 + x0' S U + (1/2) U' H U
            n = obj.n; N = obj.N;

            Qbar = kron(eye(N), obj.Q);
            Qbar((N - 1) * n + 1:N * n, (N - 1) * n + 1:N * n) = obj.Qf;   % terminal block
            Rbar = kron(eye(N), obj.R);

            P = obj.Q + obj.Lambda.' * Qbar * obj.Lambda;
            S = 2 * (obj.Lambda.' * Qbar * obj.Psi);
            H = 2 * (Rbar + obj.Psi.' * Qbar * obj.Psi);
            H = (H + H.') / 2;   % symmetrize against round-off
        end

        function [G, hFunc] = buildConstraintMatrices(obj, Gx, hx, Gu, hu)
            % Build condensed G U <= h(x0) from per-step constraints
            %   Gx * x_k <= hx   (k = 1..N),   Gu * u_k <= hu   (k = 0..N-1).
            N = obj.N;

            GxBar = kron(eye(N), Gx);
            hxBar = repmat(hx(:), N, 1);
            GuBar = kron(eye(N), Gu);
            huBar = repmat(hu(:), N, 1);

            Gtop = GxBar * obj.Psi;
            Ltop = GxBar * obj.Lambda;

            G = [Gtop; GuBar];
            hFunc = @(x0) [hxBar - Ltop * x0(:); huBar];
        end

        function c = quadraticCost(obj, x0, U)
            % J0(x0, U) for a single stacked input U (column vector).
            x0 = x0(:); U = U(:);
            c = x0.' * obj.P * x0 + x0.' * obj.S * U + 0.5 * (U.' * obj.H * U);
        end
    end
end
