clear all; clc; close all;
addpath('../uolib');

% ---------------- Switches ----------------
use_slide = false;       % true : slide-22 example (p = 250, digit 8, la = 0, seeds of the slides)
                        % false: batch settings (p = 5000, digit 3, la = 0.01), own seeds
run_own   = true;       % uo_nn_solve_st (own code)
run_ref   = true;       % uo_nn_solve    (course .p, reference)
algs      = [1 3 7];    % 1 GM, 3 BFGS, 7 SGM
run_batch = false;      % full batch of 90 runs (keep as the last line)

if ~use_slide, run_ref = false; end   % uo_nn_solve.p only accepts tr_p <= 250

% ---------------- Settings ----------------
if use_slide
    nn.num_target = 8;  nn.tr_p = 250;  nn.te_q = 250;
    nn.tr_seed = 63256; nn.te_seed = 13625;
    la = 0.0;
    par.c1 = 0.01;
    par.sg.m = 2;       par.sg.seed = 565544;
else
    nn.num_target = 3;  nn.tr_p = 5000; nn.te_q = 500;
    nn.tr_seed = 156057; nn.te_seed = 156057;
    la = 0.01;
    par.c1 = 0.1;
    par.sg.m = 10;      par.sg.seed = 156057;
end
nn.tr_freq = 0.5;  nn.te_freq = 0.0;  nn.la = la;
par.epsG = 1e-3;  par.maxiter = 100;
par.iAC = 4;  par.c2 = 0.9;
par.sg.al0 = 2;  par.sg.be = 0.3;  par.sg.emax = par.maxiter;  par.sg.eworse = 5;

% ---------------- Loss, gradient, accuracy ----------------
sig    = @(Xds)   1./(1+exp(-Xds));
y      = @(Xds,w) sig(w'*sig(Xds));
nn.L   = @(w,Xds,yds) (norm(y(Xds,w)-yds)^2)/size(yds,2) + (la*norm(w)^2)/2;
nn.gL  = @(w,Xds,yds) (2*sig(Xds)*((y(Xds,w)-yds).*y(Xds,w).*(1-y(Xds,w)))')/size(yds,2) + la*w;
nn.Acc = @(Xds,yds,wo) 100*sum(yds==round(y(Xds,wo)))/size(Xds,2);

% ---------------- Runs ----------------
% Slide 22, BFGS (isd = 3) should give: 59 log rows, L* = 2.42e-02, ||gL*|| = 5.96e-04,
% tr_acc = 97.6, te_acc = 98.4.
pr = @(tag,o) fprintf('%-4s niter = %3d  nmbatch = %5d  L* = %.3e  ||gL*|| = %.2e  tr_acc = %5.1f  te_acc = %5.1f  tex = %.2e\n', ...
                      tag, o.niter, o.nmbatch, o.Lo, o.ngLo, o.tr_acc, o.te_acc, o.tex);
for isd = algs
    par.isd = isd;
    fprintf('\n===== isd = %d =====\n', isd);
    if run_own, own = uo_nn_solve_st(nn,par); end
    if run_ref, ref = uo_nn_solve(nn,par);    end
    fprintf('\n--- isd = %d ---\n', isd);
    if run_own, pr('own',own); end
    if run_ref, pr('ref',ref); end
end

% ---------------- Full batch ----------------
if run_batch
    uo_nn_batch_st;
end