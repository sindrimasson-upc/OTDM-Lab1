% Extra numbers and figures for the case lambda = 0.
clear all; warning off;
fLOG = fopen('uo_nn_diag.log','w'); diary uo_nn_diag.log
addpath('../uolib');

% Same settings as uo_nn_batch_st
nn.tr_seed = 156057; nn.te_seed = 560571; nn.sg_seed = 156057;
nn.tr_p = 5000; nn.te_q = nn.tr_p /10; nn.tr_freq = 0.5;
nn.te_freq = 0.0;
par.epsG = 10^-3; par.maxiter = 100;
par.iAC = 4; par.c1 = 0.1; par.c2 = 0.9;
par.sg.seed = nn.sg_seed; par.sg.al0 = 2; par.sg.be = 0.3;
par.sg.m = 10; par.sg.emax = par.maxiter; par.sg.eworse = 5;
sig = @(Xds)   1./(1+exp(-Xds));
y = @(Xds,w) sig(w'*sig(Xds));
nn.Acc = @(Xds,yds,wo) 100*sum(yds==round(y(Xds,wo)))/size(Xds,2);
la = 0;
nn.la = la;
nn.L  = @(w,Xds,yds) (norm(y(Xds,w)-yds)^2)/size(yds,2) + (la*norm(w)^2)/2;
nn.gL = @(w,Xds,yds) (2*sig(Xds)*((y(Xds,w)-yds).*y(Xds,w).*(1-y(Xds,w)))')/size(yds,2)+la*w;

% Run the 30 cases, keep the iteration data of GM and BFGS and the weights
S = cell(10,3);
W = cell(10,3);
fprintf('\n digit isd  #pos  baseline   TPR    TNR   ||gL(w*)||\n');
for num_target = 1:10
    nn.num_target = num_target;
    for isd = [1 3 7]
        par.isd = isd;
        nnout = uo_nn_solve_st(nn,par);
        pred = round(y(nnout.Xte,nnout.wo));
        npos = sum(nnout.yte==1);
        nneg = sum(nnout.yte==0);
        tpr  = 100*sum(pred==1 & nnout.yte==1)/npos;
        tnr  = 100*sum(pred==0 & nnout.yte==0)/nneg;
        ng   = norm(nn.gL(nnout.wo,nnout.Xtr,nnout.ytr));   % full training gradient
        fprintf(' %4i  %3i  %4i  %7.1f  %5.1f  %5.1f   %8.2e\n', ...
            mod(num_target,10), isd, npos, 100*nneg/(npos+nneg), tpr, tnr, ng);
        if isd ~= 7
            S{num_target,isd} = nnout.sol;
        end
        W{num_target,isd} = nnout.wo;
    end
end

% cos(theta), cond(H) and line search flags of GM and BFGS
dig = [4 6 8];
fprintf('\n digit isd  min cos(th)   max cond(H)   SWC / iterations\n');
for num_target = dig
    for isd = [1 3]
        sol = S{num_target,isd};
        K = size(sol,2) - 1;
        cs  = zeros(K,1);
        kap = ones(K,1);
        for k = 1:K
            cs(k) = -sol(k).g'*sol(k).d/(norm(sol(k).g)*norm(sol(k).d));
            if isd == 3
                kap(k) = cond(sol(k).H);
            end
        end
        nSWC = sum([sol(1:K).AC] == "SWC");
        fprintf(' %4i  %3i   %10.2e   %10.2e   %4i / %4i\n', ...
            num_target, isd, min(cs), max(kap), nSWC, K);
    end
end

% Gradient norm and r_k = ||g_k+1|| / ||g_k||
figure;
for i = 1:3
    for isd = [1 3]
        ng = [S{dig(i),isd}.ng];
        r  = ng(2:end)./ng(1:end-1);
        subplot(2,3,i);   semilogy(0:length(ng)-1, ng); hold on;
        subplot(2,3,3+i); plot(1:length(r), r); hold on;
    end
    subplot(2,3,i);   title(sprintf('digit %i',dig(i))); xlabel('k'); ylabel('||\nabla L||');
    subplot(2,3,3+i); xlabel('k'); ylabel('r_k'); ylim([0 1.5]);
end
subplot(2,3,1); legend('GM','BFGS');
saveas(gcf,'fig_rk_lambda0.png');

% Cost of one loss / gradient evaluation (digit 8, full training set)
nn.num_target = 8;
[Xtr,ytr] = uo_nn_dataset(nn.tr_seed, nn.tr_p, nn.num_target, nn.tr_freq);
w = zeros(size(Xtr,1),1);
nrep = 200;
tic; for i = 1:nrep, nn.L(w,Xtr,ytr);  end; t_f  = toc/nrep;
tic; for i = 1:nrep, nn.gL(w,Xtr,ytr); end; t_g  = toc/nrep;
idx = 1:10;
tic; for i = 1:nrep, nn.gL(w,Xtr(:,idx),ytr(idx)); end; t_mb = toc/nrep;
t_it = 0.055;
fprintf('\n loss: %.2e s, gradient: %.2e s, minibatch gradient: %.2e s\n', t_f, t_g, t_mb);
fprintf(' one GM iteration = %.1f gradient evaluations\n', t_it/t_g);

% GM with 1000 iterations on digits 3 and 8
run_gm_long = true;
if run_gm_long
    par.maxiter = 1000; par.isd = 1;
    for num_target = [3 8]
        nn.num_target = num_target;
        nnout = uo_nn_solve_st(nn,par);
        fprintf('\n GM, digit %i, maxiter = 1000: iterations = %i, ngL = %.2e, tex = %.1f s\n', ...
            num_target, nnout.niter-1, nnout.ngLo, nnout.tex);
    end
end
diary off; fclose(fLOG);