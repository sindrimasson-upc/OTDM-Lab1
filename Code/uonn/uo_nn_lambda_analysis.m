%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Pattern recognition with Single Layer Neural Network (SLNN).
%
% uo_nn_lambda_analysis
%
% Extra quantities for the analysis of lambda (report, Section 3), with the
% same settings as uo_nn_batch_st:
%   - full gradient-norm history, number of SWC steps, max cond(H_k) and
%     min cos(theta_k) for GM and BFGS,
%   - ||w*||, ||grad L(w*)|| (also for SGM) and the condition number of the
%     Hessian at w* (central differences of the gradient),
%   - true/false positives and negatives on the test set,
%   - false positives on the test set split by the true digit (the test
%     images do not depend on the target, so the true digit of each image
%     is recovered from the labels for the ten targets),
%   - cosine between w* and -grad L(0) (the limit direction for large lambda),
%   - GM/BFGS iterations for 5 other states of the random generator.
% uoBLSNW32 draws random numbers, so the GM/BFGS iterations depend on the
% state of the generator. As in uo_nn_solve_st, the datasets are generated
% right before every run, so the reference runs are those of the batch.
% Writes uo_nn_lambda_analysis.csv, uo_nn_lambda_analysis.mat and the
% figure ../../Docs/fig_rk_lambda.png.
%
% Calls: uo_nn_dataset, uosol_st
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all; warning off;
addpath('../uolib');

nn.tr_seed = 156057; nn.te_seed = 560571; nn.sg_seed = 156057;
nn.tr_p = 5000; nn.te_q = nn.tr_p /10; nn.tr_freq = 0.5;
nn.te_freq = 0.0;
par.epsG = 10^-3; par.maxiter = 100;
par.iAC = 4; par.c1 = 0.1; par.c2 = 0.9;
par.sg.seed = nn.sg_seed; par.sg.al0 = 2; par.sg.be = 0.3;
par.sg.m = 10; par.sg.emax = par.maxiter; par.sg.eworse = 5;
sig = @(Xds)   1./(1+exp(-Xds));
y   = @(Xds,w) sig(w'*sig(Xds));

% true digit (0..9) of every test image
dig = zeros(1,nn.te_q);
for t = 1:10
    [~,yt] = uo_nn_dataset(nn.te_seed, nn.te_q, t, nn.te_freq);
    dig(yt==1) = mod(t,10);
end

R = [];
fCSV = fopen('uo_nn_lambda_analysis.csv','w');
fprintf(fCSV,' n;   la; isd; niter; nSWC;   maxkH;  mincos;   normw;    ngL*;   kHess;  mineig; TP; FN;  FP;  TN;  cos0; niter rng(1:5); FP by true digit 0..9\n');
for num_target = [1:10]
    for la = [0.0 0.01 0.5]
        L  = @(w,Xds,yds) (norm(y(Xds,w)-yds)^2)/size(yds,2) + (la*norm(w)^2)/2;
        gL = @(w,Xds,yds) (2*sig(Xds)*((y(Xds,w)-yds).*y(Xds,w).*(1-y(Xds,w)))')/size(yds,2)+la*w;
        for isd = [1 3 7]
            par.isd = isd;
            [Xtr,ytr] = uo_nn_dataset(nn.tr_seed, nn.tr_p, num_target, nn.tr_freq);
            [Xte,yte] = uo_nn_dataset(nn.te_seed, nn.te_q, num_target, nn.te_freq);
            P.f = @(w) L(w,Xtr,ytr);
            P.g = @(w) gL(w,Xtr,ytr);
            if isd == 7
                par.sg.Xtr = Xtr;  par.sg.ytr = ytr;
                par.sg.Xte = Xte;  par.sg.yte = yte;
                par.sg.L   = L;    par.sg.gL  = gL;
            end
            w1 = zeros(size(Xtr,1),1);
            [~,sol,opt] = evalc('uosol_st(P,w1,par)');
            r.n = mod(num_target,10); r.la = la; r.isd = isd;
            r.ng = []; r.nSWC = NaN; r.maxkH = NaN; r.mincos = NaN;
            if isd == 7
                wo = opt.wo;
                r.niter = opt.etot;
            else
                wo = sol(end).x;
                r.niter = size(sol,2) - 1;
                r.ng   = [sol.ng];
                r.nSWC = sum([sol(1:end-1).AC] == "SWC");
                if isd == 3
                    kH = zeros(1,r.niter); cs = zeros(1,r.niter);
                    for k = 1:r.niter
                        kH(k) = cond(sol(k).H);
                        cs(k) = -(sol(k).g'*sol(k).d)/(norm(sol(k).g)*norm(sol(k).d));
                    end
                    r.maxkH = max(kH); r.mincos = min(cs);
                end
            end
            r.normw = norm(wo);
            r.ngLo  = norm(gL(wo,Xtr,ytr));
            % Hessian at w* by central differences of the gradient
            n = length(wo); Hs = zeros(n); h = 1e-5;
            for i = 1:n
                e = zeros(n,1); e(i) = h;
                Hs(:,i) = (gL(wo+e,Xtr,ytr) - gL(wo-e,Xtr,ytr))/(2*h);
            end
            ev = eig((Hs+Hs')/2);
            r.mineig = min(ev);
            r.kHess  = max(abs(ev))/min(abs(ev));
            yp = round(y(Xte,wo));
            r.TP = sum(yp==1 & yte==1); r.FN = sum(yp==0 & yte==1);
            r.FP = sum(yp==1 & yte==0); r.TN = sum(yp==0 & yte==0);
            r.FPdig = histcounts(dig(yp==1 & yte==0), -0.5:1:9.5);
            r.wo = wo;
            g0 = gL(zeros(size(wo)),Xtr,ytr);
            r.cos0 = -(g0'*wo)/(norm(g0)*norm(wo));
            % same run with other states of the random generator
            r.niter_rng = NaN(1,5);
            if isd ~= 7
                for s = 1:5
                    rng(s);
                    [~,sol] = evalc('uosol_st(P,w1,par)');
                    r.niter_rng(s) = size(sol,2) - 1;
                end
            end
            R = [R r];
            fprintf(fCSV,' %1i; %4.2f; %1i; %5i; %4i; %7.1e; %7.3f; %7.2f; %7.1e; %7.1e; %7.1e; %2i; %2i; %3i; %3i; %5.3f; %s; %s\n', ...
                r.n, la, isd, r.niter, r.nSWC, r.maxkH, r.mincos, r.normw, r.ngLo, r.kHess, r.mineig, r.TP, r.FN, r.FP, r.TN, r.cos0, ...
                num2str(r.niter_rng), num2str(r.FPdig));
            fprintf('n=%d la=%4.2f isd=%d done\n', r.n, la, isd);
        end
    end
end
fclose(fCSV);
save('uo_nn_lambda_analysis.mat','R');

% Figure: gradient norm of GM and BFGS for each lambda, digits 4, 6, 8.
figure('Position',[100 100 1100 600],'Visible','off');
digs = [4 6 8]; las = [0 0.01 0.5]; algs = [1 3]; names = ["GM","BFGS"];
for a = 1:2
    for j = 1:3
        subplot(2,3,(a-1)*3+j); hold on;
        for la = las
            r = R([R.n]==digs(j) & [R.la]==la & [R.isd]==algs(a));
            semilogy(1:length(r.ng), r.ng, '-', 'LineWidth', 1.2, 'DisplayName', sprintf('\\lambda=%g',la));
        end
        set(gca,'YScale','log'); grid on; yline(1e-3,':','HandleVisibility','off');
        title(sprintf('%s, digit %d', names(a), digs(j))); xlabel('k'); ylabel('||\nabla L(w^k)||');
        if a == 1 && j == 1, legend('Location','northeast'); end
    end
end
exportgraphics(gcf,'../../Docs/fig_rk_lambda.png','Resolution',150);
