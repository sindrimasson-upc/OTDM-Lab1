%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% F.-Javier Heredia https://gnom.upc.edu/heredia
% https://creativecommons.org/licenses/by-nc/4.0/
% Pattern recognition with Single Layer Neural Network (SLNN).
%
% uo_nn_batch
%
% Main script to run a batch of recognitions.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all; warning off;
addpath('../uolib');
fLOG = fopen('uo_nn_batch_st.log','w'); diary uo_nn_batch_st.log
fprintf('[uo_nn_batch_st]  Starts\n');

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
global iheader; iheader = 1;
fCSV = fopen('uo_nn_batch_st.csv','w');
t1 = clock;
for num_target = [1:10]
    nn.num_target = num_target;
    lambda = [0.0 0.01 0.5];
    for la = lambda
        nn.la = la;
        nn.L  = @(w,Xds,yds) (norm(y(Xds,w)-yds)^2)/size(yds,2) + (la*norm(w)^2)/2;
        nn.gL = @(w,Xds,yds) (2*sig(Xds)*((y(Xds,w)-yds).*y(Xds,w).*(1-y(Xds,w)))')/size(yds,2)+la*w;
        for isd = [1 3 7]
            par.isd = isd;
            [nnout] = uo_nn_solve_st(nn,par);
            if iheader == 1
                fprintf(fCSV,' n;   la; isd;  iter/epoc;  mbatch;       L*;    ngL*;     tex; tr_acc; te_acc;\n');
            end
            if ~isempty(nnout)
                fprintf(fCSV,' %1i; %4.2f;      %1i;     %3i;   %5i; %8.2e; %7.1e; %7.1e;  %5.1f;  %5.1f;\n', ...
                    mod(num_target,10), la, isd, nnout.niter, nnout.nmbatch, nnout.Lo, nnout.ngLo, nnout.tex, nnout.tr_acc, nnout.te_acc);
            end
            iheader=0;
        end
    end
end
t2 = clock; total_t = etime(t2,t1);
fprintf('[uo_nn_batch_st]  Stops, wall time = %6.1f s.\n', total_t); fclose(fCSV);
diary off; fclose(fLOG);