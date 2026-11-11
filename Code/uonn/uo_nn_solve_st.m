%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% F.-Javier Heredia https://gnom.upc.edu/heredia
% https://creativecommons.org/licenses/by-nc/4.0/
% Pattern recognition with Single Layer Neural Network (SLNN).
%
% function [nnout] = uo_nn_solve_st(nn,par)
%
% Trains the SLNN and checks the accuracy for the problem defined in "nn"
% and "par". Input and output parameters as in the template, plus
% nn.te_freq (frequency of the target digit in the test set).
%
% Calls: uo_nn_dataset, uosol_st
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [nnout] = uo_nn_solve_st(nn,par)
Xtr=[];
ytr=[];
wo=[];
Lo=0;
tr_acc=0;
Xte=[];
yte=[];
te_acc=0;
niter=0;
tex=0;

[Xtr,ytr] = uo_nn_dataset(nn.tr_seed, nn.tr_p, nn.num_target, nn.tr_freq);
[Xte,yte] = uo_nn_dataset(nn.te_seed, nn.te_q, nn.num_target, nn.te_freq);
P.f = @(w) nn.L(w,Xtr,ytr);
P.g = @(w) nn.gL(w,Xtr,ytr);
w1  = zeros(size(Xtr,1),1);
if par.isd == 7
    P.f = @(w,varargin) flex(nn.L,Xtr,ytr,w,varargin{:});
    P.g = @(w,varargin) flex(nn.gL,Xtr,ytr,w,varargin{:});
    par.sg.Xtr = Xtr;  par.sg.ytr = ytr;
    par.sg.Xte = Xte;  par.sg.yte = yte;
    par.sg.L   = nn.L;
    par.sg.gL  = nn.gL;
end
[sol,opt] = uosol_st(P,w1,par);
tex = opt.tex;
if par.isd == 7
    wo = opt.wo;
else
    wo = sol(end).x;
end
niter = size(sol,2);
Lo  = nn.L(wo,Xtr,ytr);
gLo = nn.gL(wo,Xtr,ytr);

tr_acc = nn.Acc(Xtr,ytr,wo);

te_acc = nn.Acc(Xte,yte,wo);

nnout.sol = sol;
nnout.Xtr = Xtr;
nnout.ytr = ytr;
nnout.wo = wo;
nnout.Lo = Lo;
if par.isd == 7
    nnout.niter = opt.etot;
    nnout.nmbatch = niter;
    nnout.ngLo = 0;
else
    nnout.niter = niter;
    nnout.nmbatch = 0;
    nnout.ngLo  = norm(gLo);
end
nnout.tex = tex;
nnout.tr_acc = tr_acc;
nnout.Xte = Xte;
nnout.yte = yte;
nnout.te_acc = te_acc;

end

function v = flex(fun,Xtr,ytr,w,X,y)
if nargin < 5
    X = Xtr;  y = ytr;
end
v = fun(w,X,y);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% End Procedure uo_nn_solve
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
