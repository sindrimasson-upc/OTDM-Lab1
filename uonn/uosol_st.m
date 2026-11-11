%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% F.-Javier Heredia https://gnom.upc.edu/heredia
% https://creativecommons.org/licenses/by-nc/4.0/
%
% [sol,par] = uosol_st(P,x1,par)
%
% Template for the unconstrained optimization with first and second
% derivative methods. Added for the SLNN assignment:
%   - par.iAC = 4 : line search uoBLSNW32 with dynamic almax.
%   - par.isd = 7 : stochastic gradient method (SGM).
%   - par.tex     : running time of the optimization (without the log).
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [sol,par] = uosol_st(P,x,par)

tic;
n = size(x,1);
f = P.f;
g = P.g;
k = 1;
sol(k).x = x;

if par.isd == 7
    sg = par.sg;
    p = size(sg.Xtr,2);
    kmax= sg.emax*p/sg.m;
    kSG = sg.be*kmax;
    alSG = 0.01*sg.al0;
    rng(sg.seed);
    w = x;
    k = 0;
    e = 0;
    s = 0;
    Lbest = inf;
    while e < sg.emax && s < sg.eworse
        perm = randperm(p);
        for i = 1:sg.m:p
            idx = perm(i:i+sg.m-1);
            d = -sg.gL(w, sg.Xtr(:,idx), sg.ytr(idx));
            if k <= kSG
                al = (1 - k/kSG)*sg.al0 + (k/kSG)*alSG;
            else
                al = alSG;
            end
            w = w + al*d;
            k = k + 1;
            sol(k).al   = al;
            sol(k+1).x  = w;
        end
        e = e + 1;
        % early stopping: loss on the test set after every epoch
        Lte = sg.L(w, sg.Xte, sg.yte);
        if Lte < Lbest
            Lbest = Lte;
            wbest = w;
            ebest = e;
            kbest = k + 1;
            s = 0;
        else
            s = s + 1;
        end
    end
    par.wo = wbest;
    par.etot  = e;
    par.ebest = ebest;
    par.kbest = kbest;
    par.ktot = k + 1;
    par.tex = toc;
    ilog_sgm = false;
    if ilog_sgm
        [sol] = uosolLog(P,par,sol);
    else
        fprintf('[uosol_st] SGM: etot = %d, ebest = %d, ktot = %d, kbest = %d\n', ...
                par.etot, par.ebest, par.ktot, par.kbest);
    end
    return
end
ldescent = true;
while norm(g(x)) > par.epsG & k <= par.maxiter & (ldescent | par.isd == 4)   % maxiter iterations, as uosol.p
    if par.isd == 1
        d = -g(x);
    elseif par.isd == 3
        if k == 1
            H = eye(n);
        else
            H  = sol(k-1).H;
            s  = x - sol(k-1).x;
            y  = g(x) - sol(k-1).g;
            sy = s'*y;
            if sy > 1e-12
                rho = 1/sy;
                V   = eye(n) - rho*(y*s');
                H   = V'*H*V + rho*(s*s');
            end
        end
        d = -H*g(x);
        sol(k).H = H;
    elseif par.isd == 4
        d = -h(x)\g(x);
        sol(k).H = h(x);
    end
    ldescent = d'*g(x) < 0;
    almax = NaN;
    if     par.isd == 4
        al = 1;
        ACout = "";
    elseif par.iAC == 0
        al = -(g(x)'*d)/(d'*h(x)*d);
        ACout = "ELS";
    elseif par.iAC <= 3
        [al,ACout] = uoBLS_st(x,d,P,par);
    elseif par.iAC == 4
        if k == 1
            almax = 1;
        else
            almax = 2*(f(x) - sol(k-1).f)/(g(x)'*d);
        end
        [al,iout] = uoBLSNW32(f,g,x,d,almax,par.c1,par.c2);
        labels = ["SWC","max","eps"];
        ACout = labels(iout+1);
    end
    sol(k).x = x;
    sol(k).f = f(x);
    sol(k).g = g(x);
    sol(k).ng = norm(g(x));
    sol(k).d = d;
    sol(k).al = al;
    sol(k).almax = almax;
    sol(k).AC = ACout;
    x = x + al*d; k = k+1;
end 
sol(k).x  = x;
sol(k).f  = f(x);
sol(k).g  = g(x);
sol(k).ng = norm(g(x));
par.tex = toc;
[sol] = uosolLog(P,par,sol);

end
% [end] Function [uosol_st] %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%