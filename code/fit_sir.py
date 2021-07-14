import numpy as np
import tool as tl
import time
import copy
try:
    import lmfit
except:
    tl.error("cannot find lmfit: please see http://lmfit.github.io/lmfit-py/")

#-------------------------------#
INF = 1e32
# lmfit (default)
XTL = 1.e-8
FTL = 1.e-8
MAXFEV = 100
# lmfit (incremental)
XTLi = 0.00001
FTLi = 0.00001
MAXFEVi = 20
#-------------------------------#

def nl_fit(nlds, ftype, wtype):
    nlds = _nl_fit(nlds, wtype)
    return nlds

def _nl_fit(nlds, wtype):
    # (1) create param set
    P = _createP(nlds)
    # (2) start lmfit
    lmsol = lmfit.Minimizer(_distfunc, P, fcn_args=(nlds.data, nlds, wtype))
    res = lmsol.leastsq(xtol=XTL, ftol=FTL, maxfev=MAXFEV)
    # (3) update param set
    nlds = _updateP(res.params, nlds)
    return nlds

def _createP(nlds):
    P = lmfit.Parameters()
    P.add('beta', value=nlds.beta)
    P.add('gamma', value=nlds.gamma)
    return P

def _updateP(P, nlds):
    nlds.beta = P['beta'].value
    nlds.gamma = P['gamma'].value
    return nlds

def _distfunc(P, data, nlds, wtype):
    n = np.size(data, 0)
    # update parameter set
    nlds = _updateP(P, nlds)
    # generate seq
    dist = nlds.gen()
    # diffs
    diff = data.flatten() - dist.flatten()
    diff[np.isnan(diff)] = 0
    diff[np.isinf(diff)] = INF
    # weighted-fit
    diff = diff * tl.func_Weight(len(diff), wtype)
    return diff

