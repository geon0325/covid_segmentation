import tool as tl
import numpy as np
import time
import fit_sir

class SIR:
    def __init__(self, data):
        self.data = data
        self.init = None
    def fit(self):
        model = _fit(self)
        return model

    def fit_param(self, wtype='uniform'):
        return fit_sir.nl_fit(self, 'param', wtype)

    def setParams(self):
        _setParams(self)
    def getParams(self):
        (beta, gamma) = _getParams(self)
        return (beta, gamma)

    def gen(self, dist_len=-1):
        return _gen(self, dist_len)


def _defunc(dist, de, beta, gamma):
    de[0] = -beta * dist[0] * dist[1]
    de[1] = beta * dist[0] * dist[1] - gamma * dist[1]
    de[2] = gamma * dist[1]
    return de

def _fit(model):
    data = model.data
    model.setParams()
    model.fit_param('uniform')
    return model

def _setParams(model):
    (model.n, model.d) = np.shape(model.data)
    model.k = model.d
    model.beta = 0.00001
    model.gamma = 0.00001
    model.m0 = [model.data[0][0], model.data[0][1], model.data[0][2]]

def _getParams(model):
    beta = model.beta
    gamma = model.gamma
    return (beta, gamma)

def _gen(model, dist_len=-1):
    n = model.n; k = model.k; N = n;
    if dist_len != -1: N = dist_len
    dt = 0.1
    f1 = np.zeros(k); f2 = np.zeros(k); f3 = np.zeros(k); f4 = np.zeros(k);
    (beta, gamma) = model.getParams()
    dist = np.zeros((N, k))
    if model.init == None: 
        dist[0][0] = model.data[0][0]; 
        dist[0][1] = model.data[0][1]; 
        dist[0][2] = model.data[0][2];
    else:
        dist[0][0] = model.init[0]
        dist[0][1] = model.init[1]
        dist[0][2] = model.init[2]
    for t in range(0, N-1):
        f1 = _defunc(dist[t], f1, beta, gamma)
        f2 = _defunc(dist[t] + dt/2.0 * f1, f2, beta, gamma)
        f3 = _defunc(dist[t] + dt/2.0 * f2, f3, beta, gamma)
        f4 = _defunc(dist[t] + dt*f3, f4, beta, gamma)
        dist[t+1] = dist[t] + dt/6.0 * (f1 + 2.0*f2 + 2.0*f3 + f4)
    dist[np.isnan(dist)] = 0
    return dist
