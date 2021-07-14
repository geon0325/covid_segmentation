import numpy as np
import tool as tl
import time
from sir import SIR
from scipy.stats import norm

ZERO = 1e-100
C_F = 8

def write_result(filename, X, DIST, BOUND):
    (n, d) = np.shape(X)

    dist_comb = DIST[0]
    for i in range(1, len(DIST)):
        dist_comb = np.concatenate((dist_comb, DIST[i]))
    err = tl.RMSE(X[:,[1,2]], dist_comb[:,[1,2]])

    f = open(filename, 'w')
    f.write(str(err) + '\n')

    for i in range(len(BOUND)):
        f.write(str(BOUND[i][0]) + ',' + str(BOUND[i][1]) + '\n')
    for i in range(n):
        f_str = ''
        for j in range(1, d):
            f_str += str(X[i][j]) + ','
        for j in range(1, d):
            f_str += str(dist_comb[i][j]) + ','
        f.write(f_str[:-1] + '\n')
    f.close()

def fit_binary(X, max_bound=-1e10):
    (n, d) = np.shape(X)
    ERR = []; DIST = []; BOUND = []; MODEL = [];

    for i in range(2, n-2):
        s1 = 0; e1 = i;
        s2 = i; e2 = n;
        if max_bound != -1e10:
            if i > max_bound: continue
        try:
            model1 = _train_sir(X[s1:e1])
            dist1 = _get_dist(model1)
            err1 = tl.NORM(X[s1:e1], dist1)
            
            model2 = _train_sir(X[s2:e2])
            dist2 = _get_dist(model2)
            err2 = tl.NORM(X[s2:e2], dist2)
            
            dist = np.concatenate((dist1, dist2))
            err = tl.NORM(X, dist)
            print(err, err1, err2)

            ERR.append([err, err1, err2])
            DIST.append([dist1, dist2])
            BOUND.append(i)
            MODEL.append([model1, model2])
        except: continue

    ret_ERR = [1e10, 1e10, 1e10]; ret_DIST = None; ret_BOUND = None; ret_MODEL = None;

    for i in range(len(ERR)):
        if ERR[i][0] < ret_ERR[0]:
            ret_ERR = ERR[i]
            ret_DIST = DIST[i]
            ret_BOUND = BOUND[i]
            ret_MODEL = MODEL[i]
    return ret_ERR, ret_DIST, ret_BOUND, ret_MODEL

def _train_sir(X):
    (n, d) = np.shape(X)
    model = SIR(X)
    model.fit()
    return model

def _get_dist(model, dist_len=-1):
    dist = model.gen(dist_len)
    return dist

def get_costC(X_org, X_rec, std=-1):
    delta = X_rec - X_org
    delta[np.isnan(delta)] = 0
    if std == -1: std = np.std(delta) + ZERO
    mean = 0
    pdfs = norm.pdf(delta, mean, std)
    pdfs[pdfs > 1] = 1
    pdfs[pdfs <= ZERO] = ZERO
    costC = np.nansum(-np.log2(pdfs))
    return costC, std

def segment(X, BOUND, N):
    s = BOUND[0]; e = BOUND[1];
    (n, d) = np.shape(X[s:e])

    # without segmentation
    model = _train_sir(X[s:e])
    dist = _get_dist(model)
    costM = 2 * C_F
    costC, std = get_costC(X[s:e], dist)
    costT = costM + costC

    # with segmentation
    min_costT = costT
    min_BOUND = BOUND; min_DIST = [dist]; min_MODEL = [model];
    for i in range(2, n-2):
        s1 = s; e1 = s + i;
        s2 = s + i; e2 = e;

        try: 
            model_1 = _train_sir(X[s1:e1])
            dist_1 = _get_dist(model_1)
            costM_1 = costM
            costC_1, std_1 = get_costC(X[s1:e1], dist_1, std)

            model_2 = _train_sir(X[s2:e2])
            dist_2 = _get_dist(model_2)
            costM_2 = costM
            costC_2, std_2 = get_costC(X[s2:e2], dist_2, std)

            costT_seg = (np.log2(N) + costM_1 + costM_2) + (costC_1 + costC_2)
            print(s, '\t', e, '\t', costT, '\t\t', s+i, '\t', costT_seg)

            if costT_seg < min_costT:
                min_costT = costT_seg
                min_BOUND = [[s1, e1], [s2, e2]]
                min_DIST = [dist_1, dist_2]
                min_MODEL = [model_1, model_2]
        except: continue

    if min_costT == costT:
        return min_DIST, min_BOUND, min_MODEL
    else:
        min_DIST_1, min_BOUND_1, min_MODEL_1 = segment(X, min_BOUND[0], N)
        min_DIST_2, min_BOUND_2, min_MODEL_2 = segment(X, min_BOUND[1], N)
        return min_DIST_1 + min_DIST_2, min_BOUND_1 + min_BOUND_2, min_MODEL_1 + min_MODEL_2

def fit_sir(X, fn):
    #---------------------------------#
    tl.comment("Start learning SIR")
    #---------------------------------#
    _path = fn + 'fit_sir'
    (n, d) = np.shape(X)

    DIST, BOUND, MODEL = segment(X, [0, n], n) 
    BOUND = [[BOUND[i], BOUND[i+1]] for i in range(0, len(BOUND), 2)]
    write_result(_path + '.txt', X, DIST, BOUND)

    #---------------------------------#
    tl.comment("Finish learning SIR")
    #---------------------------------#

def forecast_sir(X, fn):
    #---------------------------------#
    tl.comment("Start learning SIR")
    #---------------------------------#
    _path = fn + 'forecast_sir'
    (n, d) = np.shape(X)
    
    alpha = 0.9
    train_size = int(n * alpha)
    test_size = n - train_size

    DIST, BOUND, MODEL = segment(X, [0, train_size], train_size)
    BOUND = [[BOUND[i], BOUND[i+1]] for i in range(0, len(BOUND), 2)]
    DIST[-1] = _get_dist(MODEL[-1], dist_len=len(DIST[-1])+test_size)
    write_result(_path + '.txt', X, DIST, BOUND)

    #---------------------------------#
    tl.comment("Finish learning SIR")
    #---------------------------------#
   
def fit_sir_single(X, fn):
    #---------------------------------#
    tl.comment("Start learning SIR")
    #---------------------------------#
    _path = fn + 'fit_sir_single'
    (n, d) = np.shape(X)
    
    model = _train_sir(X)
    dist = _get_dist(model)
    DIST = [dist]
    BOUND = [[0, n]]

    write_result(_path + '.txt', X, DIST, BOUND)

    #---------------------------------#
    tl.comment("Finish learning SIR")
    #---------------------------------#
  
def forecast_sir_single(X, fn):
    #---------------------------------#
    tl.comment("Start learning SIR")
    #---------------------------------#
    _path = fn + 'forecast_sir_single'
    (n, d) = np.shape(X)
    
    alpha = 0.9
    train_size = int(n * alpha)
    test_size = n - train_size

    model = _train_sir(X[:train_size])
    dist = _get_dist(model, dist_len=n)
    DIST = [dist]
    BOUND = [[0, train_size]]

    write_result(_path + '.txt', X, DIST, BOUND)

    #---------------------------------#
    tl.comment("Finish learning SIR")
    #---------------------------------#

def fit_sir_inc(X, fn, ERR_RAT):
    #---------------------------------#
    tl.comment("Start learning SIR")
    #---------------------------------#
    _path = fn + 'fit_sir_inc' + str("{:.2f}".format(ERR_RAT))
    (n, d) = np.shape(X)
    NUM_TRIAL = 5

    pre_dist = None
    DIST = []; BOUND = [];

    trial = 0; s_idx = 0; e_idx = 2;
    while e_idx < n:
        X_size = tl.NORM(X[s_idx:e_idx], [])

        try:
            model = _train_sir(X[s_idx:e_idx])
            dist = _get_dist(model)
            err = tl.NORM(X[s_idx:e_idx], dist)
            print(s_idx, '\t', e_idx, '\t', err, '\t', X_size * ERR_RAT)
        except: continue

        if err > X_size * ERR_RAT:
            if pre_dist is None:
                if trial < NUM_TRIAL:
                    trial += 1; continue;
                trial = 0;
                DIST.append(dist)
                BOUND.append([s_idx, e_idx])
                s_idx = e_idx; e_idx = e_idx + 2;
            else:
                DIST.append(pre_dist)
                BOUND.append([s_idx, e_idx-1])
                s_idx = e_idx - 1; e_idx = e_idx + 1;
            trial = 0; pre_dist = None
        else:
            pre_dist = dist
            e_idx += 1

    # last regime
    e_idx = n
    while True:
        try:
            model = _train_sir(X[s_idx:e_idx])
            dist = _get_dist(model)[1]
            DIST.append(dist)
            BOUND.append([s_idx, e_idx])
            err = tl.NORM(X[s_idx:e_idx], dist)
            print(s_idx, '\t', e_idx, '\t', err)
            break
        except: continue

    write_result(_path + '.txt', X, DIST, BOUND)

    #---------------------------------#
    tl.comment("Finish learning SIR")
    #---------------------------------#


