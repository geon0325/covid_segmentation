import numpy as np
import tool as tl
import time
from scipy.stats import norm
from nlds import NLDS

ZERO = 1e-100
C_F = 8
d = 2

def write_result(filename, X, DIST, BOUND, mean=None, std=None):
    (n, d) = np.shape(X)

    dist_comb = DIST[0]
    for i in range(1, len(DIST)):
        dist_comb = np.concatenate((dist_comb, DIST[i]))

    if mean is not None and std is not None:
        X = (X * std) + mean
        dist_comb = (dist_comb * std) + mean

    err = tl.RMSE(X, dist_comb)

    f = open(filename, 'w')
    f.write(str(err) + '\n')

    for i in range(len(BOUND)):
        f.write(str(BOUND[i][0]) + ',' + str(BOUND[i][1]) + '\n')
    for i in range(n):
        f_str = ''
        for j in range(d):
            f_str += str(X[i][j]) + ','
        for j in range(d):
            f_str += str(dist_comb[i][j]) + ','
        f.write(f_str[:-1] + '\n')
    f.close()

def _train_nl(X, linearFit=False):
    model = NLDS(X)
    model = model.fit(wtype='uniform', linearfit=linearFit)
    return model

def _get_dist(model, dist_len=-1, init=None, linearFit=False):
    if not linearFit:
        dist = model.gen(dist_len, init=init)
    else:
        dist = model.gen_lin(dist_len)
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

def segment(X, BOUND, linearFit, N, max_bound=-1e10):
    s = BOUND[0]; e = BOUND[1];
    (n, d) = np.shape(X[s:e])
    
    # without segmentation
    model = _train_nl(X[s:e], linearFit=linearFit)
    dist = _get_dist(model, linearFit=linearFit)[1]
    costM = C_F * (model.K**2 + (d+2+int(not linearFit))*model.K + d)
    costC, std = get_costC(X[s:e], dist)
    costT = costM + costC
    
    # with segmentation
    min_costT = costT;
    min_BOUND = BOUND; min_DIST = [dist]; min_MODEL = [model];
    for i in range(2, n-2):
        s1 = s; e1 = s + i;
        s2 = s + i; e2 = e;
        
        if max_bound != -1e10:
            if s + i > max_bound: continue

        try:
            model_1 = _train_nl(X[s1:e1], linearFit=linearFit)
            dist_1 = _get_dist(model_1, linearFit=linearFit)[1]
            costM_1 = costM
            costC_1, std_1 = get_costC(X[s1:e1], dist_1, std)

            model_2 = _train_nl(X[s2:e2], linearFit=linearFit)
            dist_2 = _get_dist(model_2, linearFit=linearFit)[1]
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
        min_DIST_1, min_BOUND_1, min_MODEL_1 = segment(X, min_BOUND[0], linearFit, N)
        min_DIST_2, min_BOUND_2, min_MODEL_2 = segment(X, min_BOUND[1], linearFit, N)
        return min_DIST_1 + min_DIST_2, min_BOUND_1 + min_BOUND_2, min_MODEL_1 + min_MODEL_2

def fit_data(X, fn, linearFit, mean, std):
    #----------------------------------#
    tl.comment("Start learning NL")
    #----------------------------------#
    _path = fn + 'fit_data'
    (n, d) = np.shape(X);
    linearFit = linearFit
    
    DIST, BOUND, MODEL = segment(X, [0, n], linearFit, n)
    BOUND = [[BOUND[i],BOUND[i+1]] for i in range(0, len(BOUND), 2)]
    if not linearFit: write_result(_path + '.txt', X, DIST, BOUND, mean, std)
    else: write_result(_path + '_lin.txt', X, DIST, BOUND, mean, std)

def forecast_data(X, fn, linearFit, mean, std):
    #----------------------------------#
    tl.comment("Start learning NL")
    #----------------------------------#
    _path = fn + 'forecast_data'
    (n, d) = np.shape(X);
    linearFit = linearFit
    
    alpha = 0.9
    train_size = int(n * alpha)
    test_size = n - train_size

    DIST, BOUND, MODEL = segment(X, [0, train_size], linearFit, train_size, max_bound=train_size-test_size)
    BOUND = [[BOUND[i],BOUND[i+1]] for i in range(0, len(BOUND), 2)]
    DIST[-1] = _get_dist(MODEL[-1], linearFit=linearFit, dist_len=len(DIST[-1])+test_size)[1]
    if not linearFit: write_result(_path + '.txt', X, DIST, BOUND, mean, std)
    else: write_result(_path + '_lin.txt', X, DIST, BOUND, mean, std)

    #----------------------------------#
    tl.comment("Finish learning NL")
    #----------------------------------#

def fit_data_single(X, fn, linearFit, mean, std):
    #----------------------------------#
    tl.comment("Start learning NL")
    #----------------------------------#
    _path = fn + 'fit_data_single'
    (n, d) = np.shape(X);
    linearFit = linearFit
    
    model = _train_nl(X, linearFit=linearFit)
    dist = _get_dist(model, linearFit=linearFit)[1]
    DIST = [dist]
    BOUND = [[0, n]]

    if not linearFit: write_result(_path + '.txt', X, DIST, BOUND, mean, std)
    else: write_result(_path + '_lin.txt', X, DIST, BOUND, mean, std)

    #----------------------------------#
    tl.comment("Finish learning NL")
    #----------------------------------#
   
def forecast_data_single(X, fn, linearFit, mean, std):
    #----------------------------------#
    tl.comment("Start learning NL")
    #----------------------------------#
    _path = fn + 'forecast_data_single'
    (n, d) = np.shape(X);
    linearFit = linearFit

    alpha = 0.9
    train_size = int(n * alpha)
    test_size = n - train_size

    model = _train_nl(X[:train_size], linearFit=linearFit)
    dist = _get_dist(model, linearFit=linearFit, dist_len=n)[1]
    DIST = [dist]
    BOUND = [[0, train_size]]

    if not linearFit: write_result(_path + '.txt', X, DIST, BOUND, mean, std)
    else: write_result(_path + '_lin.txt', X, DIST, BOUND, mean, std)

    #----------------------------------#
    tl.comment("Finish learning NL")
    #----------------------------------#
 

def fit_data_inc(X, fn, ERR_RAT, linearFit, mean, std):
    #----------------------------------#
    tl.comment("Start learning NL")
    #----------------------------------#
    _path = fn + 'fit_data_inc' + str("{:.2f}".format(ERR_RAT))
    (n, d) = np.shape(X);
    linearFit = linearFit
    NUM_TRIAL = 5

    pre_dist = None
    DIST = []; BOUND = [];
    
    trial = 0; s_idx = 0; e_idx = 2;
    while e_idx < n:
        X_size = tl.NORM(X[s_idx:e_idx], [])

        try:
            model = _train_nl(X[s_idx:e_idx], linearFit=linearFit)
            dist = _get_dist(model, linearFit=linearFit)[1]
            err = tl.NORM(X[s_idx:e_idx], dist)
            print(s_idx, '\t', e_idx, '\t', err, '\t', X_size * ERR_RAT)
        except: continue

        if err > X_size * ERR_RAT:
            if pre_dist is None:
                if trial < NUM_TRIAL:
                    trial += 1; continue;
                trial = 0
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
            model = _train_nl(X[s_idx:e_idx], linearFit=linearFit)
            dist = _get_dist(model, linearFit=linearFit)[1]
            DIST.append(dist)
            BOUND.append([s_idx, e_idx])
            err = tl.NORM(X[s_idx:e_idx], dist)
            print(s_idx, '\t', e_idx, '\t', err)
            break
        except: continue

    if not linearFit: write_result(_path + '.txt', X, DIST, BOUND, mean, std)
    else: write_result(_path + '_lin.txt', X, DIST, BOUND, mean, std)
        
    
    #----------------------------------#
    tl.comment("Finish learning NL")
    #----------------------------------#
    
