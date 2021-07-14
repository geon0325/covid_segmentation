#!/usr/bin/env python
#------------------------------------------------------------#
# Author:    Yasuko Matsubara 
# Email:     yasuko@cs.kumamoto-u.ac.jp
# URL:       http://www.cs.kumamoto-u.ac.jp/~yasuko/
# Date:      2016-09-15
#------------------------------------------------------------#
# Copyright (C) 2015--2016 Yasuko Matsubara & Yasushi Sakurai
# RegimeCast is freely available for non-commercial purposes
#------------------------------------------------------------#
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pickle
import sys
import os
import pylab as pl
from scipy.stats import norm # gaussian pdf
import zipfile 
import scipy.io
import copy

#--------------#
# ALL          #
#--------------#
ZERO=1.e-10
INF= 1.e+10
YES=1; NO=0;
#--------------#


####################
# copy, etc.
####################
def dcopy(X):
    Y=copy.deepcopy(X)
    return Y
####################
# math, etc.
####################
def log_s(x):
    if(x==0): return 0
    return 2.0*np.log2(x)+1.0;
def log2(x):
    if(x==0): return 0
    return np.log2(x)


#######################
# tools for time series  
#######################

#-----------------------------#
# input: X[n][d]    d-dim seq of length n
#        wd:        window-size
# output: Y[n][d]   smooth seq
#-----------------------------#
def smoothExp(X, wd):
    # (1) set parameter alpha
    if(wd>=1): # if wd<=1, then alpha=2/(windowsize+1)
        alpha=2.0/(wd+1.0)
    else: 
        alpha=wd
    # (2) check size: X (n x d)
    (n,d)=np.shape(X)
    # (3) exponential smoothing
    s=np.zeros((n,d))
    s[0,:]=X[0,:]
    for i in range(0,d):
        for t in range(0+1,n):
            s[t,i]=alpha*X[t,i] + (1-alpha)*s[t-1,i]
            if(np.isnan(X[t,i])):
                s[t,i]= alpha*s[t-1,i] + (1-alpha)*s[t-1,i]
    Y=s
    return Y

def smoothMA(X, wd):
    return smoothMAo(X,wd)

def smoothMAa(X, wd):
    wd = int(wd)
    if(wd==1):
        return X
    # X[n][d]
    (n,d)=np.shape(X)
    Y = np.zeros((n,d))
    for i in range(0,d):
        for t in range(0,n):
            st=t-np.floor(wd/2); ed=t+np.floor(wd/2);
            if(st<0): st=0; 
            if(ed>n): ed=n;
            Y[t,i]=np.nanmean(X[int(st):int(ed),i])
    return Y

# weightedMA
def smoothWMA(X, wd):
    wd = int(wd)
    if(wd==1):
        return X
    # X[n][d]
    (n,d)=np.shape(X)
    Y = np.zeros((n,d))
    wd2=wd/2; wlst=range(1,wd2+1)
    wt=(wlst+[wd2+1]+list(reversed(wlst)))
    for i in range(0,d):
        for t in range(0,n):
            st=t-wd/2; ed=t+wd/2;
            if(st<0): st=0; 
            if(ed>n): ed=n;
            wts=wt[0:ed-st]; 
            wts=np.asarray(wts)
            wts=wts/(1.0*np.sum(wts))*len(wts)
            Y[t,i]=np.nanmean(X[st:ed,i]*wts)
    return Y

# MA-online
def smoothMAo(X, wd):
    wd = int(wd)
    if(wd==1):
        return X
    # X[n][d]
    (n,d)=np.shape(X)
    Y = np.zeros((n,d))
    for i in range(0,d):
        Y[:,i] = np.convolve(X[:,i], np.ones(wd)/wd, mode='same')
    return Y

# weightedMA-online
def smoothWMAo(X, wd):
    wd = int(wd)
    if(wd==1):
        return X
    # X[n][d]
    (n,d)=np.shape(X)
    Y = np.zeros((n,d))
    wd2=wd; wt=np.arange(1,wd2+1)
    for i in range(0,d):
        for t in range(0,n):
            st=max(0,t-wd); ed=t;
            wts=wt[0:ed-st]; #wts=np.asarray(wts);
            wts=wts/(1.0*np.sum(wts))*len(wts)
            Y[t,i]=np.nanmean(X[st:ed,i]*wts)
    return Y



#-----------------------------#
# input: X[n][d]    d-dim seq of length n
#        wd:        window-size
# output: Y[int(n/wd)][d]   smooth seq
#-----------------------------#
def smoother(X, wd):
    wd = int(wd)
    if(wd==1):
        return X
    # X[n][d]
    (n,d)=np.shape(X)
    n2=int(n/wd)
    Y = np.zeros((n2,d))
    for i in range(0,d):
        for j in range(0,n2):
            sum=0
            for w in range(0,wd):
               sum+=X[j*wd+w][i] 
            Y[j][i]=sum/wd
    return Y
        
#-----------------------------#
# input:  X[n][d]  
# output: Y[n][d]   Z-normalized seq
#-----------------------------#
def normalizeZ(X):
    (n,d) = np.shape(X)
    mean=np.nanmean(X,0)
    std=np.nanstd(X,0) + ZERO
    Y = (X-mean)/(std);
    return Y, mean, std

def zero2nan(X):
    X[X==0]=np.nan
    return X

def nan2zero(X):
    X[np.isnan(X)]=0;
    return X

# X: org, E: est
def NORM(A,B):
    if (B==[]): B = 0.0*A
    diff = A.flatten() - B.flatten()
    return np.sqrt(np.nansum(pow(diff,2)))

def RMSER(X,E):
    errR = RMSE(X,E)/(ZERO + RMSE(X,[]))
    return errR

def RMSE(A,B):
    if(B==[]): B=0.0*A;
    diff=A.flatten() - B.flatten()
    return np.sqrt(np.nanmean(pow(diff, 2)))

def GpdfL(X, mean,var):
    #.stats.norm
    std=np.sqrt(var)
    Lhl = np.nansum(np.log(norm.pdf(X, mean, std) + ZERO))
    print(Lhl)
    return Lhl


####################
# misc 
####################
def func_Weight(n, wtype):
    n=int(n)
    if(wtype=='uniform'):
        # uniform
        return 1.0*np.ones(n)
    elif(wtype=='linear'):
        # linear 
        return 1.0*np.arange(0,n)
    elif(wtype=='linear_inv'):
        # linear  (inverse)
        return 1.0*np.arange(n,0,-1)
    elif(wtype=='exp'):
        # exponential
        T=float(n)/2.0; ticks=np.arange(0,n)
        val=np.exp(ticks/T)
        #val=(val)/sum(val)*n
        return val
    else:
        error("func_weight: usage uniform/linear")
    #return 1.0*np.ones(n) #range(0,n)


# compute W[t,r]*X[r,t,:]   (W: n x k; X: k x n x d)
# return Y (n x d)
def func_Weight_prod(W, X):
    (k,n,d)=np.shape(X)
    (nw,kw)=np.shape(W)
    if(n!=nw or k!=kw): error("len(X)!=len(W)")
    Y=np.zeros((n,d))
    for i in range(0,n):
        for j in range(0,d):
            for r in range(0,k):
                Y[i][j]+=W[i][r]*X[r][i][j]
    return Y


####################
# display
####################
def printM(X):
    if(np.ndim(X)==1): 
        # array
        a=len(X)
        for i in range(0,a):
            print(u"%.4e" % X[i])
        print("")
    elif(np.ndim(X)==2):
        # matrix
        (a,b)=np.shape(X)
        for i in range(0,a):
            for j in range(0,b):
                print(u"%.4e" % X[i][j])
            print("")
    else:
        print("NaN") 

def error(msg):
    print("====================")
    print(" error              ")
    print("--------------------")
    print("%s"%msg)
    print("====================")
    #raise
    sys.exit(0);
def warning(msg):
    print("*** Warning *** : %s"%msg)
def dotting(dot='.'):
    sys.stdout.write(dot)
    sys.stdout.flush()
def comment(msg):
    print("--------------------")
    print(" %s"%msg)
    print("--------------------")
def msg(msg):
    print(" %s"%msg)
def notfin():
    print("--------------------")
    print(" NOT FIN ")
    print("--------------------")
def figure(fid, fsize=(8,6)):
    fig = plt.figure(figsize=fsize)
    try: 
        fig.set_tight_layout(True)
    except:
        msg("cannot: fig.set_tight_layout")
    plt.ion() 
    plt.clf()
    return fig 
def resetCol(): 
    try: 
        pl.gca().set_prop_cycle(None)
    except:
        try: pl.gca().set_color_cycle(None)
        except: msg("cannot: cycle")

####################
# IO, etc.
####################
def mkdir(dir):
    # exist or not
    if(os.path.exists(dir)):
        msg('dir exists')
        return dir
    # create or not
    try: os.mkdir(dir)
    except: 
        error("cannot create dir: %s"%dir)
    return dir
def loadsq(fn):
    tmp=pl.loadtxt(fn, ndmin=2).T
    return tmp

# object IO
def load_obj(fn):
    f = open("%s"%(fn),'rb')
    obj = pickle.load(f)
    f.close()
    return obj
def save_obj(obj, fn):
    f=open("%s.obj"%(fn), "wb")
    pickle.dump(obj, f)
    f.close()
def save_txt(X, fn):
    np.savetxt(fn, X, delimiter='  ') 
# matlab "xx.mat"
def save_mat(mdicts, matfn):
    scipy.io.savemat("%s.mat"%matfn, mdict=mdicts) #%{'dat': dat, 'arr': dat})	
    


