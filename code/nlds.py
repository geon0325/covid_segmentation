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
import tool as tl
import numpy as np
from kf import KF
import pylab as pl
import fit
import copy
import time


#-----------------------------------------#
K=2                  # hidden variables k
NTRIAL=2 #1          # number of trials (default: 1)
ITER=2               # KF - em iter
ITERe=20             # KF - em iter (each)
BOUNDARY=tl.INF      # NLDS boundary
NLIN=tl.YES          # nonlinear or not
RK=tl.YES            # YES: Runge-Kutter / NO: Euler
#-----------------------------------------#
DBG=tl.NO
FULLSAVE=tl.NO
#-----------------------------------------#
# parameters and equations, etc.
#-----------------------------------------#
# param=(A0,A1,A2,B0,B1,m0)
# s'(t)= A0 + A1*s(t) + A2*S(t)
# v(t) = B0 + B1*s(t)
# s(0) = m0
#-----------------------------------------#
# Sta: latent activity s(t)
# Obs: observation v(t)
#-----------------------------------------#

class NLDS:
    def __init__(self, data):
        if(DBG): tl.msg("NLDS")
        self.data=data
        self.K=K
    # model fit (nonlinear)
    def fit(self, wtype, linearfit = False):
        nlds = _fit_optk(self, wtype, linearfit)
        return nlds 
    # model fit (linear)
    def fit_lin(self, wtype):
        return _fit_optk(self, wtype, True)

    # specific model-fit 
    #fit.nl_fit(nlds, ftype, wtype, dps)
    def fit_A2(self, wtype='uniform', dps=1):
        return fit.nl_fit(self, 'A2', wtype, dps)    
    def fit_A01(self, wtype='uniform', dps=1):
        return fit.nl_fit(self, 'A01', wtype, dps)    
    def fit_B01(self, wtype='uniform', dps=1):
        return fit.nl_fit(self, 'B01',wtype, dps)    
    def fit_m0(self, wtype='uniform', dps=1):
        return fit.nl_fit(self, 'm0', wtype, dps)

    #def rmse(self, dps): #notfin
    def rmse(self):
        data=self.data; (n,d)=np.shape(data)
        (Sta,Obs)=self.gen(n)
        return tl.RMSE(data, Obs)
    def rmser(self):
        data=self.data; (n,d)=np.shape(data)
        (Sta,Obs)=self.gen(n)
        return tl.RMSER(data, Obs)

    def setKFParams(self, kf):
        _setKFParams(self, kf)
    def getParams(self):
        (A0,A1,A2,B0,B1,m0)=_getParams(self)
        return (A0,A1,A2,B0,B1,m0)
    def gen(self, n=-1, dps=1, init=None): 
        return _gen(self,n,dps, init)
    def gen_lin(self, n=-1):
        return _gen_lin(self, n)
    def plot(self, fn='', disp=False, nf=-1):
        _plot(self, disp, nf, fn)
    def save_obj(self):
        self.plot(self.fn)
        tl.save_obj(self, "%s_nlds"%(self.fn))

#-----------------------------------------#
# differencial equations
#-----------------------------------------#
# Sta0[k], de[k]
def _defunc(Sta0, de, A0, A1, A2, k):
    # boundary value 
    if(np.max(np.abs(Sta0)>BOUNDARY)): return de*0.0
    for i in range(0,k):
        de[i]=A0[i]
        for j in range(0,k):
            de[i]+=A1[i][j]*Sta0[j] + A2[i][j][j]*Sta0[j]*Sta0[j]
            #de[i]+=A1[i][j]*Sta0[j] - np.abs(A2[i][j][j])*Sta0[j]*Sta0[j] #notfin
    return de  
#-----------------------------------------#

def _fit_optk(nlds, wtype, linearfit=False):
    data = nlds.data
    NLDSs = []; Errs = [];
    # for each k, estimate opt-params
    for trial in range(0, NTRIAL):
        # init model (k)
        nlds_k = tl.dcopy(nlds)
        # linear fitting 
        (nlds, err) = _fit_k_lin(nlds_k, K) 
        NLDSs.append(nlds_k)
        Errs.append(err)
        if Errs[len(Errs)-2] < Errs[len(Errs)-1]: break
    kbst = np.argmin(Errs)
    nlds = NLDSs[kbst]
    # non-linear fit
    if(linearfit == False): nlds = nlds.fit_A2(wtype)
    nlds = nlds.fit_m0(wtype)
    return nlds

# linear fit using kf.py
def _fit_k_lin(nlds, k, kf_i=[]):
    data = nlds.data
    # linear fit (KF)
    if(kf_i == []):
        kf = KF(data,k) # without init params
    else:
        kf = kf_i # with init params
    kf.em(ITER, ITERe) # em algorithm
    nlds.setKFParams(kf)
    # fit m0
    nlds=nlds.fit_m0()
    (Sta,Obs)=nlds.gen(len(data))
    err=tl.RMSE(Obs, data)
    return (nlds, err)

def _setKFParams(model, kf):
    #-------------#
    model.d=kf.d
    model.n=kf.n
    model.k=kf.k
    model.data=kf.data
    model.kf=kf
    #-------------#
    k=model.k
    (A,b,C,d,Q,R,m0,s0)=model.kf.getParams()
    # latent func.
    model.A0=b
    model.A1=A-np.eye(k)
    model.A2=np.zeros((k,k,k))
    # obs func.
    model.B0=d
    model.B1=C
    # init func.
    model.m0=m0
def _getParams(model):
    A0=model.A0
    A1=model.A1
    A2=model.A2
    B0=model.B0
    B1=model.B1
    m0=model.m0
    return (A0,A1,A2,B0,B1,m0)

# LDS
def _gen_lin(model, n):
    if(n==-1): n=model.n
    d=model.d 
    k=model.k
    (A,b,C,d,Q,R,m0,s0)=model.kf.getParams()
    Sta=np.zeros((n, k))
    Sta[0]=m0
    for t in range(0,n-1):
        Sta[t+1]  = (np.dot(A, Sta[t]))+b
    Obs=np.dot(C, Sta.T).T+d
    return (Sta, Obs)


def _gen(model, n, dps, init=None): 
    if(n == -1): n = model.n
    if(dps > 1): n = int(np.ceil(1.0*n/dps)); 
    dt = float(dps)
    d = model.d; k = model.k
    if(RK): f1 = np.zeros(k); f2 = np.zeros(k); f3 = np.zeros(k); f4 = np.zeros(k);
    #-----------------------------------------#
    (A0, A1, A2, B0, B1, m0) = model.getParams()
    Sta = np.zeros((n,k)); Obs = np.zeros((n,d))
    #-----------------------------------------#
    # (1) compute states - Sta[n x k]
    # latent states (t=0)
    if init is None:
        for i in range(0, k):
            Sta[0][i] = m0[i]
    else:
        for i in range(0, k):
            Sta[0][i] = init[i]
    # latent states (t>0)
    for t in range(0, n-1):
        if(RK): 
            f1 = _defunc(Sta[t], f1, A0, A1, A2, k)
            f2 = _defunc(Sta[t]+dt/2.0*f1, f2, A0, A1, A2, k)
            f3 = _defunc(Sta[t]+dt/2.0*f2, f3, A0, A1, A2, k)
            f4 = _defunc(Sta[t]+dt*f3, f4, A0, A1, A2, k)
            Sta[t+1]=Sta[t]+dt/6.0*( f1 + 2.0*f2 + 2.0*f3 + f4)
        else:
            Sta[t+1]=Sta[t]+dt*_defunc(Sta[t], Sta[t+1],A0,A1,A2,k)
    #-----------------------------------------#
    # (2) compute observations - Obs[n x d]               
    Obs = B0 + np.dot(B1, Sta.T).T
    Sta[np.isnan(Sta)] = 0; Obs[np.isnan(Obs)] = 0;
    return (Sta, Obs) 



def _plot(nlds, disp=False, nf=-1, fn=''):
    if(nf==-1): nf=int(nlds.n)
    (Sta, Obs)=nlds.gen(nf) 
    (StaL,ObsL)=nlds.gen_lin(nf) 
    err_o=tl.RMSE(nlds.data,0*nlds.data)
    err =tl.RMSE(Obs[0:len(nlds.data),:],  nlds.data)
    errL=tl.RMSE(ObsL[0:len(nlds.data),:], nlds.data)

    data=nlds.data
    mn=np.nanmin(data.flatten())
    mx=np.nanmax(data.flatten())
    mnx=abs(mx-mn)
    mx+=0.1*mnx; mn-=0.1*mnx
    fig=tl.figure(10)
    #
    tl.plt.subplot(321)
    tl.resetCol()
    tl.plt.plot(data,'--', lw=1)
    tl.resetCol()
    tl.plt.plot(Obs, '-', lw=1)
    tl.plt.xlim([0,nf])
    tl.plt.ylim([mn,mx])
    tl.plt.title("Original vs. NLDS: err %.2f (%.2f)"%(err, err/err_o))
    #
    tl.plt.subplot(322)
    tl.resetCol()
    tl.plt.plot(data,'--', lw=1)
    tl.resetCol()
    tl.plt.plot(ObsL,'-', lw=1)
    tl.plt.xlim([0,nf])
    tl.plt.ylim([mn,mx])
    tl.plt.title("Original vs. LDS: errL %.2f (%.2f)"%(errL, errL/err_o))
    #
    tl.plt.subplot(323) 
    tl.plt.plot(Sta, '-', lw=1)
    tl.plt.xlim([0,nf])
    tl.plt.title("s(t) --- Sta (NLDS) k:%d"%(nlds.k))
    #
    tl.plt.subplot(324) 
    tl.plt.plot(StaL, '-', lw=1)
    tl.plt.xlim([0,nf])
    tl.plt.title("s(t) --- Sta (LDS) k:%d"%(nlds.k))
    #
    tl.plt.subplot(325) 
    for i in range(0,len(Obs[0])):
        tl.plt.fill_between(range(0,len(Obs)), Obs[:,i]+2*err, Obs[:,i]-2*err, facecolor=[0.8,0.8,0.8], alpha=1.0, linewidth=0.0)
    tl.plt.plot(Obs, '-', lw=1)
    tl.plt.xlim([0,nf])
    tl.plt.ylim([mn,mx])
    tl.plt.title("v(t) --- Obs (NLDS)")
    #
    tl.plt.subplot(326) 
    for i in range(0,len(Obs[0])):
        tl.plt.fill_between(range(0,len(ObsL)), ObsL[:,i]+2*errL, ObsL[:,i]-2*errL, facecolor=[0.8,0.8,0.8], alpha=1.0, linewidth=0.0)
    tl.plt.plot(ObsL, '-', lw=1)
    tl.plt.xlim([0,nf])
    tl.plt.ylim([mn,mx])
    tl.plt.title("v(t) --- Obs (LDS)")
    #
    if(fn!=''): tl.plt.savefig(fn+'_nlds_fit.png')
    tl.plt.draw()
    tl.plt.show(block=disp)
    if(disp!=True): tl.plt.close()

    # save data (optional)
    if(FULLSAVE):
        tl.save_txt(data, fn+'_seq_data.txt')
        tl.save_txt(Sta, fn+'_seq_Sta.txt')
        tl.save_txt(Obs, fn+'_seq_Obs.txt')
        tl.save_txt(ObsL, fn+'_seq_ObsL.txt')



def _example1():
    A=np.arange(0,100)
    data=np.asarray([np.sin(A/10.0),np.cos(A/10.0)])
    data=data.T 
    data=tl.normalizeZ(data)
    return data
def _example2(): 
    inputfn="_dat/mocap/20_01.amc.4d"
    data=tl.loadsq(inputfn)
    data=data.T
    data=data[320:400,:]
    return data 

#---------------#
#     main      #
#---------------#
if __name__ == "__main__":
    from nlds import NLDS 

    tl.comment("load data")
    #data=_example1()
    data=_example2()
    (n,d) = np.shape(data)
    outdir="_out/tmp/nlds_out"
    wtype='uniform' #linear
    
    tl.comment("create NLDS")
    nlds=NLDS(data, outdir)
    tl.comment("start LMfit")
    tic = time.clock()
    #nlds=nlds.fit_lin(wtype)
    nlds=nlds.fit(wtype)
    toc = time.clock(); fittime= toc-tic;
    tl.comment("END LMfit")
    tl.msg("time: %f"%fittime)
    
    '''
    print ("start LMfit")
    DPS=1; niter=10
    for iter in range(0,niter): 
        tic = time.clock()
        for iter2 in range(0,niter): 
            nlds=nlds.fit_m0(wtype, DPS)
            nlds=nlds.fit_A01(wtype, DPS)
        nlds=nlds.fit_A2(wtype, DPS)
        #
        toc = time.clock(); fittime= toc-tic;
        (Sta,Obs)=nlds.gen(); err=tl.RMSE(Obs,data)
        print("time: %f, err=%f"%(fittime,err))
    
        nlds.plot()
        nlds.save_obj()
    '''

    tl.comment("plot/save model")
    nlds.save_obj()

    
