# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 16:46:47 2022

@author: franc
"""

import numpy as np
import networkx as nx
from copy import copy
from math import factorial as fac
from tqdm import tqdm
from scipy import sparse
import scratches as sc
from numba import jit

import numba as nb

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


def byn_coef(N):
    if N>=2:
        return(fac(N)/(fac(N-2)*2)) 
    else:
        return 0

    
@jit(nopython=True)
def sigmoid(th):
    return(np.exp(th)/(1+np.exp(th)))

@jit(nopython=True)
def linktype(a,b):
    if a+b<5:
        return (a*b - 1)
    else:
        return (a+b - 1)





def freederivative2(theta,N_der):
    dlnZ = N_der*sigmoid(theta)
    return(dlnZ)



    


@jit
def tweak_params(exval,theta,N_der):
    if exval - freederivative2(theta,N_der) < 0:
        theta+=-0.025
    else:
        theta+=0.025
    return(theta)


def check_condition(realvals,thetas,quants):
    prop=[]
    for j in range(len(realvals)):
        prop.append(freederivative2(thetas[j],quants[j]))
        
    prop = np.array(prop)
    cond = np.allclose(prop,realvals,atol=0.2)
    return(cond)

@jit
def greedsearch_param2(realvals,NG,NL,NI,maxiter=30000,startguess=[0,0,0,0,0,0]):
    NGG = byn_coef(NG)
    NLL = byn_coef(NL)
    NII = byn_coef(NI)
    NGL = NG*NL
    NGI = NG*NI
    NLI = NL*NI
    quants= np.array([NGG,NGL,NGI,NLL,NLI,NII])
    thetas = copy(startguess)
    condition=False
    for i in tqdm(range(maxiter)):
        for j in range(len(realvals)):
            thetas[j] = tweak_params(realvals[j], thetas[j],quants[j])
            condition = check_condition(realvals,thetas,quants)
            if condition == True:
                #print('convergence')
                break
        
    if i>=(maxiter-2):
        print('no convergence')
    for k in range(len(realvals)):
       print(freederivative2(thetas[k],quants[k]))
    return(np.array(thetas))




@jit(nopython=True)
def reversetype(lt,countlist):
    if lt == 0:
        l1 = (0, countlist[0])
        l2 = (0,countlist[0])
        return(l1,l2)
    if lt == 1:  
        l1 = (0, countlist[0])
        l2 = (countlist[0],countlist[0]+countlist[1])
        return(l1,l2)
    if lt == 2:      
        l1 = (0, countlist[0])
        l2 = (countlist[0]+countlist[1],countlist[0]+countlist[1]+countlist[2])
    if lt == 3:        
        l1 = (countlist[0],countlist[0]+countlist[1])
        l2 = (countlist[0],countlist[0]+countlist[1])
    if lt == 4:       
        l1 = (countlist[0],countlist[0]+countlist[1])
        l2 = (countlist[0]+countlist[1],countlist[0]+countlist[1]+countlist[2])
    if lt == 5:
        l1 = (countlist[0]+countlist[1],countlist[0]+countlist[1]+countlist[2])
        l2 = (countlist[0]+countlist[1],countlist[0]+countlist[1]+countlist[2])
    return(l1,l2)



@jit(nopython=True)
def Dderivative(idx, ordlist, thetE,thetD):
    val=0
    for j in range(len(ordlist)):
        val+= sigmoid(thetD[idx] + thetD[j] + thetE[linktype(ordlist[idx], ordlist[j])])
    val -= sigmoid(thetD[idx] + thetD[idx] + thetE[linktype(ordlist[idx], ordlist[idx])])
    return(val)

@jit(nopython=True)
def Ederivative(bt, countlist, thetE, thetD):
    val=0
    l1,l2 = reversetype(bt,countlist)
    if l1 != l2:
        for n in range(l1[0] , l1[1]):
            for m in range(l2[0],l2[1]):
                    val+= sigmoid(thetD[n] + thetD[m] + thetE[bt])
    elif l1 == l2:
        for n in range(l1[0] , l1[1]):
            for m in range(n,l2[1]):
                    val+= sigmoid(thetD[n] + thetD[m] + thetE[bt])
    return(val)
                    
@jit(nopython=True)
def Dderivative2(idx,thetD):
    val=0
    for j in range(len(thetD)):
        val+= sigmoid(thetD[idx] + thetD[j])
    val -= sigmoid(thetD[idx] + thetD[idx])
    return(val)           
        
        

def tweak_paramsD(dval, idx, ordlist, thetE,thetD):
    if dval - Dderivative(idx, ordlist, thetE,thetD) < 0:
        thetD[idx]+=-0.025
    else:
        thetD[idx]+=0.025
    return(thetD)

def tweak_paramsE(Eval, bt, countlist, thetE, thetD):
    if Eval - Ederivative(bt, countlist, thetE, thetD) < 0:
        thetE[bt]+=-0.025
    else:
        thetE[bt]+=0.025
    return(thetE)


def tweak_paramsD2(dval, idx, thetD):
    if dval - Dderivative2(idx,thetD) < 0:
        thetD[idx]+=-0.025
    else:
        thetD[idx]+=0.025
    return(thetD)

    
@jit
def greedsearch_paramDD(realvalsD,realvalsE,ordmat,ordlist,countlist,maxiter=300000):
    thetasD = np.zeros(len(realvalsD))
    thetasE = np.zeros(len(realvalsE))
    dlist=[]
    elist=[]
    for l in tqdm(range(maxiter)):
        for k in range(len(realvalsD)):
            thetasD = tweak_paramsD(realvalsD[k],k,ordlist,thetasE,thetasD)

        for z in range(len(realvalsE)):
            thetasE = tweak_paramsE(realvalsE[z],z,countlist,thetasE,thetasD)
    for k in range(len(realvalsD)):
        dlist.append(Dderivative(k, ordlist, thetasE, thetasD))
    for z in range(len(realvalsE)):
        elist.append(Ederivative(z,countlist,thetasE,thetasD))
    return(thetasD,thetasE,dlist,elist)

@jit
def greedsearch_paramDD_gen(realvalsD,realvalsE,ordmat,ordlist,countlist,maxiter=300000):
    thetasD = np.zeros(len(realvalsD))
    thetasE = np.zeros(len(realvalsE))
    dlist=[]
    elist=[]
    for l in tqdm(range(maxiter)):
        for k in range(countlist[0]):
            thetasD = tweak_paramsD(realvalsD[k],k,ordlist,thetasE,thetasD)

        for z in range(len(realvalsE)):
            thetasE = tweak_paramsE(realvalsE[z],z,countlist,thetasE,thetasD)
    for k in range(countlist[0]):
        dlist.append(Dderivative(k, ordlist, thetasE, thetasD))
    for z in range(len(realvalsE)):
        elist.append(Ederivative(z,countlist,thetasE,thetasD))
    return(thetasD[:countlist[0]],thetasE,dlist,elist)
            
def greedsearch_paramDD2(realvalsD,ordlist,countlist,maxiter=300000):
    thetasD = np.zeros(len(realvalsD))
    dlist=[]
    for l in tqdm(range(maxiter)):
        for k in range(len(realvalsD)):
            thetasD = tweak_paramsD2(realvalsD[k],k,thetasD)

    for k in range(len(realvalsD)):
        dlist.append(Dderivative2(k, thetasD))
    return(thetasD,dlist)

