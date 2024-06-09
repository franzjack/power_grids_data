# -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 17:04:19 2022

@author: franc
"""

import numpy as np
from copy import copy
from tqdm import tqdm
import pg_utils as sc
from numba import jit
from copy import deepcopy
import pg_utils as sc

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components



#This code can be used to sample from an ERGM ensemble
#constrained within the space of connected graphs.
#The function EEsparse should be used as an estimation method for the
#parameters of the model. The function pg_MHergm_conn is the effective
#Metropolis_Hastings sampler for constrained chains (see https://doi.org/10.48550/arXiv.1806.11276)



#-----utilities-------------


def dtype_ham(X,params):
    c=np.matmul(X,params)
    return(c)

def random_element(n):
    i = np.random.randint(n)
    j = np.random.randint(n)
    if i !=j:
        return(i,j)
    else:
        return(random_element(n))

def change_element2(newm,n):

    i,j = random_element(n)
    newm[i][j] = np.abs(newm[i][j]-1)
    newm[j][i] = newm[i][j]
    return(newm,newm[i][j],i,j)


def compute_p(prop,current):
    if prop-current>=0:
        p=1
    else:
        p = np.exp(prop-current)
    return(p)

def change_param(obs,robs,par,a,c):
    npar = copy(par)
    for i in range(len(npar)):
        change = a*np.max(np.array([np.abs(npar[i]),c]))*-np.sign(obs[i]-robs[i])
        npar[i] = npar[i] + change
   
    
    return(npar)


def generate_connected_adj(size):
    mat = np.zeros((size,size),dtype=bool)
    for i in range(size-1):
        mat[i][i+1] = 1
        mat[i+1][i] = 1
    mat[size-1][0] = 1
    mat[0][size-1] = 1
    return(mat)

def reorder_rows(a,permutation):
    idx = np.empty_like(permutation)
    idx[permutation] = np.arange(len(permutation))
    newa = deepcopy(a[idx,:])
    newaa = deepcopy(newa[:,idx])
    return(newaa)

#--------------Parameter estimation via constant parameter updating-------------------


@jit
def EEsparse(startmtx,observables,params,countlist,obs_comp,fast_obs,maxiter,alpha,c,n_step):
    n = len(startmtx)
   # bigcount = 10
    obs = obs_comp(startmtx,countlist[0],countlist[1],countlist[2])
    oldham = dtype_ham(obs,params=params)
    mtx=copy(startmtx)
    oblist = []
    count=0
    bigcount=1
    paramEE=[]
    ordlist = sc.ordered_buslist(countlist[0],countlist[1],countlist[2])
    for _ in tqdm(range(maxiter)):
        cond=1
        nmtx,move,i,j = change_element2(mtx,n)
        newobs = fast_obs(obs,nmtx,ordlist,move,i,j)
        
        newham = dtype_ham(newobs,params=params)
        
        p = compute_p(newham,oldham)
        if move==0:
            G_sparse = csr_matrix(nmtx)
            n_components = connected_components(csgraph=G_sparse, directed=False, return_labels=False)
            if n_components!=1:
                cond=0
        if (np.random.random()<p*cond) == True:
            mtx=nmtx
            oldham = copy(newham)
            obs = copy(newobs)
            oblist.append(obs)
            count+=1
            if count==n_step:
                count=0
                bigcount+=1
                params = change_param(obs,observables,params,alpha,c)
                paramEE.append(params)
                #print(params)
                
        else:
            nmtx[i][j] = nmtx[j][i] = np.abs(move-1)
                
        if bigcount%10==0:
            print(params)
            bigcount+=1
                
    EEparams=[]
    for j in range(len(observables)):
        EEparams.append(np.mean(np.array(paramEE[int(len(paramEE)/1.1):]).T[j]))
    obslist = np.array(oblist[int(len(oblist)/1.3):]).T
    print(len(obslist[0]))
    return(EEparams,obslist,np.array(paramEE).T)




#--------MH sampler-----------------------------




def pg_MHergm_conn(startmtx,observables,params,countlist,obs_comp,fast_obs,maxiter):
    n = len(startmtx)
   # bigcount = 10
    obs = obs_comp(startmtx,countlist[0],countlist[1],countlist[2])
    oldham = dtype_ham(obs,params=params)
    nmtx=copy(startmtx)
    synth=[]
    oblist = []
    move_count = 0
    ordlist = sc.ordered_buslist(countlist[0],countlist[1],countlist[2])
    for i in tqdm(range(maxiter)):
        cond=1
        nmtx,move,l,j = change_element2(nmtx,n)
        newobs = fast_obs(obs,nmtx,ordlist,move,l,j)
        newham = dtype_ham(newobs,params=params)
        p = compute_p(newham,oldham)
        if move==0:
            G_sparse = csr_matrix(nmtx)
            n_components = connected_components(csgraph=G_sparse, directed=False, return_labels=False)
            if n_components!=1:
                cond=0
        if (np.random.random()<p*cond) == True:
            mtx = copy(nmtx)
            oldham = copy(newham)
            obs = copy(newobs)
            move_count+=1
            synth.append(csr_matrix(mtx))
            oblist.append(obs)
        else:
            nmtx[l][j] = nmtx[j][l] = np.abs(move-1)
        
        i+=1
    print(len(synth))
    obslist = np.array(oblist[int(len(oblist)/1.3):]).T
    print(len(obslist[0]))
    mean_list = [np.mean(ob) for ob in obslist]
    return(mean_list,synth,obslist)






