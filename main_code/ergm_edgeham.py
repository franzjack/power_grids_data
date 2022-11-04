# -*- coding: utf-8 -*-
"""
Created on Fri May 13 10:56:56 2022

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


#code for the estimation and simulation of exponential random graph models for power grids

#-------utilities functions-----------------



def byn_coef(N):
    if N>=2:
        return(fac(N)/(fac(N-2)*2)) 
    else:
        return 0

def rearrange_rows(mtx,permutation):
    idx = np.empty_like(permutation)
    idx[permutation] = np.arange(len(permutation))
    newm = copy(mtx[idx,:])
    return(newm)
    
@jit(nopython=True)
def sigmoid(th):
    return(np.exp(th)/(1+np.exp(th)))

#----------observables counters-----------------

@jit(nopython=True)
def compute_k_triangle(mtx):
    tricount=0
    tricount2 = 0
    for i in range(len(mtx)):
        for j in range(len(mtx)):
            if mtx[i][j] ==1:
                localtricount = 0
                for k in range(len(mtx)):
                    if mtx[i][k] == 1:
                        localtricount += mtx[k][j]
                if localtricount >=2:
                    tricount2 += localtricount*(localtricount-1)/2
                tricount+=localtricount
    return(int(tricount/6),int(tricount2/2))



@jit(nopython=True)
def compute_2_triangle(mtx):
    tricount=0
    tricount2 = 0
    for i in range(len(mtx)):
        for j in range(len(mtx)):
            if mtx[i][j] ==1:
                localtricount = 0
                for k in range(len(mtx)):
                    if mtx[i][k] == 1:
                        localtricount += mtx[k][j]
                if localtricount >=2:
                    tricount2 += localtricount*(localtricount-1)/2
                tricount+=localtricount
    return(int(tricount2/2))
 
    
@jit(nopython=True)                   
def avg_degreetype(mtx,q1,q2,q3):
    d1 = np.sum(mtx[0:q1][0:len(mtx)])/q1
    d2 = np.sum(mtx[q1:q1+q2][0:len(mtx)])/q2
    d3 = np.sum(mtx[q1+q2:len(mtx)][0:len(mtx)])/q3
    return(d1,d2,d3)

def fast_obsDD(obs,move,i,j):
    dmove = 2*move -1
    obs[i] += dmove
    obs[j] += dmove
    return(obs)

@jit
def deg_distr(mtx):
    newobsD = np.array([np.sum(mtx[k]) for k in range(len(mtx))])
    return(newobsD)



@jit(nopython=True)
def check_existing_triangle(i,k,mtx):
    count=0
    for l in range(len(mtx)):
        if mtx[k][l] == 1:
            count+=mtx[l][i]
    return(count)
                            
            
@jit(nopython=True) 
def change_triang3(mtx,i,j):
    count=0
    kcount=0
    lcount=0
    rcount=0
    for k in range(len(mtx)):
        if mtx[i][k] ==1:
            if mtx[k][j] == 1:
                count+=mtx[k][j]
                lcount += check_existing_triangle(i,k,mtx)-mtx[i][j]
                rcount += check_existing_triangle(j,k,mtx)-mtx[i][j]

    if count>=2:
        kcount=count*(count-1)/2
    return(count,(kcount+lcount+rcount))

def fast_obs_simplet(past_obs,mtx,move,i,j):
    newobs = copy(past_obs)
    t,kcount = change_triang3(mtx,i,j)
    if move == 1:
        newobs[0]+=1
        newobs[1] +=  t
        k = copy(newobs[2] + kcount)
        newobs[2] = copy(k)        
    else:
        newobs[0] -=1
        newobs[1] -=  t
        k = copy(newobs[2] - kcount)
        newobs[2] = copy(k)  
    return(newobs,kcount)

@jit
def new_fast_obs_e(past_obs,mtx,ordlist,move,i,j):
    newobs = copy(past_obs)
    if move == 1:
        addtype = linktype(ordlist[i],ordlist[j])
        newobs[addtype] +=1      
    else:
        addtype = linktype(ordlist[i],ordlist[j])
        newobs[addtype] -=1
    return(newobs)


@jit
def knew_fast_obs_e(past_obs,mtx,ordlist,move,i,j):
    newobs = copy(past_obs)
    t,kcount = change_triang3(mtx,i,j)
    kcount=kcount
    if move == 1:
        addtype = linktype(ordlist[i],ordlist[j])
        newobs[addtype] +=1
        newobs[6] +=  t
        k = copy(newobs[7] + kcount)
        newobs[7] = copy(k)        
    else:
        addtype = linktype(ordlist[i],ordlist[j])
        newobs[addtype] -=1
        newobs[6] -=  t
        k = copy(newobs[7] - kcount)
        newobs[7] = copy(k)  
    return(newobs,kcount)

@jit
def alt_knew_fast_obs_e(past_obs,mtx,ordlist,move,i,j):
    newobs = copy(past_obs)
    t,kcount = change_triang3(mtx,i,j)
    k=copy(3*t - kcount)  
    if move == 1:
        addtype = linktype(ordlist[i],ordlist[j])
        newobs[addtype] +=1
        newobs[6] +=  k  
    else:
        addtype = linktype(ordlist[i],ordlist[j])
        newobs[addtype] -=1
        newobs[6] -=  k
    return(newobs,kcount)






#-----------derivatives and parameter estimation--------------------


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
            
def greedsearch_paramDD2(realvalsD,ordlist,countlist,maxiter=300000):
    thetasD = np.zeros(len(realvalsD))
    dlist=[]
    for l in tqdm(range(maxiter)):
        for k in range(len(realvalsD)):
            thetasD = tweak_paramsD2(realvalsD[k],k,thetasD)

    for k in range(len(realvalsD)):
        dlist.append(Dderivative2(k, thetasD))
    return(thetasD,dlist)



#-----------MCMC samplers----------------------


def compute_p(prop,current):
    if prop-current>=0:
        p=1
    else:
        p = np.exp(prop-current)
    return(p)

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

@jit(nopython=True)
def check_isolated(mtx,i,j):
    cond = np.sum(mtx[i])*np.sum(mtx[j])
    if cond == 0:
        return 0
    else:
        return 1



            
@jit            
def ERG_DD(startmtx,observables,params,countlist,maxiter):
    n = len(startmtx)
   # bigcount = 10
    obsE = new_avg_edge(startmtx,countlist[0],countlist[1],countlist[2])
    obsD = [np.sum(startmtx[j]) for j in range(len(startmtx))]
    obs = np.concatenate((obsD,obsE))
    oldham = dtype_ham(obs,params=params)
    mtx=copy(startmtx)
    oblist = []
    count=0
    paramEE=[]
    synth=[]
    ordlist = sc.ordered_buslist(countlist[0],countlist[1],countlist[2])
    for _ in tqdm(range(maxiter)):
        nmtx,move,i,j = change_element2(mtx,n)
        newobsE = new_fast_obs_e(obs[-6:],nmtx,ordlist,move,i,j)
        newobsD = deg_distr(nmtx)
        newobs = np.concatenate((newobsD,newobsE))
        newham = dtype_ham(newobs,params=params)
        p = compute_p(newham,oldham)
        if (np.random.random()<p) == True:
            mtx=nmtx
            oldham = copy(newham)
            obs = copy(newobs)
            oblist.append(obs)
            count+=1
            paramEE.append(params)
            synth.append(csr_matrix(mtx))
        else:
            nmtx[i][j] = nmtx[j][i] = np.abs(move-1)
    obslist = np.array(oblist[int(len(oblist)/1.3):]).T
    print(len(obslist[0]))
    return(synth,obslist,np.array(paramEE).T)               

def ERG_DD_con(startmtx,observables,params,countlist,maxiter):
    n = len(startmtx)
   # bigcount = 10
    obsE = new_avg_edge(startmtx,countlist[0],countlist[1],countlist[2])
    obsD = [np.sum(startmtx[j]) for j in range(len(startmtx))]
    obs = np.concatenate((obsD,obsE))
    oldham = dtype_ham(obs,params=params)
    mtx=copy(startmtx)
    oblist = []
    count=0
    paramEE=[]
    synth=[]
    ordlist = sc.ordered_buslist(countlist[0],countlist[1],countlist[2])
    for _ in tqdm(range(maxiter)):
        nmtx,move,i,j = change_element2(mtx,n)
        newobsE = new_fast_obs_e(obs[-6:],nmtx,ordlist,move,i,j)
        newobsD = deg_distr(nmtx)
        newobs = np.concatenate((newobsD,newobsE))
        newham = dtype_ham(newobs,params=params)
        p = compute_p(newham,oldham)
        cond=1
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
            paramEE.append(params)
            synth.append(csr_matrix(mtx))
        else:
            nmtx[i][j] = nmtx[j][i] = np.abs(move-1)
    obslist = np.array(oblist[int(len(oblist)/1.3):]).T
    print(len(obslist[0]))
    return(synth,obslist,np.array(paramEE).T)    


          
            
            
@jit            
def ERG_DD2(startmtx,observables,params,countlist,maxiter):
    n = len(startmtx)
   # bigcount = 10
    obs = np.array(deg_distr(startmtx))
    oldham = dtype_ham(obs,params=params)
    mtx=copy(startmtx)
    oblist = []
    count=0
    synth=[]
    for _ in tqdm(range(maxiter)):
        nmtx,move,i,j = change_element2(mtx,n)
        newobs = deg_distr(nmtx)
        newham = dtype_ham(newobs,params=params)
        p = compute_p(newham,oldham)
        if (np.random.random()<p) == True:
            mtx=nmtx
            oldham = copy(newham)
            #print(oldham)
            obs = copy(newobs)
            oblist.append(obs)
            count+=1

            synth.append(csr_matrix(mtx))
        else:
            nmtx[i][j] = nmtx[j][i] = np.abs(move-1)
    obslist = np.array(oblist[int(len(oblist)/1.3):]).T
    print(len(obslist[0]))
    return(synth,obslist)  
            
            
            
            

def dtype_ham(X,params):
    c=np.matmul(X,params)
    return(c)

@jit(nopython=True)
def linktype(a,b):
    if a+b<5:
        return (a*b - 1)
    else:
        return (a+b - 1)


@jit
def knew_avg_edget(mtx,q1,q2,q3):
    e_gg = np.sum(mtx[0:q1,0:q1])/2
    e_ll = np.sum(mtx[q1:(q1+q2),q1:(q1+q2)])/2
    e_ii = np.sum(mtx[(q1+q2):,(q1+q2):])/2
    e_gl = np.sum(mtx[0:q1,q1:(q1+q2)])
    e_gi = np.sum(mtx[0:q1,(q1+q2):])
    e_li = np.sum(mtx[q1:(q1+q2),(q1+q2):])
    t,tk = compute_k_triangle(mtx)
    return(np.array([e_gg,e_gl,e_gi,e_ll,e_li,e_ii,t,tk]))

@jit
def alt_knew_avg_edget(mtx,q1,q2,q3):
    e_gg = np.sum(mtx[0:q1,0:q1])/2
    e_ll = np.sum(mtx[q1:(q1+q2),q1:(q1+q2)])/2
    e_ii = np.sum(mtx[(q1+q2):,(q1+q2):])/2
    e_gl = np.sum(mtx[0:q1,q1:(q1+q2)])
    e_gi = np.sum(mtx[0:q1,(q1+q2):])
    e_li = np.sum(mtx[q1:(q1+q2),(q1+q2):])
    t,tk = compute_k_triangle(mtx)
    ak = 3*t - tk
    return(np.array([e_gg,e_gl,e_gi,e_ll,e_li,e_ii,ak]))

def obs_simplet(mtx):
    e = np.sum(mtx)/2
    t,tk = compute_k_triangle(mtx)
    return(np.array([e,t,tk]))

def new_avg_edge(mtx,q1,q2,q3):
    e_gg = np.sum(mtx[0:q1,0:q1])/2
    e_ll = np.sum(mtx[q1:(q1+q2),q1:(q1+q2)])/2
    e_ii = np.sum(mtx[(q1+q2):,(q1+q2):])/2
    e_gl = np.sum(mtx[0:q1,q1:(q1+q2)])
    e_gi = np.sum(mtx[0:q1,(q1+q2):])
    e_li = np.sum(mtx[q1:(q1+q2),(q1+q2):])
    return(np.array([e_gg,e_gl,e_gi,e_ll,e_li,e_ii]))



def MHergmPG_edges2(startmtx,observables,countlist,params,maxiter=2500):
    n = len(startmtx)
    obs = new_avg_edge(startmtx,countlist[0],countlist[1],countlist[2])
    oldham = dtype_ham(obs,params=params)
    mtx=copy(startmtx)
    synth=[]
    oblist=[]
    for i in tqdm(range(maxiter)):
        nmtx,move,i,j = change_element2(mtx,n)           
        newobs = new_avg_edge(nmtx,countlist[0],countlist[1],countlist[2])
        newham = dtype_ham(newobs,params=params)
        p = compute_p(newham,oldham)
        if (np.random.random()<p) == True:
            mtx=nmtx
            oldham = copy(newham)
            obs = copy(newobs)
            synth.append(mtx)
            oblist.append(obs)
        i+=1
    print(len(synth))
    obslist = np.array(oblist[int(len(oblist)/1.3):]).T
    print(len(obslist[0]))
    mean_list = [np.mean(ob) for ob in obslist]
    return(mean_list,synth,obslist)

def MHergmPG_simplet(startmtx,observables,params,maxiter):
    n = len(startmtx)
   # bigcount = 10
    obs = obs_simplet(startmtx)
    oldham = dtype_ham(obs,params=params)
    mtx=copy(startmtx)
    synth=[]
    oblist = []
    for i in tqdm(range(maxiter)):
        nmtx,move,i,j = change_element2(mtx,n)
        newobs,k = fast_obs_simplet(obs,nmtx,move,i,j)
        newham = dtype_ham(newobs,params=params)
        p = compute_p(newham,oldham)
        if (np.random.random()<p) == True:
            mtx=nmtx
            oldham = copy(newham)
            obs = copy(newobs)
            synth.append(mtx)
            oblist.append(obs)
        i+=1
    print(len(synth))
    obslist = np.array(oblist[int(len(oblist)/1.3):]).T
    print(len(obslist[0]))
    mean_list = [np.mean(ob) for ob in obslist]
    return(mean_list,synth,obslist)

def MHergmPG_simplet2(startmtx,params,maxiter):
    n = len(startmtx)
   # bigcount = 10
    obs = obs_simplet(startmtx)
    oldham = dtype_ham(obs,params=params)
    mtx=copy(startmtx)
    synth=[]
    oblist = []
    for i in range(maxiter):
        nmtx,move,i,j = change_element2(mtx,n)
        newobs,k = fast_obs_simplet(obs,nmtx,move,i,j)
        newham = dtype_ham(newobs,params=params)
        p = compute_p(newham,oldham)
        if (np.random.random()<p) == True:
            mtx=nmtx
            oldham = copy(newham)
            obs = copy(newobs)
            synth.append(mtx)
            oblist.append(obs)
        i+=1
    print(len(synth))
    obslist = np.array(oblist[int(len(oblist)/1.3):]).T
    print(len(obslist[0]))
    mean_list = [np.mean(ob) for ob in obslist]
    return(mean_list)

        





def MHergmPG_triang_meanK(startmtx,observables,countlist,params,maxiter):
    n = len(startmtx)
   # bigcount = 10
    obs = knew_avg_edget(startmtx,countlist[0],countlist[1],countlist[2])
    oldham = dtype_ham(obs,params=params)
    mtx=copy(startmtx)
    synth=[]
    oblist = []
    klist=[]
    move_count = 0
    ordlist = sc.ordered_buslist(countlist[0],countlist[1],countlist[2])
    for i in tqdm(range(maxiter)):
        nmtx,move,l,j = change_element2(mtx,n)
        newobs,kcount = knew_fast_obs_e(obs,nmtx,ordlist,move,l,j)
        newham = dtype_ham(newobs,params=params)
        p = compute_p(newham,oldham)
        if (np.random.random()<p) == True:
            mtx=nmtx
            oldham = copy(newham)
            obs = copy(newobs)
            move_count+=1
            synth.append(mtx)
            oblist.append(obs)
            klist.append((2*move-1)*kcount)
        
        i+=1
    print(len(synth))
    obslist = np.array(oblist[int(len(oblist)/1.3):]).T
    print(len(obslist[0]))
    mean_list = [np.mean(ob) for ob in obslist]
    return(mean_list,synth,obslist,klist)

def MHergmPG_triang_meanK_con(startmtx,observables,countlist,params,maxiter):
    n = len(startmtx)
   # bigcount = 10
    obs = knew_avg_edget(startmtx,countlist[0],countlist[1],countlist[2])
    oldham = dtype_ham(obs,params=params)
    nmtx=copy(startmtx)
    synth=[]
    oblist = []
    klist=[]
    move_count = 0
    ordlist = sc.ordered_buslist(countlist[0],countlist[1],countlist[2])
    for i in tqdm(range(maxiter)):
        cond=1
        nmtx,move,l,j = change_element2(nmtx,n)
        newobs,kcount = knew_fast_obs_e(obs,nmtx,ordlist,move,l,j)
        newham = dtype_ham(newobs,params=params)
        p = compute_p(newham,oldham)
        if move==0:
            cond = check_isolated(nmtx,l,j)
        if (np.random.random()<p*cond) == True:
            oldham = copy(newham)
            obs = copy(newobs)
            move_count+=1
            synth.append(nmtx)
            oblist.append(obs)
            klist.append((2*move-1)*kcount)
        else:
            nmtx[l][j] = nmtx[j][l] = np.abs(move-1)
        
        i+=1
    print(len(synth))
    obslist = np.array(oblist[int(len(oblist)/1.3):]).T
    print(len(obslist[0]))
    mean_list = [np.mean(ob) for ob in obslist]
    return(mean_list,synth,obslist,klist)


def MHergmPG_sparse_meanK_con(startmtx,observables,countlist,params,maxiter):
    n = len(startmtx)
   # bigcount = 10
    obs = knew_avg_edget(startmtx,countlist[0],countlist[1],countlist[2])
    oldham = dtype_ham(obs,params=params)
    nmtx=copy(startmtx)
    synth=[]
    oblist = []
    klist=[]
    move_count = 0
    ordlist = sc.ordered_buslist(countlist[0],countlist[1],countlist[2])
    for i in tqdm(range(maxiter)):
        cond=1
        nmtx,move,l,j = change_element2(nmtx,n)
        newobs,kcount = knew_fast_obs_e(obs,nmtx,ordlist,move,l,j)
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
            klist.append((2*move-1)*kcount)
        else:
            nmtx[l][j] = nmtx[j][l] = np.abs(move-1)
        
        i+=1
    print(len(synth))
    obslist = np.array(oblist[int(len(oblist)/1.3):]).T
    print(len(obslist[0]))
    mean_list = [np.mean(ob) for ob in obslist]
    return(mean_list,synth,obslist,klist)

def newedgecount(beta,n):
    eb = np.exp(beta)
    n_edges=eb*byn_coef(n)/(1-eb)
    return(n_edges)

def change_param(obs,robs,par,a,c):
    npar = copy(par)
    for i in range(len(par)):
        npar[i] = npar[i] + a*np.max(np.array([np.abs(npar[i]),c]))*-np.sign(obs[i]-robs[i])
    
    return(npar)
  
def generator(itercount,maxiter):
  while (itercount<maxiter) :
    yield    


def EE(startmtx,observables,params,maxiter,alpha,c,n_step):
    n = len(startmtx)
   # bigcount = 10
    obs = obs_simplet(startmtx)
    oldham = dtype_ham(obs,params=params)
    mtx=copy(startmtx)
    oblist = []
    count=0
    bigcount=1
    paramEE=[]
    j=0
    for i in tqdm(range(maxiter)):
        nmtx,move,i,j = change_element2(mtx,n)
        newobs,k = fast_obs_simplet(obs,nmtx,move,i,j)
        newham = dtype_ham(newobs,params=params)
        p = compute_p(newham,oldham)
        j+=1
        if (np.random.random()<p) == True:
            mtx=nmtx
            oldham = copy(newham)
            obs = copy(newobs)
            oblist.append(obs)
            count+=1
            if count==n_step:
                count=0
                bigcount+=1
                params = copy(change_param(obs,observables,params,alpha,c))
                paramEE.append(params)
                
        if bigcount%10==0:
            print(params)
            bigcount+=1
                
        i+=1
    EEparams=[]
    for j in range(len(observables)):
        EEparams.append(np.mean(np.array(paramEE[int(len(paramEE)/2):]).T[j]))
    obslist = np.array(oblist[int(len(oblist)/1.3):]).T
    print(len(obslist[0]))
    return(EEparams,obslist)


def EEfull(startmtx,observables,params,countlist,maxiter,alpha,c,n_step):
    n = len(startmtx)
   # bigcount = 10
    obs = knew_avg_edget(startmtx,countlist[0],countlist[1],countlist[2])
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
        newobs,k = knew_fast_obs_e(obs,nmtx,ordlist,move,i,j)
        newham = dtype_ham(newobs,params=params)
        p = compute_p(newham,oldham)
        if move==0:
            cond = check_isolated(nmtx,i,j)
        if (np.random.random()<p*cond) == True:
            mtx=nmtx
            oldham = copy(newham)
            obs = copy(newobs)
            oblist.append(obs)
            count+=1
            if count==n_step:
                count=0
                bigcount+=1
                params = copy(change_param(obs,observables,params,alpha,c))
                paramEE.append(params)
        else:
            nmtx[i][j] = nmtx[j][i] = np.abs(move-1)
                
        if bigcount%100==0:
            print(params)
            print(obs[6])
            bigcount+=1
                
    EEparams=[]
    for j in range(len(observables)):
        EEparams.append(np.mean(np.array(paramEE[int(len(paramEE)/2):]).T[j]))
    obslist = np.array(oblist[int(len(oblist)/1.3):]).T
    print(len(obslist[0]))
    return(EEparams,obslist,np.array(paramEE).T)


def EEsparse(startmtx,observables,params,countlist,maxiter,alpha,c,n_step):
    n = len(startmtx)
   # bigcount = 10
    obs = knew_avg_edget(startmtx,countlist[0],countlist[1],countlist[2])
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
        newobs,k = knew_fast_obs_e(obs,nmtx,ordlist,move,i,j)
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
                params = copy(change_param(obs,observables,params,alpha,c))
                paramEE.append(params)
        else:
            nmtx[i][j] = nmtx[j][i] = np.abs(move-1)
                
        if bigcount%100==0:
            print(params)
            print(obs[6])
            bigcount+=1
                
    EEparams=[]
    for j in range(len(observables)):
        EEparams.append(np.mean(np.array(paramEE[int(len(paramEE)/1.1):]).T[j]))
    obslist = np.array(oblist[int(len(oblist)/1.3):]).T
    print(len(obslist[0]))
    return(EEparams,obslist,np.array(paramEE).T)



def EEsparseDD(startmtx,observables,params,countlist,ordlist,maxiter,alpha,c,n_step):
    n = len(startmtx)
   # bigcount = 10
    obsE = new_avg_edge(startmtx,countlist[0],countlist[1],countlist[2])
    obsD = [np.sum(startmtx[j]) for j in range(len(startmtx))]
    obs = np.concatenate((obsD,obsE))
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
        newobsE = new_fast_obs_e(obs[-6:],nmtx,ordlist,move,i,j)
        newobsD = deg_distr(nmtx)
        newobs = np.concatenate((newobsD,newobsE))
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
                params = copy(change_param(obs,observables,params,alpha,c))
                paramEE.append(params)
        else:
            nmtx[i][j] = nmtx[j][i] = np.abs(move-1)
                
        if bigcount%100==0:
            print(params)
            print(obs[6])
            bigcount+=1
                
    EEparams=[]
    for k in range(len(observables)):
        EEparams.append(np.mean(np.array(paramEE[int(len(paramEE)/1.1):]).T[k]))
    obslist = np.array(oblist[int(len(oblist)/1.3):]).T
    print(len(obslist[0]))
    return(EEparams,obslist,np.array(paramEE).T)





def EEsparse_simple(startmtx,observables,params,countlist,maxiter,alpha,c,n_step):
    n = len(startmtx)
   # bigcount = 10
    obs = new_avg_edge(startmtx,countlist[0],countlist[1],countlist[2])
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
        newobs = new_fast_obs_e(obs,nmtx,ordlist,move,i,j)
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
                params = copy(change_param(obs,observables,params,alpha,c))
                paramEE.append(params)
        else:
            nmtx[i][j] = nmtx[j][i] = np.abs(move-1)
                
        if bigcount%100==0:
            print(params)
            bigcount+=1
                
    EEparams=[]
    for j in range(len(observables)):
        EEparams.append(np.mean(np.array(paramEE[int(len(paramEE)/1.1):]).T[j]))
    obslist = np.array(oblist[int(len(oblist)/1.3):]).T
    print(len(obslist[0]))
    return(EEparams,obslist,np.array(paramEE).T)




          
def alt_EEsparse(startmtx,observables,params,countlist,maxiter,alpha,c,n_step):
    n = len(startmtx)
   # bigcount = 10
    obs = alt_knew_avg_edget(startmtx,countlist[0],countlist[1],countlist[2])
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
        newobs= alt_knew_fast_obs_e(obs,nmtx,ordlist,move,i,j)
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
                params = copy(change_param(obs,observables,params,alpha,c))
                paramEE.append(params)
        else:
            nmtx[i][j] = nmtx[j][i] = np.abs(move-1)
                
        if bigcount%100==0:
            print(params)
            print(obs[6])
            bigcount+=1
                
    EEparams=[]
    for j in range(len(observables)):
        EEparams.append(np.mean(np.array(paramEE[int(len(paramEE)/1.2):]).T[j]))
    obslist = np.array(oblist[int(len(oblist)/1.3):]).T
    print(len(obslist[0]))
    return(EEparams,obslist,np.array(paramEE).T)

def ERG_simple(startmtx,observables,params,countlist,maxiter):
    n = len(startmtx)
   # bigcount = 10
    obs = new_avg_edge(startmtx,countlist[0],countlist[1],countlist[2])
    oldham = dtype_ham(obs,params=params)
    mtx=copy(startmtx)
    oblist = []
    count=0
    paramEE=[]
    synth=[]
    ordlist = sc.ordered_buslist(countlist[0],countlist[1],countlist[2])
    for _ in tqdm(range(maxiter)):
        nmtx,move,i,j = change_element2(mtx,n)
        newobs = new_fast_obs_e(obs,nmtx,ordlist,move,i,j)
        newham = dtype_ham(newobs,params=params)
        p = compute_p(newham,oldham)
        if (np.random.random()<p) == True:
            mtx=nmtx
            oldham = copy(newham)
            obs = copy(newobs)
            oblist.append(obs)
            count+=1
            paramEE.append(params)
            synth.append(csr_matrix(mtx))
        else:
            nmtx[i][j] = nmtx[j][i] = np.abs(move-1)
    obslist = np.array(oblist[int(len(oblist)/1.3):]).T
    print(len(obslist[0]))
    return(synth,obslist,np.array(paramEE).T)          
            
    

