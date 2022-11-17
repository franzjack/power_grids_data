# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 14:43:02 2022

@author: franc
"""

import numpy as np
from copy import copy
from numba import jit
import pg_betas_comp as bc



#--------utilities---------------


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
 
    
 
def linktype(a,b):
    if a+b<5:
        return (a*b - 1)
    else:
        return (a+b - 1)



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




#----------obs computation methods---------------



@jit
def obs_edg_tri_2tri(mtx,q1,q2,q3):
    e_gg = np.sum(mtx[0:q1,0:q1])/2
    e_ll = np.sum(mtx[q1:(q1+q2),q1:(q1+q2)])/2
    e_ii = np.sum(mtx[(q1+q2):,(q1+q2):])/2
    e_gl = np.sum(mtx[0:q1,q1:(q1+q2)])
    e_gi = np.sum(mtx[0:q1,(q1+q2):])
    e_li = np.sum(mtx[q1:(q1+q2),(q1+q2):])
    t,tk = compute_k_triangle(mtx)
    return(np.array([e_gg,e_gl,e_gi,e_ll,e_li,e_ii,t,tk]))



def obs_er_tri_2tri(mtx):
    e = np.sum(mtx)/2
    t,tk = compute_k_triangle(mtx)
    return(np.array([e,t,tk]))

def obs_edg(mtx,q1,q2,q3):
    e_gg = np.sum(mtx[0:q1,0:q1])/2
    e_ll = np.sum(mtx[q1:(q1+q2),q1:(q1+q2)])/2
    e_ii = np.sum(mtx[(q1+q2):,(q1+q2):])/2
    e_gl = np.sum(mtx[0:q1,q1:(q1+q2)])
    e_gi = np.sum(mtx[0:q1,(q1+q2):])
    e_li = np.sum(mtx[q1:(q1+q2),(q1+q2):])
    return(np.array([e_gg,e_gl,e_gi,e_ll,e_li,e_ii]))    
 
@jit(nopython=True)                   
def obs_avgdeg(mtx,q1,q2,q3):
    d1 = np.sum(mtx[0:q1][0:len(mtx)])/q1
    d2 = np.sum(mtx[q1:q1+q2][0:len(mtx)])/q2
    d3 = np.sum(mtx[q1+q2:len(mtx)][0:len(mtx)])/q3
    return(d1,d2,d3)



@jit
def obs_edg_ddeg(mtx,q1,q2,q3):
    obsD = deg_distr(mtx)
    obsE = obs_edg(mtx,q1,q2,q3)
    obs = np.concatenate((obsD,obsE))
    return(obs)

@jit
def obs_edg_dgen(mtx,q1,q2,q3):
    obsD = deg_distr(mtx)
    obsE = obs_edg(mtx,q1,q2,q3)
    obs = np.concatenate((obsD[:q1],obsE))
    return(obs)
 
@jit
def avg_degreetype(mx,bustypes):
    dgen = 0
    dload = 0
    dint = 0
    count_gen = 0
    count_load = 0
    count_int = 0
    for i in range(len(mx)):
        if bustypes[i] == 1:
            dgen += np.sum(mx[i])
            count_gen +=1
        elif bustypes[i] == 2:
            dload += np.sum(mx[i])
            count_load +=1
        else:
            dint += np.sum(mx[i])
            count_int +=1
    #print(count_gen,count_load,count_int)
    if count_gen==0:
        dgen=0
        count_gen=1
    if count_load==0:
        dload=0
        count_load=1
    if count_int==0:
        dint=0
        count_int=1
    return(dgen/count_gen, dload/count_load, dint/count_int,count_gen,count_load,count_int)




#-------------fast_obs methods used to compute the changes in the observables
#-------------after one step of the MC chain

def fast_obsDD(obs,move,i,j):
    dmove = 2*move -1
    obs[i] += dmove
    obs[j] += dmove
    return(obs)


@jit
def fast_obs_er_tri_2tri(past_obs,mtx,move,i,j):
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
    return(newobs)

@jit
def fast_obs_edg(past_obs,mtx,ordlist,move,i,j):
    newobs = copy(past_obs)
    if move == 1:
        addtype = linktype(ordlist[i],ordlist[j])
        newobs[addtype] +=1      
    else:
        addtype = linktype(ordlist[i],ordlist[j])
        newobs[addtype] -=1
    return(newobs)


@jit
def fast_obs_edg_tri_2tri(past_obs,mtx,ordlist,move,i,j):
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
    return(newobs)

@jit
def alt_fast_obs_edg_tri_2tri(past_obs,mtx,ordlist,move,i,j):
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
    return(newobs)

@jit
def fast_obs_ddeg(past_obs,mtx,ordlist,move,i,j):
    newobs = deg_distr(mtx)
    return(newobs)

@jit
def fast_obs_edg_ddeg(past_obs,mtx,ordlist,move,i,j):
    newobsE = fast_obs_edg(past_obs[-6:],mtx,ordlist,move,i,j)
    newobsD = deg_distr(mtx)
    newobs = np.concatenate((newobsD,newobsE))
    return(newobs)

@jit
def fast_obs_edg_dgen(past_obs,mtx,ordlist,move,i,j):
    newobsE = fast_obs_edg(past_obs[-6:],mtx,ordlist,move,i,j)
    newobsD = deg_distr(mtx)[:np.bincount(ordlist)[1]]
    newobs = np.concatenate((newobsD,newobsE))
    return(newobs)

def comp_obs_and_betas(modtype,ordmat,ordlist,countlist,maxiter=300000,startguess=np.array([1,-0.2])):
    q1,q2,q3 = countlist[0],countlist[1],countlist[2]
    if modtype == '_edg_tri_2tri':
        obs = obs_edg_tri_2tri(ordmat,q1,q2,q3)
        kbetas=bc.greedsearch_param2(obs[:6],q1,q2,q3,maxiter=maxiter)
        betas = np.append(kbetas*1.1,startguess)
        return(obs,betas)
    if modtype == '_edg_ddeg':
        kedgobs_T = obs_edg(ordmat,q1,q2,q3)
        realvals= deg_distr(ordmat)
        obs=np.concatenate((realvals,kedgobs_T))
        thetasD,thetasE,dlist,elist=bc.greedsearch_paramDD(realvals,kedgobs_T,ordmat,ordlist,countlist,maxiter=300000)
        betas = np.concatenate((thetasD,thetasE))
        return(obs,betas)
    if modtype == '_edg':
        obs = obs_edg(ordmat,q1,q2,q3)
        betas = bc.greedsearch_param2(obs,q1,q2,q3,maxiter=maxiter)
        return(obs,betas)
    if modtype == '_edg_dgen':
        kedgobs_T = obs_edg(ordmat,q1,q2,q3)
        realvals= deg_distr(ordmat)
        thetasD,thetasE,dlist,elist=bc.greedsearch_paramDD_gen(realvals,kedgobs_T,ordmat,ordlist,countlist,maxiter=300000)
        betas = np.concatenate((thetasD,thetasE))
        obs=np.concatenate((realvals[:q1],kedgobs_T))
        return(obs,betas)