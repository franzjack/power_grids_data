# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 13:12:47 2022

@author: franc
"""

import numpy as np
import networkx as nx
from copy import deepcopy
from math import factorial as fac

# --------------------------------Utils------------------------------------

def byn_coef(N):
    return(fac(N)/(fac(N-2)*2))

def compute_p(prop,current):
    if prop-current>=0:
        p=1
    else:
        p = np.exp(prop-current)
    return(p)

def generate_orderedlist(n):
    orderedlist=[]
    for i in range(n):
        orderedlist.append(i+1)
    return(orderedlist)

def random_element(n):
    i = np.random.randint(n)
    j = np.random.randint(n)
    if i !=j:
        return(i,j)
    else:
        return(random_element(n))
    

def change_element(mtx,n):
    i,j = random_element(n)
    newm = deepcopy(mtx)
    newm[i][j] = np.abs(mtx[i][j]-1)
    newm[j][i] = newm[i][j]
    return(newm)

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
    return(dgen/count_gen, dload/count_load, dint/count_int)


# --------------------------------Test Functions------------------------------------

def testobs(mx):
    G = nx.from_numpy_matrix(mx)
    my_degrees = G.degree()
    degree_values = [v for k, v in my_degrees]
    sum_of_edges = sum(degree_values)
    avg_degree = sum_of_edges/len(mx)
    return(avg_degree)



def testobs2(mx):
    return((np.sum(mx))/2)

def testham(obs,trueobs=41,n=30):
    theta = np.log(trueobs/(fac(n)/(fac(n-2)*2)-trueobs))
    return(theta*obs)


def dtype_ham(X,params):
    return((X@params))


# --------------------------------Find Optimal Parameters------------------------------------

def sigmoid(th):
    return(np.exp(th)/(1+np.exp(th)))

def freederivative(theta,N_der,N1,N2,theta1,theta2):
    dlnZ = (N_der-1)*sigmoid(2*theta/N_der) + N1*sigmoid(theta/N_der+theta1/N1) + N2*sigmoid(theta/N_der+theta2/N2)
    return(dlnZ)

def freederivative2(theta,N_der,N1,N2,theta1,theta2):
    dlnZ = N_der*sigmoid(theta) + N1*sigmoid(theta + theta1) + N2*sigmoid(theta + theta2)
    return(dlnZ)



def tweak_params(exval,theta,N_der,N1,N2,theta1,theta2):
    if exval - freederivative(theta,N_der,N1,N2,theta1,theta2) < 0:
        theta+=-0.005
    else:
        theta+=0.005
    return(theta)


def check_condition(K_G,K_L,K_I, thetaG,thetaL,thetaI,NG,NL,NI):
    xG = freederivative2(thetaG,NG,NL,NI,thetaL,thetaI)
    xL = freederivative2(thetaL,NL,NG,NI,thetaG,thetaI)
    xI = freederivative2(thetaI,NI,NL,NG,thetaL,thetaG)
    prop = np.array([xG,xL,xI])
    real = np.array([K_G,K_L,K_I])
    cond = np.allclose(prop,real,atol=0.05)
    return(cond)

"""

Scratch formulation if one wants to use the native scipy solver

def func(x):
    return [2*byn_coef(6)/(1+np.exp(2*x[0]))+132/(1+np.exp(x[0]+x[1]))+12/(1+np.exp(x[0]+x[2]))-2,
            2*byn_coef(22)/(1+np.exp(2*x[1]))+132/(1+np.exp(x[0]+x[1]))+44/(1+np.exp(x[1]+x[2]))-2.772,
            2*byn_coef(2)/(1+np.exp(2*x[2]))+12/(1+np.exp(x[0]+x[2]))+44/(1+np.exp(x[1]+x[2]))-4.5]



def func2(x):
    return [5*sigmoid(2*x[0]/6)+22*sigmoid(x[0]/6+x[1]/22)+2*sigmoid(x[0]/6+x[2]/2)-2,
            21**sigmoid(2*x[1]/22)+6*sigmoid(x[0]/6+x[1]/22)+2*sigmoid(x[1]/22+x[2]/2)-2.772,
            1*sigmoid(2*x[2]/2)+6*sigmoid(x[0]/6+x[2]/2)+22*sigmoid(x[1]/22+x[2]/2)-4.5]


 """       
    



def greedsearch_param(K_G,K_L,K_I,NG,NL,NI,maxiter=50000):
    thetaL=0.0
    thetaG=0.0
    thetaI=0.0
    condition=False
    for i in range(maxiter):
        thetaG = tweak_params(K_G, thetaG,NG,NL,NI,thetaL,thetaI)
        condition = check_condition(K_G,K_L,K_I, thetaG,thetaL,thetaI,NG,NL,NI)
        if condition == True:
            print('convergence')
            break
        thetaL = tweak_params(K_L, thetaL,NL,NG,NI,thetaG,thetaI)
        condition = check_condition(K_G,K_L,K_I, thetaG,thetaL,thetaI,NG,NL,NI)
        if condition == True:
            print('convergence')
            break
        thetaI = tweak_params(K_I, thetaI,NI,NG,NL,thetaG,thetaL)
        condition = check_condition(K_G,K_L,K_I, thetaG,thetaL,thetaI,NG,NL,NI)
        if condition == True:
            print('convergence')
            break
    if i>=1999:
        print('no convergence')
    print(freederivative(thetaG,NG,NL,NI,thetaL,thetaI))
    print(freederivative(thetaL,NL,NG,NI,thetaG,thetaI))
    print(freederivative(thetaI,NI,NL,NG,thetaL,thetaG))
    return(thetaG,thetaL,thetaI)
        
#-------------------------General Metropolis-Hastings Algorithm for ERGM-------------

def M_H_ergm(startmtx,ham,observables,comp_observables):
    condition=False
    n = len(startmtx)
    obs = comp_observables(startmtx)
    oldham = ham(obs)
    mtx=deepcopy(startmtx)
    obs = comp_observables(startmtx)
    count=0
    while(condition==False):
        nmtx = change_element(mtx,n)
        newobs = comp_observables(nmtx)
        newham = ham(newobs)
        p = compute_p(newham,oldham)
        print(newham)
        print(oldham)
        if (np.random.random()<p) == True:
            mtx=deepcopy(nmtx)
            oldham = deepcopy(newham)
        newobs = comp_observables(mtx)
        condition = np.allclose(newobs,observables,atol=0.5)
        count+=1
        #if count > 100:
        #    condition=True
         #   print(np.sum(mtx)/2)
    print(count)
    return(mtx)

#--Metropolis-Hastings Algorithm for the specific ERGM for Power Grids, Hamiltonian with 3 averages--

def M_H_ergmPG(startmtx,observables,buslist,params):
    condition=False
    n = len(startmtx)
    obs = avg_degreetype(startmtx,bustypes=buslist)
    oldham = dtype_ham(obs,params=params)
    mtx=deepcopy(startmtx)
    count=0
    synth=[]
    while(condition==False):
        nmtx = change_element(mtx,n)
        newobs = avg_degreetype(nmtx,bustypes=buslist)
        newham = dtype_ham(newobs,params=params)
        p = compute_p(newham,oldham)
        if (np.random.random()<p) == True:
            mtx=nmtx
            oldham = newham
        newobs = avg_degreetype(mtx,bustypes=buslist)
        count+=1
        if np.allclose(newobs,observables,atol=0.5)==True :
            #print(newobs)
            synth.append(mtx)
        if count > 100000:
            print('no convergence')
            break
    print(count)
    return(mtx,synth)
        
