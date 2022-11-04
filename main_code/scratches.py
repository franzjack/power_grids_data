# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 10:32:16 2022

@author: franc
"""

"""
plt.hist(entropylist)
plt.xlabel('Bus-type Entropy, synthetic graphs')
plt.ylabel('frequency')
plt.axvline(x=mtx_bus_entropy(truemat,buslist),color='red')
plt.show()
"""
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import Extract_stats as es
from tqdm import tqdm
from copy import copy,deepcopy
import mcmc_ergm as mc
from math import factorial as fac
from scipy.sparse.linalg import eigsh

def laplacian(A):
    deglist=[]
    for i in range(len(A)):
        deglist.append(np.sum(A[i]))
    D = np.diag(np.array(deglist))
    return(D-A)

def alg_conn(A):
    return(np.linalg.eigvalsh(laplacian(A))[1])



def sparse_algebraic_connectivity(net):
    #start_time = time.time()
    A = nx.incidence_matrix(net, oriented = True)
    L = A @ A.T
    ac = eigsh(L,k=2, which='SM')[0][1]
    #print("--- %s seconds ---" % (time.time() - start_time))
    return(ac)


def clustering_coeff(net):
    M = nx.DiGraph()
    for u,v in net.edges():
        if M.has_edge(u,v):
            M[u][v]['weight'] += 1
        else:
            M.add_edge(u, v, weight=1)
    
    clustering = nx.clustering(M,weight='weight')
    net_cluster = sorted(list(clustering.values()))
    return 2*(sum(net_cluster) / len(net_cluster))

def synth_selection(synlist,countlist,buslist):
    """
    Gsynlist = [nx.from_numpy_matrix(mtx) for mtx in synlist]
    boolconn = [nx.is_connected(G) for G in Gsynlist]
    conn_indx = np.where(boolconn)[0]
    mtxlist = [synlist[idx] for idx in conn_indx]
    Glist = [Gsynlist[idx] for idx in conn_indx]
    """
    mtxlist = deepcopy(synlist)
    Glist = [nx.from_numpy_matrix(mtx) for mtx in synlist]
    alg_list = [es.sparse_algebraic_connectivity(G) for G in Glist]
    clust_list = [es.clustering_coeff(G) for G in Glist]
    avg_typedeg = [mc.new_avg_degreetype(mtx,countlist[0],countlist[1],countlist[2]) for mtx in mtxlist]
    entropylist = [mtx_bus_entropy(m,buslist) for m in mtxlist]
    return(mtxlist,Glist,alg_list,clust_list,avg_typedeg,entropylist)

def synth_selection2(synlist,countlist,buslist):
    """
    Gsynlist = [nx.from_numpy_matrix(mtx) for mtx in synlist]
    boolconn = [nx.is_connected(G) for G in Gsynlist]
    conn_indx = np.where(boolconn)[0]
    mtxlist = [synlist[idx] for idx in conn_indx]
    Glist = [Gsynlist[idx] for idx in conn_indx]
    """
    Glist = [nx.from_numpy_matrix(mtx) for mtx in synlist]
    clust_list = [es.clustering_coeff(G) for G in tqdm(Glist)]
    ac_list = [alg_conn(synth) for synth in tqdm(synlist)]
    avg_typedeg = [mc.new_avg_degreetype(mtx,countlist[0],countlist[1],countlist[2]) for mtx in tqdm(synlist)]
    return(Glist,clust_list,avg_typedeg,ac_list)

def matrix_from_branch(flist,tolist,shape):
    mtx = np.zeros((shape,shape))
    for i in range(len(flist)):
        mtx[flist[i]][tolist[i]] =1
        mtx[tolist[i]][flist[i]] =1
    return(mtx)


def bus_index(bustypes):
    count_gen = 0
    count_load = 0
    count_int = 0
    it1=0
    it2=0
    new_indexes = np.zeros(len(bustypes),dtype=int)
    for i in range(len(bustypes)):
        if bustypes[i] == 1:
            new_indexes[i] = deepcopy(count_gen)
            count_gen +=1
        elif bustypes[i] == 2:
            count_load +=1
        else:
            count_int +=1
    for i in range(len(bustypes)):
        if bustypes[i] == 2:
            new_indexes[i] = deepcopy(count_gen + it1)
            it1+=1
        elif bustypes[i] == 3:
            new_indexes[i] = deepcopy(count_gen + count_load + it2)
            it2+=1
    return(new_indexes)
    
    
def linktype(a,b):
    if a+b<5:
        return a*b
    else:
        return a+b

def link_typelist(mtx, buslist):
    linklist=[]
    for i in range(len(mtx)):
        for j in range(i,len(mtx)):
                if mtx[i][j] ==1:
                    link = linktype(buslist[i],buslist[j])
                    linklist.append(link)
    return(linklist)

def ordered_buslist(q1,q2,q3):
    ordlist = np.ones((q1+q2+q3),dtype=int)
    for i in range(q1,(q1+q2)):
        ordlist[i]+=1
    for i in range((q1+q2),(q1+q2+q3)):
        ordlist[i]+=2
    return(ordlist)

def mtx_bus_entropy(mtx,buslist,mode=0):
    linklist = np.array(link_typelist(mtx,buslist))
    if mode == 0:
        link_ent_vec = np.log(np.bincount(linklist)[1:]/len(linklist)) * np.bincount(linklist)[1:]/len(linklist)
        bus_ent_vec = np.log(np.bincount(buslist)[1:]/len(buslist)) * np.bincount(buslist)[1:]/len(buslist)
        link_ent = np.sum(link_ent_vec)
        bus_ent = np.sum(bus_ent_vec)
        totW_1_ent = - (link_ent + bus_ent)
        return(totW_1_ent)
    elif mode== 1:
        link_ent_vec = np.log(np.bincount(linklist)[1:]/len(linklist)) * np.bincount(linklist)[1:]
        bus_ent_vec = np.log(np.bincount(buslist)[1:]/len(buslist)) * np.bincount(buslist)[1:]
        link_ent = np.sum(link_ent_vec)
        bus_ent = np.sum(bus_ent_vec)
        totW_1_ent = - (link_ent + bus_ent)
        return totW_1_ent
    else:
        print('Please insert a correct mode value (0 or 1)')
        return
  
    
def byn_coef(N):
    return(fac(N)/(fac(N-2)*2))  
    
def beta_edge(e, n):
    return(np.log(e/(byn_coef(n)-e)))
  
def Z_term1(b,q):
    return((1+np.exp(2*b/q))**byn_coef(q))

def Z_term2(b1,b2,q1,q2):
    return((1+np.exp(b1/q1+b2/q2))**(q1*q2))    
    

def bus_remap(bus_index,bus_i,tlist,flist):
    for j in range(len(tlist)):
        for i in range(len(bus_i)):
            if tlist[j]==bus_i[i]:
                tlist[j] = bus_index[i]
                j+1
    for j in range(len(flist)):
        for i in range(len(bus_i)):
            if flist[j]==bus_i[i]:
                flist[j] = bus_index[i]
                j+1
    return(flist,tlist)
    


def matrix_clean(mtx):
    indexes=[]
    targets=[]
    for i in range(len(mtx)):
        if np.sum(mtx[i]) <2:
            indexes.append(i)
            targets.append(np.nonzero(mtx[i]))
    newm = np.delete(mtx,indexes,0)
    newmtx = np.delete(newm,indexes,1)                       
    return(newmtx,indexes,targets)

def reorder_rows(a,permutation):
    idx = np.empty_like(permutation)
    idx[permutation] = np.arange(len(permutation))
    newa = deepcopy(a[idx,:])
    newaa = deepcopy(newa[:,idx])
    return(newaa)


def findPaths(G,u,n):
    if n==0:
        return [[u]]
    paths = [[u]+path for neighbor in G.neighbors(u) for path in findPaths(G,neighbor,n-1) if u not in path]
    return paths

def selector(synths):
    sel=[]
    ksel = copy(synths[int(len(synths)/1.5):])
    for i in range(len(ksel)):
        if i%int((len(synths[0].toarray())/2))==0:
            sel.append(ksel[i].toarray())
    return(sel)

