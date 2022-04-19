# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 17:15:21 2022

@author: franc
"""
import pandas as pd
import numpy as np
from scipy.stats import norm
import networkx as nx
from scipy import linalg
import matplotlib.pyplot as plt
import time
from scipy.sparse.linalg import eigsh


"""
First we define some utility functions: the first one is used to compute the
correct link type of a power grid given both the bus and the branches data.
The second one is just a function that fits a normal distribution and plots the results
"""
def infer_link_type(bus_data, dfnetwork_T):
    dfnetwork = dfnetwork_T.copy(deep=True)
    dfnetwork['from_type'] = 2
    dfnetwork['to_type'] = 2
    genlist = bus_data['bus_i'].loc[bus_data['bustype'] == 1].tolist()
    shuntlist = bus_data['bus_i'].loc[bus_data['bustype'] == 3].tolist()
    
    dfnetwork['from_type'].loc[dfnetwork['fbus'].isin(shuntlist)] = 3
    dfnetwork['from_type'].loc[dfnetwork['fbus'].isin(genlist)] = 1
    
    dfnetwork['to_type'].loc[dfnetwork['tbus'].isin(genlist)] = 1
    dfnetwork['to_type'].loc[dfnetwork['tbus'].isin(shuntlist)] = 3
    dfnetwork['link_type'] = dfnetwork['from_type'] + dfnetwork['to_type']
    dfnetwork['link_type'].loc[(dfnetwork['from_type'] == 1) & (dfnetwork['to_type'] == 1)] = 1
    dfnetwork['link_type'].loc[(dfnetwork['from_type'] == 1) & (dfnetwork['to_type'] == 2)]= 2
    dfnetwork['link_type'].loc[(dfnetwork['from_type'] == 2) & (dfnetwork['to_type'] == 1)] = 2
    dfnetwork['link_type'].loc[(dfnetwork['from_type'] == 1) & (dfnetwork['to_type'] == 3)] = 3
    dfnetwork['link_type'].loc[(dfnetwork['from_type'] == 3) & (dfnetwork['to_type'] == 1)] = 3
    return(dfnetwork)

def pdf_norm(data):
    mu, std = norm.fit(data)
    plt.hist(data, bins=25, density=True, alpha=0.6, color='g')

    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    plt.plot(x, p, 'k', linewidth=2)
    title = "Fit results: mu = %.2f,  std = %.2f" % (mu, std)
    plt.title(title)
    
    plt.show()
    return


"""
We define two function to compute the bus type entropy and the d-scaling property. Both
were introduced by Wang and Elyas (see doi.org/10.1109/tpwrs.2016.2634318)
"""

#here we give two possible definitions of bus type entropies
def bustype_entropy(linkdf, busdf, mode=0):
    if mode == 0:
        link_ent_vec = np.log(linkdf['link_type'].value_counts(normalize=True)) * linkdf['link_type'].value_counts(normalize=True)
        bus_ent_vec = np.log(busdf['bustype'].value_counts(normalize=True)) * busdf['bustype'].value_counts(normalize=True)
        link_ent = np.sum(link_ent_vec)
        bus_ent = np.sum(bus_ent_vec)
        totW_1_ent = - (link_ent + bus_ent)
        return(totW_1_ent)
    elif mode== 1:
        link_ent_vec = np.log(linkdf['link_type'].value_counts(normalize=True)) * linkdf['link_type'].value_counts()
        bus_ent_vec = np.log(busdf['bustype'].value_counts(normalize=True)) * busdf['bustype'].value_counts()
        link_ent = np.sum(link_ent_vec)
        bus_ent = np.sum(bus_ent_vec)
        totW_1_ent = - (link_ent + bus_ent)
        return totW_1_ent
    else:
        print('Please insert a correct mode value (0 or 1)')
        return
    

def d_scaling(linkdf_T, busdf_T,mode = 0, n_samples = 2000):
    start_time = time.time()
    real_ent = bustype_entropy(linkdf_T,busdf_T,mode = mode)
    linkdf = linkdf_T.copy(deep=True)
    busdf = busdf_T.copy(deep=True)
    random_entropies=[]
    for i in range(n_samples):
        random_types = busdf['bustype'].sample(frac=1).reset_index(drop=True)
        newdf = pd.DataFrame(columns=['bustype','bus_i'])
        newdf['bus_i'] = busdf['bus_i']
        newdf['bustype'] = random_types
        linkdf = infer_link_type(newdf, linkdf)
        randent = bustype_entropy(linkdf,newdf,mode = mode)
        random_entropies.append(randent)
    mu, std = norm.fit(random_entropies)
    d = (real_ent - mu)/std
    print("--- %s seconds ---" % (time.time() - start_time))
    return(d,mu,std,random_entropies)

"""
Here we define some common topological properties of power grids, i.e. the average degree
of each bus type, the algebraic connectivity (Fiedler eigenvalue) and
the clustering coefficient
"""

def avg_degree_bustype(net):
    netnodes = list(net.nodes(data='bustype'))
    gen_deg = 0
    load_deg = 0
    int_deg = 0
    count_gen = 0
    count_load = 0
    count_int = 0
    net_degree = list(net.degree)
    for i in range(len(netnodes)):
        if netnodes[i][1] == 1:
            gen_deg += net_degree[i][1]
            count_gen += 1
        elif netnodes[i][1] == 2:
            load_deg += net_degree[i][1]
            count_load += 1
        else:
            int_deg += net_degree[i][1]
            count_int += 1
    return(gen_deg/count_gen, load_deg/count_load, int_deg/count_int)

"""
we define two different way to compute the algebraic connectivity, however
since the power grid's Laplcacian will be a sparse matrix the second one is preferred
"""
def algebraic_connectivity(net):
    start_time = time.time()
    A = nx.incidence_matrix(net, oriented = True)
    L = A.dot(A.T).toarray()
    ac = linalg.eigh(L)[0][1]
    print("--- %s seconds ---" % (time.time() - start_time))
    return(ac)

def sparse_algebraic_connectivity(net):
    start_time = time.time()
    A = nx.incidence_matrix(net, oriented = True)
    L = A @ A.T
    ac = eigsh(L,k=2, which='SM')[0][1]
    print("--- %s seconds ---" % (time.time() - start_time))
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



    
    
            
            