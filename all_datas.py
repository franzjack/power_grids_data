# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 17:32:57 2022

@author: franc
"""

import Extract_stats as es
import load_pglib_opf as lp
import pandas as pd
#this make things work without issues, however in the future it should be removed
import warnings
warnings.filterwarnings("ignore")

"""
This program once executed creates a dataset with the chosen statistics using all the
".m" files in the same directory. It uses the load_pglib_opf module 
(https://github.com/leonlan/pglib-opf-pyparser) created by Leon Lan. Without him this
program probably wouldn't exist.'

"""

def all_data(name):
    bus,gen,branch = lp.load_pglib_opf(name)
    G = lp.to_networkx(bus,gen,branch)
    statrow = []
    statrow.append(name)
    statrow.append(len(bus))
    statrow.append(len(branch))
    d1,d2,d3 = es.avg_degree_bustype(G)
    statrow.append(d1)
    statrow.append(d2)
    statrow.append(d3)
    statrow.append(es.bustype_entropy(branch, bus))
    statrow.append(es.bustype_entropy(branch, bus,mode=1))
    statrow.append(es.sparse_algebraic_connectivity(G))
    statrow.append(es.clustering_coeff(G))
    #statrow.append(nx.average_shortest_path_length(G.to_undirected()))
    return(statrow)
        
      
if __name__ == "__main__":
       import glob
       
       allrows=[]
   
       failure = 0
       for filename in glob.iglob("./" + "**/*.m", recursive=True):
           print(filename)
           try:
               row = all_data(filename)
               allrows.append(row)
           except Exception as e:
               failure += 1
               print(Exception)
               print(f"Failed loading {filename}")
   
       if failure == 0:
           print("Succesfully loaded all instances.")
       head = ['name', 'n_bus','n_branches','d_gen','d_load','d_int','bus_entropy0','bus_entropy1','alg_conn','clust_coeff']
       complete_df = pd.DataFrame(columns=head, data=allrows, dtype=object)
