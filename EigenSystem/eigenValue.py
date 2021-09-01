# -*- coding: utf-8 -*-
""" 
@author: Samira
"""
from __future__ import division
import numpy as np
import networkx as nx
#import matplotlib.pyplot as plt
import csv
from collections import Counter
import os
from numpy import linalg as LA
import scipy.sparse as sparse
import scipy
from numba import vectorize, cuda

@vectorize(['float32(float32, float32)'], target= 'cuda')
def vectorAdd(a,b,c):
    for i in xrange(a.size):
        c[i] = a[i] + b[i]
        
def main():
    n = 320000000
    A = np.ones(n, dtype = np.float32)
    B = np.ones(n, dtype = np.float32)
    C = np.ones(n, dtype = np.float32)
    vectorAdd(A,B,C)
if __name__ == "__main__":
    main()
"""
def find_between_r( s, first, last ):
    try:
        start = s.rindex( first ) + len( first )
        end = s.rindex( last, start )
        return s[start:end]
    except ValueError:
        return ""
path_dir = "C:\Users\Samira\Dropbox\Dr_Hasan\Facebook\Adjacency_mtx"
end_str = "adj.txt"
eigen_values = {}
for file in os.listdir(path_dir):
    if file.endswith(end_str):
        print(os.path.join(path_dir, file))
        t_file = os.path.join(path_dir, file) 
        
        with open(t_file,'rt') as network:
            #lines = network.read().split(',')
            #adjacency_mtx = np.loadtxt(t_file, skiprows=1)
            reader = csv.reader(network, delimiter = ",")
            x = list(reader)
            adj_matrix = np.array(x).astype("float")
           # connected_comp = scipy.sparse.csgraph.connected_components(adj_matrix, directed=False)
            #adj_matrix = [row for row in adjacency_mtx]
            type_x = type(adj_matrix)
            #vals, vecs = sparse.linalg.eigs(adj_matrix, k=2)#LA.eig(adj_matrix)
        
       # file_name = find_between_r( t_file, path_dir, end_str)
       # eigen_values[file_name] = vals 
                    
#result_file = os.path.join(path_dir, 'eigenValues_result.txt')
#with open(result_file, 'a') as the_file:
 #   the_file.write(str(eigen_values)) 
"""