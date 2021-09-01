
# -*- coding: utf-8 -*-
""" 
@author: Samira
"""
from __future__ import division
import numpy
import networkx as nx
#import matplotlib.pyplot as plt
import csv
from collections import Counter
import os

"""
	file_ = open("res_format.txt", "w")
		
"""
def find_between_r( s, first, last ):
    try:
        start = s.rindex( first ) + len( first )
        end = s.rindex( last, start )
        return s[start:end]
    except ValueError:
        return ""
path_dir = "C:\Users\Samira\Dropbox\Dr_Hasan\Facebook"
end_str = "fb-Cal65.mtx_GUISE.txt"
sample = []
average_obj = {}
for file in os.listdir(path_dir):
    if file.endswith(end_str):
        print(os.path.join(path_dir, file))
        t_file = os.path.join(path_dir, file) 

        with open(t_file,'rt') as network:
            primary_edges = csv.reader(network, delimiter = "\t")
            edges = [edge for edge in primary_edges]
        
        G = nx.Graph() #creates a graph
        nodes = []
        for row in edges:
            G.add_edge(row[0], row[1])
            nodes.append(row[0])
        
        distinct_nodes = Counter(nodes).keys()
        
        for i in range(len(distinct_nodes)):
            sample.append("v," + distinct_nodes[i] + "," + distinct_nodes[i])
            
        for i in range(len(distinct_nodes) + len(edges)):
            j = i - len(distinct_nodes)
            sample.append("e," + str(edges[j]))
             
            
            
result_file = os.path.join(path_dir, 'sample_graph.txt')
with open(result_file, 'a') as the_file:
    the_file.write(str(sample)) 
    
"""
max_distribution = distribution[0][1]
min_distribution = distribution[-1][1]


plt.loglog(degree_sequence,'b-',marker='o')
plt.title("Degree rank plot")
plt.ylabel("degree")
plt.xlabel("rank")
"""
