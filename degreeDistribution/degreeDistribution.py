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
path_dir = "C:\Users\Samira\Dropbox\Dr_Hasan\karate\sim_graphs_karate"
end_str = ".csv"
distribution = {}
average_obj = {}
for file in os.listdir(path_dir):
    if file.endswith(end_str):
        print(os.path.join(path_dir, file))
        t_file = os.path.join(path_dir, file) 

        with open(t_file,'rt') as network:
            edges = csv.reader(network, delimiter = ",")
            # This skips the first row of the CSV file.
            next(edges)
            edges = [edge for edge in edges]
        
        G = nx.Graph() #creates a graph
        nodes = []
        for row in edges:
            G.add_edge(row[0], row[1])
            nodes.append(row[0])
        
        distinct_nodes = Counter(nodes).keys()
        #G = nx.gnp_random_graph(100,0.02)
        G2 = G.to_undirected()
        degree_sequence=sorted(nx.degree(G).values(),reverse=True) # degree sequence
        degree_sequence2=sorted(nx.degree(G2).values(),reverse=True) # degree sequence
        #print "Degree sequence", degree_sequence
        
        dmax=max(degree_sequence)
        nodes_count = len(distinct_nodes)
        
        hist, edges = numpy.histogram( degree_sequence , bins=numpy.r_[1,2,5,10,20,50,100,200,500,numpy.inf]) #Histogram with bins a percentage of degree distribution
        file_name = find_between_r( t_file, path_dir, end_str)
        distribution[file_name] = []
        
        average_obj[file_name] = reduce(lambda x, y: x + y, degree_sequence) / len(degree_sequence)
        for hist_degree in hist:
            distribution[file_name].append((hist_degree)/nodes_count)
            
result_file = os.path.join(path_dir, 'DFD_result.txt')
with open(result_file, 'a') as the_file:
    the_file.write(str(distribution))
    
#average_degree = os.path.join(path_dir, 'average_degree.txt')
#with open(average_degree, 'a') as the_file:
#    the_file.write(str(average_obj))
    
"""
max_distribution = distribution[0][1]
min_distribution = distribution[-1][1]


plt.loglog(degree_sequence,'b-',marker='o')
plt.title("Degree rank plot")
plt.ylabel("degree")
plt.xlabel("rank")
"""
