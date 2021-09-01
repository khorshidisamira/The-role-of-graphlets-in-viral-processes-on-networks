# -*- coding: utf-8 -*-
""" 
@author: Samira
"""
from __future__ import division
import sys
import os
import numpy
import networkx as nx
#import matplotlib.pyplot as plt
import csv
from collections import Counter

for file in os.listdir(sys.argv[1]):
    if file.endswith("GUISE.txt"):
        print(os.path.join(sys.argv[1], file))
        t_file = os.path.join(sys.argv[1], file)
        with open(t_file) as network: #with open('soc-wiki-Vote.txt','rt') as network:
            edges = csv.reader(network, delimiter = "\t")
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
            distribution = []
            for hist_degree in hist:
                distribution.append((hist_degree)/nodes_count)
            filename = t_file + "degreeDist_.txt"
            file_ = open(filename, "w")
            file_.write(distribution)  # python will convert \n to os.linesep
            file_.close()

"""
max_distribution = distribution[0][1]
min_distribution = distribution[-1][1]


plt.loglog(degree_sequence,'b-',marker='o')
plt.title("Degree rank plot")
plt.ylabel("degree")
plt.xlabel("rank")
"""
