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

with open("C:\Users\Samira\Dropbox\Dr_Hasan\Project_graphlet\degreeDistribution\Done\soc-buzznet.txt_GUISE.txt",'rt') as network:
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
   # file_name = find_between_r( t_file, path_dir, end_str)
    distribution= []
    for hist_degree in hist:
        distribution.append((hist_degree)/nodes_count)
