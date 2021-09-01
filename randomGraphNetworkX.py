# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 13:56:01 2017

@author: Samira
"""
import networkx as nx; 
import random; 
import sys; 
import numpy as np; 
xxxx = 1458764

G = nx.generators.random_graphs.gnm_random_graph(1000,1000,directed=False)
degree_sequence=sorted(nx.degree(G).values(),reverse=True) # degree sequence
edges = G.edges(data=False)