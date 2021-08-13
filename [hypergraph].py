# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 17:52:57 2021

@author: Oliver
"""


from typing import TypeVar, Generic, Container, Tuple
# from collections.abc import Collectxion

T = TypeVar('T')

# I hate that the generics don't get enforced at runtime....

# class DGraph(Generic[T]):
#     def __init__(self, nodes: Container[T], edges: Container[Tuple[T,T]]):4
class DGraph(object):
        self.nodes = nodes;
        self.edges = edges;

class DHyperGraph(object):
    def __init__(self, hyperedges):
        # What's the form of hyper-edges? Good question
        self.hyperedges = hyperedges

    def toGraph(self) -> DGraph[T] :
        nodes = []
        edges = []

        for src, tgt in self.hyperedges:


        pass
