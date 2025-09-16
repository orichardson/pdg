"""
This module contains tools and algorithms for investigating the qualitative
sturcture of distributions relative to (directed) hypergraphs. 
"""

from .rv import Variable as Var
from .dist import RawJointDist as RJD

from collections import namedtuple
from collections.abc import Collection, Mapping
from typing import Any
from operator import mul
from functools import reduce

import networkx as nx
import itertools as itt
import numpy as np
from scipy.sparse import coo_array
import torch

Arc = namedtuple('Arc', ['srcs', 'tgts', 'weight'], defaults=[1])
Arc.scope = property(lambda self: itt.chain(self.srcs, self.tgts))
Arc.w = property(lambda self: self.weight)


class HyperGraph:
    def __init__(self, hyperedges : Mapping[Any, Collection] | Collection[Collection] ):
        if isinstance(hyperedges, HyperGraph):
            hyperedges = hyperedges.hyperedges
        elif not isinstance(hyperedges, Mapping):
            hyperedges = dict(enumerate(hyperedges))

        self.hyperedges = hyperedges
    
    def __iter__(self):
        return iter(self.hyperedges.values())


def _init_tensor(shape, init_mode, require_grad=True):
    if init_mode == 'unif':
        t = torch.zeros(shape)
    elif init_mode == 'random':
        t = torch.tensor(np.random.gumbel(size=shape))
    else:
        raise ValueError("Unknown Init Mode: "+repr(init_mode))
    
    t.requires_grad = require_grad
    return t


def fit_factorization(mu : RJD, hg, MAX_ITERS=500, init_mode='random', **optim_kwargs):
    hg = HyperGraph(hg)
    """
    Does a given distribution μ(A,B,C) factor as μ(A,B,C) = f1(A,B) f2(B,C) f3(C,A) for suitable functions f1-3? Not all distributions do, but this property is not just a question of independence. 
    
    This method helps investigate such factorization properties of distributions along a hypergraph, by "training" such functions f1-3, to match the given distribution. 
    
    Parameters
    ----------
    mu: dist.RawJointDist
        the distribution of interest. Can it be represented as the product of factors along the given hypergraph structure?
    hg: HyperGraph
        the hypergraph of interest, describing the desired factorization.
    MAX_ITERS : int
        the number of optimization iterations. 
        
    Returns
    -------
    best: dist.RawJointDist
        The best approximation to mu that was found.
    """
    # varlookup = { V.name : V for V in mu.varlist }
    # name2idx = { V.name : i for (i,V) in enumerate(mu.varlist) }

    mudata = torch.tensor(mu.data.reshape(-1))
    
    logfactors = []
    for varsubset in hg:
        # it's very important that the shapes of these tensors line up properly, so that we can add them together. 
        localshape = tuple(len(X) if X.name in varsubset else 1 for X in mu.varlist)
        logfactors.append(_init_tensor(localshape,init_mode))

    ozr = torch.optim.Adam( logfactors, **optim_kwargs)
    for it in range(MAX_ITERS):
        ozr.zero_grad()

        unnorm = sum(logfactors)
        normed = torch.softmax(unnorm.reshape(-1), 0) #
        loss = ( normed * (torch.log(normed) - torch.log(mudata.data))).sum()
        
        loss.backward();
        ozr.step()

        if it%50 == 0:
            print(loss.detach().item())

    # return normed, logfactors
    return RJD(normed, mu.varlist)


##########################################################
#   Now, methods for dealing with directed hypergraphs.
##########################################################

class DHyperGraph(object):
    """
    This class is a very skeletal description of directed hypergraphs.
    
    TODO: merge this code with the functionality in the hyperflow repository, to give a single library for talking about directed hypergraphs.
    """
    def __init__(self, hyperarcs : Mapping[Any, tuple[Collection,Collection]] | Collection[tuple[Collection,Collection]], 
                    nodes=None):
        # hyperarcs = mapping { label :  (srcs, tgts), }
        #   where each (srcs, tgts) are both collections. 
        if not isinstance(hyperarcs, Mapping):
            hyperarcs = dict(enumerate(hyperarcs))

        self.hyperarcs = {}
        for (l,(srcs,tgts)) in hyperarcs.items():
            if not isinstance(srcs, Collection): srcs = [srcs]
            if not isinstance(tgts, Collection): tgts = [tgts]
            self.hyperarcs[l] = Arc(srcs,tgts)
        
        # print(self.hyperarcs)
        if nodes is None:
            self.nodes = set()
            for a in self.hyperarcs.values():
                self.nodes |= set(a.scope)
        else:
            self.nodes = set(nodes)

        # all hyperedges must be contained with the node set, even if manually specified.
        # assert( all(N in self.nodes for a in self.hyperarcs for N in a.scope ))
        assert( all(N in self.nodes for a in self for N in a.scope ))

    def __iter__(self):
        return iter(self.hyperarcs.values())
    
    @property
    def labeled_arcs(self):
        return self.hyperarcs.items()

    def to_nxDiGraph(self) -> nx.DiGraph:
        G = nx.MultiDiGraph()

        G.add_nodes_from(self.nodes)
        # new_joint_nodes = set()

        for a in self:
            # for S in [a.srcs, a.tgts]:
            #     if len(S) > 1:
            #         G.add_node(frozenset(S))
            # S = frozenset(a.srcs) if len(a.srcs) > 1 else next(iter(a.srcs))
            # T = frozenset(a.tgts) if len(a.tgts) > 1 else next(iter(a.tgts))
            # G.add_edge(S,T)
            G.add_edge(frozenset(a.srcs), frozenset(a.tgts))
            
        # for n in G.nod
        # TODO: add implied arcs from supersets to subsets, and return.
        raise NotImplementedError()
    
    def SDef(self, mu):
        return - mu.H(...) + sum(mu.H(*a.tgts,'|', *a.srcs) for a in self)


## TODO: Implement QDG class when it becomse useful.
## QDG = weighted hypergraph
# class QDG(DHyperGraph):
#     def __init__(self, whyperarcs):
#         # self.whyperedges = 
#         pass

def _all_fns(set_from, set_to):
    # return list(map(dict, itt.product(*[[(s,t) for t in set_to] for s in set_from])))
    return list( itt.product(*[[(s,t) for t in set_to] for s in set_from]))

def _t1(n):  # a tuple of ones of length n
    return tuple(1 for i in range(n))


def find_witness( mu : RJD, Ar: DHyperGraph, N_ITERS=500, 
                 evenly=False, init_mode='unif', 
                 tol=1E-6, lr=.9,
                 verbose=False):
    """
    Can the given distribution have arisen from a causal model with independent mechanisms whose shape is the given hypergraph? This method is brute-force attempt to answer that question. 
    
    Given a hypergraph (Xn, Ar) whose nodes are variable names,
        and a joint distributiion μ(X) over values of all relevant
        variables in the hypergraph, this method aims to find an
        extended distribution ̄μ(X, U) that, in a sense, serves as 
        
        See Definition 2 (page 3) of _Qualitative Mechanism Independence_
        (https://arxiv.org/abs/2501.15488). 
        
    ALGORITHM
    ---------
    The procedure for finding such a witness is based roughly on the ideas behind equation (3) and  the surrounding material in section 4.2  of that paper. But it introduces a significant innovation.
    
    In this approach, the mutual independence of the U's (F's) and the fact that S_a -->> T_a (i.e., that the Targets of a determined by the Sources of a) are hard constraints of the parameterization, and we train to match μ. However, that space is far too over-constrained to optimize over, so we also introduce a possible "null" value for each variable, which typically has the effect of relaxing our optimization problem to sub-joint distributions.  
    
    Ultimately the optimization objective is the KL between the induced marginal over the original variables and \mu, plus an additional penalty for use of the null value.
    
    Parameters
    ----------
    mu : dist.RawJointDist
        the distribution μ(X) of interest
    Ar : DirectedHypergraph
        the hypergraph (Xn, Ar) of interest
    evenly : bool, optional
        Determines whether or not to constrain to witnesses with a certain
        "uniformity" property --- i.e., those that are uniform over the solutions
        to some causal model. 
    tol: float
    lr : float
        learning rate for the Adam optimizer
    
    Returns
    -------
    mu_bar : dist.RawJointDist
        The extended joint distribution that was as close as possible to being a QIM- witness.
        
    Fs: list[rv.Variable]
        A list of the additional variables $U = (U_a)$, one per hyperarc of Ar.
        The values of $Fs[a_i] = U_a$ are functions from the source variables of a to the targets of a. 
        
    params: list[torch.Tensor]
        the parameters of the optimization. There are two kinds:
         - logQ_normalized, the centered log univariate distributions over each function variable U_a
         - logQQs: the distribution over fixedpoints of the functions.    
    """ 
    varlookup = { V.name : V for V in mu.varlist }
    name2idx = { V.name : i for (i,V) in enumerate(mu.varlist) }
    mutorch = mu.torchify()


    Xnull = [ Var(set(X) | { None }, name=X.name+"'") for X in mu.varlist]
    Fs = []
    logQ_params = []

    if verbose: print("SETUP...")

    for i, (l, a) in enumerate(Ar.labeled_arcs):
        if verbose: print("\t", a)

        Svals = list(itt.product(*[varlookup[Sn] for Sn in a.srcs]))
        Tvals = list(itt.product(*[varlookup[Tn] for Tn in a.tgts]))

        F = Var(_all_fns(Svals, Tvals), name="F_"+str(l))
        Fs.append(F) 

        logQ_params.append(_init_tensor(len(F), init_mode))

    Xind = {}
    if not evenly: logQQs = {}

    if verbose: 
        print("LOGQQ SETUP...")
        print("now, loop that takes %d iterations" % int(np.prod([len(X) for X in Fs])) )
        print("each constructs R of shape ", tuple(1 for X in Xnull))
        counter = 0

    for fs in itt.product(*Fs):
        if verbose:
            counter += 1
            if counter % 100 == 0:
                print(counter, end='\r')
        # Step 1: figure out the set of fixedpts 
        # build relation on X's for this setting of f's. 
        # That is, want R[X1... Xn] = [1 if f_a(S_a) = T_a forall a, else 0].

        # R = np.ones(tuple(1 for X in Xnull))
        R = np.ones(tuple(len(X) for X in Xnull))

        for f,a in zip(fs, Ar):
            localshape = tuple(len(X)+1 if X.name in a.scope else 1 for X in mu.varlist)
            Rlocal = np.zeros(localshape)

            Sjoint = itt.product(*[varlookup[Sn] for Sn in a.srcs])
            Tjoint = itt.product(*[varlookup[Tn] for Tn in a.tgts])

            for s,t in itt.product(Sjoint,Tjoint):
                idxs = [0] * len(mu.varlist)
                for j,Sn in enumerate(a.srcs):
                    idxs[name2idx[Sn]] = varlookup[Sn].ordered.index(s[j])
                for j,Tn in enumerate(a.tgts):
                    idxs[name2idx[Tn]] = varlookup[Tn].ordered.index(t[j])

                Rlocal[tuple(idxs)] = (dict(f)[s] == t)
            
            Rlocal[tuple(-1 for i in localshape)] = 1
            
            R = R * Rlocal
        
        # R_coo = coo_array(R)
        R_coo = torch.tensor(R).to_sparse()
        nnz = len(R_coo.values())
        # QQs[fs] = torch.ones(nnz) / nnz
        if not evenly: logQQs[fs] = _init_tensor(nnz, init_mode)

        Xind[fs] = R_coo.indices()
        # print(Xind[fs], R_coo.shape)


    if verbose: print("SETUP COMPLETE")
    if len(logQ_params) == 0 and evenly: # dummy optimizer for paramatricity when no params
        ozr =torch.optim.Adam( [torch.tensor([0.01],requires_grad=True)], lr=lr)
    else:
        # ozr = torch.optim.Adam( logQ_params + ([] if evenly else list(logQQs.values())), lr=9E-1)
        ozr = torch.optim.Adam( logQ_params + ([] if evenly else list(logQQs.values())), lr=lr)
        # ozr = torch.optim.Adam( logQ_params, lr=1E-2)
    # torch.autograd.set_detect_anomaly(True)
    
    for it in range(N_ITERS):
        # if verbose: print(it)
        logQ_normalized = [logQ-torch.logsumexp(logQ,0) for logQ in logQ_params]

        # distdata = torch.zeros(tuple(len(X)+1 for X in mu.varlist))
        distdata = torch.zeros(mu.data.shape)
        # ozr.zero_grad(set_to_none=True)
        ozr.zero_grad()

        for fs in itt.product(*Fs):
            # total log probability that (\U = fs), since they're independent
            logq = sum((logQ_normalized[ai][Fs[ai].ordered.index(f)] for ai,f in enumerate(fs)),
                       start=torch.tensor(0.))
            # print(logq)
            
            # Xind[fs].shape =  [original tensor.ndim,  nnz]
            if evenly:
                nnz = Xind[fs].shape[1]
                fp_selection = torch.ones(nnz) 

                for j in range(nnz):
                    if any(Xind[fs][i,j] == len(X) for i,X in enumerate(mu.varlist)):
                        fp_selection[j] = 0
                
                if fp_selection.sum() == 0:
                    fp_selection = torch.ones(nnz)
                # if nnz > 1 :
                #     fp_selection[-1] = 0
                
            else:
                fp_selection = torch.exp(logQQs[fs])

            # print(fp_selection)
            # print(Xind[fs])
            distslice = torch.sparse_coo_tensor(Xind[fs], 
                fp_selection / fp_selection.sum() * torch.exp(logq)).to_dense()
            distslice2 = distslice
            # idxslice = tuple((slice(0,-1) if distslice.shape[i] > 1 else 0) for i in range(distslice.ndim))
            idxslice = tuple((slice(0,-1) if distslice.shape[i] > 1 else 0) for i in range(distslice.ndim))
            # print(distslice2, distslice2.shape)
            # print(distslice.shape, distdata.shape)
            # print(distslice2[idxslice].shape)
            distdata += distslice2[idxslice]

        # loss = RJD(distdata, mu.varlist, use_torch=True) // mu.torchify() + \
        #     (1-distdata.sum()) * torch.log()
        # loss = torch.sum((distdata - mutorch.data)**2) + (1-distdata.sum())**2
        # loss = mu.torchify() // RJD(distdata, mu.varlist, use_torch=True) 
        loss = (mutorch.data * (torch.log(mutorch.data) - torch.log(distdata))).sum()
        # [Zhu&Rower] Extended KL:   D[p||q] = E_p[ log p - log q + q - p]
        # here, p = mutorch, q = distdata.  So the  following is backwards!!

        loss += 1 - distdata.sum() 
        # loss += distdata.sum()  - 1
            # + \
            # (1-distdata.sum()) * torch.log()

        if abs(distdata.detach().sum() - 1) < tol and loss.detach().item() < tol:
            print(loss.detach().item())
            break

        loss.backward()
        ozr.step()
        if it % 50 == 0:
            print(loss.detach().item())

    # rjd = RJD.unif    
    # return Fs, Qs
    return (RJD(distdata, mu.varlist, use_torch=True).npify(), Fs, 
            logQ_normalized) + (() if evenly else (logQQs,))
