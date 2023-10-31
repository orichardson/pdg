from .rv import Variable as Var, ConditionRequest, Unit
from .pdg import PDG
from .dist import RawJointDist as RJD

from collections import namedtuple
from collections.abc import Collection, Mapping, Collection

import networkx as nx
import itertools as itt
import numpy as np
from scipy.sparse import coo_array
import torch

Arc = namedtuple('Arc', ['srcs', 'tgts', 'weight'], defaults=[1])
Arc.scope = property(lambda self: itt.chain(self.srcs, self.tgts))
Arc.w = property(lambda self: self.weight)


class DHyperGraph(object):
    def __init__(self, hyperedges : Mapping[tuple[Collection,Collection]] | Collection[tuple[Collection,Collection]], 
                    nodes=None):
        # hyperedges = mapping { label :  (srcs, tgts), }
        #   where each (srcs, tgts) are both collections. 
        if not isinstance(hyperedges, Mapping):
            hyperedges = dict(enumerate(hyperedges))

        # self.hyperedges = { 
        #     l: Arc(srcs, tgts) for (l,(srcs,tgts)) in hyperedges.items()
        # }
        self.hyperedges = {}
        for (l,(srcs,tgts)) in hyperedges.items():
            if not isinstance(srcs, Collection): srcs = [srcs]
            if not isinstance(tgts, Collection): tgts = [tgts]
            self.hyperedges[l] = Arc(srcs,tgts)

        # print(self.hyperedges)
        
        if nodes is None:
            self.nodes = set()
            for a in self.hyperedges.values():
                self.nodes |= set(a.scope)
        else:
            self.nodes = set(nodes)

        assert( all(N in self.nodes for N in a.scope for a in self.hyperedges))

        # hyperedges

    def __iter__(self):
        return iter(self.hyperedges.values())
    
    @property
    def labeled_arcs(self):
        return self.hyperedges.items()

    def to_nxDiGraph(self) -> nx.DiGraph:
        G = nx.MultiDiGraph()

        G.add_nodes_from(self.nodes)
        new_joint_nodes = set()

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
        


# QDG = weighted hypergraph
class QDG(DHyperGraph):
    def __init__(self, whyperedges):
        # self.whyperedges = 
        pass

def all_fns(set_from, set_to):
    # return list(map(dict, itt.product(*[[(s,t) for t in set_to] for s in set_from])))
    return list( itt.product(*[[(s,t) for t in set_to] for s in set_from]))

def t1(n):  # a tuple of ones of length n
    return tuple(1 for i in range(n))


def find_witness( mu : RJD, Ar: DHyperGraph, N_ITERS=500, verbose=False):
    varlookup = { V.name : V for V in mu.varlist }
    name2idx = { V.name : i for (i,V) in enumerate(mu.varlist) }
    M = len(Ar.hyperedges)
    mutorch = mu.torchify()

    Xnull = [ Var(set(X) | { None }, name=X.name+"'") for X in mu.varlist]
    Fs = []
    logQ_params = []

    if verbose: print("SETUP...")

    for i, (l, a) in enumerate(Ar.labeled_arcs):
        if verbose: print("\t", a)
        # Svals = itt.product(varlookup[Sn] for Sn in Sns)
        # Sjoint = Var.product(varlookup[Sn] for Sn in a.srcs)
        # Tjoint = Var.product(varlookup[Tn] for Tn in a.tgts)
        Svals = list(itt.product(*[varlookup[Sn] for Sn in a.srcs]))
        Tvals = list(itt.product(*[varlookup[Tn] for Tn in a.tgts]))

        # F = Var(all_fns(Sjoint,Tjoint), name="F_"+str(l))
        F = Var(all_fns(Svals, Tvals), name="F_"+str(l))
        Fs.append(F) 

        # Q = torch.ones(len(F)) / len(F)
        # logQ = torch.zeros(len(F))
        # logQ = torch.zeros(len(F)) - np.log(len(F)+ 0.)
        logQ = torch.zeros(len(F))
        # logQ = logQ -torch.logsumexp(logQ, 0)
        # Q = Q.reshape(*t1(i), len(F), *t1(M-i-1))
        logQ.requires_grad = True
        # logQ_normd = logQ - torch.logsumexp(logQ,0)
        # logQ_normd = logQ - torch.sum(logQ)
        logQ_params.append(logQ)
        # logQ_normalized.append(logQ_normd)
        # logQ_normalized.append(logQ - torch.logsumexp(logQ,0))
        # logQ_normalized.append(logQ - torch.sum(logQ*))


    Xind = {}
    logQQs = {}

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

        R = np.ones(tuple(1 for X in Xnull))
        # R = np.ones(tuple(len(X) for X in Xnull))
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
        logQQs[fs] = torch.zeros(nnz)
        logQQs[fs].requires_grad = True
        Xind[fs] = R_coo.indices()


    if verbose: print("SETUP COMPLETE")
    ozr = torch.optim.Adam( logQ_params + list(logQQs.values()), lr=5E-1)
    # ozr = torch.optim.Adam( logQ_params, lr=1E-2)
    # torch.autograd.set_detect_anomaly(True)
    
    for it in range(N_ITERS):
        if verbose: print(it)
        logQ_normalized = [logQ-torch.logsumexp(logQ,0) for logQ in logQ_params]

        # distdata = torch.zeros(tuple(len(X)+1 for X in mu.varlist))
        distdata = torch.zeros(mu.data.shape)
        ozr.zero_grad(set_to_none=True)
        # ozr.zero_grad()

        for fs in itt.product(*Fs):
            logq = sum(logQ_normalized[ai][Fs[ai].ordered.index(f)] for ai,f in enumerate(fs))
            
            fp_selection = torch.exp(logQQs[fs])
            distslice = torch.sparse_coo_tensor(Xind[fs], 
                fp_selection / fp_selection.sum() * torch.exp(logq)).to_dense()
            distslice2 = distslice
            distdata += distslice2[tuple(slice(0,-1) for i in range(distslice.ndim))]

        # loss = RJD(distdata, mu.varlist, use_torch=True) // mu.torchify() + \
        #     (1-distdata.sum()) * torch.log()
        # loss = torch.sum((distdata - mutorch.data)**2) + (1-distdata.sum())**2
        # loss = mu.torchify() // RJD(distdata, mu.varlist, use_torch=True) 
        loss = (mutorch.data * (torch.log(mutorch.data) - torch.log(distdata))).sum()
            # + \
            # (1-distdata.sum()) * torch.log()
        loss.backward()
        ozr.step()
        if it % 50 == 0:
            print(loss.detach().item())

    # rjd = RJD.unif    
    # return Fs, Qs
    return RJD(distdata, mu.varlist, use_torch=True).npify(), Fs, logQ_normalized, logQQs