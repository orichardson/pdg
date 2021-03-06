# %load_ext autoreload
# %autoreload 2
# %pwd

x = 3

import math
import numpy as np
# import pandas as pd

from dist import RawJointDist, CPT
from rv import Variable, binvar, Unit
from pdg import PDG

#%% ############""" Example Code """##############

M = PDG()
PS = binvar('PS')
S = binvar('S')
SH = binvar('SH')
C = binvar('C')

M += CPT.from_ddict(Unit, PS, {'⋆': 0.3})
M += CPT.from_ddict(PS, S, { 'ps': 0.4, '~ps' : 0.2})
M += CPT.from_ddict(PS, SH, { 'ps': 0.8, '~ps' : 0.3})
M += CPT.from_ddict(S & SH, C,
    { ('s','sh') : 0.6, ('s','~sh') : 0.4,
      ('~s','sh'): 0.1, ('~s','~sh'): 0.01} )

CPT.det(PS, S, {'ps' : 's', '~ps' : 's'})


#  assert(all(M.edgedata[('PS', 'S',0)]['cpd'] == M[S|PS]))


#%% ############# Basic RawDist Tests ##############

unif = RawJointDist.unif([Unit, PS, S,SH,C])
mu2 = RawJointDist.random(M.vars)
Pr = M.factor_product()

len(M.edgedata)
Pr.H(C | SH,)

# A,B,C,D,E =
varis = [Variable.alph(chr(i+65), i+2) for i in range(5)]
mu3 = RawJointDist.random(varis)

V1, V2, V3, V4, V5 = varis

mu3.H(V4)

# mu3._query_mode = "dataframe"
# mu3[A | Unit]

# mu2(("~ps", PS))
mu2[SH,S | PS]

mu2[SH,S,PS | PS, Unit].shape

# pd.Series(mu2[PS,S,SH])

#%% #####################################################

d = M.factor_product()

import networkx as nx
# %matplotlib inline

nx.draw(M.graph)
M[C|S*SH]
d[S,SH | PS]

list(M.graph.edges())
# M += "p", CPT.det(Unit, Variable.alph("A", 3), {"⋆":'a1'})


# M + CPT.from_ddict(Unit, )

assert(np.allclose( M[C|S*SH],  d[C|S*SH],  d[C|S,SH]))
# d._query_mode = "dataframe"
d[C|S,SH]
M.score(RawJointDist.unif(d.varlist))
M.score(mu2)
M.score(Pr)

# d[PS|Unit].columns.to_flat_index()
# fd = d[PS|Unit].columns.to_flat_index().map(lambda s: s[0])

# np.ma.log((M[PS|Unit] / X).to_numpy())


from scipy import optimize
M.score(mu2)


mscore = M._build_fast_scorer()
# optimize.minimize(M.score, mu3)
mscore(mu2.data.reshape(-1))
mscore(Pr.data)
res = optimize.minimize(mscore, mu2.data.copy())

abs(res.x).sum()
"""
    # Building part by part in a context.


    # build by hypergraph
    G = load(" 1 -> PS ; PS -> S ; PS -> SH ; SH, S -> C")
    PDG.build(graph = G, n_vals = lambda var: 2)

    # pdg.add_random_links(n = 4, H = 0.5)

    # query consistency
    pdg.is_consistent()

    #
    data = pdg.samples(n = 1000) # each data[i]  = (ps, s, sh, c)...

    ## replace everything with 0,1, use matrices.
    # pdg.to_dense() ;
    ## replace values with lower case variable names.
    # pdg.to_labeled()

    fg = pdg.to_factor_graph()
    print(fg.Pr.toCPT() == pdg.max_entropy_dist.toCPT())



    # for each spanning tree, can conver to BN
    for tree in rooted_spanning_trees(pdg.G):
        print("BN: ", BayesNet(pdg.subgr(tree)) )
        # CONJECTURE: Every Distirbution is in the convex hull
        # of the distributions generated by spanning trees.
"""
