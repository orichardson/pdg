%load_ext autoreload
%autoreload 2

import numpy as np
import pandas as pd
import itertools

from dist import RawJointDist, CPT
from rv import Variable, binvar, Unit
from pdg import *


pm = {True: '+', False : '-'}
# In[ ]

M = PDG()

Y = Variable(['+', '-'])

M += 'Y', Variable(['+', '-'])

nbits = 3
M += 'X', Variable( itertools.product([-1,1], repeat=nbits))
nexp = 2**3
M += 'H', Variable(map(tuple,np.random.rand(nexp, nbits)))

locals().update(**M.vars)

(H*X).__dict__
len(H*X)

CPT.det(H*X, Y, lambda hx: pm[np.array(hx[0]).dot(hx[1]) > 0])

M += CPT.det(H*X, Y, lambda hx: pm[np.array(hx[0]).dot(hx[1]) > 0])
# 
# M += CPT.from_ddict(Unit, PS, {'⋆': 0.3})
# M += CPT.from_ddict(PS, S, { 'ps': 0.4, '~ps' : 0.2})
# M += CPT.from_ddict(PS, SH, { 'ps': 0.8, '~ps' : 0.3})
# M += CPT.from_ddict(S * SH, C, 
#     { ('s','sh') : 0.6, ('s','~sh') : 0.4,
#       ('~s','sh'): 0.1, ('~s','~sh'): 0.01} )
# 
# 
# # Right now it's just a the BN distribution we expect
# T = binvar("T")
# M += T
M.graph.edges

# del M[H*X,Y]
mu1 = M.factor_product()

s = [np.array(h).dot(x) > 0 for x in X for h in H]
[(x,h) for x in X for h in H if np.array(h).dot(x) <= 0]
sum(s) / len(s)

mu1[Y | X]
# np.unique(mu1.data)
# mu1.info_diagram(X,Y)
