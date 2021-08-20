import numpy as np
import pandas as pd

import sys
sys.path.append(sys.path[0] + '/..')

from dist import RawJointDist, CPT
from rv import Variable, binvar, Unit
from pdg import *
import itertools

# start with small.


M = PDG()
X1 = Variable.alph("X1_", 2)
X2 = Variable.alph("X2_", 2)
Y = Variable.alph("Y", 2)

M += "p", CPT.make_random(X1,Y)
M += "q", CPT.make_random(X2,Y)
alpharange = np.linspace(0,1,5)

dists = np.zeros((len(alpharange), len(alpharange), *M.dshape))
dists.shape
for (i,a1),(j,a2) in itertools.product(enumerate(alpharange),enumerate(alpharange)):
    wM = M.with_params(alpha={'p': a1, 'q': a2})
    
    disti = wM.optimize_score()
    dists[i,j,...] = disti.data
    print('*'*50)
    print(wM._opt_rslt)
from matplotlib import pyplot as plt
import seaborn as sns

greens = sns.light_palette("green", as_cmap=True)
d1 = RawJointDist(dists[0,0], M.varlist)
# d1._query_mode = "dataframe"
d1[Y|X1,X2].style.background_gradient(greens,axis=None).hide_index()
plt.matshow(wM[Y|X2],cmap='Blues')    
wM[Y|X1]
disti[Y|X1]
M[Y |X2]
d[Y|X2,Unit]
d2 = M.factor_product()

M.score(d)
M.score(d2)
d2._query_mode = "dataframe"

d2[Y|X1,X2]
d[Y|X1,X2]


########### What about for different values of alpha? ########
