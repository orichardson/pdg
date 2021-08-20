%load_ext autoreload
%autoreload 2
%pwd
%cd examples

import numpy as np
import pandas as pd

# from dist import RawJointDist, CPT
# from rv import Variable, binvar, Unit
# from pdg import *

from dist import RawJointDist, CPT
from rv import Variable, binvar, Unit
from pdg import *

############""" Example Code """##############

M = PDG()
PS = binvar('PS')
S = binvar('S')
SH = binvar('SH')
C = binvar('C')


M += CPT.from_ddict(Unit, PS, {'⋆': 0.3})
M += CPT.from_ddict(PS, S, { 'ps': 0.4, '~ps' : 0.2})
M += CPT.from_ddict(PS, SH, { 'ps': 0.8, '~ps' : 0.3})
M += CPT.from_ddict(S * SH, C, 
    { ('s','sh') : 0.6, ('s','~sh') : 0.4,
      ('~s','sh'): 0.1, ('~s','~sh'): 0.01} )
      

# Right now it's just a the BN distribution we expect
T = binvar("T")
M += T

mu1 = M.factor_product()

# But now we add tanning


T2C =  CPT.from_ddict(T, C, { 't' : .3, '~t' : .05})
list((M+T2C).graph.edges(keys=True))
# Add
mu2 = (M + T2C).factor_product()

mu2[S | PS]
mu1[S | PS]
## Note: mu1 fits BOTH pdgs better than mu2!

M.score(mu1) # -1
M.score(mu2) # .56 -1
(M).score(mu2) - (M).score(mu1)

(M+T2C).score(mu1)
(M+T2C).score(mu2)
(M+T2C).score(mu2) - (M+T2C).score(mu1)


M.matches(mu1)
M.matches(mu2)




# # Graph: alpha vs score for both distributions
# alpha, beta = np.meshgrid(np.linspace(0, 1, 20), np.linspace(0,2,20))
# 
# scores = np.zeros(alpha.shape)
# # for x,y in zip(xcoord.flatten(), ycoord.flatten()):
# #     scores[x,y] = 
# import itertools
# for xy in itertools.product(*[list(range(n)) for n in scores.shape]):
#     # print(xy)
#     scores[xy] = M.score(mu1, weightMods=(beta[xy], -beta[xy] + alpha[xy], 0,0))
#     scores.min()


from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
greens = sns.light_palette("green", as_cmap=True)


mu1.prob_matrix(given=None).shape
mu1.H(...)
mu1.I(C, given=None)
mu1.I(C | SH,S)
# mu1[C | PS, S, SH].style.background_gradient(cmap=greens, axis=None)
# plt.matshow(scores)

#################################################
# TODO: 
# - duplicated edges.
# - 
... is ...
