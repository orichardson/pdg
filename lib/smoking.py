# %load_ext autoreload
# %autoreload 2

# import numpy as np
# import pandas as pd

from dist import CPT
from rv import binvar, Unit
from pdg import PDG

############""" Example Code """##############

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
      
# μ = M.factor_product(repr='atomic')
# M.Inc(μ)
# list(M.edges("Xn Yn α"))
# M.IDef(μ, ed_vector=True)
# μ.H(S,SH, PS, C)
# 
# 
# M.IDef(μ, ed_vector=True).sum()
# μ.I(C, PS | S,SH)
# # observe the un-blocking of D-separation....
# μ.I(S,SH | PS)
# μ.I(S, SH | PS, C)
# 
# 
# 
# # Right now it's just a the BN distribution we expect
# T = binvar("T")
# M += T
