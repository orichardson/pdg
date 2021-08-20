import numpy as np
# import pandas as pd

from dist import RawJointDist, CPT
from rv import Variable, binvar, Unit
from pdg import PDG


# Graph: 1 -> A,  1 -> B, AB -> A, AB -> B
M = PDG()
A = Variable.alph('A', 3) 
B = Variable.alph('B', 2) 

M += CPT.from_matrix(Unit, A, [[.2, .3, .5]])
M += CPT.from_matrix(Unit, B, [[.1, .9]])

d = M.factor_product()
d[A|B]

M += A * B

# All of these look good inline.
M[A|Unit]
M[A | A*B]
M[B | A*B]


d2 = M.factor_product()

M.edgedata.keys()
assert(np.allclose(d2[A|Unit], d[A|Unit]))
assert(np.allclose(d2[Unit,A,B], d[Unit,A,B]))
# 
# for X,Y,cpd,*_ in M:
#     print(X,Y,cpd)
