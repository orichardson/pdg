# %load_ext autoreload
# %autoreload 2
import numpy as np
from operator import mul
from functools import reduce

from lib import A,B,C,D
from pdg import PDG
from dist import RawJointDist as RJD, CPT


_base = PDG()
_base += A,B,C,D

# P : RJD = RJD.random(_base.varlist)

with_rand_cpts : PDG  = _base.copy()
consist_with_P : PDG = _base.copy()
missing_parents : PDG = _base.copy()

adj = {'AB', 'BD', 'CD', 'AC'}
ϕ = [np.random.random(tuple((len(v) if v.name in ed else 1) for v in _base.varlist)) for ed in adj]

P = RJD(reduce(mul, ϕ), [A,B,C,D]).normalize()

# P.I(A,D | B,C)
Ed = [ ( _base(' '.join(sorted(x for X in adj for x in X if Y in X if x != Y))),
        _base(Y) ) for Y in 'ABCD' ]


# random edges
for PaY,Y in Ed:
    with_rand_cpts += CPT.make_random(PaY, Y)
    consist_with_P += P[Y | PaY]

# _prodpfactors = consist_with_P.factor_product()
from numpy.random import choice

for PaY,Y in Ed:
    # remove a random parent
    parent_names = set(PaY.name.split('×'))
    n2kill = choice([*parent_names])
    SubPaY = _base( ' '.join(parent_names - { n2kill } ))
    missing_parents += P[Y | SubPaY]
