# [ ] 

import numpy as np
from operator import mul
import math

def primes():
    i = 1
    # for counter in range(100):
    while(True):
        i += 1
        for j in range(2,min(int(math.sqrt(i))+1,i)):
            if i % j == 0:
                break
        else:
            yield i
            
from ..pdg import PDG
from ..rv import Variable as Var
from ..dist import CPT, RawJointDist as RJD

_loc = locals()

def _ns_save(v : Var):
    _loc[v.name] = v

for (N,p) in zip('ABCDE', primes()) :
    # _ns_save( Var([N.lower()+str(i) for i in range(p)], name=N, default_value=N.lower()+str('0')) )
    _ns_save(Var.alph(N, p))

for c in range(ord('A'), ord('L')):
    for n in range(2,4): # [2,4) = [2, 3]
        _ns_save (Var.alph(chr(c)+str(n), n))
