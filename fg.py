from operator import mul
from functools import reduce
from itertools import combinations
import numpy as np

from .rv import Variable as Var
from .dist import CPT, RawJointDist as RJD

# from typing import Iterable

from pgmpy.models import MarkovNetwork

def canonical_varlist(factors):
    subs = '₀₁₂₃₄₅₆₇₈₉'
    return [Var.alph('X%s'%''.join(subs[int(d)] for d in str(i)), n) for (i,n) 
        in enumerate(reduce(mul,factors).shape)]

class FactorGraph:
    """ 
    TODO :: What about densitites? Variables that are not 
        finite? Or generally, implicit variables?
    """
    
    def __init__(self, factors, varlist='generate'):
        self._varlist = varlist
        self.factors = factors
        
        if varlist == 'generate':
            self._varlist = canonical_varlist(factors)
        
    @property
    def vars(self):
        if self._varlist != None:
            return self._varlist
            # Test to make sure the right dimensions
            ## actually, don't. Might be expensive for no reason.
            # for i,v in enumerate(self._varlist):
            #     assert 
        else:
            return canonical_varlist(self.factors)
            
        
    @property
    def dist(self, **rjd_kwarg):
        """ warning: scales exponentially! """
        data = reduce(mul, self.factors)
        data /= data.sum()
        
        return RJD(data, self.vars, **rjd_kwarg)
    
    @property    
    def Z(self):
        """ warning: scales exponentially !"""
        return reduce(mul, self.factors).sum()

    def scope_card(self, f, names=True):
        return zip(*[(v.name if names else v, len(v)) 
                for (i,v) in enumerate(self._varlist) if f.shape[i] > 1])
    
    def scope(self, f, names=True):
        scope, _ = self.scope_card(f,names)
        return list(scope)
        
    
    # def neighbors(vars):
    #     return 
        
    
    def gibbs_marginal_estimate(self, vars, init_sample=None, iters=1):
        sample = init_sample if init_sample is not None \
            else {v.name : np.random.choice(v) for v in self.vars}
            
        raise NotImplemented
        
        for it in range(iters):
            for v in self.vars: # go over ALL variables;
                # sample from product of neighbor functions
                pass
                
                
        # esimate Pr(vars)    
                
    def to_pgmpy_markov_net(self):
        from pgmpy.models import MarkovNetwork
        from pgmpy.factors.discrete import DiscreteFactor
        
        mn = MarkovNetwork()

        mn.add_nodes_from([V.name for V in self._varlist])
        for f in self.factors:
            # print([(i,v) for (i,v) in enumerate(self._varlist) if f.shape[i] > 1])
            scope, card = zip(*[(v.name, len(v)) 
                for (i,v) in enumerate(self._varlist) if f.shape[i] > 1])

            mn.add_edges_from(combinations(scope,2));
            mn.add_factors( DiscreteFactor(scope, card, f ))

        # raise NotImplemented
        return mn
            
        
        
        
        
class ExpFam(FactorGraph):
    pass

