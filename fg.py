from operator import mul
from functools import reduce
import numpy as np

from .rv import Variable as Var
from .dist import CPT, RawJointDist as RJD

# from typing import Iterable
from operator import mul
from functools import reduce



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
    def dist(self, use_torch=False):
        """ warning: scales exponentially! """
        data = reduce(mul, self.factors)
        data /= data.sum()
        
        return RJD(data, self.vars, use_torch=use_torch)
    
    @property    
    def Z(self):
        """ warning: scales exponentially !"""
        return reduce(mul, self.factors).sum()
        
    
    # def neighbors(vars):
    #     return 
        
    
    def gibbs_marginal_estimate(self, vars, init_sample=None, iters=1):
        sample = init_sample if init_sample is not None \
            else {v.name : np.random.choice(v) for v in self.vars}
        
        for it in range(iters):
            for v in self.vars: # go over ALL variables;
                # sample from product of neighbor functions
                pass
                
                
        # esimate Pr(vars)    
                
            
        
        
        
        
class ExpFam(FactorGraph):
    pass
    
        
        
            
