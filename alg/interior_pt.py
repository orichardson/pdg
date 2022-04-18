
from ..pdg import PDG
import cvxpy as cp
import numpy as np


class TreeCoveredPDG:  pass


def cvx_opt( treeM : TreeCoveredPDG ) :
    pass
    
    
def marginals(mu, varshape) :
    
    
def cvx_opt_component( M : PDG ) :
    n = np.prod(M.dshape)
    mu = cp.Variable(n, nonneg=True)
    t = { L : cp.Variable(n) for L in M.edges("l") }
    
    tol_constraints = [
        cp.constraints.exponential.ExpCone(-t[L], mu, p) 
            for L,X,Y,p in M.edges("l,Xn,Yn,P")
    ]
    
    prob = cp.Problem( 
        cp.Minimize( sum(βL * sum(t[L]) for βL,L in M.edges("β,l") ) ),
            [sum(mu) == 1] + tol_constraints )
    prob.solve()    
    
    return mu.value
