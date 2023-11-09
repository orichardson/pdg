import numpy as np
import itertools
import functools, operator

from scipy import sparse
from scipy.sparse.linalg import svds as sparse_svd

def prod(factors):
    return functools.reduce(operator.mul, factors, 1)
    

def twiddle( t : tuple, idx, newVal):
    l = list(t)
    l[idx] = newVal
    return tuple(l)


Construct = {"dense": np.zeros, "csc": sparse.csc_matrix, "lil": sparse.lil_matrix }

def matricize(X, n, repr='lil'):
    N = len(X.shape)
    
    M = Construct[repr]((X.shape[n], prod(X.shape[m] for m in range(N) if m != n)), dtype=X.dtype)
    
    for II in itertools.product(*[range(I) for I in X.shape]):
        j = sum(II[k] * prod(X.shape[m] for m in range(k) if m != n) for k in range(N) if k != n)
        M[II[n], j] = X[II]
        
    return M

def modeMult(T, M, n):
    In = T.shape[n]
    retshape = twiddle(T.shape, n, M.shape[0])
    
    result = np.zeros(retshape)
    
    for indices in itertools.product(*[range(I) for I in retshape]):
        for i_n in range(In):
            I = twiddle(indices, n, i_n)            
            result[indices] += T[I] * M[indices[n], i_n]
    
    return result


def tensorAdapt(X, As, transpose = True):
    assert len(As) == len(X.shape), \
        "the order of the tensor (%d) must equal the number of matrices (%d)" %(len(As), len(X.shape))
        
    # we will left-associate to compute this product. It should be associative because the orders are
    # all different.
    Y = X
    for n,A in enumerate(As):
        Y = modeMult(Y, (A.T if transpose else A), n)
    return Y


def leftsing(mat, k, dense=False):
    if k >= min(mat.shape) or dense:
        U, S, V = np.linalg.svd(mat.todense() if sparse.issparse(mat) else mat)
    else:
        U, S, V = sparse_svd(mat.asfptype(), k=k)

    return U[:, :k]    

"""
Computing the tucker decomposition of a tensor X into a core tensor G, plus face
matrices A, B, and C. 
"""
def HOSVD(X, ranks) : # higher order SVD
    As = []
    
    for n,rn in enumerate(ranks):
        mat = matricize(X, n)
        As.append( leftsing(mat, rn) ) # the rn leading left singular vectors of X_n
        
    G = tensorAdapt(X, As)
    return (G, As)
    
    

def HOOI(X, ranks, MAX_ITERS=20, tol=1E-7):
    N = len(X.shape)
    G, As = HOSVD(X, ranks) # Use HOSVD as a starting point
    
    objective = []
    
    for iters in range(MAX_ITERS) :
        for n,rn in enumerate(ranks) :
            Y = X
            for m,A in enumerate(As) :
                if m != n :
                    Y = modeMult(Y, A.T, m)   
        
            As[n] = leftsing( matricize(Y, n), k = rn)
        
        # I only need to compute this at the end, but it makes estimating the objective
        # easier by reusing the code I already have:
        
        GG = tensorAdapt(X, As)
        objective.append( np.linalg.norm(X - tensorAdapt(GG, As, transpose=False)) )
        
        if len(objective) > 1 and abs(objective[-1] - objective[-2]) < tol:
            break;
            
    GG = tensorAdapt(X, As)
    return (GG, As, objective)