
from ..pdg import PDG
from ..dist import RawJointDist as RJD

import cvxpy as cp
from cvxpy.constraints.exponential import ExpCone
import numpy as np

from collections.abc import Iterable
from collections import namedtuple
import itertools

from operator import mul
from functools import reduce

# from collections import namedtuple
# Tree
# class TreeCoveredPDG:  pass


# def pgmpy_junction_tree_clusters( M : PDG ):

    

def mk_projector(dshape, IDXs) -> np.array : 
    """
    :param dshape: a shape for the joint distribution, say (d_1, ..., d_n)
    :param IDXs: the list of indices, say [d_{i1}, ... ]
    :return: a numpy array
    """
    nvars = len(dshape)
    IDXs = list(IDXs)
    allIDXs = [i for i in range(nvars)]
    A_proj_IDXs = np.zeros(list(dshape) + [dshape[i] for i in IDXs])
    np.einsum(A_proj_IDXs, allIDXs+IDXs, allIDXs)[:] = 1
    return A_proj_IDXs.reshape(np.prod(dshape), np.prod([dshape[i] for i in IDXs]))

def marginalize_via_projector(mu, M, varis):
    return mu.T @ mk_projector(M.dshape, M._idxs(*varis))

def marginalize(mu, shape, IDXs):
     # to prevent strangeness with raveling for unit, just return the sum
    if len(IDXs) == 0: 
        return cp.sum(mu, keepdims=True)

    postshape = tuple(shape[i] for i in IDXs)
    elts = [0] *  np.prod( postshape, dtype=int) 
        # preallocate list of the appropriate size. Will eventually hold 
        # scalar cp.expressions, which I will aggregate with cp.bmat
    ## will be converted to a cp.expression by operator overloading after addition.

    for i, mui in enumerate(mu):
        v_idx = np.unravel_index(i, shape)  # virtual index
        # print('FLAT: ', i, '\t VIRTUAL: ', v_idx, '\t in shape ', shape)
        idx = np.ravel_multi_index(tuple(v_idx[j] for j in IDXs), postshape)
        # print('  |=> post.virtual = ', tuple(v_idx[j] for j in IDXs), '\t in shape', postshape)
        # print('  |=> flat.idx = ', idx)
        elts[idx] = elts[idx] + mui

    return cp.hstack(elts)

    

def cpd2joint(cpt, mu_X):
    P = cpt.to_numpy()
    # print("P shape: ", P.shape, "\t μ_X shape : ", mu_X.shape)
    return cp.vstack([ P[i,:] * mu_X[i] for i in range(mu_X.shape[0])] ).T    

def n_copies( mu_X, n ):
    """ an analogue of cpd2joint, which just does broadcasting, 
        effectively with a "cpd" that is constant ones. """
    # do a hstack, because we're not iterating over indices.
    return cp.hstack( [ mu_X for _ in range(n) ] ).T
    
def cvx_opt_joint( M : PDG,  also_idef=True) :
    n = np.prod(M.dshape)
    mu = cp.Variable(n, nonneg=True)
    t = { L : cp.Variable(p.to_numpy().size) for (L,p) in M.edges("l,P") if 'π' not in L }
    # t = { L : cp.Variable(n) for L in M.edges("l") if 'π' not in L }
    
    beta_tol_constraints = [
        ExpCone(-t[L], 
            #    mu.T @ mk_projector(M.dshape, M._idxs(X,Y)), 
               marginalize(mu, M.dshape, M._idxs(X,Y)),
               cp.vec(cpd2joint(p, marginalize(mu, M.dshape, M._idxs(X))) ))
            #    cp.vec(cpd2joint(p, mu.T @ mk_projector(M.dshape, M._idxs(X))) )) 
            for L,X,Y,p in M.edges("l,X,Y,P") if 'π' not in L
    ]
    
    prob = cp.Problem( 
        cp.Minimize( sum(βL * sum(t[L]) for βL,L in M.edges("β,l") if 'π' not in L) ),
            [sum(mu) == 1] + beta_tol_constraints)
    prob.solve() 
    

    # now, do the same thing again, but with prob.mu as initialization and new constraints
    if also_idef: 
        # first, save old solution.
        oldmu_dist = RJD(mu.value.copy(), M.varlist) # : np.ndarray
        oldmuPr = oldmu_dist.conditional_marginal
        # conditional marginals must match old ones
        cm_constraints = [
                marginalize(mu, M.dshape, M._idxs(X,Y)) == 
                cp.vec(cpd2joint(oldmuPr(Y|X), marginalize(mu, M.dshape, M._idxs(X)) ))
            for L,X,Y,p in M.edges("l,X,Y,P") if 'π' not in L
        ]
        # the new objective is KL( mu || prod of marginals of oldmu.Pr )
        # Or directly: use 1 instead of p to get only form.
        Pr = oldmu_dist.prob_matrix
        # fp = reduce(mul, [ Pr(Y|X) for X,Y in M.edges("X,Y")])
        fp = np.prod( [ Pr(Y|X)**α for X,Y,α in M.edges("X,Y,α")] )
        tt = cp.Variable(n)

        new_prob = cp.Problem(
            cp.Minimize(sum(tt)),
            cm_constraints + [ ExpCone(-tt, mu, fp.reshape(-1) ), sum(mu) == 1]
        )
        new_prob.solve();
        #########
        ## UPDATE: this doesn't include the final -H(mu) term! How to fix?
        ###
        #  alpha_tol_constraints = [
        #     cp.constraints.exponential.ExpCone(
        #         -t[L], 
        #         marginalize(mu, M.dshape, M._idxs(X,Y)),
        #         cp.vec(n_copies(marginalize(mu, M.dshape, M._idxs(X)), len(Y)) ))
        #     for L,X,Y in M.edges("l,X,Y") if 'π' not in L
        #  ]
        # new_prob = cp.Problem(
        #     cp.Minimize(sum(αL * sum(t[L]) for αL,L in M.edges("α,l") if 'π' not in L)),
        #     cm_constraints + alpha_tol_constraints + [sum(mu) == 1]
        # )
        # new_prob.solve();
        ###########

        
    
    ## save problem, etc as properties of method so you can get them afterwards.
    cvx_opt_joint.prob = prob
    cvx_opt_joint.t = t
    
    return RJD(mu.value, M.varlist)


# like cvx_opt_joint, but in parallel over all clusters, with consistency constraints
def cvx_opt_clusters( M, varname_clusters,  also_idef=True) :
    if(varname_clusters == None):
        varname_clusters = jtree_clusters(M)

    Cs = varname_clusters
    m = len(varname_clusters)
    
    edgemap = {} # label -> cluster index
    # var_clusters = []
    cluster_shapes = [tuple(len(M.vars[Vn]) for Vn in C) for C in Cs]
    
    for L, X, Y in M.edges("l,X,Y"):
        for i,cluster in enumerate(Cs):
            atoms = (X & Y).atoms
            # if all((N.name in cluster or N.is1) for N in atoms):
            if all((N.name in cluster) for N in atoms):
                edgemap[L] = i
                break;
        else:
            raise ValueError("Invalid Cluster Tree: an edge (%s: %s → %s) is not contained in any cluster"
                % (L,X.name,Y.name) )

    mus = [ cp.Variable(np.prod(shape)) for shape in cluster_shapes]
    # mu = cp.Variable(n, nonneg=True) 
    ts = { L : cp.Variable(p.to_numpy().size) for (L,p) in M.edges("l,P") if 'π' not in L }
    # t = { L : cp.Variable(n) for L in M.edges("l") if 'π' not in L }
    
    tol_constraints = []
    for L,X,Y,p in M.edges("l,X,Y,P"):
        if 'π' not in L:
            i = edgemap[L]
            C = varname_clusters[i]            
            
            idxs_XY = [C.index(N.name) for N in (X&Y).atoms]
            idxs_X = [C.index(N.name) for N in X.atoms]
            # idxs_XY = [C.index(N.name) for N in (X&Y).atoms if not N.is1]
            # idxs_X = [C.index(N.name) for N in X.atoms if not N.is1]
            
            # idxs_XY = [j for j,vn in enumerate(C) if M.vars[vn] in (X & Y).atoms]
            # print("For edge %s: %s → %s, in cluster "%(L,X.name,Y.name), C,
            #     "idxs_XY =", idxs_XY)
            # idxs_X = [j for j,vn in enumerate(C) if M.vars[vn] in X.atoms]
            
            expcone = cp.constraints.exponential.ExpCone(-ts[L], 
            #    mus[i].T @ mk_projector(cluster_shapes[i], idxs_XY), 
               marginalize(mus[i], cluster_shapes[i], idxs_XY), 
            #    cp.vec(cpd2joint(p, mus[i].T @ mk_projector(cluster_shapes[i], idxs_X)) )) 
               cp.vec(cpd2joint(p,marginalize(mus[i], cluster_shapes[i], idxs_X)) )) 
# 

            tol_constraints.append(expcone)

    loc_constraints = []
    
    for i in range(m):
        for j in range(m):
            common = set(Cs[i]) & set(Cs[j])
            if len(common) > 0:
                i_idxs = [k for k,vn in enumerate(Cs[i]) if vn in common]
                j_idxs = [k for k,vn in enumerate(Cs[j]) if vn in common]
                # ishareproj = mk_projector(cluster_shapes[i], i_idxs)
                # jshareproj = mk_projector(cluster_shapes[j], j_idxs)
                
                marg_constraint = marginalize(mus[i], cluster_shapes[i], i_idxs)\
                        == marginalize(mus[j], cluster_shapes[j], j_idxs)
                loc_constraints.append(marg_constraint)
    
    prob = cp.Problem( 
        cp.Minimize( sum(βL * sum(ts[L]) for βL,L in M.edges("β,l") if 'π' not in L) ),
            [sum(mus[i]) == 1 for i in range(m)] + tol_constraints + loc_constraints )
    prob.solve()    
    
    print(prob.value)

    if also_idef:
        

        raise NotImplemented
    
    ## save problem, etc as properties of method so you can get them afterwards.
    # return RJD(mu.value, M.varlist)
    return namedtuple("ClusterPseudomarginals", ['marginals', 'value'])(
        marginals= [ RJD(mus[i].value, [M.vars[vn] for vn in C]) for i,C in enumerate(Cs)],
        # prob=prob,
        value=prob.value)
