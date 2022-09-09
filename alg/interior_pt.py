
from msvcrt import open_osfhandle
from termios import CS6
from ..pdg import PDG
from ..dist import RawJointDist as RJD

import cvxpy as cp
from cvxpy.constraints.exponential import ExpCone

import networkx as nx
import numpy as np

from collections.abc import Iterable
from collections import namedtuple
import itertools 
from operator import mul
from functools import reduce

# from collections import namedtuple
# Tree
# class TreeCoveredPDG:  pass



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
		
	# print(shape,IDXs,end='\n',flush=True)
	##
	
	print('\tmarginalizing : ', mu.size, shape, IDXs)
	II = np.arange(mu.size)
	v_idx = np.unravel_index(II, shape)
	idx = np.ravel_multi_index(tuple(v_idx[j] for j in IDXs), postshape)

	# print('\n\t',end='')
	for i in range(len(elts)):
		print('\t[%d / %d] components' % (i,len(elts)),end='\r' if i < len(elts)-1 else '\n',flush=True)
		idxs_i = np.nonzero((idx==i))
		elts[i] = cp.sum(mu[idxs_i])
	
	# scalar cp.expressions, which I will aggregate with cp.bmat
	# for i, mui in enumerate(mu):
	# 	v_idx = np.unravel_index(i, shape)  # virtual index
	# 	# print('FLAT: ', i, '\t VIRTUAL: ', v_idx, '\t in shape ', shape)
	# 	idx = np.ravel_multi_index(tuple(v_idx[j] for j in IDXs), postshape)
	# 	# print('  |=> post.virtual = ', tuple(v_idx[j] for j in IDXs), '\t in shape', postshape)
	# 	# print('  |=> flat.idx = ', idx)
	# 	elts[idx] = elts[idx] + mui

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

# does the all-at-once	
def cvx_opt_joint( M : PDG,  also_idef=True, **solver_kwargs) :
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
	prob.solve(**solver_kwargs) 
	

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
		new_prob.solve(**solver_kwargs);
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
def cvx_opt_clusters( M, also_idef=True, 
		varname_clusters = None, cluster_edges = None,
		dry_run=False, **solver_kwargs) :
	if(varname_clusters == None):
		print("no clusters given; using pgmpy junction tree to find some.")

		jtree = M.to_markov_net().to_junction_tree()
		varname_clusters = list(jtree.nodes())
		cluster_edges = list(jtree.edges())

		print("FOUND: ",varname_clusters)

	Cs = [tuple(C) for C in varname_clusters]
	cluster_shapes = [tuple(len(M.vars[Vn]) for Vn in C) for C in Cs]
	m = len(varname_clusters)

	if cluster_edges == None:
		complete_graph = nx.Graph()
		for i in range(m):
			for j in range(i+1,m):
				common = set(Cs[i]) & set(Cs[j])
				num_sepset_params = np.prod([len(M.vars[X]) for X in common])

				complete_graph.add_edge(Cs[i], Cs[j], weight=-len(common))
				# complete_graph.add_edge(Cs[i], Cs[j], weight=-num_sepset_params)
				# complete_graph.add_edge(i, j, weight=-num_sepset_params)
		
		cluster_edges = nx.minimum_spanning_tree(complete_graph).edges()

	## Now, we have to make sure that the "running intersection property" (?) 
	# is satisifed. We can't have a cluster tree
	#  (ab)  -- (d) -- (ac)   
	# because the middle node doesn't have a, so we wouldn't enforce marginal constraints
	# properly, were we to only look along edges. So formally, we require that, for every
	# clusters i and j, we have C_i \cap C_j must be contained in every node in the unique
	# path from C_i to C_j in the cluster tree.
	
	edgemap = {} # label -> cluster index
	# var_clusters = []
    
	
	sorted_clusters = sorted(enumerate(Cs), key=lambda iC: np.prod(cluster_shapes[iC[0]]))
	print(sorted_clusters)

	for L, X, Y in M.edges("l,X,Y"):
		# for i,cluster in enumerate(Cs):
		for i,cluster in sorted_clusters:
			atoms = (X & Y).atoms
			# if all((N.name in cluster or N.is1) for N in atoms):
			if all((N.name in cluster) for N in atoms):
				edgemap[L] = i
				break;
		else:
			raise ValueError("Invalid Cluster Tree: an edge (%s: %s → %s) is not contained in any cluster"
				% (L,X.name,Y.name) )

	mus = [ cp.Variable(np.prod(shape)) for shape in cluster_shapes]
	ts = { L : cp.Variable(p.to_numpy().size) for (L,p) in M.edges("l,P") if 'π' not in L }
	
	tol_constraints = []
	k = 0
	for L,X,Y,p in M.edges("l,X,Y,P"):
		if 'π' not in L:
			k += 1
			print("[%d] adding constr "%k, L, ': ', X.name , '->',Y.name, end='\n',flush=True)

			i = edgemap[L]
			# print(L, i, X.name, Y.name)
			C = varname_clusters[i]            
			
			idxs_XY = [C.index(N.name) for N in (X&Y).atoms]
			idxs_X = [C.index(N.name) for N in X.atoms]
			
			print('\tabout to marginalize', end='\n',flush=True)
			xymarginal = marginalize(mus[i], cluster_shapes[i], idxs_XY)
			print('\tabout to construct expcone', end='\n',flush=True)
			print('\tmus[i].shape ', mus[i].shape, end='\n',flush=True)
			expcone = cp.constraints.exponential.ExpCone(-ts[L], 
			#    mus[i].T @ mk_projector(cluster_shapes[i], idxs_XY), 
			   xymarginal, 
			#    cp.vec(cpd2joint(p, mus[i].T @ mk_projector(cluster_shapes[i], idxs_X)) )) 
			   cp.vec(cpd2joint(p,marginalize(mus[i], cluster_shapes[i], idxs_X)) )) 
			print('\texpcone constructed', end='\n',flush=True)
			tol_constraints.append(expcone)

			if dry_run:
				break

	local_marg_constraints = []
	
	# for i in range(m):
	# 	for j in range(i+1,m):
	for Ci, Cj in cluster_edges:
		# common = set(Cs[i]) & set(Cs[j])
		print(Ci,Cj)
		common = set(Ci) & set(Cj)
		if len(common) > 0:
			i, j = Cs.index(Ci), Cs.index(Cj)
			print("adding common constr between (",i,",",j,') : ', common,flush=True)

			i_idxs = [k for k,vn in enumerate(Ci) if vn in common]
			j_idxs = [k for k,vn in enumerate(Cj) if vn in common]
			# ishareproj = mk_projector(cluster_shapes[i], i_idxs)
			# jshareproj = mk_projector(cluster_shapes[j], j_idxs)
			
			marg_constraint = marginalize(mus[i], cluster_shapes[i], i_idxs)\
					== marginalize(mus[j], cluster_shapes[j], j_idxs)
			local_marg_constraints.append(marg_constraint)
	
	prob = cp.Problem( 
		cp.Minimize( sum(βL * sum(ts[L]) for βL,L in M.edges("β,l") if 'π' not in L) ),
			[sum(mus[i]) == 1 for i in range(m)]
			+ [mus[i] >= 0 for i in range(m)]
			+ tol_constraints + local_marg_constraints )

	if dry_run:
		return prob

	prob.solve(**solver_kwargs)    
	
	fp = None
	if also_idef:
		# hmmm do we need to use Bethe entropy? Or Kikuchi approximations? Might not be cvx....
		#  ... unless clusters are tree. But don't we need to prove it's cvx with the type system? 
		old_cluster_rjds = {}
		for i,C in enumerate(Cs):
			old_cluster_rjds[i] = RJD(mus[i].value.copy(), [M.vars[n] for n in C])


		cpds = {}
		cm_constraints = []
		for L,X,Y,p in M.edges("l,X,Y,P"):
			if 'π' not in L:
				i = edgemap[L]
				C = Cs[i]
				sh = cluster_shapes[i]  
				
				idxs_XY = [C.index(N.name) for N in (X&Y).atoms]
				idxs_X = [C.index(N.name) for N in X.atoms]

				cpd = old_cluster_rjds[edgemap[L]].conditional_marginal(Y|X)
				cpds[L] = cpd
				# print("Pr(", Y, "|", X ,")")
				# print()
				# print(cpd)
				# print(p)
				# print()
				# print(old_cluster_rjds[edgemap[L]][X,Y])
				# print()

				cm_constraints.append(
					marginalize(mus[i], sh, idxs_XY)
						==
					cp.vec(cpd2joint(cpd, marginalize(mus[i], sh, idxs_X)))
				)


		## next, we're going to add the IDef terms.
		tts = [ cp.Variable( np.prod(shape)) for shape in cluster_shapes ]
		tol_constraints = []
		# We're going to need the Bethe entropy along the tree, 
		# which involves calculating the sepset beliefs,
		# and to keep the program convex, we need at most one per cluster. 
		# So. First we find a root, which doesn't get such a term,
		# and then apply the appropriate correction term as we traverse the tree.
		print(Cs)
		print(m, 'cluster; \t edges: ', cluster_edges)
		Gr = nx.Graph(cluster_edges).to_directed()
		Gr.add_nodes_from(Cs)
		nx.set_edge_attributes(Gr, 
			{e : {'weight' : np.prod([len(M.vars[x]) for x in e[1]]) }
				for e in Gr.edges() })
		print(Gr.edges(data=True))
		ab = nx.minimum_spanning_arborescence(Gr)


		for i in range(m):
			# fp = np.prod( [np.ones(cluster_shapes[i])]+[old_cluster_rjds[i].prob_matrix(Y|X) **α 
			# 	for X,Y,L,α in M.edges("X,Y,L,α") if edgemap[L] == i] )
			fp = reduce(mul, 
				[np.ones(cluster_shapes[i])]+
				[old_cluster_rjds[i].prob_matrix(Y|X) **α 
					for X,Y,L,α in M.edges("X,Y,L,α") if edgemap[L] == i  
						and 'π' not in L
						] 
			)

			correction_elts = [1] * mus[i].size
			for j in range(m):
				# if i != j: # for use with square roots
				# if i < j: # only single-count corrections?
				# if i < j \
				# 		and ((Cs[i],Cs[j]) in cluster_edges or (Cs[j], Cs[i]) in cluster_edges):
				if (Cs[j], Cs[i]) in ab.edges():
					print("Correction along edge (%d-%d)"%(i,j))
					common = set(Cs[i]) & set(Cs[j])
					idxs = [k for k,vn in enumerate(Cs[i]) if vn in common]
					if len(common) > 0:
						new_term = marginalize(mus[i], cluster_shapes[i], idxs)
						# new_term = cp.sqrt(marginalize(mus[i], cluster_shapes[i], idxs))
						
						# want correction *= new_term, but no broadcasting b/c indices lost
						# ... sooo instead ....
						
						#(a.0) pre-compute the shape of the shared sepset 
						newterm_vshape = tuple(cluster_shapes[i][k] for k in idxs)
						for w,mu_w in enumerate(mus[i]):
							#1. unravel the joint cluster i's world w into a value for each variable
							v_idx = np.unravel_index(w, cluster_shapes[i])
							#2. figure out what the appropriate flattened index into the
							# marginal probability (new_term) should be. 
							idx = np.ravel_multi_index(tuple(v_idx[j] for j in idxs), newterm_vshape)
							correction_elts[w] *= new_term[idx]
						# (b.0) prepcompute the shape of shared subset
						# newterm_vshape = tuple(cluster_shapes[i][k] for k in idxs)
						# II  = np.arange(mus[i].size)
						# v_idx = np.unravel_index(II , cluster_shapes[i])
						# idx = np.ravel_multi_index(tuple(v_idx[k] for k in idxs), newterm_vshape)
						# correction_elts

						## the below is taken from marginalize for context.
						# II = np.arange(mu.size)
						# v_idx = np.unravel_index(II, shape)
						# idx = np.ravel_multi_index(tuple(v_idx[j] for j in IDXs), postshape)
						# # print('\n\t',end='')
						# for i in range(len(elts)):
						# 	print('\t[%d / %d] components' % (i,len(elts)),end='\r',flush=True)
						# 	idxs_i = np.nonzero((idx==i))
						# 	elts[i] = cp.sum(mu[idxs_i])

			correction = cp.hstack(correction_elts)
			print(Cs[i], cluster_shapes[i])
			print('correction shape: ', correction.shape)
			print(correction.is_dcp())
			print(correction.is_affine())
			print('fp.shape: ', fp.shape)
			
			tol_constraints.append(ExpCone(
				-tts[i],
				mus[i],
				# fp.reshape(-1) * correction ## FIXME this doesn't work
				cp.multiply(fp.reshape(-1), correction)
				# fp.reshape(-1)
			));

		new_prob =cp.Problem(
			cp.Minimize(sum(sum(tts[i]) for i in range(m))),
			tol_constraints + 
			cm_constraints + # conditional marginals
			local_marg_constraints +  # marg constraints
			[sum(mus[i]) == 1 for i in range(m)]
			# the positivity constraint below is redundant because of the expcone shapes.
			# + [mus[i] >= 0 for i in range(m)]
		)
		new_prob.solve(**solver_kwargs)
		# raise NotImplemented
	
	## save problem, etc as properties of method so you can get them afterwards.
	# return RJD(mu.value, M.varlist)
	return namedtuple("ClusterPseudomarginals", ['marginals', 'value'])(
		marginals= [ RJD(mus[i].value, [M.vars[vn] for vn in C]) for i,C in enumerate(Cs)],
		# prob=prob,
		# fp = fp,
		value=prob.value)


# def jtree_clusters(M : PDG):
# 	return M.to_markov_net().to_junction_tree().nodes();[ ]

def cvx_opt_qq_joint(M, gamma=1, **solver_kwargs): 
	"""both the qualitative, and the quantitative terms together""" 
    
	raise NotImplemted

	n = np.prod(M.dshape)
	mu = cp.Variable(n, nonneg=True)
	tb = { L : cp.Variable(p.to_numpy().size) for (L,p) in M.edges("l,P") if 'π' not in L }
	ta = { L : cp.Variable(p.to_numpy().size) for (L,p) in M.edges("l,P") if 'π' not in L }
	# t = { L : cp.Variable(n) for L in M.edges("l") if 'π' not in L }
	
	beta_tol_constraints = [
		ExpCone(-tb[L], 
			#    mu.T @ mk_projector(M.dshape, M._idxs(X,Y)), 
			   marginalize(mu, M.dshape, M._idxs(X,Y)),
			   cp.vec(cpd2joint(p, marginalize(mu, M.dshape, M._idxs(X))) ))
			#    cp.vec(cpd2joint(p, mu.T @ mk_projector(M.dshape, M._idxs(X))) )) 
			for L,X,Y,p in M.edges("l,X,Y,P") if 'π' not in L
	]


	## TODO VET ME
	alpha_tol_constraints = [
		ExpCone(-ta[L], 
			#    mu.T @ mk_projector(M.dshape, M._idxs(X,Y)), 
				marginalize(mu, M.dshape, M._idxs(X,Y)),
                ### TODO FIXME <<<<~!!!
				cp.vec(cpd2joint(np.ones(...), marginalize(mu, M.dshape, M._idxs(X))) ))
			for L,X,Y,p in M.edges("l,X,Y,P") if 'π' not in L
	]

	
	prob = cp.Problem( 
		cp.Minimize( 
			sum(βL * sum(tb[L]) for βL,L in M.edges("β,l") if 'π' not in L) 
			+ sum(αL * sum(tα[L]) for αL,L in M.edges("α,l") if 'π' not in L) ),
		[sum(mu) == 1] 
			+ beta_tol_constraints + alpha_tol_constraints
	)
	prob.solve(**solver_kwargs) 


	## save problem, etc as properties of method so you can get them afterwards.
	cvx_opt_joint.prob = prob
	cvx_opt_joint.t = t
	
	return RJD(mu.value, M.varlist)