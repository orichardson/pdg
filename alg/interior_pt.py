
from ..pdg import PDG
from ..dist import RawJointDist as RJD

import networkx as nx
import numpy as np
import cvxpy as cp
from cvxpy.constraints.exponential import ExpCone

from collections.abc import Iterable
from collections import namedtuple
from operator import mul
from functools import reduce
import itertools 

############### UTILITY METHODS ############

def _mk_projector(dshape, IDXs) -> np.array : 
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

def _marginalize_via_projector(mu, M, varis):
	return mu.T @ _mk_projector(M.dshape, M._idxs(*varis))

def _marginalize(mu, shape, IDXs):
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
	
	# print('\tmarginalizing : ', mu.size, shape, IDXs)
	II = np.arange(mu.size)
	v_idx = np.unravel_index(II, shape)
	idx = np.ravel_multi_index(tuple(v_idx[j] for j in IDXs), postshape)

	# print('\n\t',end='')
	for i in range(len(elts)):
		# print('\t[%d / %d] components' % (i,len(elts)),end='\r' if i < len(elts)-1 else '\n',flush=True)
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

def _cpd2joint(cpt, mu_X):
	P = cpt.to_numpy()
	# print("P shape: ", P.shape, "\t μ_X shape : ", mu_X.shape)
	return cp.vstack([ P[i,:] * mu_X[i] for i in range(mu_X.shape[0])] ).T    

def _cpd2joint_np(cpt, mu_X):
	P = cpt
	# print("P shape: ", P.shape, "\t μ_X shape : ", mu_X.shape)
	return cp.vstack([ P[i,:] * mu_X[i] for i in range(mu_X.shape[0])] ).T    

def _combine_iters(it1, it2, sel): 
	i1 = iter(it1)
	i2 = iter(it2)
	for s in sel: 
		if s: yield next(i1) 
		else: yield next(i2)

def _dup2shape(cvxpy_expr, shape, idxs):
	"""takes a cvxpy expression `cvxpy_expr` representing flattened marginals along dimensions `idxs` in
	the context of a joint n-dimensional array of shape `shape`, and
	:returns: a version of cvxpy_expr suitable for element-wise multiplication, i.e., of that same joint shape."""
	n = np.prod(shape)
	expr_vshape = tuple(shape[i] for i in idxs)
	m = np.prod(expr_vshape)

	assert m == cvxpy_expr.size, "indices " + str(tuple(shape[i] for i in idxs)) + \
		" not compatible with expression of size "+ str(cvxpy_expr.size)

	unwound = list(cvxpy_expr)
	
	# not_idx = tuple(i for i in range(len(shape)) if i not in idxs)
	not_shape = tuple(shape[i] for i in range(len(shape)) if i not in idxs)

	expr_coords = np.unravel_index(np.arange(m), expr_vshape)
	to_ret = np.zeros(shape, dtype='object')
	

	# prematurely "optimized" because I didn't want to iterate
	# through everything
	for non_coords in np.ndindex(*not_shape):
		full_coords = tuple(_combine_iters(
			expr_coords,
			non_coords, 
			[i in idxs for i in range(len(shape))]
		))
		flat_full_coords = np.ravel_multi_index(full_coords, shape)
		np.put(to_ret, flat_full_coords, unwound)

	# put everything back together. Better than iterating? Who knows...
	return cp.hstack(to_ret.reshape(-1))

def _n_copies( mu_X, n ):
	""" an analogue of cpd2joint, which just does broadcasting, 
		effectively with a "cpd" that is constant ones. """
	# do a hstack, because we're not iterating over indices.
	return cp.hstack( [ mu_X for _ in range(n) ] ).T

############# INFERENCE ALGORITHMS ############
# optimizes over joint distributions
def cvx_opt_joint( M : PDG,  also_idef=True, **solver_kwargs) :
	n = np.prod(M.dshape)
	mu = cp.Variable(n, nonneg=True)
	t = { L : cp.Variable(p.to_numpy().size) for (L,p) in M.edges("l,P") if 'π' not in L }
	# t = { L : cp.Variable(n) for L in M.edges("l") if 'π' not in L }
	
	beta_tol_constraints = [
		ExpCone(-t[L], 
			#    mu.T @ _mk_projector(M.dshape, M._idxs(X,Y)), 
			   _marginalize(mu, M.dshape, M._idxs(X,Y)),
			   cp.vec(_cpd2joint(p, _marginalize(mu, M.dshape, M._idxs(X))) ))
			#    cp.vec(_cpd2joint(p, mu.T @ _mk_projector(M.dshape, M._idxs(X))) )) 
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
				_marginalize(mu, M.dshape, M._idxs(X,Y)) == 
				cp.vec(_cpd2joint(oldmuPr(Y|X), _marginalize(mu, M.dshape, M._idxs(X)) ))
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
		#         _marginalize(mu, M.dshape, M._idxs(X,Y)),
		#         cp.vec(_n_copies(_marginalize(mu, M.dshape, M._idxs(X)), len(Y)) ))
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
def cvx_opt_clusters( M : PDG, also_idef=True, 
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

	mus = [ cp.Variable(np.prod(shape), nonneg=True) for shape in cluster_shapes]
	ts = { L : cp.Variable(p.to_numpy().size) for (L,p) in M.edges("l,P") if 'π' not in L }
	
	tol_constraints = []
	k = 0
	for L,X,Y,p in M.edges("l,X,Y,P"):
		if 'π' not in L:
			k += 1
			# print("[%d] adding constr "%k, L, ': ', X.name , '->',Y.name, end='\n',flush=True)

			i = edgemap[L]
			# print(L, i, X.name, Y.name)
			C = varname_clusters[i]            
			
			idxs_XY = [C.index(N.name) for N in (X&Y).atoms]
			idxs_X = [C.index(N.name) for N in X.atoms]
			
			# print('\tabout to marginalize', end='\n',flush=True)
			xymarginal = _marginalize(mus[i], cluster_shapes[i], idxs_XY)
			# print('\tabout to construct expcone', end='\n',flush=True)
			# print('\tmus[i].shape ', mus[i].shape, end='\n',flush=True)
			expcone = cp.constraints.exponential.ExpCone(-ts[L], 
			#    mus[i].T @ _mk_projector(cluster_shapes[i], idxs_XY), 
			   xymarginal, 
			#    cp.vec(_cpd2joint(p, mus[i].T @ _mk_projector(cluster_shapes[i], idxs_X)) )) 
			   cp.vec(_cpd2joint(p,_marginalize(mus[i], cluster_shapes[i], idxs_X)) )) 
			# print('\texpcone constructed', end='\n',flush=True)
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
			# print("adding common constr between (",i,",",j,') : ', common,flush=True)

			i_idxs = [k for k,vn in enumerate(Ci) if vn in common]
			j_idxs = [k for k,vn in enumerate(Cj) if vn in common]
			# ishareproj = _mk_projector(cluster_shapes[i], i_idxs)
			# jshareproj = _mk_projector(cluster_shapes[j], j_idxs)
			
			marg_constraint = _marginalize(mus[i], cluster_shapes[i], i_idxs)\
					== _marginalize(mus[j], cluster_shapes[j], j_idxs)
			local_marg_constraints.append(marg_constraint)
	
	prob = cp.Problem( 
		cp.Minimize( sum(βL * sum(ts[L]) for βL,L in M.edges("β,l") if 'π' not in L) ),
			[sum(mus[i]) == 1 for i in range(m)]
			# + [mus[i] >= 0 for i in range(m)] ## subsumed by nonneg = True
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
					_marginalize(mus[i], sh, idxs_XY)
						==
					cp.vec(_cpd2joint(cpd, _marginalize(mus[i], sh, idxs_X)))
				)


		## next, we're going to add the IDef terms.
		tts = [ cp.Variable( np.prod(shape)) for shape in cluster_shapes ]
		tol_constraints = []
		# We're going to need the Bethe entropy along the tree, 
		# which involves calculating the sepset beliefs,
		# and to keep the program convex, we need at most one per cluster. 
		# So. First we find a root, which doesn't get such a term,
		# and then apply the appropriate correction term as we traverse the tree.
		# print(Cs)
		# print(m, 'cluster; \t edges: ', cluster_edges)
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
					# print("Correction along edge (%d-%d)"%(i,j))
					common = set(Cs[i]) & set(Cs[j])
					idxs = [k for k,vn in enumerate(Cs[i]) if vn in common]
					if len(common) > 0:
						new_term = _marginalize(mus[i], cluster_shapes[i], idxs)
						# new_term = cp.sqrt(_marginalize(mus[i], cluster_shapes[i], idxs))
						
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
			# print(Cs[i], cluster_shapes[i])
			# print('correction shape: ', correction.shape)
			# print(correction.is_dcp())
			# print(correction.is_affine())
			# print('fp.shape: ', fp.shape)
			
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
			# the positivity constraint below is redundant because of the expcone shapes (and also nonneg=True)
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

# custom implementation of the CCCP
def cccp_opt_joint(M, gamma=1, max_iters=20, **solver_kwargs): 
	"""both the qualitative, and the quantitative terms together.
	Custom implementation of the cccp.""" 
	n = np.prod(M.dshape)
	mu = cp.Variable(n, nonneg=True)

	# keys for both dicts (s, t) are the "convex" edge labels,
	#   i.e., those with β >= α γ
	t = {}
	s = {} 

	def mu_marg(*varis):
		return _marginalize(mu, M.dshape, M._idxs(*varis))
	

	cvx_tol_constraints = []
	cave_edges = []

	for L,X,Y,α,β,p in M.edges("l,X,Y,α,β,P"):
		if 'π' in L: continue
		sL = (β - gamma * α)

		if sL >= 0: # this part is convex
			print("(vex)   ", L, '\t', sL)

			t[L] = cp.Variable(p.to_numpy().size)
			cone = ExpCone(-t[L], 
					mu_marg(X,Y),
					cp.vec(_cpd2joint_np(np.ones(p.shape), mu_marg(X)))	) 
			s[L] = sL
			cvx_tol_constraints.append(cone)

		else: # this part is concave
			cave_edges.append((L,X,Y,sL))
			print("(cave)  ", L, '\t', sL)


	from ..dist import zz1_div, z_mult, D_KL

	def g(x : RJD) : # concave part (actual val just for printing really)
		val = 0
		for L,X,Y,sL in cave_edges:
			muxy = x.prob_matrix(X,Y)
			mux = x.prob_matrix(X)
			vmat = sL * z_mult( muxy , np.ma.log( np.ma.divide(muxy, mux) ))
			val += vmat.sum()
		return val 

	def grad_g(x : RJD): # the gradient is more important
		val = np.zeros(x.data.shape)

		for L,X,Y,sL in cave_edges:
			muxy = x.prob_matrix(X,Y)
			mux = x.prob_matrix(X)
			# vmat = sL * z_mult( muxy , np.ma.log( np.ma.divide(muxy, mux) ))
			vmat = sL *  np.ma.log( np.ma.divide(muxy, mux))
			val += vmat - z_mult(muxy, vmat).sum()

		return val.reshape(-1)


	hard_constraints = []
	logprobs = 0
	for L,X,Y,β,p in M.edges("l,X,Y,β,P"):
		if 'π' not in L:
			p = p.to_numpy()
			zero = (p == 0)
			# lp = - np.ma.log(p)
			lp = - np.ma.log(p)

			if(np.any(zero)):
				lp = np.where(zero, 0, lp)

				hard_constraints.append(
					mu_marg(X,Y)[np.argwhere(p.reshape(-1)==0)] == 0
				)

			lin_lp = cp.sum( cp.multiply(lp.reshape(-1), mu_marg(X,Y)) )
			logprobs += β * lin_lp



	## add entropy constraints
	global_t_ent = cp.Variable(n)
	prev_val = np.inf

	for it in range(max_iters):
		if mu.value is None:
			frozen = M.genΔ(RJD.unif)
		else:
			frozen = RJD(mu.value.copy(), M.varlist)


		linearized = cp.sum(
			cp.multiply((mu - frozen.data.reshape(-1)), grad_g(frozen))
		)
		prob = cp.Problem( 
			cp.Minimize( 
				logprobs +
				linearized + 
				sum( s[L] * sum(t[L]) for L,tL in t.items() ) 
				+ gamma * cp.sum(global_t_ent) # entropy term
			),
			cvx_tol_constraints 
				+ hard_constraints
				+ [sum(mu) == 1]
				+ [ ExpCone(-global_t_ent, mu, np.ones(mu.shape)) ] 
		)

		prob.solve(**solver_kwargs)

		print('obj: ', prob.value + g(frozen), '\t\t tv distance', np.max(np.absolute(mu.value-frozen.data.reshape(-1)))  )
		if(prob.value == prev_val) or (np.allclose(mu.value, frozen.data.reshape(-1), rtol=1E-4, atol=1E-8)):
			break
		prev_val = prob.value

	return RJD(mu.value, M.varlist)

# custom CCCP but with the frozen variable as a parameter.
# Currently is slower because the problem is not DPP in parameter :(
def cccp_opt_joint_parameterized(M, gamma=1, max_iters=20, **solver_kwargs): 
	""" CCCP algorithm for \gamma > 0, with a parameter. 
	Currently oes not result in a speed-up because the problem does not
	satisfy the DPP grammar in the parameter. """ 
	n = np.prod(M.dshape)
	mu = cp.Variable(n, nonneg=True)

	# keys for both dicts (s, t) are the "convex" edge labels,
	#   i.e., those with β >= α γ
	t = {}
	s = {} 

	def mu_marg(*varis):
		return _marginalize(mu, M.dshape, M._idxs(*varis))
	

	cvx_tol_constraints = []
	cave_edges = []

	for L,X,Y,α,β,p in M.edges("l,X,Y,α,β,P"):
		if 'π' in L: continue
		sL = (β - gamma * α)

		if sL >= 0: # this part is convex
			# print("(vex)   ", L, '\t', sL)
			t[L] = cp.Variable(p.to_numpy().size)
			cone = ExpCone(-t[L], 
					mu_marg(X,Y),
					cp.vec(_cpd2joint_np(np.ones(p.shape), mu_marg(X)))	) 
			s[L] = sL
			cvx_tol_constraints.append(cone)

		else: # this part is concave
			# print("(cave)  ", L, '\t', sL)
			cave_edges.append((L,X,Y,sL))


	def g_p(nu) : # concave part, for cvxpy expressions
		val = 0
		for L,X,Y,sL in cave_edges:
			xyind = M._idxs(X,Y)
			nuxy = _marginalize(nu, M.dshape, xyind)
			nux = _marginalize(nu, M.dshape, M._idxs(X))
			ones = np.ones((len(X), len(Y)))
			nux_same = cp.vec(_cpd2joint_np(ones, nux))

			vmat = sL * cp.multiply(nuxy, cp.log( nuxy ) - cp.log(nux_same) )
			val += cp.sum(vmat)
		return val 
	
	def grad_g_p(nu):
		val = np.zeros(nu.shape)

		for L,X,Y,sL in cave_edges:
			xyind = M._idxs(X,Y)
			nuxy = _marginalize(nu, M.dshape, xyind) # affine in nu
			nux = _marginalize(nu, M.dshape, M._idxs(X)) # affine in nu
			ones = np.ones((len(X), len(Y)))
			nux_same = cp.vec(_cpd2joint_np(ones, nux)) # should also be affine in nu

			local_mat = sL * (cp.log( nuxy ) - cp.log(nux_same)) # definitely not affine in nu
			vmat = _dup2shape(local_mat, M.dshape, xyind)
			val += vmat - cp.sum(cp.multiply(nuxy, local_mat))

		return val


	hard_constraints = []
	logprobs = 0
	for L,X,Y,β,p in M.edges("l,X,Y,β,P"):
		if 'π' not in L:
			p = p.to_numpy()
			zero = (p == 0)
			# lp = - np.ma.log(p)
			lp = - np.ma.log(p)

			if(np.any(zero)):
				lp = np.where(zero, 0, lp)

				hard_constraints.append(
					mu_marg(X,Y)[np.argwhere(p.reshape(-1)==0)] == 0
				)

			lin_lp = cp.sum( cp.multiply(lp.reshape(-1), mu_marg(X,Y)) )
			logprobs += β * lin_lp



	## add entropy constraints
	global_t_ent = cp.Variable(n)
	prev_val = np.inf

	frozen = cp.Parameter(mu.size)

	linearized = cp.sum(
		cp.multiply((mu-frozen), grad_g_p(frozen))
	)
	prob = cp.Problem( 
		cp.Minimize( 
			logprobs +
			linearized + 
			sum( s[L] * sum(t[L]) for L,tL in t.items() ) 
			+ gamma * cp.sum(global_t_ent) # entropy term
		),
		cvx_tol_constraints 
			+ hard_constraints
			+ [sum(mu) == 1]
			+ [ ExpCone(-global_t_ent, mu, np.ones(mu.shape)) ] 
	)

	for it in range(max_iters):
		if mu.value is None:
			frozen.value = M.genΔ(RJD.unif).data.reshape(-1)
		else:
			frozen.value = mu.value.copy()

		prob.solve(**solver_kwargs)

		# print(prob.value + g_p(frozen.value))
		# if 'verbose' in solver_kwargs and solver_kwargs['verbose']:
		print('obj: ', prob.value + g_p(frozen.value), 
			'\t\t tv distance', np.sum(np.absolute(mu.value-frozen.data.reshape(-1)))  )
		if(prob.value == prev_val) or \
				np.sum(np.absolute(mu.value-frozen.data.reshape(-1))) < 1E-6:
			# (np.allclose(mu.value, frozen.value, rtol=1E-6, atol=1E-9)):
			break
		prev_val = prob.value
			
	# prob.solve(method='dccp', **solver_kwargs) 
	# prob.solve(**solver_kwargs)

	return RJD(mu.value, M.varlist)


# best of all worlds: cccp on top of clusters
def cccp_opt_clusters( M : PDG, gamma=1, max_iters=20,
		varname_clusters = None, cluster_edges = None,
		**solver_kwargs) :
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

	
	sorted_clusters = sorted(enumerate(Cs), key=lambda iC: np.prod(cluster_shapes[iC[0]]))
	print(sorted_clusters)

	# assign each edge L to a cluster, prioritizing smaller clusters.
	edgemap = {} # label -> cluster index
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

	mus = [ cp.Variable(np.prod(shape), nonneg=True) for shape in cluster_shapes ]
	ts = {}
	ss = {}
	
	k = 0
	cvx_tol_constraints = []
	hard_constraints = []
	cave_edges = []
	logprobs = 0

	for L,X,Y,α,β,p in M.edges("l,X,Y,α,β,P"):
		if 'π' in L: continue
		sL = (β - gamma * α)

		i = edgemap[L]
		C = Cs[i]

		idxs_XY = [C.index(N.name) for N in (X&Y).atoms]
		idxs_X = [C.index(N.name) for N in X.atoms]

		mu_xy = _marginalize(mus[i], cluster_shapes[i], idxs_XY)
		mu_x = _marginalize(mus[i], cluster_shapes[i], idxs_X)

		if sL >= 0: # this part is convex
			# print("(vex)   ", L, '\t', sL)
			ts[L] = cp.Variable(p.to_numpy().size)
			cone = ExpCone(-ts[L], 
						   mu_xy,
						   cp.vec(_cpd2joint_np(np.ones(p.shape), mu_x))	) 
			ss[L] = sL
			cvx_tol_constraints.append(cone)

		else: # this part is concave
			# print("(cave)  ", L, '\t', sL)
			cave_edges.append((L,X,Y,sL))

		# either way, we then add the linear part.
		p = p.to_numpy().reshape(-1)
		zero = (p == 0)
		lp = - np.ma.log(p)

		if np.any(zero):
			lp = np.where(zero, 0, lp)
			hard_constraints.append( mu_xy[np.argwhere(zero)] == 0 )

		logprobs += β * cp.sum( cp.multiply(lp, mu_xy ) )


	local_marg_constraints = []
	
	for Ci, Cj in cluster_edges:
		common = set(Ci) & set(Cj)
		if len(common) > 0:
			i, j = Cs.index(Ci), Cs.index(Cj)
			# print("adding common constr between (",i,",",j,') : ', common,flush=True)

			i_idxs = [k for k,vn in enumerate(Ci) if vn in common]
			j_idxs = [k for k,vn in enumerate(Cj) if vn in common]
			
			local_marg_constraints.append(
					_marginalize(mus[i], cluster_shapes[i], i_idxs)
						== 
					_marginalize(mus[j], cluster_shapes[j], j_idxs) )
	

	from ..dist import z_mult

	def g( nus ) : # concave part (actual val just for printing really)
		val = 0
		for L,X,Y,sL in cave_edges:
			i = edgemap[L]
			muxy = nus[i].prob_matrix(X,Y)
			mux = nus[i].prob_matrix(X)
			vmat = sL * z_mult( muxy , np.ma.log( np.ma.divide(muxy, mux) ))
			val += vmat.sum()
		return val 

	def grad_g( nus ): # the gradient is more important
		# map(print, nus)
		vals = [ np.zeros(nu.data.shape) for nu in nus ]

		for i,nu in enumerate(nus):
			for L,X,Y,sL in cave_edges:
				# if not all((N.name in Cs[i]) for N in (X&Y).atoms): continue
				## contrast with: 
				if i != edgemap[L]: continue
				## are they different?

				nuxy = nu.prob_matrix(X,Y)
				nux = nu.prob_matrix(X)
				# vmat = sL * z_mult( muxy , np.ma.log( np.ma.divide(muxy, mux) ))
				vmat = sL *  np.ma.log( np.ma.divide(nuxy, nux))
				vals[i] += vmat - z_mult(nuxy, vmat).sum()
			
			vals[i] = vals[i].reshape(-1)

		return vals

	## add entropy constraints
	tts_ent = [ cp.Variable( np.prod(shape)) for shape in cluster_shapes ]
	ent_tol_constraints = []

	Gr = nx.Graph(cluster_edges).to_directed()
	Gr.add_nodes_from(Cs)
	nx.set_edge_attributes(Gr, 
		{e : {'weight' : np.prod([len(M.vars[x]) for x in e[1]]) }
			for e in Gr.edges() })
	print(Gr.edges(data=True))
	ab = nx.minimum_spanning_arborescence(Gr)
	root = next(C for C,d in ab.in_degree() if d==0) # this is the root cluster


	for (C_i, C_j) in ab.edges():
		j = Cs.index(C_j)
		common = set(C_j) & set(C_i)
		j_common_idxs = [k for k,vn in enumerate(C_j) if vn in common]

		common_marg = _marginalize(mus[j], cluster_shapes[j], j_common_idxs)

		ent_tol_constraints.append(ExpCone(
			-tts_ent[j],
			mus[j],
			_dup2shape(common_marg, cluster_shapes[j], j_common_idxs)
		))

	# one last cone to add
	j = Cs.index(root)
	ent_tol_constraints.append(ExpCone( -tts_ent[j], mus[j], np.ones(mus[j].shape)))


	prev_val = np.inf

	for it in range(max_iters):
		if it == 0:
			frozens = [ RJD.unif([M.vars[vn] for vn in C]) for i,C in enumerate(Cs) ] 
		else:
			frozens = [ RJD(mus[i].value, [M.vars[vn] for vn in C]) for i,C in enumerate(Cs)]


		linearized = sum(
			# cp.multiply((mu - frozen.data.reshape(-1)), grad_g(frozen))
			cp.sum( cp.multiply((m - f.data.reshape(-1)), gg ))
			for m, f, gg in zip( mus, frozens, grad_g(frozens) )
		)
		prob = cp.Problem( 
			cp.Minimize( 
				logprobs +
				linearized + 
				sum( ss[L] * sum(tL) for L,tL in ts.items() )  
				+ gamma * sum(cp.sum(tt) for tt in tts_ent) #entropy term
			),
			cvx_tol_constraints 
				+ hard_constraints
				+ local_marg_constraints # new for clusters
				+ [ cp.sum(mu) == 1 for mu in mus ]
				# + [ ExpCone(-global_t_ent, mu, np.ones(mu.shape)) ] 
				# TODO FIXME must be done in the same way as also_idef in cluster_opt
				+ ent_tol_constraints
		)

		prob.solve(**solver_kwargs)

		print('obj: ', prob.value + g(frozens), '\t\t tv distance', 
				max( np.sum(np.absolute(mu.value-frozen.data.reshape(-1)))
				  for mu,frozen in zip(mus, frozens))  )

		if(prob.value == prev_val) or all(
			# np.allclose(mu.value, frozen.data.reshape(-1), rtol=1E-6, atol=1E-9)
			np.sum(np.absolute(mu.value-frozen.data.reshape(-1))) <= 1E-6
				for mu, frozen in zip(mus, frozens)):
			break

		prev_val = prob.value

	
	## save problem, etc as properties of method so you can get them afterwards.
	# return RJD(mu.value, M.varlist)
	return namedtuple("ClusterPseudomarginals", ['marginals', 'value'])(
		marginals= [ RJD(mus[i].value, [M.vars[vn] for vn in C]) for i,C in enumerate(Cs)],
		# prob=prob,
		# fp = fp,
		value=prob.value)


# Direct encoding of the objective. Cvxpy cannot solve!
def _cvx_opt_direct(M, gamma=1, **solver_kwargs):
	"""
	DO NOT USE!  Will Raise Exception: Problem is not DCCP. 
	"""
	n = np.prod(M.dshape)
	mu = cp.Variable(n, nonneg=True)
	objective = 0
	
	for L,β,α,X,Y,p in M.edges("l,β,α,X,Y,P"):
		if 'π' not in L:
			muXY = _marginalize(mu, M.dshape, M._idxs(X,Y))
			muX = _marginalize(mu, M.dshape, M._idxs(X))	
			## next calculation might be buggy..
			muX_broadcast2XY = cp.vec(_n_copies(muX, len(Y)) )
			muXpY_X = cp.vec(_cpd2joint(p, muX))
			print(muXY.shape, muX.shape, muX_broadcast2XY.shape, muXpY_X.shape, 
			cp.log(muXY).shape, cp.log(muXpY_X).shape)
			print((cp.log(muXY) - cp.log(muXpY_X)).shape)
			print( muXY.shape )

			objective += β * cp.sum( cp.multiply(muXY,  cp.log(muXY) - cp.log(muXpY_X) ) )
			objective -= α * cp.sum( cp.multiply(muXY, cp.log(muXY) - cp.log(muX_broadcast2XY)))

	prob = cp.Problem(
		cp.Minimize( objective),
		[sum(mu) == 1]
	)
	prob.solve(**solver_kwargs) 
	return RJD(mu.value, M.varlist)