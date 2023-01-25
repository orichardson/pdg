
from ..pdg import PDG
from ..dist import ClusterDist, RawJointDist as RJD

import networkx as nx
import numpy as np
import cvxpy as cp
from cvxpy.constraints.exponential import ExpCone

from collections.abc import Iterable
from collections import namedtuple
from operator import mul
from functools import reduce

############### UTILITY METHODS ############

def _mk_projector(dshape, IDXs) -> np.array : 
	"""
	A slow (depricated) way of marginalizing: make a large matrix that's 
	mostly full of zeros that computes the appropriate sums

	:param `dshape`: a shape for the joint distribution, say (d_1, ..., d_n)
	:param `IDXs`: the list of indices, say [d_{i1}, ... ]
	:return: a numpy array
	"""
	nvars = len(dshape)
	IDXs = list(IDXs)
	allIDXs = [i for i in range(nvars)]
	A_proj_IDXs = np.zeros(list(dshape) + [dshape[i] for i in IDXs])
	np.einsum(A_proj_IDXs, allIDXs+IDXs, allIDXs)[:] = 1
	return A_proj_IDXs.reshape(np.prod(dshape), np.prod([dshape[i] for i in IDXs]))

def _marginalize_via_projector(mu, M, varis):
	""" A drop-in replacement for `_marginalize` that uses the `_mk_projector` method. """
	return mu.T @ _mk_projector(M.dshape, M._idxs(*varis))

def _marginalize(mu, shape, IDXs):
	""" Given a (flat) cvxpy expression `mu` (a joint distribution), which implicitly we think
	 of as having a certain `shape`, return its (flat) marginal on the indices in the list `IDXs`.
	"""
	if len(IDXs) == 0: 
	 	# to prevent strangeness with "raveling" when marginalizing to keep nothing,
		# just return the sum, still with a dimension
		return cp.sum(mu, keepdims=True)

	postshape = tuple(shape[i] for i in IDXs)
	# preallocate list of the appropriate size. Will eventually hold 
	# scalar cp.expressions; will be converted to a cp.expression by operator overloading after addition.
	elts = [0] *  np.prod( postshape, dtype=int) 
		
	II = np.arange(mu.size)
	v_idx = np.unravel_index(II, shape)
	idx = np.ravel_multi_index(tuple(v_idx[j] for j in IDXs), postshape)

	for i in range(len(elts)):
		idxs_i = np.nonzero((idx==i))
		elts[i] = cp.sum(mu[idxs_i])
	
	## commmented out below: a slower element-wise version of this code
	## with many calls to np.ravel / np.unravel
	#
	# for i, mui in enumerate(mu):
	# 	v_idx = np.unravel_index(i, shape)  # virtual index
	# 	# print('FLAT: ', i, '\t VIRTUAL: ', v_idx, '\t in shape ', shape)
	# 	idx = np.ravel_multi_index(tuple(v_idx[j] for j in IDXs), postshape)
	# 	# print('  |=> post.virtual = ', tuple(v_idx[j] for j in IDXs), '\t in shape', postshape)
	# 	# print('  |=> flat.idx = ', idx)
	# 	elts[idx] = elts[idx] + mui

	return cp.hstack(elts)

def _cpd2joint(cpt, mu_X):
	return cp.vstack([ cpt.to_numpy()[i,:] * mu_X[i] for i in range(mu_X.shape[0])] ).T    

def _cpd2joint_np(cpt, mu_X):
	return cp.vstack([ cpt[i,:] * mu_X[i] for i in range(mu_X.shape[0])] ).T    

def _combine_iters(it1, it2, sel): 
	""" merges the two iterators, at each point using `sel` (an iterable of 0/1 or True/False)
	to decide which iterator (`i1` or `i2`) to take from next. """
	i1 = iter(it1)
	i2 = iter(it2)
	for s in sel: 
		if s: yield next(i1) 
		else: yield next(i2)

def _dup2shape(cvxpy_expr, shape, idxs):
	"""takes a cvxpy expression `cvxpy_expr` representing flattened marginals along dimensions `idxs` in
	the context of a joint n-dimensional array of shape `shape`, and 
	returns a version of `cvxpy_expr` suitable for element-wise multiplication, 
		i.e., of that same joint shape."""
	n = np.prod(shape)
	expr_vshape = tuple(shape[i] for i in idxs)
	m = np.prod(expr_vshape)

	assert m == cvxpy_expr.size, "indices " + str(tuple(shape[i] for i in idxs)) + \
		" not compatible with expression of size "+ str(cvxpy_expr.size)

	unwound = list(cvxpy_expr)
	
	not_shape = tuple(shape[i] for i in range(len(shape)) if i not in idxs)

	expr_coords = np.unravel_index(np.arange(m), expr_vshape)
	to_ret = np.zeros(shape, dtype='object')

	# perhaps prematurely "optimized" because I didn't want to iterate
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
	""" an analogue of `_cpd2joint`, which just does broadcasting, 
		effectively with a "cpd" that is constant ones. 
	WARNING: may not quite be correct!! """
	# do a hstack, because we're not iterating over indices.
	return cp.hstack( [ mu_X for _ in range(n) ] ).T


def _find_cluster_graph(M : PDG, 
		varname_clusters = None, cluster_edges =None,
		verbose=False
	):
	"""
	Given a PDG `M`, and possibly a specification of which clusters and / or cluster
	edges we want to see in the graph, find a cluster graph (ideally a cluster tree)
	that has the running intersection property.
	"""
	if varname_clusters is None:
		if verbose: print("no clusters given; using pgmpy junction tree to find some.")
		
		mn = M.to_markov_net()
		component_jtrees = [mn.subgraph(N).to_junction_tree() for N in nx.connected_components(mn)]

		# jtree = 
		# varname_clusters = list(jtree.nodes())
		# cluster_edges = list(jtree.edges())
		varname_clusters = [N for jt in component_jtrees for N in jt.nodes() ]
		cluster_edges = [N for jt in component_jtrees for N in jt.edges() ]

		if verbose: print("FOUND: ",varname_clusters)

	Cs = [tuple(C) for C in varname_clusters]
	m = len(varname_clusters)

	if cluster_edges is None:
		complete_graph = nx.Graph()
		for i in range(m):
			for j in range(i+1,m):
				common = set(Cs[i]) & set(Cs[j])
				complete_graph.add_edge(Cs[i], Cs[j], weight=-len(common))

				## might want to weight by number of params (like below), but might not give
				## the running intersection property, so instead do the above.
				#
				# num_sepset_params = np.prod([len(M.vars[X]) for X in common])
 				# complete_graph.add_edge(Cs[i], Cs[j], weight=-num_sepset_params)
		
		cluster_edges = nx.minimum_spanning_tree(complete_graph).edges()

	cluster_shapes = [tuple(len(M.vars[Vn]) for Vn in C) for C in Cs]

	edgemap = {} # label -> cluster index
	
	sorted_clusters = sorted(enumerate(Cs), key=lambda iC: np.prod(cluster_shapes[iC[0]]))
	if verbose: print(sorted_clusters)

	for L, X, Y in M.edges("l,X,Y"):
		for i,cluster in sorted_clusters:
			if all((N.name in cluster) for N in (X & Y).atoms):
				edgemap[L] = i
				break
		else:
			raise ValueError("Invalid Cluster Tree: an edge (%s: %s → %s) is not contained in any cluster"
				% (L,X.name,Y.name) )

	return Cs, cluster_edges, edgemap, cluster_shapes

############# INFERENCE ALGORITHMS ############
# optimizes over joint distributions
def cvx_opt_joint( M : PDG,  also_idef=True, **solver_kwargs) :
	""" Given a pdg `M`, do convex optimization to find a distribution that minimizes Inc.
	if `also_idef=True`, then find the unique such distribution that also minimizes IDef 
	subject to these constraints, by solving a second optimization problem afterwards."""
	n = np.prod(M.dshape)
	mu = cp.Variable(n, nonneg=True)
	t = { L : cp.Variable(p.to_numpy().size) for (L,p) in M.edges("l,P") if 'π' not in L }
	
	beta_tol_constraints = [
		ExpCone(-t[L], 
			   _marginalize(mu, M.dshape, M._idxs(X,Y)),
			   cp.vec(_cpd2joint(p, _marginalize(mu, M.dshape, M._idxs(X))) ))
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
		# the new objective is KL( mu || prod of marginals of oldmu.Pr ^ \alpha )
		Pr = oldmu_dist.prob_matrix
		# fp = np.prod( [ Pr(Y|X)**α for X,Y,α in M.edges("X,Y,α")] )
		fp = reduce(mul, [np.ones(M.dshape)] + [ Pr(Y|X)**α for X,Y,α in M.edges("X,Y,α")] )
		tt = cp.Variable(n)

		new_prob = cp.Problem(
			cp.Minimize(sum(tt)),
			cm_constraints + [ ExpCone(-tt, mu, fp.reshape(-1) ), sum(mu) == 1]
		)
		new_prob.solve(**solver_kwargs);
		
	## save problem, etc as properties of method so you can get them afterwards.
	cvx_opt_joint.prob = prob
	cvx_opt_joint.t = t
	
	return RJD(mu.value, M.varlist)

# like cvx_opt_joint, but in parallel over all clusters, with consistency constraints
def cvx_opt_clusters( M : PDG, also_idef=True, 
		varname_clusters = None, cluster_edges = None,
		# dry_run=False, 
		**solver_kwargs) :
	""" Given a pdg `M`, do a convex optimization to find (a compact representation of) a 
	joint distribution which minimizes Inc. This is done by only tracking the cluster marginals
	along a cluster tree; it can be shown that any minimizer of Inc for \gamma > 0 must be of
	this form. 

	This method is a generalization of `cvx_opt_joint`. 
	"""
	verb = 'verbose' in solver_kwargs and solver_kwargs['verbose']

	Cs, cluster_edges, edgemap, cluster_shapes = _find_cluster_graph(M, varname_clusters, cluster_edges, verb) 
	m = len(Cs)

	mus = [ cp.Variable(np.prod(shape), nonneg=True) for shape in cluster_shapes]
	ts = { L : cp.Variable(p.to_numpy().size) for (L,p) in M.edges("l,P") if 'π' not in L }
	
	if verb: 
		print("total parameters: ", sum(m.size for m in mus), flush=True)
		print("adding exponential cones... ", end='', flush=True)
	
	tol_constraints = []
	for L,X,Y,p in M.edges("l,X,Y,P"):
		if 'π' not in L:
			i = edgemap[L]
			C = Cs[i]            
			
			idxs_XY = [C.index(N.name) for N in (X&Y).atoms]
			idxs_X = [C.index(N.name) for N in X.atoms]
			
			xymarginal = _marginalize(mus[i], cluster_shapes[i], idxs_XY)

			tol_constraints.append(ExpCone(-ts[L], xymarginal, 
			   cp.vec(_cpd2joint(p,_marginalize(mus[i], cluster_shapes[i], idxs_X)) )) )

			# if dry_run:
			# 	break
	
	if verb:
		print("... done!", flush=True)
		print("Adding local marginal constraints ... ", end='', flush=True)

	local_marg_constraints = []
	

	for Ci, Cj in cluster_edges:
		common = set(Ci) & set(Cj)
		if len(common) > 0:
			# Add a constraint that the marginals of these two clusters agree.
			i, j = Cs.index(Ci), Cs.index(Cj)

			i_idxs = [k for k,vn in enumerate(Ci) if vn in common]
			j_idxs = [k for k,vn in enumerate(Cj) if vn in common]
			
			marg_constraint = _marginalize(mus[i], cluster_shapes[i], i_idxs)\
					== _marginalize(mus[j], cluster_shapes[j], j_idxs)
			local_marg_constraints.append(marg_constraint)

	if verb:
		print("... done!", flush=True)
		print("Constructing problem.")

	prob = cp.Problem( 
		cp.Minimize( sum(βL * sum(ts[L]) for βL,L in M.edges("β,l") if 'π' not in L) ),
			[sum(mus[i]) == 1 for i in range(m)]
			# + [mus[i] >= 0 for i in range(m)] ## subsumed by nonneg = True
			+ tol_constraints + local_marg_constraints )

	# if dry_run:
	# 	return prob

	prob.solve(**solver_kwargs)    
	
	fp = None
	if also_idef:
		if verb: print("Now beginning IDef computation")
		# Also implement the Bethe entropy along this tree
		old_cluster_rjds = {}
		for i,C in enumerate(Cs):
			old_cluster_rjds[i] = RJD(mus[i].value.copy(), [M.vars[n] for n in C])


		cpds = {}
		cm_constraints = []
		for L,X,Y,p in M.edges("l,X,Y,P"):
			if 'π' in L: continue

			i = edgemap[L]
			C = Cs[i]
			sh = cluster_shapes[i]  
			
			idxs_XY = [C.index(N.name) for N in (X&Y).atoms]
			idxs_X = [C.index(N.name) for N in X.atoms]

			cpd = old_cluster_rjds[edgemap[L]].conditional_marginal(Y|X)
			cpds[L] = cpd

			# constrain to be the same conditional marginals as we had before
			cm_constraints.append(
				_marginalize(mus[i], sh, idxs_XY)
					==
				cp.vec(_cpd2joint(cpd, _marginalize(mus[i], sh, idxs_X)))
			)


		## next, we're going to add the IDef terms.
		tts = [ cp.Variable( np.prod(shape)) for shape in cluster_shapes ]
		tol_constraints = []
		# We're going to need the Bethe entropy along the tree, which involves calculating
		# the sepset beliefs, and to keep the program convex, we need at most one per cluster.
		# So, we find a directed spanning tree, called an "aboresence", and then apply
		# the appropriate correction to the tail of each edge in the tree.

		Gr = nx.Graph(cluster_edges).to_directed()
		Gr.add_nodes_from(Cs)
		nx.set_edge_attributes(Gr, 
			{e : { # we want a spanning aboressence that selects the biggest possible
				   # root node, because we don't have to marginalize to get a
				   # correction term for it.
				   'weight' : np.prod([len(M.vars[x]) for x in e[1]])  }
				for e in Gr.edges() }
		)
		ab = nx.minimum_spanning_arborescence(Gr)


		for i in range(m):
			fp = reduce(mul, 
				# first term is just here to make sure this value broadcasts to the 
				# right shape, even if not all variables are used (e.g., for an empty PDG)
				[np.ones(cluster_shapes[i])]+
				[old_cluster_rjds[i].prob_matrix(Y|X) **α 
					for X,Y,L,α in M.edges("X,Y,L,α") if edgemap[L] == i  
						and 'π' not in L
						] 
			)

			# correction_elts = [1] * mus[i].size
			correction = np.ones(mus[i].shape)
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
						
						# want correction *= new_term, but no broadcasting b/c indices lost
						# ... sooo instead ....
						
						##(a.0) pre-compute the shape of the shared sepset 
						# newterm_vshape = tuple(cluster_shapes[i][k] for k in idxs)
						# for w,mu_w in enumerate(mus[i]):
						# 	##(a.1) unravel the joint cluster i's world w into a value for each variable
						# 	v_idx = np.unravel_index(w, cluster_shapes[i])
						# 	##(a.2) figure out what the appropriate flattened index into the
						# 	# marginal probability (new_term) should be. 
						# 	idx = np.ravel_multi_index(tuple(v_idx[j] for j in idxs), newterm_vshape)
						# 	correction_elts[w] *= new_term[idx]
							
						## (b.0) prepcompute the shape of shared subset
						newterm_vshape = tuple(cluster_shapes[i][k] for k in idxs)
						v_idx = np.unravel_index(np.arange(mus[i].size) , cluster_shapes[i])
						idx = np.ravel_multi_index(tuple(v_idx[k] for k in idxs), newterm_vshape)
						correction = cp.multiply(correction, new_term[idx])

			# correction = cp.hstack(correction_elts)

			tol_constraints.append(ExpCone(
				-tts[i],
				mus[i],
				cp.multiply(fp.reshape(-1), correction)
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
	
	# return RJD(mu.value, M.varlist)
	cd = ClusterDist([ RJD(mus[i].value, [M.vars[vn] for vn in C]) for i,C in enumerate(Cs)])
	return namedtuple("ClusterPseudomarginals",
			['marginals', 'cluster_dist', 'inc','idef'])(
		marginals= [ RJD(mus[i].value, [M.vars[vn] for vn in C]) for i,C in enumerate(Cs)],
		# prob=prob,
		# fp = fp,
		cluster_dist = cd,
		inc=prob.value,
		# idef=new_prob.value if also_idef else M.IDef(cd)
		idef=new_prob.value if also_idef else float('inf') # too hard to compute
	)

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
	# for L,X,Y,β,p in M.edges("l,X,Y,β,P"):
	for L,X,Y,β,p in M.edges("l,X,Y,α,P"):
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
			# oops, I think wrong quantity here. 
			# logprobs += β * lin_lp
			logprobs += α * gamma * lin_lp



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
	# for L,X,Y,β,p in M.edges("l,X,Y,β,P"):
	for L,X,Y,α,p in M.edges("l,X,Y,α,P"):
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
			# logprobs += β * lin_lp
			logprobs += α * gamma * lin_lp


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

	return RJD(mu.value, M.varlist)


# best of all worlds: cccp on top of clusters
def cccp_opt_clusters( M : PDG, gamma=1, max_iters=20,
		varname_clusters = None, cluster_edges = None,
		**solver_kwargs) :
	verb = 'verbose' in solver_kwargs and solver_kwargs['verbose']

	Cs, cluster_edges, edgemap, cluster_shapes = _find_cluster_graph(M, varname_clusters, cluster_edges, verb)
	mus = [ cp.Variable(np.prod(shape), nonneg=True) for shape in cluster_shapes ]

	ts = {}
	ss = {}
	
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

		# logprobs += β * cp.sum( cp.multiply(lp, mu_xy ) )
		logprobs += α * gamma * cp.sum( cp.multiply(lp, mu_xy ) )

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
				if i != edgemap[L]: continue
				## contrast with: 
				# if not all((N.name in Cs[i]) for N in (X&Y).atoms): continue
				## ... which seems to be unstable and give the wrong gradient sometimes

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
			## TODO FIXME THIS DUP IS VERY SLOW
			_dup2shape(common_marg, cluster_shapes[j], j_common_idxs)
		))

	# one last cone to add
	j = Cs.index(root)
	ent_tol_constraints.append(ExpCone( -tts_ent[j], mus[j], np.ones(mus[j].shape)))


	prev_val = np.inf

	for it in range(max_iters):
		if it == 0:
			frozens = [ RJD.unif([M.vars[vn] for vn in C]) for C in Cs ] 
		else:
			frozens = [ RJD(mus[i].value, [M.vars[vn] for vn in C]) for i,C in enumerate(Cs)]


		linearized = sum(
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
				+ ent_tol_constraints
		)

		prob.solve(**solver_kwargs)

		if verb: print('obj: ', prob.value + g(frozens), '\t\t tv distance', 
				        max( np.sum(np.absolute(mu.value-frozen.data.reshape(-1)))
				            for mu,frozen in zip(mus, frozens))  )

		if(prob.value == prev_val) or all(
			np.sum(np.absolute(mu.value-frozen.data.reshape(-1))) <= 1E-6
				for mu, frozen in zip(mus, frozens)):
			break

		prev_val = prob.value


	cd = ClusterDist([ RJD(mus[i].value, [M.vars[vn] for vn in C]) for i,C in enumerate(Cs)])
	return namedtuple("ClusterPseudomarginals", ['marginals', "cluster_dist", 'value'])(
		marginals= [ RJD(mus[i].value, [M.vars[vn] for vn in C]) for i,C in enumerate(Cs)],
		cluster_dist = cd,
		value=prob.value)


# Direct encoding of the objective. Cvxpy cannot solve!
def _cvx_opt_direct(M, gamma=1, **solver_kwargs):
	"""
	DO NOT USE!  Will Raise Exception: Problem is not DCP nor DCCP. 
	It is just here to show that direct optimization of the
	objective cannot be done even with the built-in dccp algorithm.
	"""
	mu = cp.Variable(np.prod(M.dshape), nonneg=True)
	objective = 0
	
	for L,β,α,X,Y,p in M.edges("l,β,α,X,Y,P"):
		if 'π' not in L:
			muXY = _marginalize(mu, M.dshape, M._idxs(X,Y))
			muX = _marginalize(mu, M.dshape, M._idxs(X))	

			## next calculation might be buggy..
			muX_broadcast2XY = cp.vec(_n_copies(muX, len(Y)) )
			muXpY_X = cp.vec(_cpd2joint(p, muX))

			objective += β * cp.sum( cp.multiply(muXY,  cp.log(muXY) - cp.log(muXpY_X) ) )
			objective -= α * cp.sum( cp.multiply(muXY, cp.log(muXY) - cp.log(muX_broadcast2XY)))

	prob = cp.Problem( cp.Minimize( objective),	[sum(mu) == 1] )
	prob.solve(**solver_kwargs) 

	return RJD(mu.value, M.varlist)