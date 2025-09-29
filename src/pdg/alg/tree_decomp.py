from ..pdg import PDG

import networkx as nx
import numpy as np

def tree_decompose(M : PDG, 
		varname_clusters = None, cluster_edges =None,
		verbose=False
	):
	"""
	Given a PDG `M`, and possibly a specification of which clusters and / or cluster
	edges we want to see in the graph, find a tree decomposition. Return a tuple

    (list of clusters, list of edges (pairs of clusters), 
        map from PDG edges to clusters,
        shape for each cluster.)

    where each cluster is a tuple of strings (the names of the variables). 
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
	
    # want to assign each edge to smallest cluster containing it...
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