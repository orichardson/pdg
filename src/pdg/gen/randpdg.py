import networkx as nx
import random

import itertools as itt

from ..rv import Variable as Var
from ..dist import CPT
from ..pdg import PDG

def find_cliques_size_k(G, k):
	""" based on https://stackoverflow.com/a/58782120/13480314 """
	for clique in nx.find_cliques(G):
		if len(clique) == k:
			yield tuple(clique)
		elif len(clique) > k:
			yield from itt.combinations(clique,k)

def random_k_tree(n, k):
	if n <= k+1:
		G = nx.complete_graph(n)
		ctree = nx.Graph(); ctree.add_node(tuple(G.nodes()))
		return G, ctree

	G = nx.complete_graph(k + 1)
	ctree = nx.Graph()
	ctree.add_node(tuple(G.nodes()))

	while len(G.nodes()) < n:
		kcq = random.choice( list(find_cliques_size_k(G,k)))
		
		newnode = len(G.nodes())
		newcluster = kcq + (newnode, )

		G.add_node(newnode)
		ctree.add_node(newcluster)

		G.add_edges_from( (n, newnode) for n in kcq )
		ctree.add_edges_from((C, newcluster) for C in ctree.nodes() if len(set(C) & set(newcluster)) == k)

	ctree_tree = nx.minimum_spanning_tree(ctree)
	return G, ctree_tree


def rand_PDG( tw_range = [1,4],
        edge_range = [8,15],
        n_vars_range = [8, 30], 
        n_val_range = [2,3],
        n_src_range = [0,3],
        n_tgt_range = [1,2]):
    n = random.randint(*n_vars_range)
    k = random.randint(*tw_range)
    m = random.randint(*edge_range)

    g, ctree = random_k_tree(n,k)

    pdg = PDG()

    var_names = iter(itt.chain(
		(chr(i + ord('A')) for i in range(26)) ,
		("X%d_"%v for v in itt.count()) ))

    for _, vn in zip(range(n), var_names):
        pdg += Var.alph(vn, random.randint(*n_val_range))

    while len(pdg.edgedata) < m:
        # c1,c2 = random.choice(list(ctree.edges()))
        c1 = random.choice(list(ctree.nodes()))
        # c2 = random.choice(list(ctree[c1]))
        # options = [pdg.varlist[i] for i in set(c1) | set(c2)]
        options = [pdg.varlist[i] for i in c1]
        # options = random.choice(list(ctree.nodes()))

        try:
            src = random.sample(options, k=random.randint(*n_src_range))
            # print('remaining', [ v for v in pdg.varlist if v not in src])
            # print('args.tgt_range: ', args.tgt_range)
            tgt = random.sample([ v for v in options if v not in src], k=random.randint(*n_tgt_range))
        except ValueError:
            continue # if there wasn't space, try again. 

        # pdg += CPT.make_random( reduce(and_, src, initial=Unit), reduce(and_, tgt, initial=Unit) )
        print(f"{Var.product(src).name:>20} --> {Var.product(tgt).name:<20}")
        pdg += CPT.make_random( Var.product(src), Var.product(tgt))

    nx.relabel_nodes(ctree, {C:tuple(pdg.varlist[i].name for i in C) for C in ctree.nodes()}, copy=False)

    return pdg, ctree