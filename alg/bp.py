from pgmpy.inference.ExactInference import BeliefPropagation
import networkx as nx

def avg_init_pgmpy_BP_calibrate(model, operation='marginalize'):
    bp = BeliefPropagation(model)

    # bp.calibrate()
    ###### REDOING BP CALIBRATION ##################
    bp.clique_beliefs = {
        clique: bp.junction_tree.get_factors(clique)
        for clique in bp.junction_tree.nodes()
    }

    bp.sepset_beliefs = {}
    for (U,V) in bp.junction_tree.edges():
        common = frozenset(U)&frozenset(V)
        f1 =  getattr(bp.clique_beliefs[U], operation)(
                list(frozenset(U) - common), inplace=False)
        f2 =  getattr(bp.clique_beliefs[V], operation)(
                list(frozenset(V) - common), inplace=False)
        bp.sepset_beliefs[frozenset((U,V))] = (f1 + f2) *0.5

    for clique in bp.junction_tree.nodes():
        if not bp._is_converged(operation=operation):
            neighbors = bp.junction_tree.neighbors(clique)
            # update root's belief using neighbor clique's beliefs
            # upward pass
            for neighbor_clique in neighbors:
                bp._update_beliefs(neighbor_clique, clique, operation=operation)
            bfs_edges = nx.algorithms.breadth_first_search.bfs_edges(
                bp.junction_tree, clique
            )
            # update the beliefs of all the nodes starting from the root to leaves using root's belief
            # downward pass
            for edge in bfs_edges:
                bp._update_beliefs(edge[0], edge[1], operation=operation)
        else:
            break
    
    return bp
