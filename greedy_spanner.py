from datatypes import Graph
from itertools import izip

def greedy_spanner(graph, dilation, refine_existing_edges=False):
    """Adds edges to graph until its dilation ratio is less than dilation.
    Takes O(n^2 * log(n) * (n log(n) + k) time, where k is the amount of inserted edges.
    
    If refine_existing_edges is set to True, the greedy spanner operates only
    on the edges present in graph. The returned graph will have a subset of the
    edges of the graph parameter. If refine_existing_edges is set to False, the
    edges that are already in the graph are left as they are and the algorithm
    adds additional edges when required. If refine_existing_eges is True and the
    input graph has O(n) edges then the running time is O(n^2 * log(n)).
    
    WARNING: When refine_existing_edges is True, it is likely but NOT GUARANTEED
    that the dilation will be below dilation, even if the input graph has a dilation
    ratio below dilation."""

    if refine_existing_edges:
        idxs, pairs = graph.shortest_edges()
        graph.clear_edges()
    else:
        idxs, pairs = graph.closest_euclidian_node_pairs()
    
    for idx in idxs:
        v1 = graph.vertices[int(pairs[idx, 0])]
        v2 = graph.vertices[int(pairs[idx, 1])]
        
        dist = ((v1.x-v2.x)**2 + (v1.y-v2.y)**2)**0.5
        graph_dist = graph.bfs_distance(v1, v2, dist * dilation)
        
        if graph_dist is None:
            graph.add_edge(v1, v2)
            
def greedy_theta_spanner(graph, dilation):
    from theta import theta_graph
    
    theta_graph(graph, dilation)
    greedy_spanner(graph, dilation, True)

if __name__ == "__main__":
    import random
    import numpy as np
    import time

    from data.train_stations import train_stations_by_label_luxembourg

    s = train_stations_by_label_luxembourg

    #g = Graph.random_graph(100, 0)
    g = Graph.labeled_datapoints_graph({k: v for k, v in s.iteritems() if v is not None})
    #g = Graph.datapoints_graph((v for v in s.values() if v is not None))
    greedy_spanner(g, 1.3)
    g.plot()
