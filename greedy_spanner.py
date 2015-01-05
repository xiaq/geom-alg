from datatypes import Graph
from itertools import izip

def greedy_spanner(graph, dilation):
    """Adds edges to graph until its dilation ratio is less than dilation.
    Takes O(n^2 * (n log(n) + k) time, where k is the amount of inserted edges."""

    idxs, pairs = graph.iter_closest_euclidian_node_pairs()
    
    for idx in idxs:
        v1 = graph.vertices[int(pairs[idx, 0])]
        v2 = graph.vertices[int(pairs[idx, 1])]
        
        dist = ((v1.x-v2.x)**2 + (v1.y-v2.y)**2)**0.5
        graph_dist = graph.bfs_distance(v1, v2, dist * dilation)
        
        if graph_dist is None:
            graph.add_edge(v1, v2)

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
    

##    print "Dilation;#Vertices;#Edges in result;% of edges in result;Construction time"
##
##    for r in (1.5, 1.75, 2, 2.5, 3, 4, 5):
##        for n in range(25, 351, 25):
##            for i in range(3):
##                try:
##                    g = Graph()
##                    
##                    for i in xrange(n):
##                        g.add_vertex(random.random() * 100, random.random() * 100)
##
##                    t = time.clock()    
##                    greedy_spanner(g, r)
##                    t = time.clock() - t
##                    print "%f;%d;%d;%f;%f" % (r, n, g.n_edges(), 2.0*g.n_edges()/(n*(n-1)), t)
##                except:
##                    print "%f;%d;-;-;-" % (r, n)
