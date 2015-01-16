import math
import operator
import itertools

from datatypes import Graph

def distance(u, v):
    return (u.x-v.x)**2 + (u.y-v.y)**2

def yao_graph(graph, dilation):
    """
    Construct a Yao graph that is a t-spanner (t = dilation).
    """
    # Find a suitable k
    theta = math.asin(1 - 1.0/(dilation**2))
    k = int(math.ceil(2*math.pi / theta))
    theta = 2*math.pi / k

    for v in graph.vertices:
        ts = [None] * k
        for u in graph.vertices:
            if u == v:
                continue
            c = int(math.floor(math.atan2(u.y - v.y, u.x - v.x) / theta))
            d = distance(u, v)
            if ts[c] is None or ts[c][0] > d:
                ts[c] = (d, u)

        for t in ts:
            if t is None:
                continue
            graph.add_edge(v, t[1], True)

if __name__ == "__main__":
    import random
    import numpy as np
    import time

    from data.train_stations import train_stations_by_label_luxembourg

    s = train_stations_by_label_luxembourg

    #g = Graph.random_graph(100, 0)
    g = Graph.labeled_datapoints_graph({k: v for k, v in s.iteritems() if v is not None})
    #g = Graph.datapoints_graph((v for v in s.values() if v is not None))
    yao_graph(g, 1.3)
    print g.nr_intersections()
    g.plot()
