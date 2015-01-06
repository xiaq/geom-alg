import math
import operator
import itertools

from datatypes import Graph

def unit_vector(d):
    return (math.cos(d), math.sin(d))

def theta_graph(graph, dilation):
    """
    Construct a Theta graph that is a t-spanner (t = dilation).
    """
    # Find a suitable k
    theta = math.asin(1 - 1.0/(dilation**2))
    k = int(math.ceil(2*math.pi / theta))
    theta = 2*math.pi / k
    bisectors = [unit_vector((c+0.5)/k * 2*math.pi) for c in range(k)]

    for v in graph.vertices:
        # for each point u, find the cone u is in, the distance of the
        # projection of u on the bisector to v
        fs = []
        for u in graph.vertices:
            if u == v:
                continue
            c = int(math.floor(math.atan2(u.y - v.y, u.x - v.x) / theta))
            b = bisectors[c]
            d = (u.x - v.x) * b[0] + (u.y - v.y) * b[1]
            if d <= 0:
                break
            fs.append((c, d, u))

        for c, g in itertools.groupby(sorted(fs), operator.itemgetter(0)):
            graph.add_edge(v, next(g)[2])

if __name__ == "__main__":
    import random
    import numpy as np
    import time

    from data.train_stations import train_stations_by_label_luxembourg

    s = train_stations_by_label_luxembourg

    #g = Graph.random_graph(100, 0)
    g = Graph.labeled_datapoints_graph({k: v for k, v in s.iteritems() if v is not None})
    #g = Graph.datapoints_graph((v for v in s.values() if v is not None))
    theta_graph(g, 1.3)
    g.plot()
