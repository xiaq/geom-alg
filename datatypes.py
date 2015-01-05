from collections import namedtuple
from priodict import PriorityDictionary
from itertools import chain
import heapq

Vertex = namedtuple('Vertex', ['x', 'y', 'id', 'label'])

class Edge(object):

    __slots__ = ('v1', 'v2', 'length_sq', 'length')

    def __init__(self, v1, v2):
        self.v1 = v1
        self.v2 = v2
    
        self.length_sq = (v1.x - v2.x)**2 + (v1.y - v2.y)**2
        self.length = self.length_sq**0.5
        
    def __cmp__(self, other):
        return cmp(self.length_sq, other.length_sq)

    def other(self, v):
        """Returns, given one endpoint of the edge, the other endpoint."""
        
        if v == self.v1:
            return self.v2
        else:
            return self.v1
        
class Graph(object):

    def __init__(self):
        self.vertices = []
        self.edges = []
        self.n = 0
        
    def add_vertex(self, x, y, label=None):
        id = self.n
        self.n += 1
        self.vertices.append(Vertex(x=x, y=y, id=id, label=label))
        self.edges.append([])
        return id
        
    def add_edge(self, v1, v2, check_exists=False):
        if type(v1) == int:
            v1 = self.vertices[v1]
        if type(v2) == int:
            v2 = self.vertices[v2]
            
        if check_exists:
            v1_e = self.edges[v1.id]
            v2_e = self.edges[v2.id]
            if len(v1_e) < len(v2_e):
                for e in v1_e:
                    if e.other(v1) == v2:
                        return
            else:
                for e in v2_e:
                    if e.other(v2) == v1:
                        return
            
        edge = Edge(v1, v2)
        self.edges[v1.id].append(edge)
        self.edges[v2.id].append(edge)
        
    def n_vertices(self):
        return self.n
        
    def n_edges(self):
        return int(0.5 * sum(len(edges) for edges in self.edges))
        
    def numpy_coordinates(self):
        """Returns the coordinates of all vertices in this graph in a numpy n-by-2 matrix."""
    
        from numpy import empty
        
        coordinates = empty((self.n, 2)) # Create a new numpy array with random contents
        for v in self.vertices:
            coordinates[v.id, 0] = v.x
            coordinates[v.id, 1] = v.y
            
        return coordinates
        
    def iter_vertices(self):
        """Returns an iterator for all vertices, ordered by ID / insertion order."""
        
        return iter(self.vertices)
    
    def iter_edges(self):
        """Returns an iterator for all edges. First reports all edges of the first vertex,
        then all edges of the second vertex that were not yet reported, et cetera. Never
        reports an edge twice."""
        
        for idx, edges in enumerate(self.edges):
            for edge in edges:
                if edge.v1.id == idx: # condition to ensure that we return each edge only once
                    yield edge
        
    def iter_neighbors(self, vertex):
        """Returns an iterator for all neighbors of a vertex, in (neighbor, distance) pairs."""
    
        if type(vertex) == Vertex:
            v_idx = vertex.id
        elif type(vertex) == int:
            v_idx = vertex
            vertex = self.vertices[v_idx]
        else:
            raise TypeError, "Vertex argument must be either a Vertex object or an int, not %s" % type(vertex)
            
        return ((edge.other(vertex), edge.length) for edge in self.edges[v_idx])
        
    def iter_closest_euclidian_node_pairs(self):
        """Returns an iterator for all pairs of nodes, sorted on actual euclidian distance
        between these two nodes. Does not take edges of the graph into account.
        
        Running time is O(n*n*log(n))"""
        
        from numpy import empty
        
        # Keep two arrays: one containing only the distances and one containing all data.
        # We will end up sorting the first and passing its IDs with
        all_distances_sq = empty(0.5*self.n*(self.n-1))
        index_pair = empty((0.5*self.n*(self.n-1), 2))
        
        i = 0
        for v_idx, v in enumerate(self.vertices):
            for w_idx in xrange(v_idx + 1, len(self.vertices)):
                w = self.vertices[w_idx]
                all_distances_sq[i] = (v.x-w.x)**2 + (v.y-w.y)**2
                index_pair[i] = v_idx, w_idx
                i += 1
                
        sorted_idxs = all_distances_sq.argsort()
        return sorted_idxs, index_pair
        
    def closest_neighbor(self, vertex):
        """Returns the closes neighbor of a vertex, or None if it has degree zero."""
    
        if type(vertex) == Vertex:
            vertex = vertex.id
        return min((edge.length, edge.other(vertex)) for edge in self.edges[vertex])[1] if self.edges[vertex] else None
        
    def bfs_distance(self, start, end, maxlen=None):
        """Computes and returns the distance from node 'start' to node 'end'. If no path
        between them exists, returns None.
        
        Optionally, maxlen may be specified in which case the algorithm will optimize to
        not further explore paths longer than maxlen. If maxlen is specified it will never
        return a path length of more than maxlen."""

        if type(start) == int:
            start = self.vertices[start]
        if type(end) == int:
            end = self.vertices[end]

        # Based on http://code.activestate.com/recipes/119466-dijkstras-algorithm-for-shortest-paths/
    
        D = {} # dictionary of final distances
        Q = PriorityDictionary() # est.dist. of non-final vert.
        Q[start] = 0
        
        while Q:
            v = Q.smallest()
            D[v] = Q[v]
            Q.pop_smallest()
            
            if v == end: 
                break
            
            for w, dist in self.iter_neighbors(v):
                vwLength = D[v] + dist
                if (maxlen is None or vwLength <= maxlen) and w not in D and (w not in Q or vwLength < Q[w]):
                    Q[w] = vwLength
                else:
                    pass
                    #print "Not appending", D, (maxlen is None or vwLength <= maxlen), w not in D, (w not in Q or vwLength < Q[w])
        
        if end in D and (maxlen is None or D[end] <= maxlen):
            #print "not return none"
            return D[end]
        else:
            return None
            
    def plot(self):
        import matplotlib.pyplot as plt
        
        # Plot edges
        # Use None-separated list as explained on http://exnumerus.blogspot.nl/2011/02/how-to-quickly-plot-multiple-line.html
        xlist = []
        ylist = []
        for e in self.iter_edges():
            xlist.extend([e.v1.x, e.v2.x])
            xlist.append(None)
            ylist.extend([e.v1.y, e.v2.y])
            ylist.append(None)
            
        plt.plot(xlist, ylist)
            
        # Plot vertices
        coords = list((v.x, v.y) for v in self.iter_vertices())
        x, y = zip(*coords)
        plt.scatter(x, y)

        # Plot vertex labels
        for vertex in self.iter_vertices():
            if vertex.label:
                plt.annotate(vertex.label, (vertex.x, vertex.y))

        plt.show()
        
    @classmethod
    def random_graph(cls, n_vertices, n_edges):
        import random
    
        g = cls()
        
        for i in xrange(n_vertices):
            g.add_vertex(random.random() * 100, random.random() * 100)
            
        for i in xrange(n_edges):
            g.add_edge(random.randint(0, n_vertices-1), random.randint(0, n_vertices-1), True)
            
        return g

    @classmethod
    def datapoints_graph(cls, points, limit=None):
        g = cls()

        for p in points:
            if limit and g.n_vertices() >= limit:
                break
            g.add_vertex(p[0], p[1])

        return g

    @classmethod
    def labeled_datapoints_graph(cls, points, limit=None):
        g = cls()

        for label, p in points.iteritems():
            if limit and g.n_vertices() >= limit:
                break
            g.add_vertex(p[0], p[1], label)

        return g

if __name__ == "__main__":
    import numpy as np
    import random
    g = Graph()
    
    for i in xrange(100):
        g.add_vertex(random.random() * 100, random.random() * 100)
        
    for i in xrange(300):
        id1 = random.randint(0, 99)
        id2 = random.randint(0, 99)
        
        if id1 == id2:
            continue
            
        g.add_edge(id1, id2, True)
