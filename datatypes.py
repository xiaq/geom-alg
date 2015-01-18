from collections import namedtuple
from priodict import PriorityDictionary
from itertools import chain
from heapq import heappush, heappop
import bisect
from blist import *

class Vertex(namedtuple('Vertex', ['x', 'y', 'id', 'label'])):
    def __cmp__(self, other):
        if isinstance(other, Vertex):
            return cmp((self.y, self.x), (other.y, other.x))
        elif isinstance(other, Edge):
            if other.v1 < other.v2:
                x1, y1 = other.v1.x, other.v1.y
                x2, y2 = other.v2.x, other.v2.y
            else:
                x1, y1 = other.v2.x, other.v2.y
                x2, y2 = other.v1.x, other.v1.y
            return cmp((y2 - y1) * self.x - (x2 - x1) * self.y,
                       (y2 - y1) * x1     - (x2 - x1) * y1)
        else:
            raise TypeError('Vertex can only be compared to Vertex or Edge')
            
Event = namedtuple('Event', ['vertex', 'edge1', 'edge2', 'type'])

START, END, INTERSECT = range(3)

class Edge(object):

    __slots__ = ('v1', 'v2', 'length_sq', 'length')

    def __init__(self, v1, v2):
        self.v1 = v1
        self.v2 = v2
    
        self.length_sq = (v1.x - v2.x)**2 + (v1.y - v2.y)**2
        self.length = self.length_sq**0.5
        
    def other(self, v):
        """Returns, given one endpoint of the edge, the other endpoint."""
        
        if v == self.v1:
            return self.v2
        else:
            return self.v1
     
    def __repr__(self):
        return 'Edge(%r, %r, %f, %f)' % (self.v1, self.v2, self.length_sq, self.length)

def getLineIntersection(edge1, edge2):
    """ 
    Returns the intersection point if the lines intersect, otherwise None. Based on: http://stackoverflow.com/questions/563198/how-do-you-detect-where-two-line-segments-intersect
    """
    if edge1.v1 in (edge2.v1, edge2.v2) or edge1.v2 in (edge2.v1, edge2.v2):
        return None
    p0_x, p0_y = edge1.v1.x, edge1.v1.y
    p1_x, p1_y = edge1.v2.x, edge1.v2.y
    p2_x, p2_y = edge2.v1.x, edge2.v1.y
    p3_x, p3_y = edge2.v2.x, edge2.v2.y
    
    s1_x = p1_x - p0_x
    s1_y = p1_y - p0_y
    s2_x = p3_x - p2_x
    s2_y = p3_y - p2_y

    try:
        s = (-s1_y * (p0_x - p2_x) + s1_x * (p0_y - p2_y)) / float(-s2_x * s1_y + s1_x * s2_y)
        t = ( s2_x * (p0_y - p2_y) - s2_y * (p0_x - p2_x)) / float(-s2_x * s1_y + s1_x * s2_y)
    except:
        print edge1
        print edge2
        raise

    if 1e-4 < s < 1-1e-4 and 1e-4 < t < 1-1e-4:
        # Collision detected
        i_x = p0_x + (t * s1_x);
        i_y = p0_y + (t * s1_y);
        return Vertex(i_x, i_y, -1, None)

    return None # No collision

class Graph(object):
    # This class is not thread-safe.

    def __init__(self):
        self.vertices = []
        self.edges = []
        self.n = 0
        
    def add_vertex(self, x, y, label=None):
        """Adds a vertex with given coordinates and label. Returns the vertex ID
        that can be used to refer to it later with functions like add_edge.
        
        Vertex IDs are guaranteed to be successive numbers starting at zero, so they
        can be used as array indices, for example."""
    
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
        
    def clear_edges(self):
        for idx in xrange(self.n_vertices()):
            self.edges[idx] = []
        
    def n_vertices(self):
        return self.n
        
    def n_edges(self):
        return int(0.5 * sum(len(edges) for edges in self.edges))
        
    def weight(self):
        """Computes and returns the sum of the length of all edges."""
        
        return sum(edge.length for edge in self.iter_edges())
        
    def max_edge_degree(self):
        return max(len(e_lst) for e_lst in self.edges)
        
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
        
    def closest_euclidian_node_pairs(self):
        """Returns numpy arrays for all pairs of nodes, sorted on actual euclidian distance
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
        
    def shortest_edges(self):
        """Returns numpy arrays containing all edge lengths, sorted from
        shortest to longest."""
        
        from numpy import empty
        
        n_edges = self.n_edges()
        all_lengths_sq = empty(n_edges)
        index_pair = empty((n_edges, 2))
        
        for i, edge in enumerate(self.iter_edges()):
            all_lengths_sq[i] = edge.length_sq
            index_pair[i] = edge.v1.id, edge.v2.id
            
        sorted_idxs = all_lengths_sq.argsort()
        return sorted_idxs, index_pair
        
    def closest_neighbor(self, vertex):
        """Returns the closes neighbor of a vertex, or None if it has degree zero."""
    
        if type(vertex) == Vertex:
            vertex = vertex.id
        return min((edge.length, edge.other(vertex)) for edge in self.edges[vertex])[1] if self.edges[vertex] else None
        
    def euclidian_distance(self, v, w, squared=False):
        if type(v) == int:
            v = self.vertices[v]
        if type(w) == int:
            w = self.vertices[w]
    
        if squared:
            return (v.x-w.x)**2 + (v.y-w.y)**2
        else:
            return ((v.x-w.x)**2 + (v.y-w.y)**2)**0.5
        
    def bfs_distance(self, start, end, maxlen=None, weighted=True):
        """Computes and returns the distance from node 'start' to node 'end'. If no path
        between them exists, returns None. Specify end=None to compute the distance from
        'start' to all other nodes. In that case, the return value is a Numpy-array such
        that array[i]=bfs_distance(start, i, maxlen), only the vertices to which no path
        (shorter than maxlen) was found will be set to -1 instead of None.
        
        Optionally, maxlen may be specified in which case the algorithm will optimize to
        not further explore paths longer than maxlen. If maxlen is specified it will never
        return a path length of more than maxlen, or set such a value in the return array
        if end=None.
        
        Weighted may be set to False to not take edge lengths into account, in this case
        the result will be an integer which states the amount of edges from on the shortest
        path. The meaning of maxlen does not change.
        
        Time complexity is O(|E| log |V|) (for |E|>|V|)."""

        if type(start) == int:
            start = self.vertices[start]
        if type(end) == int:
            end = self.vertices[end]
            
        # Based on http://code.activestate.com/recipes/119466-dijkstras-algorithm-for-shortest-paths/
        
        # Determine variable that will hold distances. If end is specified, we take a dictionary
        # because we assume that we will not visit most nodes. If end is not specified we take an
        # array because we will access all nodes (and also because that is what we'll be returning).
        if end is None:
            import numpy as np
            D = np.ones((self.n_vertices(),)) * -1
        else:
            D = {} # dictionary of final distances
        
        Q = PriorityDictionary() # est.dist. of non-final vert.
        Q[start.id] = 0
        
        while Q:
            v = Q.smallest()
            D[v] = Q[v]
            Q.pop_smallest()
            
            # Stop searching if we have an end node
            if end is not None and v == end.id:
                break
            
            for w, dist in self.iter_neighbors(v):
                if weighted:
                    vwLength = D[v] + dist
                else:
                    vwLength = D[v] + 1
                    
                if ((maxlen is None or vwLength <= maxlen) # This checks that the discovered path is not longer than maxlen
                        and (end is None or w.id not in D)    # This, combined with the next line, checks whether w is not yet in D
                        and (end is not None or D[w.id] == -1)
                        and (w.id not in Q or vwLength < Q[w.id])): # This checks whether we already have a shorter path to w in Q
                        
                    Q[w.id] = vwLength
                
                # Otherwise we already have a path to w that is shorter _or_ the path we just discovered is too long
        
        if end is None:
            return D
        else:
            if end.id in D and (maxlen is None or D[end.id] <= maxlen):
                return D[end.id]
            else:
                return None
            
    def dilation_ratio(self):
        """Returns the dilation ratio of the graph, which is the largest dilation
        ratio between any two nodes in the graph. Returns None if the graph is
        disconnected.
        
        Note: Returns infinity if there are duplicate points which are not connected
        directly (when the euclidian distance is zero and the graph distance is positive).
        
        Time complexity is O(|V|*|E|*log |V|)"""
    
        import numpy as np
        
        dilation_max = 1
        
        for v in xrange(self.n_vertices()):
            # Do single-source distances to all other points in O(n log n) time, then compare then all to the Euclidian distances
            graph_dist = self.bfs_distance(v, None)
            
            # Compute all Euclidian distances, and see if the dilation for this pair is greater than dilation_max
            # No need to check lower half of the triangle (including diagonal)
            for w in xrange(v+1, self.n_vertices()):
                # If v != w and graph_dist[w.id]=0 then w is unreachable from v
                if graph_dist[w] == -1:
                    return None
                    
                euclidian_dist = self.euclidian_distance(v, w)
                if euclidian_dist > 0:
                    dilation = graph_dist[w] / euclidian_dist
                else:
                    if graph_dist[w] == 0:
                        dilation = 1
                    else:
                        # We found a point with infinite dilation, no need to process
                        # the rest of the points because you can't get higher dilation
                        # than infinity.
                        return np.inf
                    
                if dilation > dilation_max:
                    dilation_max = dilation
                    
        return dilation_max
        
    def diameter(self):
        """Returns the dilation ratio of the graph, which is the largest dilation
        ratio between any two nodes in the graph. Returns None if the graph is
        disconnected.
        
        Time complexity is O(|V|*|E|*log |V|)"""
    
        import numpy as np
        
        diameter = 1
        
        for v in xrange(self.n_vertices()):
            graph_dist = self.bfs_distance(v, None, None, False)
            eccentricity = max(graph_dist)
            diameter = max((diameter, eccentricity))
                    
        return diameter
        
    def nr_intersections(self):
        """
        Returns the number of intersections of the graph using a brute force way.
        """
        numIntersections = 0
        print len(list(self.iter_edges()))
        for edge1 in self.iter_edges():
            for edge2 in self.iter_edges():
                if edge1.v1.id >= edge2.v1.id:
                    continue
                intersection = getLineIntersection(edge1, edge2)
                
                if intersection is not None:
                    #yield intersection
                    numIntersections += 1
        
        return numIntersections
        
    def nr_intersectionsSweepLine(self):
        """Computes and returns the number of intersections between edges
        in this graph."""
        
        T = blist([])
        Q = []
        for edge in self.iter_edges():
            if edge.v1 < edge.v2:
                heappush(Q, Event(edge.v1, edge, None, START))
                heappush(Q, Event(edge.v2, edge, None, END))
        
        numIntersections = 0
        lastPopped = None
        while len(Q) > 0:
            while True:
                event = heappop(Q)
                if event != lastPopped:
                    lastPopped = event
                    break
            
            if event.type == START:
                i = bisect.bisect_left(T, event.vertex)
                T.insert(i, event.edge1)
                
                if i - 1 >= 0:
                    intersectionLeft = getLineIntersection(event.edge1, T[i-1])
                    if intersectionLeft is not None:
                        # insert in Q
                        heappush(Q, Event(intersectionLeft, event.edge1, T[i-1], INTERSECT))
                
                if i + 1 < len(T):
                    intersectionRight = getLineIntersection(event.edge1, T[i+1])
                    if intersectionRight is not None:
                        # insert in Q
                        heappush(Q, Event(intersectionRight, event.edge1, T[i+1], INTERSECT))
                
            elif event.type == END:
                i = bisect.bisect_left(T, event.vertex)
                if event.edge2 == T[i]:
                    i -= 1
                if event.edge1 != T[i]:
                    raise Exception("Assumption violated")
                del T[i]
                if i-1 >= 0 and i < len(T):
                    intersectionNeighbors = getLineIntersection(T[i-1], T[i])
                    if intersectionNeighbors is not None and intersectionNeighbors.y > event.vertex.y:
                        # insert in Q
                        heappush(Q, Event(intersectionNeighbors, T[i-1], T[i], INTERSECT))
                
            elif event.type == INTERSECT:
                i = bisect.bisect_left(T, event.vertex)
                if event.edge2 == T[i]:
                    i -= 1
                if event.edge1 != T[i]:
                    raise Exception("Assumption violated")
                if i+1 >= len(T):
                    print i
                    print T
                    print event
                T[i], T[i+1] = T[i+1], T[i]
                
                if i - 1 >= 0:
                    intersectionLeft = getLineIntersection(T[i-1], T[i])
                    if intersectionLeft is not None:
                        # insert in Q
                        heappush(Q, Event(intersectionLeft, T[i-1], T[i], INTERSECT))
                        
                if i + 2 < len(T):
                    intersectionRight = getLineIntersection(T[i+1], T[i+2])
                    if intersectionRight is not None:
                        # insert in Q
                        heappush(Q, Event(intersectionRight, T[i+1], T[i+2], INTERSECT))
                
                numIntersections += 1
                
            else:
                raise ValueError('Only START, END and INTERSECT events are supported.')
        
        return numIntersections
            
    def plot(self, transpose=False):
        import matplotlib.pyplot as plt
        
        # Plot edges
        # Use None-separated list as explained on http://exnumerus.blogspot.nl/2011/02/how-to-quickly-plot-multiple-line.html
        xlist = []
        ylist = []
        for e in self.iter_edges():
            xlist.extend([e.v1.x, e.v2.x, None])
            ylist.extend([e.v1.y, e.v2.y, None])

        if transpose:
            plt.plot(ylist, xlist)
        else:
            plt.plot(xlist, ylist)
            
        # Plot vertices
        coords = list((v.x, v.y) for v in self.iter_vertices())
        x, y = zip(*coords)
        if transpose:
            plt.scatter(y, x)
        else:
            plt.scatter(x, y)

        # Plot vertex labels
        for vertex in self.iter_vertices():
            if vertex.label:
                if transpose:
                    plt.annotate(vertex.label, (vertex.y, vertex.x))
                else:
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
    def datapoints_graph(cls, points, limit=None, removeDuplicates=None):
        g = cls()
        
        if removeDuplicates:
            points = list(set(points))

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
        
    @classmethod
    def from_data_challenge(cls, filename, removeDuplicates=None):
        g = cls()
        
        with open(filename, "r") as f:
            nr_points = int(f.readline().strip("\r\n"))
            ratio_a_b = map(int, f.readline().strip("\r\n").split())
            
            points = set()
            numPointsRead = 0
            for line in f.readlines():
                if not line:
                    continue
                    
                x, y = map(float, line.strip("\r\n").split())
                points.add((x,y))
                numPointsRead += 1
            
            if removeDuplicates:
                points = list(set(points))
            
            for p in points:
                g.add_vertex(p[0],p[1])
                
        if numPointsRead != g.n_vertices():
            print "Warning: there are %s points removed from file '%s' because of duplicate points. There are %s points left." % (numPointsRead - g.n_vertices(), filename, g.n_vertices())
        elif numPointsRead != nr_points:
            print "Warning: there should be %s points but %s were read from the file." % (nr_points, g.n_vertices())
            
        return g, float(ratio_a_b[0]) / float(ratio_a_b[1])
        
    def to_data_challenge(self, filename, dilation_ratio=1):
        """Serializes the vertices in this graph (not the edges!) to a data
        challenge file. A data challenge file needs some dilation ratio with
        it, this can be specified in the dilation_ratio parameter."""
        
        with open(filename, "w") as f:
            f.write("%d\n" % self.n_vertices())
            f.write("%d 1000\n" % int(dilation_ratio * 1000)) # Arbitrary dilation ratio
            
            for v in self.vertices:
                f.write("%s %s\n" % (v.x, v.y))

if __name__ == "__main__":
    import random
    g = Graph()
    
    for i in xrange(100):
        g.add_vertex(random.random() * 100, random.random() * 100)
        
    for i in xrange(300):
        id1 = random.randint(0, 99)
        id2 = random.randint(0, 99)
        
        if id1 == id2:
            continue
            
        #g.add_edge(id1, id2, True)

    g_trivial = Graph()
    g_trivial.add_vertex(1,1)
    g_trivial.add_vertex(1,2)
    g_trivial.add_vertex(2,2)
    g_trivial.add_edge(0,1)
    g_trivial.add_edge(1,2)
