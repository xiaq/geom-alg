from QuadTree import QuadTree
from QuadTree import Point
from QuadTree import bounding_box
from QuadTree import createQuadTree
from numpy import sqrt

from datatypes import Graph

def is_well_separated(u, v, s):
    """
    Determine if the points in the quadtrees u and v are s-well-separated.

    This means that dist(ball(u), ball(v)) >= s * max(radius(ball(u)), radius(ball(v))
    where ball(u) represents a ball which encloses the points of u. The ball has center
    the arithmetic mean of the points and radius the distance between the center and
    a corner of the bounding box of the points.
    """

    def dist(a,b):
        """
        Compute the Euclidean distance between points a and b.
        """
        dx = b.x - a.x
        dy = b.y - a.y

        return sqrt(dx**2 + dy**2)

    bbU = bounding_box(u.points)
    bbV = bounding_box(v.points)

    centerU = Point((bbU['xMin'] + bbU['xMax'])/float(2), (bbU['yMin'] + bbU['yMax'])/float(2))
    radiusU = dist(centerU, Point(bbU['xMin'], bbU['yMin']))
    
    centerV = Point((bbV['xMin'] + bbV['xMax'])/float(2), (bbV['yMin'] + bbV['yMax'])/float(2))
    radiusV = dist(centerV, Point(bbV['xMin'], bbV['yMin']))

    distUV = dist(centerU, centerV) - radiusU - radiusV

    return distUV >= s * max(radiusU, radiusV)



def wsPairs(u, v, s):
    """
    Calculates all s-well-separated pairs where one cell of the pairs is in quadtree u
    and the other cell of the pair is in quadtree v.
    """

    if len(u.points) == 0 or len(v.points) == 0:
        return []
    elif is_well_separated(u,v,s) and u != v:
        return [(u.points[0], v.points[0])]
    else:
        if u.depth > v.depth:
            temp = v
            v = u
            u = temp
        pairs = []

        if u.children != None:
            if u == v:
                # Make sure to not recurse on the same pair twice to
                # prevent the same wspd-pairs being generated
                pairs.extend(wsPairs(u.children[0], v.children[0], s))
                pairs.extend(wsPairs(u.children[0], v.children[1], s))
                pairs.extend(wsPairs(u.children[0], v.children[2], s))
                pairs.extend(wsPairs(u.children[0], v.children[3], s))
                pairs.extend(wsPairs(u.children[1], v.children[1], s))
                pairs.extend(wsPairs(u.children[1], v.children[2], s))
                pairs.extend(wsPairs(u.children[1], v.children[3], s))
                pairs.extend(wsPairs(u.children[2], v.children[2], s))
                pairs.extend(wsPairs(u.children[2], v.children[3], s))
                pairs.extend(wsPairs(u.children[3], v.children[3], s))
            else:
                for child in u.children:
                    pairs.extend(wsPairs(child, v, s))
                    
        elif v.children != None: # u is a leaf but v still has points so check if we can recurse on v
            temp = v
            v = u
            u = temp

            if u == v:
                # Make sure to not recurse on the same pair twice to
                # prevent the same wspd-pairs being generated
                pairs.extend(wsPairs(u.children[0], v.children[0], s))
                pairs.extend(wsPairs(u.children[0], v.children[1], s))
                pairs.extend(wsPairs(u.children[0], v.children[2], s))
                pairs.extend(wsPairs(u.children[0], v.children[3], s))
                pairs.extend(wsPairs(u.children[1], v.children[1], s))
                pairs.extend(wsPairs(u.children[1], v.children[2], s))
                pairs.extend(wsPairs(u.children[1], v.children[3], s))
                pairs.extend(wsPairs(u.children[2], v.children[2], s))
                pairs.extend(wsPairs(u.children[2], v.children[3], s))
                pairs.extend(wsPairs(u.children[3], v.children[3], s))
            else:
                for child in u.children:
                    pairs.extend(wsPairs(child, v, s))
            
        return pairs

def wspd_spanner(graph, dilation):
    """
    Calculates a spanner with the specified dilation ratio using well-separated points
    which are based on a quadtree.
    """
    points = list(graph.iter_vertices())
    quadtree = createQuadTree(points)
    s = 4 * float(dilation+1) / float(dilation-1)
    wspd_pairs = wsPairs(quadtree, quadtree, s)

    for pair in wspd_pairs:
        graph.add_edge(pair[0].id, pair[1].id, False)

    #graph.plot()


#interesting points:
""" 
points = [
Point(11.42345273512,8.1526698833),
Point(80.8117542451,86.1762624158),
Point(24.7525670094,8.22273162758),
Point(15.6036472629,8.23681644359)]

g = Graph()
for p in points:
    g.add_vertex(p.x, p.y)

g2 = Graph.random_graph(700, 0)

import time

t_start = time.clock()
wspd_spanner(g2, dilation = 3) 
t_elapsed = time.clock() - t_start

print t_elapsed

print g2.numDoubleEdges
"""
