from collections import namedtuple

class BoundaryType(object):

        NE = 1 #North-East
        NW = 2 #North-West
        SW = 3 #South-West
        SE = 4 #South-East
        Root = 5
        
Point = namedtuple('Point', ['x', 'y'])

class Boundary(object):
    """Boundary to represent an area of the quadtree."""
    
    __slots__ = ('Cx', 'Cy', 'halfWidth','type')
    
    def __init__(self, x, y, halfWidth, type):
        self.Cx = x
        self.Cy = y
        self.halfWidth = halfWidth
        self.type = type #type from BoundaryType
    
    def contains_point(self, Px, Py):
        """Returns whether the area bounded by the boundary contains the point (Px,Py)."""
        if self.type == BoundaryType.NE:
            return Px > self.Cx - self.halfWidth and Py > self.Cy - self.halfWidth
        elif self.type == BoundaryType.NW:
            return Px <= self.Cx + self.halfWidth and Py > self.Cy - self.halfWidth
        elif self.type == BoundaryType.SW:
            return Px <= self.Cx + self.halfWidth and Py <= self.Cy + self.halfWidth
        elif self.type == BoundaryType.SE:
            return Px > self.Cx - self.halfWidth and Py <= self.Cy + self.halfWidth;
        elif self.type == BoundaryType.Root:
            return Px <= self.Cx + self.halfWidth and Px >= self.Cx - self.halfWidth and Py <= self.Cy + self.halfWidth and Py >= self.Cy - self.halfWidth;
        else:
            print 'Unknown boundary type'
            return false
    
    def __str__(self):
        return str(self.Cx) + ',' + str(self.Cy) + ':  ' + str(self.halfWidth)
        
class Quadtree(object):

    __slots__ = ('parent', 'boundary', 'points', 'children', 'depth', 'representative')
    
    quadtree_node_capacity = 1

    def __init__(self, parent, boundary):
        if type(boundary) == Boundary:
            self.boundary = boundary
        else:
            raise Exception('Unknown boundary.')
            
        self.parent = parent
        self.points = []
        self.children = [None, None, None, None] #order: North-East, North-West, South-West, South-East
        self.representative = None
        
        if parent == None:
            self.depth = 0
        else:
            self.depth = parent.depth + 1
    
    def insert_point(self, x, y):
        """Insert point (x,y) into the quadtree and subdivide the quadtree recursively if necessary."""
        
        if not self.boundary.contains_point(x,y):
            return False
        
        if self.children[0] == None and len(self.points) < self.quadtree_node_capacity:
            self.points.append(Point(x=x, y=y))
            if self.representative == None:
                self.representative = Point(x=x, y=y)
            return True
        
        if self.children[0] == None:
            self.subdivide()
            
        if self.children[0].boundary.contains_point(x,y):
            return self.children[0].insert_point(x,y)
        elif self.children[1].boundary.contains_point(x,y):
            return self.children[1].insert_point(x,y)
        elif self.children[2].boundary.contains_point(x,y):
            return self.children[2].insert_point(x,y)
        elif self.children[3].boundary.contains_point(x,y):
            return self.children[3].insert_point(x,y)
        else:
            print 'Unknown child:  ' + str(x) + ' ' + str(y)
            return False
        
    
    def subdivide(self):
        """Subdivide the quadtree into four quadrants and divide the points 
        of the original quadtree over its children."""
        
        if self.children[0] == None:
            newHalfWidth = self.boundary.halfWidth / float(2)
            centerX = self.boundary.Cx
            centerY = self.boundary.Cy
            
            self.children = [
                Quadtree(self, Boundary(centerX + newHalfWidth, centerY + newHalfWidth, newHalfWidth, BoundaryType.NE)),
                Quadtree(self, Boundary(centerX - newHalfWidth, centerY + newHalfWidth, newHalfWidth, BoundaryType.NW)),
                Quadtree(self, Boundary(centerX - newHalfWidth, centerY - newHalfWidth, newHalfWidth, BoundaryType.SW)),
                Quadtree(self, Boundary(centerX + newHalfWidth, centerY - newHalfWidth, newHalfWidth, BoundaryType.SE))]
                
            for point in self.points:
                if self.children[0].boundary.contains_point(point.x, point.y):
                    self.children[0].insert_point(point.x, point.y)
                elif self.children[1].boundary.contains_point(point.x, point.y):
                    self.children[1].insert_point(point.x, point.y)
                elif self.children[2].boundary.contains_point(point.x, point.y):
                    self.children[2].insert_point(point.x, point.y)
                elif self.children[3].boundary.contains_point(point.x, point.y):
                    self.children[3].insert_point(point.x, point.y)
                else:
                    print 'Unknown child'
            self.points = []
            
    def find_neighbor(self, direction):
        raise Exception('Not implemented')
                
    def iter_points(self):
        """Returns an iterator for all points in the quadtree. The order is 
        depth-first-search on the quadtree with the quadrants in the 
        order NE, NW, SW, SE."""
        
        toReturn = list(self.points)
        for child in self.children:
            if child == None:
                break
            toReturn.extend(child.iter_points())
        return iter(toReturn)
    
    def _returnInnerBoundaryQuadtree(self):
        """Returns for each quadtree level its inner boundary line segments."""
        if self.children[0] == None:
            return [],[]
            
        halfWidth = self.boundary.halfWidth
        Cx = self.boundary.Cx
        Cy = self.boundary.Cy
        xList = [Cx, Cx, None, Cx - halfWidth, Cx + halfWidth, None]
        yList = [Cy - halfWidth, Cy + halfWidth, None, Cy, Cy, None]
        
        for child in self.children:
            if child.children[0] != None:
                xListChild, yListChild = child._returnInnerBoundaryQuadtree()
                xList.extend(xListChild)
                yList.extend(yListChild)
        
        return xList, yList
                
    def plot(self):
        import numpy as np
        import matplotlib.pyplot as plt
        
        # plot quadtree
        # Use None-separated list as explained on http://exnumerus.blogspot.nl/2011/02/how-to-quickly-plot-multiple-line.html
        halfWidth = self.boundary.halfWidth
        Cx = self.boundary.Cx
        Cy = self.boundary.Cy
        
        if self.boundary.type == BoundaryType.Root:
            plt.plot([Cx - halfWidth, Cx - halfWidth, None],[Cy - halfWidth, Cy + halfWidth, None])
            plt.plot([Cx - halfWidth, Cx + halfWidth, None],[Cy - halfWidth, Cy - halfWidth, None])
            plt.plot([Cx + halfWidth, Cx + halfWidth, None],[Cy - halfWidth, Cy + halfWidth, None])
            plt.plot([Cx - halfWidth, Cx + halfWidth, None],[Cy + halfWidth, Cy + halfWidth, None])
        
        xList,yList = self._returnInnerBoundaryQuadtree()
        plt.plot(xList,yList)
        
        # plot points
        try:
            self.iter_points().next()
            
            coords = list((v.x, v.y) for v in self.iter_points())
            x, y = zip(*coords)
            plt.scatter(x, y)
            
        except StopIteration:
            print 'No points inserted.'

        plt.show()
        
    @classmethod
    def random_quadtree(cls, Cx, Cy, halfWidth, n_points):
        import random
    
        q = cls(None, Boundary(Cx, Cy, halfWidth, BoundaryType.Root))
        
        for i in xrange(n_points):
            q.insert_point(random.random() * 100, random.random() * 100)
            
        return q
        
        
'''        
b = Boundary(50, 50, 50, BoundaryType.Root)
q = Quadtree(None, b)

import random

for i in range(100):
    x,y = random.random() * 100, random.random() * 100
    #print str(x) + ',' + str(y)
    q.insert_point(x,y)
    
q.plot()
'''
