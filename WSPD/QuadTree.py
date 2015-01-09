from collections import namedtuple

Point = namedtuple('Point', ['x', 'y'])

class BoundaryType(object):

    NE = 1 #North-East
    NW = 2 #North-West
    SW = 3 #South-West
    SE = 4 #South-East
    Root = 5


class Boundary(object):
    """Boundary to represent an area of the quadtree."""
    
    __slots__ = ('Cx', 'Cy', 'halfWidth','type')
    
    def __init__(self, x, y, halfWidth, type):
        self.Cx = x
        self.Cy = y
        self.halfWidth = halfWidth
        self.type = type #type from BoundaryType
    
    def contains_point(self, Px, Py, typeOfBoundary = None):
        """Returns whether the area bounded by the boundary contains the point (Px,Py).

        Note: it is not checked if the point lies in the outer bounding box. To check this,
        use typeOfBoundary = BoundaryType.root."""

        if typeOfBoundary == None:
            typeOfBoundary = self.type
            
        if typeOfBoundary == BoundaryType.NE:
            return Px > self.Cx - self.halfWidth and Py > self.Cy - self.halfWidth
        elif typeOfBoundary == BoundaryType.NW:
            return Px <= self.Cx + self.halfWidth and Py > self.Cy - self.halfWidth
        elif typeOfBoundary == BoundaryType.SW:
            return Px <= self.Cx + self.halfWidth and Py <= self.Cy + self.halfWidth
        elif typeOfBoundary == BoundaryType.SE:
            return Px > self.Cx - self.halfWidth and Py <= self.Cy + self.halfWidth;
        elif typeOfBoundary == BoundaryType.Root:
            return Px <= self.Cx + self.halfWidth and Px >= self.Cx - self.halfWidth and Py <= self.Cy + self.halfWidth and Py >= self.Cy - self.halfWidth;
        else:
            print 'Unknown boundary type'
            return false
    
def __str__(self):
        return str(self.Cx) + ',' + str(self.Cy) + ':  ' + str(self.halfWidth)

class QuadTree(object):

    __slots__ = ('parent', 'points', 'children', 'depth', 'boundary')

    def __init__(self, parent, boundary, points):
        self.parent = parent
        self.points = points
        if not isinstance(boundary, Boundary):
            raise Exception('Unknown boundary: ' + boundary)
        self.boundary = boundary

        if parent == None:
            self.depth = 0
        else:
            self.depth = parent.depth + 1

        if len(points) <= 1:
            self.children = None
        else:
            NE_points, NW_points, SW_points, SE_points = [], [], [], []

            newHalfWidth = self.boundary.halfWidth / float(2)
            centerX = self.boundary.Cx
            centerY = self.boundary.Cy

            NE_boundary = Boundary(centerX + newHalfWidth, centerY + newHalfWidth, newHalfWidth, BoundaryType.NE)
            NW_boundary = Boundary(centerX - newHalfWidth, centerY + newHalfWidth, newHalfWidth, BoundaryType.NW)
            SW_boundary = Boundary(centerX - newHalfWidth, centerY - newHalfWidth, newHalfWidth, BoundaryType.SW)
            SE_boundary = Boundary(centerX + newHalfWidth, centerY - newHalfWidth, newHalfWidth, BoundaryType.SE)

            for p in points:
                if NE_boundary.contains_point(p.x, p.y):
                    NE_points.append(p)
                elif NW_boundary.contains_point(p.x, p.y):
                    NW_points.append(p)
                elif SW_boundary.contains_point(p.x, p.y):
                    SW_points.append(p)
                elif SE_boundary.contains_point(p.x, p.y):
                    SE_points.append(p)
                else:
                    raise Exception('Unknown child quadrant for point ' + p.x + ',' + p.y)

            self.children = [
                QuadTree(self, NE_boundary, NE_points),
                QuadTree(self, NW_boundary, NW_points),
                QuadTree(self, SW_boundary, SW_points),
                QuadTree(self, SE_boundary, SE_points)]

    def find_neighbor(self, direction):
        raise Exception('Not implemented')
    
    def _returnInnerBoundaryQuadtree(self):
        """Returns for each quadtree level its inner boundary line segments."""
        if self.children == None:
            return [],[]
            
        halfWidth = self.boundary.halfWidth
        Cx = self.boundary.Cx
        Cy = self.boundary.Cy
        xList = [Cx, Cx, None, Cx - halfWidth, Cx + halfWidth, None]
        yList = [Cy - halfWidth, Cy + halfWidth, None, Cy, Cy, None]
        
        for child in self.children:
            if child.children != None:
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
            coords = list((v.x, v.y) for v in self.points)
            x, y = zip(*coords)
            plt.scatter(x, y)
            
        except StopIteration:
            print 'No points inserted.'

        plt.show()
    
    @classmethod
    def random_quadtree(cls, Cx, Cy, halfWidth, n_points):
        import random

        points = []
        for i in xrange(n_points):
            points.append(Point(random.random() * halfWidth * 2, random.random() *  halfWidth * 2))

        q = createQuadTree(points)
            
        return q

def bounding_box(points):
    """Returns a dictionary of the minimal and maximum x and y-coordinate of the points."""
        
    return {'xMin': min(p.x for p in points),
            'yMin': min(p.y for p in points),
            'xMax': max(p.x for p in points),
            'yMax': max(p.y for p in points)}

def createQuadTree(points):
    """Create a quadtree of the points with extra free space in every direction."""
        
    bb = bounding_box(points)
    lengthLongestDimension = max(bb['xMax'] - bb['xMin'], bb['yMax'] - bb['yMin'])
    boundary = Boundary(float(bb['xMin'] + bb['xMax'])/2, float(bb['yMin'] + bb['yMax'])/2, lengthLongestDimension/float(5)*3, BoundaryType.Root)

    return QuadTree(None, boundary, points)

points = [Point(1,1), Point(50,80)]

Q = QuadTree.random_quadtree(50,50,50,20)

Q.plot()
