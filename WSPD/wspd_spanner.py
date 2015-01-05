from datatypes import Quadtree
from datatypes import Boundary
from datatypes import BoundaryType

class WSPD_spanner(object):
    """NOT FINISHED!
       Although all functions are implemented, the output of wdpd_spanner() 
       is not yet correct. To be continued.
    """

    def __init__(self, quadtree, dilation):
        self.quadtree = quadtree
        self.dilation = dilation
        self.s = 4*(float(dilation+1)/float(dilation-1))

    def wsPairs(self, u, v):
        """Generates well-separated pairs from u and v."""
        if u.representative == None or v.representative == None:
            return []
        elif self._check_s_well_separated(u, v) and u != v:
            if u.depth == v.depth:
                if u.representative.x > v.representative.x or (u.representative.x == v.representative.x and u.representative.y >= v.representative.y):
                    return [(u, v)]
                else:
                    return []
            else:
                return [(u.representative, v.representative)]
        else:
            if u.depth > v.depth:
                temp = u
                u = v
                v = temp
            if u.children[0] == None:
                return []
            else:
                toReturn = []
                for child in u.children:
                    toReturn.extend(self.wsPairs(child, v))
                return toReturn

    def _check_s_well_separated(self, u, v):
        """Checks whether u and v are s-well separated."""
        
        radiusU = 0 if u.children[0] == None else u.boundary.halfWidth
        radiusV = 0 if v.children[0] == None else v.boundary.halfWidth
        maxRadius = (2**0.5) * max(radiusU, radiusV)
        
        distBetweenDisks = ((u.boundary.Cx - v.boundary.Cx)**2 + (u.boundary.Cy - v.boundary.Cy)**2)**0.5 - radiusU - radiusV
        return distBetweenDisks >= self.s * maxRadius

    def wspd_spanner(self):
        """..."""
        list = self.wsPairs(self.quadtree, self.quadtree)
        print 'Number of well-separated pairs: ' + str(len(list))
        for (l1,l2) in list:
            pass
            print str(l1.boundary) + '   --   ' + str(l2.boundary)

if __name__ == "__main__":
    import random

    q = Quadtree.random_quadtree(50,50,50,4)
    
    WSPD = WSPD_spanner(q,1.5)
    WSPD.wspd_spanner()
    q.plot()
    
