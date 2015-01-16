def plot(points, links=True, dots=True):
    """If dots is set to True, draws dots on the points. If links is True,
    draws links between points. If both are false nothing is drawn."""
    
    import matplotlib.pyplot as plt
    
    # Plot links
    if links:
        # Use None-separated list as explained on http://exnumerus.blogspot.nl/2011/02/how-to-quickly-plot-multiple-line.html
        xlist = []
        ylist = []
        for i in range(0, len(points)-1):
            xlist.extend([points[i][0], points[i+1][0], None])
            ylist.extend([points[i][1], points[i+1][1], None])
        
        plt.plot(xlist, ylist)
        
    # Plot points
    if points:
        x, y = zip(*points)
        plt.scatter(x, y)

    plt.show()