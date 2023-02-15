import shapely.geometry as g

a = g.Polygon(((0., 0.), (0., 1.), (1., 1.), (1., 0.), (0., 0.)))
b = g.LineString(((-1,-1), (2,2)))
print(a.intersection(b))