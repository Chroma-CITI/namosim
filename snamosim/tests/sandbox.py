from shapely.geometry import Polygon, MultiPoint, Point, LineString
import shapely.affinity as affinity
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

if __name__ == '__main__':
    p1 = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])

    pfin = affinity.rotate(geom=affinity.translate(
        geom=p1, xoff=1., yoff=2., zoff=0.),
        angle=215., origin=(0., 0.), use_radians=False
    )

    pmoitie = affinity.rotate(geom=affinity.translate(
        geom=p1, xoff=1./2., yoff=2./2., zoff=0.),
        angle=215./2., origin=(0., 0.), use_radians=False
    )

    pquart = affinity.rotate(geom=affinity.translate(
        geom=p1, xoff=1./4., yoff=2./4., zoff=0.),
        angle=215./4., origin=(0., 0.), use_radians=False
    )

    ptroisquarts = affinity.rotate(geom=affinity.translate(
        geom=p1, xoff=1.*3./4., yoff=2.*3./4., zoff=0.),
        angle=215.*3./4., origin=(0., 0.), use_radians=False
    )

    polygon_states = [p1, pquart, pmoitie, ptroisquarts, pfin]

    fig, ax = plt.subplots()

    for p in polygon_states:
        ax.plot(*p.exterior.xy, color='grey')

    points_states = zip(*[list(polygon.exterior.coords) for polygon in polygon_states])

    for point_states in points_states:
        x, y = zip(*[point for point in point_states])
        ax.scatter(x, y)

    ax.axis('equal')
    fig.show()
    print()
