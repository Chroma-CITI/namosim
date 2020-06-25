import mapbox_earcut as earcut
import fcl
import numpy as np
from shapely.geometry import Polygon
import matplotlib.pyplot as plt


def triangulate(polygon):
    if isinstance(polygon, Polygon):
        verts = np.array(list(polygon.exterior.coords)).reshape(-1, 2)
        rings = np.array([verts.shape[0]])
        triangles_vertices = verts[earcut.triangulate_float64(verts, rings)]
        triangles = [triangles_vertices[n:n + 3] for n in range(0, len(triangles_vertices), 3)]
        return np.array(triangles)


def triangulate_3d(polygon):
    if isinstance(polygon, Polygon):
        verts = np.array(list(polygon.exterior.coords)).reshape(-1, 2)
        rings = np.array([verts.shape[0]])
        triangles_vertices = verts[earcut.triangulate_float64(verts, rings)]
        triangle_unique_vertices = np.array(list({tuple(vertex) for vertex in triangles_vertices}))
        triangles = [triangles_vertices[n:n + 3] for n in range(0, len(triangles_vertices), 3)]
        triangles_tris = [triangles_vertices[n:n + 3] for n in range(0, len(triangles_vertices), 3)]
        return triangle_unique_vertices, triangles_tris


if __name__ == '__main__':
    obs_01_s = Polygon([(1.5, 1.5), (1.5, 2.), (2., 2.), (2., 1.5)])
    obs_02_m = Polygon([(0., 1.), (0., 2.), (1., 2.), (1., 1.)])

    # plt.plot(*obs_01_s.exterior.xy)
    # plt.plot(*obs_02_m.exterior.xy)

    tri_obs_01_s = triangulate(obs_01_s)
    tri_obs_01_s_3d = triangulate_3d(obs_01_s)

    for tri in tri_obs_01_s:
        x, y = [point[0] for point in tri], [point[1] for point in tri]
        plt.plot(x, y)

    plt.axis('equal')
    # plt.axhline(y=0)
    # plt.axvline(x=0)
    plt.show()
