"""
Ear Clipping Algorithm

Implementation by Github user mrbaozi: https://github.com/mrbaozi

Released under MIT License: https://github.com/mrbaozi/triangulation/blob/master/LICENSE

Original repository: https://github.com/mrbaozi/triangulation
"""


def is_convex(a, b, c):
    # only convex if traversing anti-clockwise!
    crossp = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    if crossp >= 0:
        return True
    return False


def in_triangle(a, b, c, p):
    barycentric_coeffs = [0, 0, 0]
    eps = 0.0000001
    # calculate barycentric coefficients for point p
    # eps is needed as error correction since for very small distances denom->0
    barycentric_coeffs[0] = ((b[1] - c[1]) * (p[0] - c[0]) + (c[0] - b[0]) * (p[1] - c[1])) \
        / (((b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])) + eps)
    barycentric_coeffs[1] = ((c[1] - a[1]) * (p[0] - c[0]) + (a[0] - c[0]) * (p[1] - c[1])) \
        / (((b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])) + eps)
    barycentric_coeffs[2] = 1 - barycentric_coeffs[0] - barycentric_coeffs[1]
    # check if p lies in triangle (a, b, c)
    for x in barycentric_coeffs:
        if x > 1 or x < 0:
            return False
    return True


def is_clockwise(poly):
    # initialize sum with last element
    _sum = (poly[0][0] - poly[len(poly) - 1][0]) * \
        (poly[0][1] + poly[len(poly) - 1][1])
    # iterate over all other elements (0 to n-1)
    for i in range(len(poly) - 1):
        _sum += (poly[i + 1][0] - poly[i][0]) * (poly[i + 1][1] + poly[i][1])
    if _sum > 0:
        return True
    return False


def get_ear(poly):
    size = len(poly)
    if size < 3:
        return []
    if size == 3:
        tri = (poly[0], poly[1], poly[2])
        del poly[:]
        return tri
    for i in range(size):
        tritest = False
        p1 = poly[(i - 1) % size]
        p2 = poly[i % size]
        p3 = poly[(i + 1) % size]
        if is_convex(p1, p2, p3):
            for x in poly:
                if not (x in (p1, p2, p3)) and in_triangle(p1, p2, p3, x):
                    tritest = True
            if not tritest:
                del poly[i % size]
                return p1, p2, p3
    print('get_ear(): no ear found')
    return []


def triangulate(poly_pts):
    pts_to_triangulate = poly_pts[::-1] if is_clockwise(poly_pts) else poly_pts[:]
    triangles = []
    while len(pts_to_triangulate) >= 3:
        a = get_ear(pts_to_triangulate)
        if not a:
            break
        triangles.append(a)
    return triangles
