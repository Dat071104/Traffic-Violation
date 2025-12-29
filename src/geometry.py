def cross_sign(p, a, b):
    return (b[0]-a[0])*(p[1]-a[1]) - (b[1]-a[1])*(p[0]-a[0])


def point_in_polygon(pt, poly):
    x, y = pt
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i+1) % n]
        
        if ((y1 > y) != (y2 > y)) and (x < (x2-x1) * (y-y1) / (y2-y1 + 1e-9) + x1):
            inside = not inside
    return inside