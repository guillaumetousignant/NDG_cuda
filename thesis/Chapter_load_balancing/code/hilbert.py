import numpy as np
import numpy.typing as npt

def xy2d(n: int, xy: npt.ArrayLike) -> int:
    d = 0
    s = n/2
    while s > 0:
        rxy = np.array([(xy[0] & s) > 0,
                        (xy[1] & s) > 0], dtype=bool)

        d += s * s * ((3 * rxy[0]) ^ rxy[1])
        xy = rot(n, xy, rxy)
        s /= 2
    
    return d

def d2xy(n: int, d: int) -> npt.ArrayLike:
    xy = np.zeros(2, dtype=int)
    s = 1
    while s < n:
        rxy = np.array([1 & int(d/2), 0], dtype=int)
        rxy[1] = 1 & (int(d) ^ rxy[0])
        xy = rot(s, xy, rxy)
        xy[0] += s * rxy[0]
        xy[1] += s * rxy[1]
        d /= 4
        s *= 2
    
    return xy

def rot(n: int, xy: npt.ArrayLike, rxy: npt.ArrayLike) -> npt.ArrayLike:
    if rxy[1] == 0:
        if rxy[0] == 1:
            xy[0] = n-1 - xy[0]
            xy[1] = n-1 - xy[1]

        # Swap x and y
        xy[0], xy[1] = xy[1], xy[0]
    
    return xy
