#include "functions/Hilbert.h"

//convert (x,y) to d
auto SEM::Hilbert::xy2d(int n, std::array<int, 2> xy) -> int {
    int d = 0;
    for (int s = n/2; s > 0; s /= 2) {
        const std::array<int, 2> rxy {(xy[0] & s) > 0,
                                      (xy[1] & s) > 0};

        d += s * s * ((3 * rxy[0]) ^ rxy[1]);
        rot(n, xy, rxy);
    }
    return d;
}

//convert d to (x,y)
auto SEM::Hilbert::d2xy(int n, int d) -> std::array<int, 2> {
    std::array<int, 2> xy{0, 0};
    for (int s = 1; s < n; s *= 2) {
        std::array<int, 2> rxy{1 & (d/2), 0};
        rxy[1] = 1 & (d ^ rxy[0]);
        rot(s, xy, rxy);
        xy[0] += s * rxy[0];
        xy[1] += s * rxy[1];
        d /= 4;
    }
    return xy;
}

//rotate/flip a quadrant appropriately
auto SEM::Hilbert::rot(int n, std::array<int, 2>& xy, std::array<int, 2> rxy) -> void {
    if (rxy[1] == 0) {
        if (rxy[0] == 1) {
            xy[0] = n-1 - xy[0];
            xy[1] = n-1 - xy[1];
        }

        //Swap x and y
        std::swap(xy[0], xy[1]);
    }
}