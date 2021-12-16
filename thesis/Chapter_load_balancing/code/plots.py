import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import hilbert

K = [2, 4, 8]
x_min = 0
x_max = 1
y_min = 0
y_max = 1
points_colour = np.array([197, 134, 192])/255
elements_colour = np.array([37, 37, 37])/255
curve_colour = np.array([106, 153, 85])/255
points_width = 12
elements_width = 3
curve_width = 5
points_size = 12
points_shape = "."

save_path = Path.cwd().parent / "media"
save_path.mkdir(parents=True, exist_ok=True)

# This is basically the mesh generator, without boundaries
for k in K:
    x_res = k
    y_res = k
    x_node_res = x_res + 1
    y_node_res = y_res + 1
    n_elements = x_res * y_res
    n_nodes = x_node_res * y_node_res

    x = np.zeros(n_nodes)
    y = np.zeros(n_nodes)

    for i in range(x_node_res):
        for j in range(y_node_res):
            x[i * y_node_res + j] = x_min + i * (x_max - x_min)/(x_node_res - 1)
            y[i * y_node_res + j] = y_min + j * (y_max - y_min)/(y_node_res - 1)

    n_sides = 4
    elements = np.zeros((n_elements, n_sides), dtype=np.uint)
    elements_center_x = np.zeros(n_elements)
    elements_center_y = np.zeros(n_elements)

    for i in range(n_elements):
        xy = hilbert.d2xy(k, i)
        elements[i][0] = y_node_res * (xy[0] + 1) + xy[1]
        elements[i][1] = y_node_res * (xy[0] + 1) + xy[1] + 1
        elements[i][2] = y_node_res * xy[0] + xy[1] + 1
        elements[i][3] = y_node_res * xy[0] + xy[1]

        elements_center_x[i] = (x[elements[i][0]] + x[elements[i][1]] + x[elements[i][2]] + x[elements[i][3]])/4
        elements_center_y[i] = (y[elements[i][0]] + y[elements[i][1]] + y[elements[i][2]] + y[elements[i][3]])/4

    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.1)
    ax.set_aspect(1)
    ax.axis('off')
    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)

    for i in range(n_elements):
        ax.plot([x[elements[i][0]], x[elements[i][1]], x[elements[i][2]], x[elements[i][3]], x[elements[i][0]]], [y[elements[i][0]], y[elements[i][1]], y[elements[i][2]], y[elements[i][3]], y[elements[i][0]]], color=elements_colour, linewidth=elements_width, label="Elements" if i == 0 else "")

    ax.plot(x, y, color=points_colour, linestyle="None", linewidth=points_width, marker=points_shape, markersize=points_size, label="Nodes")

    ax.plot(elements_center_x, elements_center_y, color=curve_colour, linewidth=curve_width, label="Hilbert curve")

    fig.savefig(save_path / f"hilbert_curve_K{k}.svg", format='svg', transparent=True)

plt.show()