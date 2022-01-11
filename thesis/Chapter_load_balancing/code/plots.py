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
background_curve_colour = np.array([181, 206, 168])/255
H_colour = np.array([244, 71, 61])/255
A_colour = np.array([79, 193, 255])/255
R_colour = np.array([78, 201, 176])/255
B_colour = np.array([197, 134, 192])/255
points_width = 12
elements_width = 3
curve_width = 5
background_curve_width = 5
points_size = 12
points_shape = "."
hilbert_font_size = 36
arrow_width = 0.025
arrow_head_length = 0.05

save_path = Path(__file__).parent.parent / "media"
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

# Hilbert splits
H_fig = plt.figure(figsize=(4, 10))
H_ax = H_fig.add_subplot(1, 1, 1)
H_ax.set_xlim(-0.01, 1.01)
H_ax.set_ylim(-0.01, 2.51)
H_ax.set_aspect(1)
H_ax.axis('off')
H_fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)

H_ax.plot([0, 1, 1, 0, 0], [1.5, 1.5, 2.5, 2.5, 1.5], color=elements_colour, linewidth=elements_width, label="Elements")

H_ax.text(0.5, 2, "H", fontfamily="Fira Code", fontsize=hilbert_font_size, horizontalalignment="center", verticalalignment="center", color=H_colour)

H_ax.plot([0, 0.5, 0.5, 0, 0], [0, 0, 0.5, 0.5, 0], color=elements_colour, linewidth=elements_width)
H_ax.plot([0.5, 1, 1, 0.5, 0.5], [0, 0, 0.5, 0.5, 0], color=elements_colour, linewidth=elements_width)
H_ax.plot([0.5, 1, 1, 0.5, 0.5], [0.5, 0.5, 1, 1, 0.5], color=elements_colour, linewidth=elements_width)
H_ax.plot([0, 0.5, 0.5, 0, 0], [0.5, 0.5, 1, 1, 0.5], color=elements_colour, linewidth=elements_width)

H_ax.arrow(0.5, 1.375, 0, -0.25, length_includes_head=True, width=arrow_width, head_length=arrow_head_length, color=elements_colour)
H_ax.arrow(0.25, 0.375, 0, 0.25, length_includes_head=True, width=arrow_width, head_length=arrow_head_length, color=curve_colour)
H_ax.arrow(0.375, 0.75, 0.25, 0, length_includes_head=True, width=arrow_width, head_length=arrow_head_length, color=curve_colour)
H_ax.arrow(0.75, 0.625, 0, -0.25, length_includes_head=True, width=arrow_width, head_length=arrow_head_length, color=curve_colour)

H_ax.text(0.25, 0.25, "A", fontfamily="Fira Code", fontsize=hilbert_font_size, horizontalalignment="center", verticalalignment="center", color=H_colour)
H_ax.text(0.75, 0.25, "B", fontfamily="Fira Code", fontsize=hilbert_font_size, horizontalalignment="center", verticalalignment="center", color=H_colour)
H_ax.text(0.75, 0.75, "H", fontfamily="Fira Code", fontsize=hilbert_font_size, horizontalalignment="center", verticalalignment="center", color=H_colour)
H_ax.text(0.25, 0.75, "H", fontfamily="Fira Code", fontsize=hilbert_font_size, horizontalalignment="center", verticalalignment="center", color=H_colour)

H_fig.savefig(save_path / f"hilbert_split_H.svg", format='svg', transparent=True)

A_fig = plt.figure(figsize=(4, 10))
A_ax = A_fig.add_subplot(1, 1, 1)
A_ax.set_xlim(-0.01, 1.01)
A_ax.set_ylim(-0.01, 2.51)
A_ax.set_aspect(1)
A_ax.axis('off')
A_fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)

A_ax.plot([0, 1, 1, 0, 0], [1.5, 1.5, 2.5, 2.5, 1.5], color=elements_colour, linewidth=elements_width, label="Elements")

A_ax.text(0.5, 2, "A", fontfamily="Fira Code", fontsize=hilbert_font_size, horizontalalignment="center", verticalalignment="center", color=A_colour)

A_ax.plot([0, 0.5, 0.5, 0, 0], [0, 0, 0.5, 0.5, 0], color=elements_colour, linewidth=elements_width)
A_ax.plot([0.5, 1, 1, 0.5, 0.5], [0, 0, 0.5, 0.5, 0], color=elements_colour, linewidth=elements_width)
A_ax.plot([0.5, 1, 1, 0.5, 0.5], [0.5, 0.5, 1, 1, 0.5], color=elements_colour, linewidth=elements_width)
A_ax.plot([0, 0.5, 0.5, 0, 0], [0.5, 0.5, 1, 1, 0.5], color=elements_colour, linewidth=elements_width)

A_ax.arrow(0.5, 1.375, 0, -0.25, length_includes_head=True, width=arrow_width, head_length=arrow_head_length, color=elements_colour)
A_ax.arrow(0.375, 0.25, 0.25, 0, length_includes_head=True, width=arrow_width, head_length=arrow_head_length, color=curve_colour)
A_ax.arrow(0.75, 0.375, 0, 0.25, length_includes_head=True, width=arrow_width, head_length=arrow_head_length, color=curve_colour)
A_ax.arrow(0.625, 0.75, -0.25, 0, length_includes_head=True, width=arrow_width, head_length=arrow_head_length, color=curve_colour)

A_ax.text(0.25, 0.25, "H", fontfamily="Fira Code", fontsize=hilbert_font_size, horizontalalignment="center", verticalalignment="center", color=A_colour)
A_ax.text(0.75, 0.25, "A", fontfamily="Fira Code", fontsize=hilbert_font_size, horizontalalignment="center", verticalalignment="center", color=A_colour)
A_ax.text(0.75, 0.75, "A", fontfamily="Fira Code", fontsize=hilbert_font_size, horizontalalignment="center", verticalalignment="center", color=A_colour)
A_ax.text(0.25, 0.75, "R", fontfamily="Fira Code", fontsize=hilbert_font_size, horizontalalignment="center", verticalalignment="center", color=A_colour)

A_fig.savefig(save_path / f"hilbert_split_A.svg", format='svg', transparent=True)

R_fig = plt.figure(figsize=(4, 10))
R_ax = R_fig.add_subplot(1, 1, 1)
R_ax.set_xlim(-0.01, 1.01)
R_ax.set_ylim(-0.01, 2.51)
R_ax.set_aspect(1)
R_ax.axis('off')
R_fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)

R_ax.plot([0, 1, 1, 0, 0], [1.5, 1.5, 2.5, 2.5, 1.5], color=elements_colour, linewidth=elements_width, label="Elements")

R_ax.text(0.5, 2, "R", fontfamily="Fira Code", fontsize=hilbert_font_size, horizontalalignment="center", verticalalignment="center", color=R_colour)

R_ax.plot([0, 0.5, 0.5, 0, 0], [0, 0, 0.5, 0.5, 0], color=elements_colour, linewidth=elements_width)
R_ax.plot([0.5, 1, 1, 0.5, 0.5], [0, 0, 0.5, 0.5, 0], color=elements_colour, linewidth=elements_width)
R_ax.plot([0.5, 1, 1, 0.5, 0.5], [0.5, 0.5, 1, 1, 0.5], color=elements_colour, linewidth=elements_width)
R_ax.plot([0, 0.5, 0.5, 0, 0], [0.5, 0.5, 1, 1, 0.5], color=elements_colour, linewidth=elements_width)

R_ax.arrow(0.5, 1.375, 0, -0.25, length_includes_head=True, width=arrow_width, head_length=arrow_head_length, color=elements_colour)
R_ax.arrow(0.75, 0.625, 0, -0.25, length_includes_head=True, width=arrow_width, head_length=arrow_head_length, color=curve_colour)
R_ax.arrow(0.625, 0.25, -0.25, 0, length_includes_head=True, width=arrow_width, head_length=arrow_head_length, color=curve_colour)
R_ax.arrow(0.25, 0.375, 0, 0.25, length_includes_head=True, width=arrow_width, head_length=arrow_head_length, color=curve_colour)

R_ax.text(0.25, 0.25, "R", fontfamily="Fira Code", fontsize=hilbert_font_size, horizontalalignment="center", verticalalignment="center", color=R_colour)
R_ax.text(0.75, 0.25, "R", fontfamily="Fira Code", fontsize=hilbert_font_size, horizontalalignment="center", verticalalignment="center", color=R_colour)
R_ax.text(0.75, 0.75, "B", fontfamily="Fira Code", fontsize=hilbert_font_size, horizontalalignment="center", verticalalignment="center", color=R_colour)
R_ax.text(0.25, 0.75, "A", fontfamily="Fira Code", fontsize=hilbert_font_size, horizontalalignment="center", verticalalignment="center", color=R_colour)

R_fig.savefig(save_path / f"hilbert_split_R.svg", format='svg', transparent=True)

B_fig = plt.figure(figsize=(4, 10))
B_ax = B_fig.add_subplot(1, 1, 1)
B_ax.set_xlim(-0.01, 1.01)
B_ax.set_ylim(-0.01, 2.51)
B_ax.set_aspect(1)
B_ax.axis('off')
B_fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)

B_ax.plot([0, 1, 1, 0, 0], [1.5, 1.5, 2.5, 2.5, 1.5], color=elements_colour, linewidth=elements_width, label="Elements")

B_ax.text(0.5, 2, "B", fontfamily="Fira Code", fontsize=hilbert_font_size, horizontalalignment="center", verticalalignment="center", color=B_colour)

B_ax.plot([0, 0.5, 0.5, 0, 0], [0, 0, 0.5, 0.5, 0], color=elements_colour, linewidth=elements_width)
B_ax.plot([0.5, 1, 1, 0.5, 0.5], [0, 0, 0.5, 0.5, 0], color=elements_colour, linewidth=elements_width)
B_ax.plot([0.5, 1, 1, 0.5, 0.5], [0.5, 0.5, 1, 1, 0.5], color=elements_colour, linewidth=elements_width)
B_ax.plot([0, 0.5, 0.5, 0, 0], [0.5, 0.5, 1, 1, 0.5], color=elements_colour, linewidth=elements_width)

B_ax.arrow(0.5, 1.375, 0, -0.25, length_includes_head=True, width=arrow_width, head_length=arrow_head_length, color=elements_colour)
B_ax.arrow(0.625, 0.75, -0.25, 0, length_includes_head=True, width=arrow_width, head_length=arrow_head_length, color=curve_colour)
B_ax.arrow(0.25, 0.625, 0, -0.25, length_includes_head=True, width=arrow_width, head_length=arrow_head_length, color=curve_colour)
B_ax.arrow(0.375, 0.25, 0.25, 0, length_includes_head=True, width=arrow_width, head_length=arrow_head_length, color=curve_colour)

B_ax.text(0.25, 0.25, "B", fontfamily="Fira Code", fontsize=hilbert_font_size, horizontalalignment="center", verticalalignment="center", color=B_colour)
B_ax.text(0.75, 0.25, "H", fontfamily="Fira Code", fontsize=hilbert_font_size, horizontalalignment="center", verticalalignment="center", color=B_colour)
B_ax.text(0.75, 0.75, "R", fontfamily="Fira Code", fontsize=hilbert_font_size, horizontalalignment="center", verticalalignment="center", color=B_colour)
B_ax.text(0.25, 0.75, "B", fontfamily="Fira Code", fontsize=hilbert_font_size, horizontalalignment="center", verticalalignment="center", color=B_colour)

B_fig.savefig(save_path / f"hilbert_split_B.svg", format='svg', transparent=True)

# Hilbert levels
l0_fig = plt.figure(figsize=(4, 4))
l0_ax = l0_fig.add_subplot(1, 1, 1)
l0_ax.set_xlim(-0.1, 1.1)
l0_ax.set_ylim(-0.1, 1.1)
l0_ax.set_aspect(1)
l0_ax.axis('off')
l0_fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)

l0_ax.plot([0, 1, 1, 0, 0], [0, 0, 1, 1, 0], color=elements_colour, linewidth=elements_width, label="Elements")

l0_ax.text(0.5, 0.5, "H", fontfamily="Fira Code", fontsize=hilbert_font_size, horizontalalignment="center", verticalalignment="center", color=elements_colour)

l0_fig.savefig(save_path / f"hilbert_level_0.svg", format='svg', transparent=True)

l1_fig = plt.figure(figsize=(4, 4))
l1_ax = l1_fig.add_subplot(1, 1, 1)
l1_ax.set_xlim(-0.1, 1.1)
l1_ax.set_ylim(-0.1, 1.1)
l1_ax.set_aspect(1)
l1_ax.axis('off')
l1_fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)

l1_ax.plot([0, 0.5, 0.5, 0, 0], [0, 0, 0.5, 0.5, 0], color=elements_colour, linewidth=elements_width, label="Elements")
l1_ax.plot([0.5, 1, 1, 0.5, 0.5], [0, 0, 0.5, 0.5, 0], color=elements_colour, linewidth=elements_width)
l1_ax.plot([0.5, 1, 1, 0.5, 0.5], [0.5, 0.5, 1, 1, 0.5], color=elements_colour, linewidth=elements_width)
l1_ax.plot([0, 0.5, 0.5, 0, 0], [0.5, 0.5, 1, 1, 0.5], color=elements_colour, linewidth=elements_width)

l1_ax.plot([0.25, 0.25, 0.75, 0.75], [0.25, 0.75, 0.75, 0.25], color=background_curve_colour, linewidth=background_curve_width, label="Hilbert curve")

l1_ax.text(0.25, 0.25, "A", fontfamily="Fira Code", fontsize=hilbert_font_size, horizontalalignment="center", verticalalignment="center", color=H_colour)
l1_ax.text(0.75, 0.25, "B", fontfamily="Fira Code", fontsize=hilbert_font_size, horizontalalignment="center", verticalalignment="center", color=H_colour)
l1_ax.text(0.75, 0.75, "H", fontfamily="Fira Code", fontsize=hilbert_font_size, horizontalalignment="center", verticalalignment="center", color=H_colour)
l1_ax.text(0.25, 0.75, "H", fontfamily="Fira Code", fontsize=hilbert_font_size, horizontalalignment="center", verticalalignment="center", color=H_colour)

l1_fig.savefig(save_path / f"hilbert_level_1.svg", format='svg', transparent=True)

l2_fig = plt.figure(figsize=(4, 4))
l2_ax = l2_fig.add_subplot(1, 1, 1)
l2_ax.set_xlim(-0.1, 1.1)
l2_ax.set_ylim(-0.1, 1.1)
l2_ax.set_aspect(1)
l2_ax.axis('off')
l2_fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)

l2_ax.plot([0, 0.25, 0.25, 0, 0], [0, 0, 0.25, 0.25, 0], color=elements_colour, linewidth=elements_width, label="Elements")
l2_ax.plot([0.25, 0.5, 0.5, 0.25, 0.25], [0, 0, 0.25, 0.25, 0], color=elements_colour, linewidth=elements_width)
l2_ax.plot([0.25, 0.5, 0.5, 0.25, 0.25], [0.25, 0.25, 0.5, 0.5, 0.25], color=elements_colour, linewidth=elements_width)
l2_ax.plot([0, 0.25, 0.25, 0, 0], [0.25, 0.25, 0.5, 0.5, 0.25], color=elements_colour, linewidth=elements_width)

l2_ax.plot([0.5, 0.75, 0.75, 0.5, 0.5], [0, 0, 0.25, 0.25, 0], color=elements_colour, linewidth=elements_width)
l2_ax.plot([0.75, 1, 1, 0.75, 0.75], [0, 0, 0.25, 0.25, 0], color=elements_colour, linewidth=elements_width)
l2_ax.plot([0.75, 1, 1, 0.75, 0.75], [0.25, 0.25, 0.5, 0.5, 0.25], color=elements_colour, linewidth=elements_width)
l2_ax.plot([0.5, 0.75, 0.75, 0.5, 0.5], [0.25, 0.25, 0.5, 0.5, 0.25], color=elements_colour, linewidth=elements_width)

l2_ax.plot([0.5, 0.75, 0.75, 0.5, 0.5], [0.5, 0.5, 0.75, 0.75, 0.5], color=elements_colour, linewidth=elements_width)
l2_ax.plot([0.75, 1, 1, 0.75, 0.75], [0.5, 0.5, 0.75, 0.75, 0.5], color=elements_colour, linewidth=elements_width)
l2_ax.plot([0.75, 1, 1, 0.75, 0.75], [0.75, 0.75, 1, 1, 0.75], color=elements_colour, linewidth=elements_width)
l2_ax.plot([0.5, 0.75, 0.75, 0.5, 0.5], [0.75, 0.75, 1, 1, 0.75], color=elements_colour, linewidth=elements_width)

l2_ax.plot([0, 0.25, 0.25, 0, 0], [0.5, 0.5, 0.75, 0.75, 0.5], color=elements_colour, linewidth=elements_width)
l2_ax.plot([0.25, 0.5, 0.5, 0.25, 0.25], [0.5, 0.5, 0.75, 0.75, 0.5], color=elements_colour, linewidth=elements_width)
l2_ax.plot([0.25, 0.5, 0.5, 0.25, 0.25], [0.75, 0.75, 1, 1, 0.75], color=elements_colour, linewidth=elements_width)
l2_ax.plot([0, 0.25, 0.25, 0, 0], [0.75, 0.75, 1, 1, 0.75], color=elements_colour, linewidth=elements_width)

l2_ax.plot([0.125, 0.375, 0.375, 0.125, 0.125, 0.125, 0.375, 0.375, 0.625, 0.625, 0.875, 0.875, 0.875, 0.625, 0.625, 0.875], 
           [0.125, 0.125, 0.375, 0.375, 0.625, 0.875, 0.875, 0.625, 0.625, 0.875, 0.875, 0.625, 0.375, 0.375, 0.125, 0.125], 
           color=background_curve_colour, linewidth=background_curve_width, label="Hilbert curve")

l2_ax.text(0.125, 0.125, "H", fontfamily="Fira Code", fontsize=hilbert_font_size, horizontalalignment="center", verticalalignment="center", color=A_colour)
l2_ax.text(0.375, 0.125, "A", fontfamily="Fira Code", fontsize=hilbert_font_size, horizontalalignment="center", verticalalignment="center", color=A_colour)
l2_ax.text(0.375, 0.375, "A", fontfamily="Fira Code", fontsize=hilbert_font_size, horizontalalignment="center", verticalalignment="center", color=A_colour)
l2_ax.text(0.125, 0.375, "R", fontfamily="Fira Code", fontsize=hilbert_font_size, horizontalalignment="center", verticalalignment="center", color=A_colour)

l2_ax.text(0.625, 0.125, "B", fontfamily="Fira Code", fontsize=hilbert_font_size, horizontalalignment="center", verticalalignment="center", color=B_colour)
l2_ax.text(0.875, 0.125, "H", fontfamily="Fira Code", fontsize=hilbert_font_size, horizontalalignment="center", verticalalignment="center", color=B_colour)
l2_ax.text(0.875, 0.375, "R", fontfamily="Fira Code", fontsize=hilbert_font_size, horizontalalignment="center", verticalalignment="center", color=B_colour)
l2_ax.text(0.625, 0.375, "B", fontfamily="Fira Code", fontsize=hilbert_font_size, horizontalalignment="center", verticalalignment="center", color=B_colour)

l2_ax.text(0.625, 0.625, "A", fontfamily="Fira Code", fontsize=hilbert_font_size, horizontalalignment="center", verticalalignment="center", color=H_colour)
l2_ax.text(0.875, 0.625, "B", fontfamily="Fira Code", fontsize=hilbert_font_size, horizontalalignment="center", verticalalignment="center", color=H_colour)
l2_ax.text(0.875, 0.875, "H", fontfamily="Fira Code", fontsize=hilbert_font_size, horizontalalignment="center", verticalalignment="center", color=H_colour)
l2_ax.text(0.625, 0.875, "H", fontfamily="Fira Code", fontsize=hilbert_font_size, horizontalalignment="center", verticalalignment="center", color=H_colour)

l2_ax.text(0.125, 0.625, "A", fontfamily="Fira Code", fontsize=hilbert_font_size, horizontalalignment="center", verticalalignment="center", color=H_colour)
l2_ax.text(0.375, 0.625, "B", fontfamily="Fira Code", fontsize=hilbert_font_size, horizontalalignment="center", verticalalignment="center", color=H_colour)
l2_ax.text(0.375, 0.875, "H", fontfamily="Fira Code", fontsize=hilbert_font_size, horizontalalignment="center", verticalalignment="center", color=H_colour)
l2_ax.text(0.125, 0.875, "H", fontfamily="Fira Code", fontsize=hilbert_font_size, horizontalalignment="center", verticalalignment="center", color=H_colour)

l2_fig.savefig(save_path / f"hilbert_level_2.svg", format='svg', transparent=True)

plt.show()