import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import sem

N = [4, 6]
points_colour = np.array([197, 134, 192])/255
outline_colour = np.array([0, 0, 0])
lines_colour = np.array([37, 37, 37])/255
faces_colour = np.array([86, 156, 214])/255
face_points_colour = np.array([156, 220, 254])/255
points_width = 12
outline_width = 2
lines_width = 1
faces_width = 4
points_size = 12
points_shape = "."

save_path = Path.cwd().parent / "media"
save_path.mkdir(parents=True, exist_ok=True)

nodes_4 = sem.LegendreGaussNodesAndWeights(N[0])[0]
nodes_6 = sem.LegendreGaussNodesAndWeights(N[1])[0]

nodes = [nodes_4, nodes_6]

p_fig = plt.figure(figsize=(8.5,3))
p_ax = p_fig.add_subplot(1, 1, 1)
p_ax.set_xlim(-1.1, 5.1)
p_ax.set_ylim(-1.1, 1.1)
p_ax.set_aspect(1)
p_ax.axis('off')
p_fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)

p_ax.plot([-1, 1, 1, -1, -1], [-1, -1, 1, 1, -1], color=outline_colour, linewidth=outline_width)

for i in range(N[0] + 1):
    p_ax.plot([nodes[0][i], nodes[0][i]], [-1, 1], color=lines_colour, linewidth=lines_width)
    p_ax.plot([-1, 1], [nodes[0][i], nodes[0][i]], color=lines_colour, linewidth=lines_width)

for i in range(N[0] + 1):
    for j in range(N[0] + 1):
        p_ax.plot(nodes[0][i], nodes[0][j], color=points_colour, linewidth=points_width, marker=points_shape, markersize=points_size)
    
p_ax.plot([3, 5, 5, 3, 3], [-1, -1, 1, 1, -1], color=outline_colour, linewidth=outline_width)

for i in range(N[1] + 1):
    p_ax.plot([nodes[1][i] + 4, nodes[1][i] + 4], [-1, 1], color=lines_colour, linewidth=lines_width)
    p_ax.plot([3, 5], [nodes[1][i], nodes[1][i]], color=lines_colour, linewidth=lines_width)

for i in range(N[1] + 1):
    for j in range(N[1] + 1):
        p_ax.plot(nodes[1][i] + 4, nodes[1][j], color=points_colour, linewidth=points_width, marker=points_shape, markersize=points_size)

p_ax.arrow(1.5, 0, 1, 0, length_includes_head=False, width=0.05, head_length=0.1, color=outline_colour)

p_fig.savefig(save_path / f"p-adaptivity_N{N[0]}_N{N[1]}.svg", format='svg', transparent=True)

h_fig = plt.figure(figsize=(8.5,3))
h_ax = h_fig.add_subplot(1, 1, 1)
h_ax.set_xlim(-1.1, 5.1)
h_ax.set_ylim(-1.1, 1.1)
h_ax.set_aspect(1)
h_ax.axis('off')
h_fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)

h_ax.plot([-1, 1, 1, -1, -1], [-1, -1, 1, 1, -1], color=outline_colour, linewidth=outline_width)

for i in range(N[0] + 1):
    h_ax.plot([nodes[0][i], nodes[0][i]], [-1, 1], color=lines_colour, linewidth=lines_width)
    h_ax.plot([-1, 1], [nodes[0][i], nodes[0][i]], color=lines_colour, linewidth=lines_width)

for i in range(N[0] + 1):
    for j in range(N[0] + 1):
        h_ax.plot(nodes[0][i], nodes[0][j], color=points_colour, linewidth=points_width, marker=points_shape, markersize=points_size)

h_ax.plot([3, 4, 4, 3, 3], [-1, -1, 0, 0, -1], color=outline_colour, linewidth=outline_width)
h_ax.plot([4, 5, 5, 4, 4], [-1, -1, 0, 0, -1], color=outline_colour, linewidth=outline_width)
h_ax.plot([4, 5, 5, 4, 4], [0, 0, 1, 1, 0], color=outline_colour, linewidth=outline_width)
h_ax.plot([3, 4, 4, 3, 3], [0, 0, 1, 1, 0], color=outline_colour, linewidth=outline_width)

for i in range(N[0] + 1):
    h_ax.plot([nodes[0][i]/2 + 3.5, nodes[0][i]/2 + 3.5], [-1, 0], color=lines_colour, linewidth=lines_width)
    h_ax.plot([nodes[0][i]/2 + 4.5, nodes[0][i]/2 + 4.5], [-1, 0], color=lines_colour, linewidth=lines_width)
    h_ax.plot([nodes[0][i]/2 + 3.5, nodes[0][i]/2 + 3.5], [0, 1], color=lines_colour, linewidth=lines_width)
    h_ax.plot([nodes[0][i]/2 + 4.5, nodes[0][i]/2 + 4.5], [0, 1], color=lines_colour, linewidth=lines_width)
    h_ax.plot([3, 4], [nodes[0][i]/2 - 0.5, nodes[0][i]/2 - 0.5], color=lines_colour, linewidth=lines_width)
    h_ax.plot([4, 5], [nodes[0][i]/2 - 0.5, nodes[0][i]/2 - 0.5], color=lines_colour, linewidth=lines_width)
    h_ax.plot([4, 5], [nodes[0][i]/2 + 0.5, nodes[0][i]/2 + 0.5], color=lines_colour, linewidth=lines_width)
    h_ax.plot([3, 4], [nodes[0][i]/2 + 0.5, nodes[0][i]/2 + 0.5], color=lines_colour, linewidth=lines_width)

for i in range(N[0] + 1):
    for j in range(N[0] + 1):
        h_ax.plot(nodes[0][i]/2 + 3.5, nodes[0][j]/2 - 0.5, color=points_colour, linewidth=points_width, marker=points_shape, markersize=points_size)
        h_ax.plot(nodes[0][i]/2 + 4.5, nodes[0][j]/2 - 0.5, color=points_colour, linewidth=points_width, marker=points_shape, markersize=points_size)
        h_ax.plot(nodes[0][i]/2 + 4.5, nodes[0][j]/2 + 0.5, color=points_colour, linewidth=points_width, marker=points_shape, markersize=points_size)
        h_ax.plot(nodes[0][i]/2 + 3.5, nodes[0][j]/2 + 0.5, color=points_colour, linewidth=points_width, marker=points_shape, markersize=points_size)

h_ax.arrow(1.5, 0, 1, 0, length_includes_head=False, width=0.05, head_length=0.1, color=outline_colour)

h_fig.savefig(save_path / f"h-adaptivity_N{N[0]}.svg", format='svg', transparent=True)

hp_fig = plt.figure(figsize=(4.95,3.55))
hp_ax = hp_fig.add_subplot(1, 1, 1)
hp_ax.set_xlim(-1.1, 2.5)
hp_ax.set_ylim(-1.3, 1.3)
hp_ax.set_aspect(1)
hp_ax.axis('off')
hp_fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)

hp_ax.plot([-1, 1, 1, -1, -1], [-1, -1, 1, 1, -1], color=outline_colour, linewidth=outline_width)

for i in range(N[0] + 1):
    hp_ax.plot([nodes[0][i], nodes[0][i]], [-1, 1], color=lines_colour, linewidth=lines_width)
    hp_ax.plot([-1, 1], [nodes[0][i], nodes[0][i]], color=lines_colour, linewidth=lines_width)

for i in range(N[0] + 1):
    for j in range(N[0] + 1):
        hp_ax.plot(nodes[0][i], nodes[0][j], color=points_colour, linewidth=points_width, marker=points_shape, markersize=points_size)

hp_ax.plot([1.4, 2.4, 2.4, 1.4, 1.4], [-1.2, -1.2, -0.2, -0.2, -1.2], color=outline_colour, linewidth=outline_width)
hp_ax.plot([1.4, 2.4, 2.4, 1.4, 1.4], [0.2, 0.2, 1.2, 1.2, 0.2], color=outline_colour, linewidth=outline_width)

for i in range(N[1] + 1):
    hp_ax.plot([nodes[1][i]/2 + 1.9, nodes[1][i]/2 + 1.9], [-1.2, -0.2], color=lines_colour, linewidth=lines_width)
    hp_ax.plot([1.4, 2.4], [nodes[1][i]/2 - 0.7, nodes[1][i]/2 - 0.7], color=lines_colour, linewidth=lines_width)

for i in range(N[0] + 1):
    hp_ax.plot([nodes[0][i]/2 + 1.9, nodes[0][i]/2 + 1.9], [0.2, 1.2], color=lines_colour, linewidth=lines_width)
    hp_ax.plot([1.4, 2.4], [nodes[0][i]/2 + 0.7, nodes[0][i]/2 + 0.7], color=lines_colour, linewidth=lines_width)

for i in range(N[1] + 1):
    for j in range(N[1] + 1):
        hp_ax.plot(nodes[1][i]/2 + 1.9, nodes[1][j]/2 - 0.7, color=points_colour, linewidth=points_width, marker=points_shape, markersize=points_size)

for i in range(N[0] + 1):
    for j in range(N[0] + 1):
        hp_ax.plot(nodes[0][i]/2 + 1.9, nodes[0][j]/2 + 0.7, color=points_colour, linewidth=points_width, marker=points_shape, markersize=points_size) 

hp_ax.plot([1.2, 1.2], [-1.1, -0.1], color=faces_colour, linewidth=faces_width)
hp_ax.plot([1.2, 1.2], [0.1, 1.1], color=faces_colour, linewidth=faces_width)
hp_ax.plot([1.4, 2.4], [0, 0], color=faces_colour, linewidth=faces_width)

for i in range(N[0] + 1):
    hp_ax.plot(1.2, nodes[0][i]/2 + 0.6, color=face_points_colour, linewidth=points_width, marker=points_shape, markersize=points_size) 

for i in range(N[1] + 1):
    hp_ax.plot(1.2, nodes[1][i]/2 - 0.6, color=face_points_colour, linewidth=points_width, marker=points_shape, markersize=points_size) 
    hp_ax.plot(nodes[1][i]/2 + 1.9, 0, color=face_points_colour, linewidth=points_width, marker=points_shape, markersize=points_size) 

hp_fig.savefig(save_path / f"hp-adaptivity_N{N[0]}_N{N[1]}.svg", format='svg', transparent=True)

plt.show()
    