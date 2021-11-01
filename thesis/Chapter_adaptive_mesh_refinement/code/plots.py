import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import sem

N = [4, 6]
points_colour = np.array([197, 134, 192])/255
outline_colour = np.array([0, 0, 0])
lines_colour = np.array([37, 37, 37])/255

save_path = Path.cwd().parent / "media"
save_path.mkdir(parents=True, exist_ok=True)

nodes_4 = sem.LegendreGaussNodesAndWeights(N[0])[0]
nodes_6 = sem.LegendreGaussNodesAndWeights(N[1])[0]

nodes = [nodes_4, nodes_6]

p_fig, p_ax = plt.subplots(1, 1)
p_ax.set_xlim(-1.1, 5.1)
p_ax.set_ylim(-1.1, 1.1)
p_ax.set_aspect(1)
p_ax.axis('off')

p_ax.plot([-1, 1, 1, -1, -1], [-1, -1, 1, 1, -1], color=outline_colour, linewidth=2)

for i in range(N[0] + 1):
    p_ax.plot([nodes[0][i], nodes[0][i]], [-1, 1], color=lines_colour, linewidth=1)
    p_ax.plot([-1, 1], [nodes[0][i], nodes[0][i]], color=lines_colour, linewidth=1)

for i in range(N[0] + 1):
    for j in range(N[0] + 1):
        p_ax.plot(nodes[0][i], nodes[0][j], color=points_colour, linewidth=6, marker="o", markersize=4)
    
p_ax.plot([3, 5, 5, 3, 3], [-1, -1, 1, 1, -1], color=outline_colour, linewidth=2)

for i in range(N[1] + 1):
    p_ax.plot([nodes[1][i] + 4, nodes[1][i] + 4], [-1, 1], color=lines_colour, linewidth=1)
    p_ax.plot([3, 5], [nodes[1][i], nodes[1][i]], color=lines_colour, linewidth=1)

for i in range(N[1] + 1):
    for j in range(N[1] + 1):
        p_ax.plot(nodes[1][i] + 4, nodes[1][j], color=points_colour, linewidth=6, marker="o", markersize=4)

p_ax.arrow(1.5, 0, 1, 0, length_includes_head=False, width=0.05, head_length=0.1, color=outline_colour)

p_fig.savefig(save_path / f"p-adaptivity_N{N[0]}_N{N[1]}.svg", format='svg', transparent=True)
p_fig.savefig(save_path / f"p-adaptivity_N{N[0]}_N{N[1]}.png", format='png', transparent=True)

h_fig, h_ax = plt.subplots(1, 1)
h_ax.set_xlim(-1.1, 5.1)
h_ax.set_ylim(-1.1, 1.1)
h_ax.set_aspect(1)
h_ax.axis('off')

h_ax.plot([-1, 1, 1, -1, -1], [-1, -1, 1, 1, -1], color=outline_colour, linewidth=2)

for i in range(N[0] + 1):
    h_ax.plot([nodes[0][i], nodes[0][i]], [-1, 1], color=lines_colour, linewidth=1)
    h_ax.plot([-1, 1], [nodes[0][i], nodes[0][i]], color=lines_colour, linewidth=1)

for i in range(N[0] + 1):
    for j in range(N[0] + 1):
        h_ax.plot(nodes[0][i], nodes[0][j], color=points_colour, linewidth=4, marker="o", markersize=4)

h_ax.plot([3, 4, 4, 3, 3], [-1, -1, 0, 0, -1], color=outline_colour, linewidth=2)
h_ax.plot([4, 5, 5, 4, 4], [-1, -1, 0, 0, -1], color=outline_colour, linewidth=2)
h_ax.plot([4, 5, 5, 4, 4], [0, 0, 1, 1, 0], color=outline_colour, linewidth=2)
h_ax.plot([3, 4, 4, 3, 3], [0, 0, 1, 1, 0], color=outline_colour, linewidth=2)

for i in range(N[0] + 1):
    h_ax.plot([nodes[0][i]/2 + 3.5, nodes[0][i]/2 + 3.5], [-1, 0], color=lines_colour, linewidth=1)
    h_ax.plot([nodes[0][i]/2 + 4.5, nodes[0][i]/2 + 4.5], [-1, 0], color=lines_colour, linewidth=1)
    h_ax.plot([nodes[0][i]/2 + 3.5, nodes[0][i]/2 + 3.5], [0, 1], color=lines_colour, linewidth=1)
    h_ax.plot([nodes[0][i]/2 + 4.5, nodes[0][i]/2 + 4.5], [0, 1], color=lines_colour, linewidth=1)
    h_ax.plot([3, 4], [nodes[0][i]/2 - 0.5, nodes[0][i]/2 - 0.5], color=lines_colour, linewidth=1)
    h_ax.plot([4, 5], [nodes[0][i]/2 - 0.5, nodes[0][i]/2 - 0.5], color=lines_colour, linewidth=1)
    h_ax.plot([4, 5], [nodes[0][i]/2 + 0.5, nodes[0][i]/2 + 0.5], color=lines_colour, linewidth=1)
    h_ax.plot([3, 4], [nodes[0][i]/2 + 0.5, nodes[0][i]/2 + 0.5], color=lines_colour, linewidth=1)

for i in range(N[0] + 1):
    for j in range(N[0] + 1):
        h_ax.plot(nodes[0][i]/2 + 3.5, nodes[0][j]/2 - 0.5, color=points_colour, linewidth=4, marker="o", markersize=4)
        h_ax.plot(nodes[0][i]/2 + 4.5, nodes[0][j]/2 - 0.5, color=points_colour, linewidth=4, marker="o", markersize=4)
        h_ax.plot(nodes[0][i]/2 + 4.5, nodes[0][j]/2 + 0.5, color=points_colour, linewidth=4, marker="o", markersize=4)
        h_ax.plot(nodes[0][i]/2 + 3.5, nodes[0][j]/2 + 0.5, color=points_colour, linewidth=4, marker="o", markersize=4)

h_ax.arrow(1.5, 0, 1, 0, length_includes_head=False, width=0.05, head_length=0.1, color=outline_colour)

h_fig.savefig(save_path / f"h-adaptivity_N{N[0]}.svg", format='svg', transparent=True)
h_fig.savefig(save_path / f"h-adaptivity_N{N[0]}.png", format='png', transparent=True)

plt.show()
    