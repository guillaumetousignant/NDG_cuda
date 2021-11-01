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

points_fig, points_ax = plt.subplots(1, 1)
points_ax.set_xlim(-1.1, 5.1)
points_ax.set_ylim(-1.1, 1.1)
points_ax.set_aspect(1)
points_ax.axis('off')

points_ax.plot([-1, 1, 1, -1, -1], [-1, -1, 1, 1, -1], label="N = 4 outline", color=outline_colour, linewidth=2)

for i in range(N[0] + 1):
    points_ax.plot([nodes[0][i], nodes[0][i]], [-1, 1], color=lines_colour, linewidth=1)
    points_ax.plot([-1, 1], [nodes[0][i], nodes[0][i]], color=lines_colour, linewidth=1)

for i in range(N[0] + 1):
    for j in range(N[0] + 1):
        points_ax.plot(nodes[0][i], nodes[0][j], color=points_colour, linewidth=5, marker="o", markersize=4)
    
points_ax.plot([3, 5, 5, 3, 3], [-1, -1, 1, 1, -1], label="N = 6 outline", color=outline_colour, linewidth=2)

for i in range(N[1] + 1):
    points_ax.plot([nodes[1][i] + 4, nodes[1][i] + 4], [-1, 1], color=lines_colour, linewidth=1)
    points_ax.plot([3, 5], [nodes[1][i], nodes[1][i]], color=lines_colour, linewidth=1)

for i in range(N[1] + 1):
    for j in range(N[1] + 1):
        points_ax.plot(nodes[1][i] + 4, nodes[1][j], color=points_colour, linewidth=5, marker="o", markersize=4)

points_ax.arrow(1.5, 0, 1, 0, length_includes_head=False, width=0.05, head_length=0.1, color=outline_colour)

points_fig.savefig(save_path / f"collocation_points_N{N[0]}_N{N[1]}.svg", format='svg', transparent=True)
points_fig.savefig(save_path / f"collocation_points_N{N[0]}_N{N[1]}.png", format='png', transparent=True)

plt.show()
    