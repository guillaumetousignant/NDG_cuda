import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import sem

N = range(6)
n_points = 1000
x = np.linspace(-1, 1, n_points)
polynomial_width = 2
nodes_width = 2
elements_width = 3
nodes_linestyle = "--"
nodes_colour = np.array([244, 71, 61])/255
elements_colour = np.array([37, 37, 37])/255
arrow_colour = np.array([200, 200, 200])/255
normals_colour = np.array([78, 201, 176])/255
polynomial_color_map = plt.get_cmap("viridis")
interpolant_color_map = plt.get_cmap("rainbow")
arrow_width = 0.025
arrow_head_length = 0.05
axes_font_size = 24
graduations_font_size = 16
normals_font_size = 16

save_path = Path(__file__).parent.parent / "media"
save_path.mkdir(parents=True, exist_ok=True)

# Polynomials
polynomials_fig = plt.figure(figsize=(10.5, 7))
polynomials_ax = polynomials_fig.add_subplot(1, 1, 1)
polynomials_ax.set_xlim(-1.1, 1.1)
polynomials_ax.set_ylim(-1.1, 1.1)
polynomials_ax.set_aspect(0.667)
polynomials_ax.axis('on')
polynomials_ax.set_xlabel("x")
polynomials_ax.set_ylabel("$L_n(x)$")
polynomials_ax.grid()

for n in N:
    normalised_N = (n - N[0])/(N[-1] - N[0])

    y = np.zeros(x.shape)
    for i in range(n_points):
        y[i] = sem.LegendrePolynomialAndDerivative(n, x[i])[0]

    polynomials_ax.plot(x, y, linewidth=polynomial_width, label=f"$L_{n}(x)$", color=polynomial_color_map(normalised_N))

polynomials_ax.legend()

polynomials_fig.savefig(save_path / f"polynomials.svg", format='svg', transparent=True)

# Interpolants
[nodes, weights] = sem.LegendreGaussNodesAndWeights(N[-1])
barycentric_weights = sem.BarycentricWeights(nodes)
interpolants = np.zeros((N[-1] + 1, n_points))

interpolants_fig = plt.figure(figsize=(10.5, 7))
interpolants_ax = interpolants_fig.add_subplot(1, 1, 1)
interpolants_ax.set_xlim(-1.1, 1.1)
interpolants_ax.set_ylim(-1.1, 1.1)
interpolants_ax.set_aspect(0.667)
interpolants_ax.axis('on')
interpolants_ax.set_xlabel("x")
interpolants_ax.set_ylabel("$l_n(x)$")
interpolants_ax.grid()

for i in range(n_points):
    interpolants[:, i] = sem.LagrangeInterpolatingPolynomials(x[i], nodes, barycentric_weights)

for n in N:
    normalised_N = (n - N[0])/(N[-1] - N[0])
    interpolants_ax.plot(x, interpolants[n, :], linewidth=polynomial_width, label=f"$l_{n}(x)$", color=interpolant_color_map(normalised_N))

interpolants_ax.vlines(nodes, -1, 1, linewidth=nodes_width, linestyle=nodes_linestyle, color=nodes_colour, label="collocation points")

interpolants_ax.legend()

interpolants_fig.savefig(save_path / f"interpolants.svg", format='svg', transparent=True)

# Simple domain
domain_fig = plt.figure(figsize=(4, 4))
domain_ax = domain_fig.add_subplot(1, 1, 1)
domain_ax.set_xlim(-1.5, 1.5)
domain_ax.set_ylim(-1.5, 1.5)
domain_ax.set_aspect(1)
domain_ax.axis('off')

domain_ax.arrow(-1.5, 0, 3, 0, length_includes_head=True, width=arrow_width, head_length=arrow_head_length, color=arrow_colour)
domain_ax.arrow(0, -1.5, 0, 3, length_includes_head=True, width=arrow_width, head_length=arrow_head_length, color=arrow_colour)

domain_ax.plot([-1, 1, 1, -1, -1], [-1, -1, 1, 1, -1], color=elements_colour, linewidth=elements_width, label="Domain")

domain_ax.text(1.4, -0.2, "x", fontfamily="Fira Code", fontsize=axes_font_size, horizontalalignment="center", verticalalignment="center", color=elements_colour)
domain_ax.text(-0.2, 1.4, "y", fontfamily="Fira Code", fontsize=axes_font_size, horizontalalignment="center", verticalalignment="center", color=elements_colour)

domain_ax.text(-1, -1.1, "-1", fontfamily="Fira Code", fontsize=graduations_font_size, horizontalalignment="center", verticalalignment="top", color=elements_colour)
domain_ax.text(1, -1.1, "1", fontfamily="Fira Code", fontsize=graduations_font_size, horizontalalignment="center", verticalalignment="top", color=elements_colour)
domain_ax.text(-1.1, -1, "-1", fontfamily="Fira Code", fontsize=graduations_font_size, horizontalalignment="right", verticalalignment="center", color=elements_colour)
domain_ax.text(-1.1, 1, "1", fontfamily="Fira Code", fontsize=graduations_font_size, horizontalalignment="right", verticalalignment="center", color=elements_colour)

domain_ax.arrow(1, 0, 0.4, 0, length_includes_head=True, width=arrow_width, head_length=arrow_head_length, color=normals_colour)
domain_ax.arrow(-1, 0, -0.4, 0, length_includes_head=True, width=arrow_width, head_length=arrow_head_length, color=normals_colour)
domain_ax.arrow(0, -1, 0, -0.4, length_includes_head=True, width=arrow_width, head_length=arrow_head_length, color=normals_colour)
domain_ax.arrow(0, 1, 0, 0.4, length_includes_head=True, width=arrow_width, head_length=arrow_head_length, color=normals_colour)

domain_ax.text(0.1, -1.2, "$n_0$", fontfamily="Fira Code", fontsize=normals_font_size, horizontalalignment="left", verticalalignment="center", color=normals_colour)
domain_ax.text(1.2, 0.1, "$n_1$", fontfamily="Fira Code", fontsize=normals_font_size, horizontalalignment="center", verticalalignment="bottom", color=normals_colour)
domain_ax.text(0.1, 1.2, "$n_2$", fontfamily="Fira Code", fontsize=normals_font_size, horizontalalignment="left", verticalalignment="center", color=normals_colour)
domain_ax.text(-1.2, 0.1, "$n_3$", fontfamily="Fira Code", fontsize=normals_font_size, horizontalalignment="center", verticalalignment="bottom", color=normals_colour)

domain_fig.savefig(save_path / f"domain.svg", format='svg', transparent=True)

plt.show()