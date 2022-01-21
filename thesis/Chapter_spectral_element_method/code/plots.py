import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import sem

N = range(6)
n_points = 1000
x = np.linspace(-1, 1, n_points)
polynomial_width = 2
nodes_width = 2
nodes_linestyle = "--"
nodes_colour = np.array([244, 71, 61])/255

save_path = Path(__file__).parent.parent / "media"
save_path.mkdir(parents=True, exist_ok=True)

# Polynomials
polynomials_fig = plt.figure(figsize=(7, 7))
polynomials_ax = polynomials_fig.add_subplot(1, 1, 1)
polynomials_ax.set_xlim(-1.1, 1.1)
polynomials_ax.set_ylim(-1.1, 1.1)
polynomials_ax.set_aspect(1)
polynomials_ax.axis('on')
polynomials_ax.set_xlabel("x")
polynomials_ax.set_ylabel("$L_n(x)$")
polynomials_ax.grid()

for n in N:
    y = np.zeros(x.shape)
    for i in range(n_points):
        y[i] = sem.LegendrePolynomialAndDerivative(n, x[i])[0]

    polynomials_ax.plot(x, y, linewidth=polynomial_width, label=f"$L_{n}(x)$")

polynomials_ax.legend()

polynomials_fig.savefig(save_path / f"polynomials.svg", format='svg', transparent=True)

# Interpolants
[nodes, weights] = sem.LegendreGaussNodesAndWeights(N[-1])
barycentric_weights = sem.BarycentricWeights(nodes)
interpolants = np.zeros((N[-1] + 1, n_points))

interpolants_fig = plt.figure(figsize=(7, 7))
interpolants_ax = interpolants_fig.add_subplot(1, 1, 1)
interpolants_ax.set_xlim(-1.1, 1.1)
interpolants_ax.set_ylim(-1.1, 1.1)
interpolants_ax.set_aspect(1)
interpolants_ax.axis('on')
interpolants_ax.set_xlabel("x")
interpolants_ax.set_ylabel("$l_n(x)$")
interpolants_ax.grid()

for i in range(n_points):
    interpolants[:, i] = sem.LagrangeInterpolatingPolynomials(x[i], nodes, barycentric_weights)

for n in N:
    interpolants_ax.plot(x, interpolants[n, :], linewidth=polynomial_width, label=f"$l_{n}(x)$")

interpolants_ax.vlines(nodes, -1, 1, linewidth=nodes_width, linestyle=nodes_linestyle, color=nodes_colour, label="collocation points")

interpolants_ax.legend()

interpolants_fig.savefig(save_path / f"interpolants.svg", format='svg', transparent=True)

plt.show()