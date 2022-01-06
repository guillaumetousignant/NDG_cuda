# Plots meshes with the data returned from the Mesh2D_t print() function

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from pathlib import Path
import sys
import argparse
import math
import naca

class Mesh:
    def __init__(self, 
                nodes_x: npt.ArrayLike, 
                nodes_y: npt.ArrayLike, 
                elements_nodes: npt.ArrayLike,
                airfoil_nodes: npt.ArrayLike,
                boundary_nodes: npt.ArrayLike):

        self.nodes_x = nodes_x
        self.nodes_y = nodes_y
        self.elements_nodes = elements_nodes
        self.airfoil_nodes = airfoil_nodes
        self.boundary_nodes = boundary_nodes

def cart2pol(x: float, y: float) -> tuple[float, float]:
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def pol2cart(rho: float, phi: float) -> tuple[float, float]:
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)

def generate_mesh(profile: str, 
        r_res: int = 64, 
        θ_half_res: int = 32, 
        diameter: float = 128, 
        exponent: float = 1,
        x_offset: float = 0,
        y_offset: float = 0,
        uneven_spacing: bool = True) -> Mesh:

    x, y = naca.naca(profile, θ_half_res, False, uneven_spacing)

    θ_res = len(x) 
    radius = diameter/2

    n_nodes = r_res * θ_res
    n_elements = (r_res - 1) * θ_res
    n_boundary = θ_res

    nodes_x = np.zeros(n_nodes)
    nodes_y = np.zeros(n_nodes)
    elements_nodes = np.zeros((n_elements, 4), dtype=np.uint32)
    airfoil_nodes = np.zeros((n_boundary, 2), dtype=np.uint32)
    boundary_nodes = np.zeros((n_boundary, 2), dtype=np.uint32)

    for i in range(θ_res):
        nodes_x[i] = x[i]
        nodes_y[i] = y[i]
        airfoil_nodes[i] = [i + 1 if i + 1 < θ_res else 0, i]
        boundary_nodes[i] = [(r_res - 1) * θ_res + i, (r_res - 1) * θ_res + i + 1 if i + 1 < θ_res else (r_res - 1) * θ_res]

        (r_init, θ) = cart2pol(x[i] + x_offset, y[i] + y_offset)
        r_span = radius - r_init

        for j in range(1, r_res):
            dist = (j/(r_res))**exponent
            r = r_init + dist * r_span

            x_current, y_current = pol2cart(r, θ)
            nodes_x[j * θ_res + i] = x_current - x_offset
            nodes_y[j * θ_res + i] = y_current - y_offset

            elements_nodes[(j - 1) * θ_res + i] = [(j - 1) * θ_res + i, 
                                                   (j - 1) * θ_res + i + 1 if i + 1 < θ_res else (j - 1) * θ_res,
                                                   j * θ_res + i + 1 if i + 1 < θ_res else j * θ_res,
                                                   j * θ_res + i]

    return Mesh(nodes_x, nodes_y, elements_nodes, airfoil_nodes, boundary_nodes)
    
def write_mesh(mesh: Mesh, 
        output_path: Path,
        boundary: str,
        airfoil: str):
    
    with open(output_path, 'w') as file:
        file.write(f"NDIME= 2\nNELEM= {mesh.elements_nodes.shape[0]}\n")
        for i in range(mesh.elements_nodes.shape[0]):
            file.write(f"9 {mesh.elements_nodes[i][0]} {mesh.elements_nodes[i][1]} {mesh.elements_nodes[i][2]} {mesh.elements_nodes[i][3]} {i}\n")

        file.write(f"NPOIN= {mesh.nodes_x.shape[0]}\n")
        for i in range(mesh.nodes_x.shape[0]):
            file.write(f"{mesh.nodes_x[i]:9f} {mesh.nodes_y[i]:9f} {i}\n")

        file.write(f"NMARK= 2\nMARKER_TAG= {boundary}\nMARKER_ELEMS= {mesh.boundary_nodes.shape[0]}\n")
        for i in range(mesh.boundary_nodes.shape[0]):
            file.write(f"3 {mesh.boundary_nodes[i][0]} {mesh.boundary_nodes[i][1]}\n")

        file.write(f"MARKER_TAG= {airfoil}\nMARKER_ELEMS= {mesh.airfoil_nodes.shape[0]}\n")
        for i in range(mesh.airfoil_nodes.shape[0]):
            file.write(f"3 {mesh.airfoil_nodes[i][0]} {mesh.airfoil_nodes[i][1]}\n")

def plot_mesh(mesh: Mesh, 
    title: str = "Mesh", 
    show_elements: bool = True, 
    show_nodes: bool = True, 
    show_ghosts: bool = True, 
    element_numbers: bool = True, 
    node_numbers: bool = True, 
    ghost_numbers: bool = True, 
    boundary_numbers: bool = True,
    show_title: bool = True,
    show_axis: bool = True,
    show_legend: bool = True,
    show_curve: bool = True,
    airfoil_tag: str = "AIRFOIL",
    boundary_tag: str = "FARFIELD"):
    
    points_colour = np.array([197, 134, 192])/255
    elements_colour = np.array([37, 37, 37])/255
    ghosts_colour = np.array([244, 71, 71])/255
    boundaries_colour = np.array([78, 201, 176])/255
    curve_colour = np.array([106, 153, 85])/255
    points_width = 12
    elements_width = 3
    ghosts_width = 5
    curve_width = 5
    points_size = 12
    points_shape = "."
    ghost_offset = 0.3
    points_font_size = 12
    elements_font_size = 14
    ghosts_font_size = 14
    boundaries_font_size = 6
    points_text_offset = [-0.005, -0.005]
    elements_text_offset = [0, 0]
    ghosts_text_offset = 0.2
    boundaries_text_offset = [0.2, 0.15]

    fig = plt.figure()
    fig.canvas.manager.set_window_title(title)
    ax = fig.add_subplot(1, 1, 1)
    ax.set_aspect(1)
    if show_axis:
        ax.set_xlabel("x")
        ax.set_ylabel("y")
    else: 
        ax.axis('off')

    if show_title:
        ax.set_title("Mesh")

    if show_ghosts:
        for i in range(mesh.airfoil_nodes.shape[0]):
            x = [mesh.nodes_x[mesh.airfoil_nodes[i][0]], mesh.nodes_x[mesh.airfoil_nodes[i][1]]]
            y = [mesh.nodes_y[mesh.airfoil_nodes[i][0]], mesh.nodes_y[mesh.airfoil_nodes[i][1]]]
            x_avg = (x[0] + x[1])/2
            y_avg = (y[0] + y[1])/2
            x = [x[0] * (1 - ghost_offset) + x_avg * ghost_offset, x[1] * (1 - ghost_offset) + x_avg * ghost_offset]
            y = [y[0] * (1 - ghost_offset) + y_avg * ghost_offset, y[1] * (1 - ghost_offset) + y_avg * ghost_offset]
            norm = math.sqrt(x_avg**2 + y_avg**2)
            length = math.sqrt((x[1] - x[0])**2 + (y[1] - y[0])**2)

            dx = x[1] - x[0]
            dy = y[1] - y[0]
            if (abs(dx) > 0 or abs(dy) > 0):
                norm = math.sqrt(dx**2 + dy**2)
                dx /= norm
                dy /= norm
            normal = [-dy, dx]

            ax.plot(x, y, color=ghosts_colour, linewidth=ghosts_width, label=airfoil_tag if i == 0 else "")
            if ghost_numbers:
                ax.text(x_avg + normal[0] * ghosts_text_offset * length, y_avg + normal[1] * ghosts_text_offset * length, f"{i}", fontfamily="Fira Code", fontsize=ghosts_font_size, horizontalalignment="center", verticalalignment="center", color=ghosts_colour)
        
        for i in range(mesh.boundary_nodes.shape[0]):
            x = [mesh.nodes_x[mesh.boundary_nodes[i][0]], mesh.nodes_x[mesh.boundary_nodes[i][1]]]
            y = [mesh.nodes_y[mesh.boundary_nodes[i][0]], mesh.nodes_y[mesh.boundary_nodes[i][1]]]
            x_avg = (x[0] + x[1])/2
            y_avg = (y[0] + y[1])/2
            x = [x[0] * (1 - ghost_offset) + x_avg * ghost_offset, x[1] * (1 - ghost_offset) + x_avg * ghost_offset]
            y = [y[0] * (1 - ghost_offset) + y_avg * ghost_offset, y[1] * (1 - ghost_offset) + y_avg * ghost_offset]
            norm = math.sqrt(x_avg**2 + y_avg**2)
            length = math.sqrt((x[1] - x[0])**2 + (y[1] - y[0])**2)

            dx = x[1] - x[0]
            dy = y[1] - y[0]
            if (abs(dx) > 0 or abs(dy) > 0):
                norm = math.sqrt(dx**2 + dy**2)
                dx /= norm
                dy /= norm
            normal = [-dy, dx]

            ax.plot(x, y, color=ghosts_colour, linewidth=ghosts_width, label=boundary_tag if i == 0 else "")
            if ghost_numbers:
                ax.text(x_avg + normal[0] * ghosts_text_offset * length, y_avg + normal[1] * ghosts_text_offset * length, f"{i}", fontfamily="Fira Code", fontsize=ghosts_font_size, horizontalalignment="center", verticalalignment="center", color=ghosts_colour)

    if show_elements:
        for i in range(mesh.elements_nodes.shape[0]):
            ax.plot([mesh.nodes_x[mesh.elements_nodes[i][0]], mesh.nodes_x[mesh.elements_nodes[i][1]], mesh.nodes_x[mesh.elements_nodes[i][2]], mesh.nodes_x[mesh.elements_nodes[i][3]], mesh.nodes_x[mesh.elements_nodes[i][0]]], [mesh.nodes_y[mesh.elements_nodes[i][0]], mesh.nodes_y[mesh.elements_nodes[i][1]], mesh.nodes_y[mesh.elements_nodes[i][2]], mesh.nodes_y[mesh.elements_nodes[i][3]], mesh.nodes_y[mesh.elements_nodes[i][0]]], color=elements_colour, linewidth=elements_width, label="Elements" if i == 0 else "")
            if element_numbers:
                ax.text((mesh.nodes_x[mesh.elements_nodes[i][0]] + mesh.nodes_x[mesh.elements_nodes[i][1]] + mesh.nodes_x[mesh.elements_nodes[i][2]] + mesh.nodes_x[mesh.elements_nodes[i][3]])/4 + elements_text_offset[0], (mesh.nodes_y[mesh.elements_nodes[i][0]] + mesh.nodes_y[mesh.elements_nodes[i][1]] + mesh.nodes_y[mesh.elements_nodes[i][2]] + mesh.nodes_y[mesh.elements_nodes[i][3]])/4 + elements_text_offset[1], f"{i}", fontfamily="Fira Code", fontsize=elements_font_size, horizontalalignment="center", verticalalignment="center", color=elements_colour)

    if show_nodes:
        ax.plot(mesh.nodes_x, mesh.nodes_y, color=points_colour, linestyle="None", linewidth=points_width, marker=points_shape, markersize=points_size, label="Nodes")
        if node_numbers:
            for i in range(mesh.nodes_x.shape[0]):
                ax.text(mesh.nodes_x[i] + points_text_offset[0], mesh.nodes_y[i] + points_text_offset[1], f"{i}", fontfamily="Fira Code", fontsize=points_font_size, horizontalalignment="right", verticalalignment="top", color=points_colour)

    if show_curve:
        elements_center_x = np.zeros(mesh.elements_nodes.shape[0])
        elements_center_y = np.zeros(mesh.elements_nodes.shape[0])
        for i in range(mesh.elements_nodes.shape[0]):
            elements_center_x[i] = (mesh.nodes_x[mesh.elements_nodes[i][0]] + mesh.nodes_x[mesh.elements_nodes[i][1]] + mesh.nodes_x[mesh.elements_nodes[i][2]] + mesh.nodes_x[mesh.elements_nodes[i][3]])/4
            elements_center_y[i] = (mesh.nodes_y[mesh.elements_nodes[i][0]] + mesh.nodes_y[mesh.elements_nodes[i][1]] + mesh.nodes_y[mesh.elements_nodes[i][2]] + mesh.nodes_y[mesh.elements_nodes[i][3]])/4
        ax.plot(elements_center_x, elements_center_y, color=curve_colour, linewidth=curve_width, label="Hilbert curve")

    if boundary_numbers and show_ghosts:
        for i in range(mesh.airfoil_nodes.shape[0]):
            element_node0_x = mesh.nodes_x[mesh.airfoil_nodes[i][0]]
            element_node1_x = mesh.nodes_x[mesh.airfoil_nodes[i][1]]
            element_node0_y = mesh.nodes_y[mesh.airfoil_nodes[i][0]]
            element_node1_y = mesh.nodes_y[mesh.airfoil_nodes[i][1]]
            x_avg = (element_node0_x + element_node1_x)/2
            y_avg = (element_node0_y + element_node1_y)/2
            dx = element_node1_x - element_node0_x
            dy = element_node1_y - element_node0_y
            normal_x = -dy
            normal_y = dx
            ax.text(x_avg + normal_x * boundaries_text_offset[0] + abs(dx) * boundaries_text_offset[1], y_avg + normal_y * boundaries_text_offset[0] + abs(dy) * boundaries_text_offset[1], f"{airfoil_tag} {i}", fontfamily="Fira Code", fontsize=boundaries_font_size, horizontalalignment="center", verticalalignment="center", color=boundaries_colour)
        
        for i in range(mesh.boundary_nodes.shape[0]):
            element_node0_x = mesh.nodes_x[mesh.boundary_nodes[i][0]]
            element_node1_x = mesh.nodes_x[mesh.boundary_nodes[i][1]]
            element_node0_y = mesh.nodes_y[mesh.boundary_nodes[i][0]]
            element_node1_y = mesh.nodes_y[mesh.boundary_nodes[i][1]]
            x_avg = (element_node0_x + element_node1_x)/2
            y_avg = (element_node0_y + element_node1_y)/2
            dx = element_node1_x - element_node0_x
            dy = element_node1_y - element_node0_y
            normal_x = -dy
            normal_y = dx
            ax.text(x_avg + normal_x * boundaries_text_offset[0] + abs(dx) * boundaries_text_offset[1], y_avg + normal_y * boundaries_text_offset[0] + abs(dy) * boundaries_text_offset[1], f"{boundary_tag} {i}", fontfamily="Fira Code", fontsize=boundaries_font_size, horizontalalignment="center", verticalalignment="center", color=boundaries_colour)
        
        ax.plot([], [], color=boundaries_colour, linestyle="None", linewidth=0, marker="$n$", markersize=boundaries_font_size, label="Boundary conditions")

    if show_legend:
        ax.legend()

def main(argv: list[str]):
    parser = argparse.ArgumentParser(description="Generates airfoil meshes using the SU2 format.")
    parser.add_argument('-f', '--filename', type=Path, default='./meshes/mesh.su2', help='path to the resulting mesh')
    parser.add_argument('-p', '--profile', type=str, default="0012", help='4 or 5 digit NACA profile')
    parser.add_argument('-r', '--r_res', type=int, default=64, help='number of nodes in the r direction')
    parser.add_argument('-t', '--theta_res', type=int, default=32, help='number of nodes in the θ direction on each side of the airfoil')
    parser.add_argument('-d', '--diameter', type=float, default=128, help='diameter of the resulting mesh')
    parser.add_argument('-e', '--exponent', type=float, default=1, help='grouping of the elements, above 1 is closer to the airfoil, below 1 is further')
    parser.add_argument('-b', '--boundary', type=str, default='FARFIELD', help='marker tag of the outside boundary')
    parser.add_argument('-a', '--airfoil', type=str, default='WALL', help='marker tag of the airfoil')
    parser.add_argument('-x', '--xoffset', type=float, default=0, help='x offset to center airfoil')
    parser.add_argument('-y', '--yoffset', type=float, default=0, help='y offset to center airfoil')
    parser.add_argument('--uneven', type=bool, default=True, action=argparse.BooleanOptionalAction, help='uniform spacing')
    parser.add_argument('--plot', type=bool, default=False, action=argparse.BooleanOptionalAction, help='display mesh')
    parser.add_argument('--elements', type=bool, default=True, action=argparse.BooleanOptionalAction, help='show/hide elements')
    parser.add_argument('--nodes', type=bool, default=True, action=argparse.BooleanOptionalAction, help='show/hide nodes')
    parser.add_argument('--ghosts', type=bool, default=True, action=argparse.BooleanOptionalAction, help='show/hide ghosts')
    parser.add_argument('--element-numbers', type=bool, default=True, action=argparse.BooleanOptionalAction, help='show/hide element numbers')
    parser.add_argument('--node-numbers', type=bool, default=True, action=argparse.BooleanOptionalAction, help='show/hide node numbers')
    parser.add_argument('--ghost-numbers', type=bool, default=True, action=argparse.BooleanOptionalAction, help='show/hide ghost numbers')
    parser.add_argument('--boundary-numbers', type=bool, default=True, action=argparse.BooleanOptionalAction, help='show/hide boundary numbers')
    parser.add_argument('--title', type=bool, default=True, action=argparse.BooleanOptionalAction, help='show/hide title')
    parser.add_argument('--axis', type=bool, default=True, action=argparse.BooleanOptionalAction, help='show/hide axes')
    parser.add_argument('--legend', type=bool, default=True, action=argparse.BooleanOptionalAction, help='show/hide legend')
    parser.add_argument('--curve', type=bool, default=True, action=argparse.BooleanOptionalAction, help='show/hide Hilbert curve')
    parser.add_argument('-v', '--version', action='version', version='%(prog)s 1.0')
    args = parser.parse_args(argv)

    mesh = generate_mesh(args.profile, args.r_res, args.theta_res, args.diameter, args.exponent, args.xoffset, args.yoffset, args.uneven)
    write_mesh(mesh, args.filename, args.boundary, args.airfoil)

    if args.plot:
        plot_mesh(mesh, airfoil_tag=args.airfoil, boundary_tag=args.boundary, show_elements=args.elements, show_nodes=args.nodes, show_ghosts=args.ghosts, element_numbers=args.element_numbers, node_numbers=args.node_numbers, ghost_numbers=args.ghost_numbers, boundary_numbers=args.boundary_numbers, show_title=args.title, show_axis=args.axis, show_legend=args.legend, show_curve=args.curve)
        plt.show()

if __name__ == "__main__":
    main(sys.argv[1:])