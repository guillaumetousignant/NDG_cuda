# Plots meshes with the data returned from the Mesh2D_t print() function

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import re
from pathlib import Path
import sys
import argparse
import math

class Elements:
    def __init__(self,
                n: int,
                n_total: int,
                nodes: npt.ArrayLike, 
                type: npt.ArrayLike,
                rotation: npt.ArrayLike,
                min_length: npt.ArrayLike,
                N: npt.ArrayLike) :

        self.n = n
        self.n_total = n_total
        self.nodes = nodes
        self.type = type
        self.rotation = rotation
        self.min_length = min_length
        self.N = N

class Face_elements:
    def __init__(self,
                L: npt.ArrayLike,
                R: npt.ArrayLike):

        self.L = L
        self.R = R

class Face_elements_side:
    def __init__(self,
                L: npt.ArrayLike,
                R: npt.ArrayLike):

        self.L = L
        self.R = R

class Face_normals:
    def __init__(self,
                x: npt.ArrayLike,
                y: npt.ArrayLike):

        self.x = x
        self.y = y

class Face_tangents:
    def __init__(self,
                x: npt.ArrayLike,
                y: npt.ArrayLike):

        self.x = x
        self.y = y

class Face_offsets:
    def __init__(self,
                L: npt.ArrayLike,
                R: npt.ArrayLike):

        self.L = L
        self.R = R

class Face_scales:
    def __init__(self,
                L: npt.ArrayLike,
                R: npt.ArrayLike):

        self.L = L
        self.R = R
        
class Faces:
    def __init__(self,
                n: int,
                nodes: npt.ArrayLike, 
                elements_L: npt.ArrayLike,
                elements_R: npt.ArrayLike,
                elements_side_L: npt.ArrayLike,
                elements_side_R: npt.ArrayLike,
                N: npt.ArrayLike,
                length: npt.ArrayLike,
                normals_x: npt.ArrayLike,
                normals_y: npt.ArrayLike,
                tangents_x: npt.ArrayLike,
                tangents_y: npt.ArrayLike,
                offsets_L: npt.ArrayLike,
                offsets_R: npt.ArrayLike,
                scales_L: npt.ArrayLike,
                scales_R: npt.ArrayLike,
                refine: npt.ArrayLike) :

        self.n = n
        self.nodes = nodes
        self.elements = Face_elements(elements_L, elements_R)
        self.elements_side = Face_elements_side(elements_side_L, elements_side_R)
        self.N = N
        self.length = length
        self.normals = Face_normals(normals_x, normals_y)
        self.tangents = Face_tangents(tangents_x, tangents_y)
        self.offsets = Face_offsets(offsets_L, offsets_R)
        self.scales = Face_offsets(scales_L, scales_R)
        self.refine = refine 

class Nodes:
    def __init__(self,
                n: int,
                x: npt.ArrayLike,
                y: npt.ArrayLike):

        self.n = n
        self.x = x
        self.y = y

class Boundary:
    def __init__(self,
                n: int,
                elements: npt.ArrayLike):

        self.n = n
        self.elements = elements

class Interfaces:
    def __init__(self,
                n: int,
                destination: npt.ArrayLike,
                origin: npt.ArrayLike,
                origin_side: npt.ArrayLike):

        self.n = n
        self.destination = destination
        self.origin = origin
        self.origin_side = origin_side

class Outgoing_MPI_Interfaces:
    def __init__(self,
                n_elements: int,
                size: npt.ArrayLike,
                offset: npt.ArrayLike,
                elements: npt.ArrayLike,
                elements_side: npt.ArrayLike):

        self.n_elements = n_elements
        self.size = size
        self.offset = offset
        self.elements = elements
        self.elements_side = elements_side

class Incoming_MPI_Interfaces:
    def __init__(self,
                n_elements: int,
                size: npt.ArrayLike,
                offset: npt.ArrayLike,
                elements: npt.ArrayLike):

        self.n_elements = n_elements
        self.size = size
        self.offset = offset
        self.elements = elements

class MPI_Interfaces:
    def __init__(self,
                n: int,
                process: npt.ArrayLike,
                n_outgoing: int,
                n_incoming: int,
                size_outgoing: npt.ArrayLike,
                size_incoming: npt.ArrayLike,
                offset_outgoing: npt.ArrayLike,
                offset_incoming: npt.ArrayLike,
                elements_outgoing: npt.ArrayLike,
                elements_incoming: npt.ArrayLike,
                elements_side_outgoing: npt.ArrayLike):

        self.n = n
        self.process = process
        self.outgoing =  Outgoing_MPI_Interfaces(n_outgoing, size_outgoing, offset_outgoing, elements_outgoing, elements_side_outgoing)
        self.incoming =  Incoming_MPI_Interfaces(n_incoming, size_incoming, offset_incoming, elements_incoming)

class Mesh:
    def __init__(self, 
                n_elements: int, 
                n_elements_total: int, 
                n_faces: int, 
                n_nodes: int, 
                n_walls: int, 
                n_symmetries: int, 
                n_inflows: int, 
                n_outflows: int, 
                n_interfaces: int, 
                n_mpi_origins: int, 
                n_mpi_destinations: int, 
                nodes_x: npt.ArrayLike, 
                nodes_y: npt.ArrayLike, 
                elements_nodes: npt.ArrayLike,
                faces_nodes: npt.ArrayLike,
                faces_elements_L: npt.ArrayLike,
                faces_elements_R: npt.ArrayLike,
                faces_elements_side_L: npt.ArrayLike,
                faces_elements_side_R: npt.ArrayLike,
                elements_type: npt.ArrayLike,
                elements_rotation: npt.ArrayLike,
                elements_min_length: npt.ArrayLike,
                elements_N: npt.ArrayLike,
                faces_N: npt.ArrayLike,
                faces_length: npt.ArrayLike,
                face_normals_x: npt.ArrayLike,
                face_normals_y: npt.ArrayLike,
                face_tangents_x: npt.ArrayLike,
                face_tangents_y: npt.ArrayLike,
                face_offsets_L: npt.ArrayLike,
                face_offsets_R: npt.ArrayLike,
                face_scales_L: npt.ArrayLike,
                face_scales_R: npt.ArrayLike,
                face_refine: npt.ArrayLike,
                self_interfaces_destinations: npt.ArrayLike,
                self_interfaces_origins: npt.ArrayLike,
                self_interfaces_origins_side: npt.ArrayLike,
                wall_boundaries: npt.ArrayLike,
                symmetry_boundaries: npt.ArrayLike,
                inflow_boundaries: npt.ArrayLike,
                outflow_boundaries: npt.ArrayLike,
                n_mp_interfaces: int,
                mpi_interfaces_process: npt.ArrayLike,
                mpi_interfaces_outgoing_size: npt.ArrayLike,
                mpi_interfaces_incoming_size: npt.ArrayLike,
                mpi_interfaces_outgoing_offset: npt.ArrayLike,
                mpi_interfaces_incoming_offset: npt.ArrayLike,
                mpi_interfaces_outgoing_index: npt.ArrayLike,
                mpi_interfaces_outgoing_side: npt.ArrayLike,
                mpi_interfaces_incoming_index: npt.ArrayLike):

        self.elements = Elements(n_elements, n_elements_total, elements_nodes, elements_type, elements_rotation, elements_min_length, elements_N)
        self.faces = Faces(n_faces, faces_nodes, faces_elements_L, faces_elements_R, faces_elements_side_L, faces_elements_side_R, faces_N, faces_length,face_normals_x, face_normals_y, face_tangents_x, face_tangents_y, face_offsets_L, face_offsets_R, face_scales_L, face_scales_R, face_refine)
        self.nodes = Nodes(n_nodes, nodes_x, nodes_y)
        self.wall = Boundary(n_walls, wall_boundaries)
        self.symmetry = Boundary(n_symmetries, symmetry_boundaries)
        self.inflow = Boundary(n_inflows, inflow_boundaries)
        self.outflow = Boundary(n_outflows, outflow_boundaries)
        self.interfaces = Interfaces(n_interfaces, self_interfaces_destinations, self_interfaces_origins, self_interfaces_origins_side)
        self.mpi_interfaces = MPI_Interfaces(n_mp_interfaces, mpi_interfaces_process, n_mpi_origins, n_mpi_destinations, mpi_interfaces_outgoing_size, mpi_interfaces_incoming_size, mpi_interfaces_outgoing_offset, mpi_interfaces_incoming_offset, mpi_interfaces_outgoing_index, mpi_interfaces_incoming_index, mpi_interfaces_outgoing_side)

def read_file(filename: Path) -> Mesh:
    with open(filename, 'r') as file:
        lines = file.readlines()

        # Not actually needed, since they have a known length and always start the line
        n_elements_finder = re.compile(r"N elements: \d*")
        n_elements_total_finder = re.compile(r"N elements and ghosts: \d*")
        n_faces_finder = re.compile(r"N faces: \d*")
        n_nodes_finder = re.compile(r"N nodes: \d*")
        n_walls_finder = re.compile(r"N wall boundaries: \d*")
        n_symmetries_finder = re.compile(r"N symmetry boundaries: \d*")
        n_inflows_finder = re.compile(r"N inflow boundaries: \d*")
        n_outflows_finder = re.compile(r"N outflow boundaries: \d*")
        n_interfaces_finder = re.compile(r"N interfaces: \d*")
        n_mpi_origins_finder = re.compile(r"N mpi incoming interfaces: \d*")
        n_mpi_destinations_finder = re.compile(r"N mpi outgoing interfaces: \d*")
        process_finder = re.compile(r"MPI interface to process \d*")
        outgoing_size_finder = re.compile(r" of outgoing size \d*")
        incoming_size_finder = re.compile(r", incoming size \d*")
        outgoing_offset_finder = re.compile(r", outgoing offset \d*")
        incoming_offset_finder = re.compile(r" and incoming offset \d*")

        n_elements_match = n_elements_finder.search(lines[0])
        n_elements_total_match = n_elements_total_finder.search(lines[1])
        n_faces_match = n_faces_finder.search(lines[2])
        n_nodes_match = n_nodes_finder.search(lines[3])
        n_walls_match = n_walls_finder.search(lines[4])
        n_symmetries_match = n_symmetries_finder.search(lines[5])
        n_inflows_match = n_inflows_finder.search(lines[6])
        n_outflows_match = n_outflows_finder.search(lines[7])
        n_interfaces_match = n_interfaces_finder.search(lines[8])
        n_mpi_origins_match = n_mpi_origins_finder.search(lines[9])
        n_mpi_destinations_match = n_mpi_destinations_finder.search(lines[10])

        n_elements = int(n_elements_match.group(0)[12:])
        n_elements_total = int(n_elements_total_match.group(0)[23:])
        n_faces = int(n_faces_match.group(0)[9:])
        n_nodes = int(n_nodes_match.group(0)[9:])
        n_walls = int(n_walls_match.group(0)[19:])
        n_symmetries = int(n_symmetries_match.group(0)[23:])
        n_inflows = int(n_inflows_match.group(0)[21:])
        n_outflows = int(n_outflows_match.group(0)[22:])
        n_interfaces = int(n_interfaces_match.group(0)[14:])
        n_mpi_origins = int(n_mpi_origins_match.group(0)[27:])
        n_mpi_destinations = int(n_mpi_destinations_match.group(0)[27:])

        nodes_x = np.zeros(n_nodes)
        nodes_y = np.zeros(n_nodes)
        elements_nodes = np.zeros(4 * n_elements_total, dtype=np.uint32)
        faces_nodes = np.zeros(2 * n_faces, dtype=np.uint32)
        faces_elements_L = np.zeros(n_faces, dtype=np.uint32)
        faces_elements_R = np.zeros(n_faces, dtype=np.uint32)
        faces_elements_side_L = np.zeros(n_faces, dtype=np.uint32)
        faces_elements_side_R = np.zeros(n_faces, dtype=np.uint32)
        elements_type = np.zeros(n_elements_total, dtype=np.unicode_)
        elements_rotation = np.zeros(n_elements_total, dtype=np.uint8)
        elements_min_length = np.zeros(n_elements_total)
        elements_N = np.zeros(n_elements_total, dtype=np.int64)
        faces_N = np.zeros(n_faces, dtype=np.int64)
        faces_length = np.zeros(n_faces)
        face_normals_x = np.zeros(n_faces)
        face_normals_y = np.zeros(n_faces)
        face_tangents_x = np.zeros(n_faces)
        face_tangents_y = np.zeros(n_faces)
        face_offsets_L = np.zeros(n_faces)
        face_offsets_R = np.zeros(n_faces)
        face_scales_L = np.zeros(n_faces)
        face_scales_R = np.zeros(n_faces)
        face_refine = np.zeros(n_faces, dtype=np.bool8)
        self_interfaces_destinations = np.zeros(n_interfaces, dtype=np.uint32)
        self_interfaces_origins = np.zeros(n_interfaces, dtype=np.uint32)
        self_interfaces_origins_side = np.zeros(n_interfaces, dtype=np.uint32)
        wall_boundaries = np.zeros(n_walls, dtype=np.uint32)
        symmetry_boundaries = np.zeros(n_symmetries, dtype=np.uint32)
        inflow_boundaries = np.zeros(n_inflows, dtype=np.uint32)
        outflow_boundaries = np.zeros(n_outflows, dtype=np.uint32)

        line_index = 15

        for i in range(n_nodes):
            line = lines[line_index + i]
            words = line.split()

            nodes_x[i] = float(words[3][1:-1])
            nodes_y[i] = float(words[4][0:-1])
        
        line_index += n_nodes + 2

        for i in range(n_elements_total):
            line = lines[line_index + i]
            words = line.split()

            elements_nodes[4 * i]     = int(words[3])
            elements_nodes[4 * i + 1] = int(words[4])
            elements_nodes[4 * i + 2] = int(words[5])
            elements_nodes[4 * i + 3] = int(words[6])

        line_index += n_elements_total + 2

        for i in range(n_faces):
            line = lines[line_index + i]
            words = line.split()

            faces_nodes[2 * i]     = int(words[3])
            faces_nodes[2 * i + 1] = int(words[4])

        line_index += n_faces + 2

        for i in range(n_faces):
            line = lines[line_index + i]
            words = line.split()

            faces_elements_L[i] = int(words[3])
            faces_elements_R[i] = int(words[4])

        line_index += n_faces + 2

        for i in range(n_faces):
            line = lines[line_index + i]
            words = line.split()

            faces_elements_side_L[i] = int(words[3])
            faces_elements_side_R[i] = int(words[4])

        line_index += n_faces + 3

        for i in range(n_elements_total):
            line = lines[line_index + i]
            words = line.split()

            elements_type[i] = words[3][0]

        line_index += n_elements_total + 2

        for i in range(n_elements_total):
            line = lines[line_index + i]
            words = line.split()

            elements_rotation[i] = int(words[3])

        line_index += n_elements_total + 2

        for i in range(n_elements_total):
            line = lines[line_index + i]
            words = line.split()

            elements_min_length[i] = float(words[3])

        line_index += n_elements_total + 2

        for i in range(n_elements_total):
            line = lines[line_index + i]
            words = line.split()

            elements_N[i] = int(words[3])

        line_index += n_elements_total + 2

        for i in range(n_faces):
            line = lines[line_index + i]
            words = line.split()

            faces_N[i] = int(words[3])

        line_index += n_faces + 2

        for i in range(n_faces):
            line = lines[line_index + i]
            words = line.split()

            faces_length[i] = float(words[3])

        line_index += n_faces + 2

        for i in range(n_faces):
            line = lines[line_index + i]
            words = line.split()

            face_normals_x[i] = float(words[3][1:-1])
            face_normals_y[i] = float(words[4][0:-1])

        line_index += n_faces + 2

        for i in range(n_faces):
            line = lines[line_index + i]
            words = line.split()

            face_tangents_x[i] = float(words[3][1:-1])
            face_tangents_y[i] = float(words[4][0:-1])

        line_index += n_faces + 2

        for i in range(n_faces):
            line = lines[line_index + i]
            words = line.split()

            face_offsets_L[i] = float(words[3])
            face_offsets_R[i] = float(words[4])

        line_index += n_faces + 2

        for i in range(n_faces):
            line = lines[line_index + i]
            words = line.split()

            face_scales_L[i] = float(words[3])
            face_scales_R[i] = float(words[4])

        line_index += n_faces + 2

        for i in range(n_faces):
            line = lines[line_index + i]
            words = line.split()

            face_refine[i] = bool(words[3])

        line_index += n_faces + 2

        for i in range(n_interfaces):
            line = lines[line_index + i]
            words = line.split()

            self_interfaces_destinations[i] = int(words[3])
            self_interfaces_origins[i] = int(words[4])
            self_interfaces_origins_side[i] = int(words[5])

        line_index += n_interfaces + 2

        n_mp_interfaces = 0

        mpi_interfaces_process = []
        mpi_interfaces_outgoing_size = []
        mpi_interfaces_incoming_size = []
        mpi_interfaces_outgoing_offset = []
        mpi_interfaces_incoming_offset = []
        mpi_interfaces_outgoing_index = []
        mpi_interfaces_outgoing_side = []
        mpi_interfaces_incoming_index = []

        while lines[line_index].startswith("	MPI interface to process"):
            process_match = process_finder.search(lines[line_index])
            outgoing_size_match = outgoing_size_finder.search(lines[line_index])
            incoming_size_match = incoming_size_finder.search(lines[line_index])
            outgoing_offset_match = outgoing_offset_finder.search(lines[line_index])
            incoming_offset_match = incoming_offset_finder.search(lines[line_index])

            process = int(process_match.group(0)[25:])
            outgoing_size = int(outgoing_size_match.group(0)[18:])
            incoming_size = int(incoming_size_match.group(0)[16:])
            outgoing_offset = int(outgoing_offset_match.group(0)[18:])
            incoming_offset = int(incoming_offset_match.group(0)[21:])

            mpi_interfaces_process.append(process)
            mpi_interfaces_outgoing_size.append(outgoing_size)
            mpi_interfaces_incoming_size.append(incoming_size)
            mpi_interfaces_outgoing_offset.append(outgoing_offset)
            mpi_interfaces_incoming_offset.append(incoming_offset)
            mpi_interfaces_outgoing_index.append(np.zeros(outgoing_size, dtype=np.uint32))
            mpi_interfaces_outgoing_side.append(np.zeros(outgoing_size, dtype=np.uint32))
            mpi_interfaces_incoming_index.append(np.zeros(incoming_size, dtype=np.uint32))

            line_index += 2

            for i in range(outgoing_size):
                line = lines[line_index + i]
                words = line.split()

                mpi_interfaces_outgoing_index[n_mp_interfaces][i] = int(words[4])
                mpi_interfaces_outgoing_side[n_mp_interfaces][i] = int(words[5])

            line_index += outgoing_size + 1

            for i in range(incoming_size):
                line = lines[line_index + i]
                words = line.split()

                mpi_interfaces_incoming_index[n_mp_interfaces][i] = int(words[4])

            line_index += incoming_size

            n_mp_interfaces += 1

        line_index += 2

        for i in range(n_walls):
            line = lines[line_index + i]
            words = line.split()

            wall_boundaries[i] = int(words[3])

        line_index += n_walls + 2

        for i in range(n_symmetries):
            line = lines[line_index + i]
            words = line.split()

            symmetry_boundaries[i] = int(words[3])

        line_index += n_symmetries + 2

        for i in range(n_inflows):
            line = lines[line_index + i]
            words = line.split()

            inflow_boundaries[i] = int(words[3])

        line_index += n_inflows + 2

        for i in range(n_outflows):
            line = lines[line_index + i]
            words = line.split()

            outflow_boundaries[i] = int(words[3])

        line_index += n_outflows + 2

    return Mesh(n_elements, 
                n_elements_total, 
                n_faces, 
                n_nodes, 
                n_walls, 
                n_symmetries, 
                n_inflows, 
                n_outflows, 
                n_interfaces, 
                n_mpi_origins, 
                n_mpi_destinations, 
                nodes_x, 
                nodes_y, 
                elements_nodes,
                faces_nodes,
                faces_elements_L,
                faces_elements_R,
                faces_elements_side_L,
                faces_elements_side_R,
                elements_type,
                elements_rotation,
                elements_min_length,
                elements_N,
                faces_N,
                faces_length,
                face_normals_x,
                face_normals_y,
                face_tangents_x,
                face_tangents_y,
                face_offsets_L,
                face_offsets_R,
                face_scales_L,
                face_scales_R,
                face_refine,
                self_interfaces_destinations,
                self_interfaces_origins,
                self_interfaces_origins_side,
                wall_boundaries,
                symmetry_boundaries,
                inflow_boundaries,
                outflow_boundaries,
                n_mp_interfaces,
                mpi_interfaces_process,
                mpi_interfaces_outgoing_size,
                mpi_interfaces_incoming_size,
                mpi_interfaces_outgoing_offset,
                mpi_interfaces_incoming_offset,
                mpi_interfaces_outgoing_index,
                mpi_interfaces_outgoing_side,
                mpi_interfaces_incoming_index)

def plot_mesh(mesh: Mesh, 
    title: str = "Mesh", 
    show_elements: bool = True, 
    show_faces: bool = True, 
    show_nodes: bool = True, 
    show_ghosts: bool = True, 
    element_numbers: bool = True, 
    face_numbers: bool = True, 
    node_numbers: bool = True, 
    ghost_numbers: bool = True, 
    boundary_numbers: bool = True,
    show_title: bool = True,
    show_axis: bool = True,
    show_legend: bool = True,
    show_curve: bool = True):
    
    points_colour = np.array([197, 134, 192])/255
    elements_colour = np.array([37, 37, 37])/255
    faces_colour = np.array([86, 156, 214])/255
    ghosts_colour = np.array([244, 71, 71])/255
    boundaries_colour = np.array([78, 201, 176])/255
    curve_colour = np.array([106, 153, 85])/255
    points_width = 12
    elements_width = 3
    faces_width = 1
    ghosts_width = 5
    curve_width = 5
    points_size = 12
    points_shape = "."
    faces_offset = 0.2
    ghost_offset = 0.3
    points_font_size = 12
    elements_font_size = 14
    faces_font_size = 10
    ghosts_font_size = 14
    boundaries_font_size = 6
    points_text_offset = [-0.005, -0.005]
    elements_text_offset = [0, 0]
    faces_text_offset = 0.1
    ghosts_text_offset = 0.2
    boundaries_text_offset = [0.2, 0.15]
    outgoing_boundaries_text_offset = [0.2, 0]

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
        for i in range(mesh.elements.n, mesh.elements.n_total):
            x = [mesh.nodes.x[mesh.elements.nodes[4 * i]], mesh.nodes.x[mesh.elements.nodes[4 * i + 1]]]
            y = [mesh.nodes.y[mesh.elements.nodes[4 * i]], mesh.nodes.y[mesh.elements.nodes[4 * i + 1]]]
            x_avg = (x[0] + x[1])/2
            y_avg = (y[0] + y[1])/2
            x = [x[0] * (1 - ghost_offset) + x_avg * ghost_offset, x[1] * (1 - ghost_offset) + x_avg * ghost_offset]
            y = [y[0] * (1 - ghost_offset) + y_avg * ghost_offset, y[1] * (1 - ghost_offset) + y_avg * ghost_offset]
            norm = math.sqrt(x_avg**2 + y_avg**2)

            dx = x[1] - x[0]
            dy = y[1] - y[0]
            if (abs(dx) > 0 or abs(dy) > 0):
                norm = math.sqrt(dx**2 + dy**2)
                dx /= norm
                dy /= norm
            normal = [-dy, dx]

            ax.plot(x, y, color=ghosts_colour, linewidth=ghosts_width, label="Ghost elements" if i == mesh.elements.n else "")
            if ghost_numbers:
                ax.text(x_avg + normal[0] * ghosts_text_offset * mesh.elements.min_length[i], y_avg + normal[1] * ghosts_text_offset * mesh.elements.min_length[i], f"{i}", fontfamily="Fira Code", fontsize=ghosts_font_size, horizontalalignment="center", verticalalignment="center", color=ghosts_colour)

    if show_elements:
        for i in range(mesh.elements.n):
            ax.plot([mesh.nodes.x[mesh.elements.nodes[4 * i]], mesh.nodes.x[mesh.elements.nodes[4 * i + 1]], mesh.nodes.x[mesh.elements.nodes[4 * i + 2]], mesh.nodes.x[mesh.elements.nodes[4 * i + 3]], mesh.nodes.x[mesh.elements.nodes[4 * i]]], [mesh.nodes.y[mesh.elements.nodes[4 * i]], mesh.nodes.y[mesh.elements.nodes[4 * i + 1]], mesh.nodes.y[mesh.elements.nodes[4 * i + 2]], mesh.nodes.y[mesh.elements.nodes[4 * i + 3]], mesh.nodes.y[mesh.elements.nodes[4 * i]]], color=elements_colour, linewidth=elements_width, label="Elements" if i == 0 else "")
            if element_numbers:
                ax.text((mesh.nodes.x[mesh.elements.nodes[4 * i]] + mesh.nodes.x[mesh.elements.nodes[4 * i + 1]] + mesh.nodes.x[mesh.elements.nodes[4 * i + 2]] + mesh.nodes.x[mesh.elements.nodes[4 * i + 3]])/4 + elements_text_offset[0], (mesh.nodes.y[mesh.elements.nodes[4 * i]] + mesh.nodes.y[mesh.elements.nodes[4 * i + 1]] + mesh.nodes.y[mesh.elements.nodes[4 * i + 2]] + mesh.nodes.y[mesh.elements.nodes[4 * i + 3]])/4 + elements_text_offset[1], f"{i}", fontfamily="Fira Code", fontsize=elements_font_size, horizontalalignment="center", verticalalignment="center", color=elements_colour)

    if show_faces:
        for i in range(mesh.faces.n):
            x = [mesh.nodes.x[mesh.faces.nodes[2 * i]], mesh.nodes.x[mesh.faces.nodes[2 * i + 1]]]
            y = [mesh.nodes.y[mesh.faces.nodes[2 * i]], mesh.nodes.y[mesh.faces.nodes[2 * i + 1]]]
            x_avg = (x[0] + x[1])/2
            y_avg = (y[0] + y[1])/2
            x = [x[0] * (1 - faces_offset) + x_avg * faces_offset, x[1] * (1 - faces_offset) + x_avg * faces_offset]
            y = [y[0] * (1 - faces_offset) + y_avg * faces_offset, y[1] * (1 - faces_offset) + y_avg * faces_offset]
            ax.plot(x, y, color=faces_colour, linewidth=faces_width, label="Faces" if i == 0 else "")
            if face_numbers:
                ax.text(x_avg + mesh.faces.normals.x[i] * faces_text_offset * mesh.faces.length[i], y_avg + mesh.faces.normals.y[i] * faces_text_offset * mesh.faces.length[i], f"{i}", fontfamily="Fira Code", fontsize=faces_font_size, horizontalalignment="center", verticalalignment="center", color=faces_colour)

    if show_nodes:
        ax.plot(mesh.nodes.x, mesh.nodes.y, color=points_colour, linestyle="None", linewidth=points_width, marker=points_shape, markersize=points_size, label="Nodes")
        if node_numbers:
            for i in range(mesh.nodes.n):
                ax.text(mesh.nodes.x[i] + points_text_offset[0], mesh.nodes.y[i] + points_text_offset[1], f"{i}", fontfamily="Fira Code", fontsize=points_font_size, horizontalalignment="right", verticalalignment="top", color=points_colour)

    if show_curve:
        elements_center_x = np.zeros(mesh.elements.n)
        elements_center_y = np.zeros(mesh.elements.n)
        for i in range(mesh.elements.n):
            elements_center_x[i] = (mesh.nodes.x[mesh.elements.nodes[4 * i]] + mesh.nodes.x[mesh.elements.nodes[4 * i + 1]] + mesh.nodes.x[mesh.elements.nodes[4 * i + 2]] + mesh.nodes.x[mesh.elements.nodes[4 * i + 3]])/4
            elements_center_y[i] = (mesh.nodes.y[mesh.elements.nodes[4 * i]] + mesh.nodes.y[mesh.elements.nodes[4 * i + 1]] + mesh.nodes.y[mesh.elements.nodes[4 * i + 2]] + mesh.nodes.y[mesh.elements.nodes[4 * i + 3]])/4
        ax.plot(elements_center_x, elements_center_y, color=curve_colour, linewidth=curve_width, label="Hilbert curve")

    if boundary_numbers and show_ghosts:
        for i in range(mesh.mpi_interfaces.n):
            for j in range(mesh.mpi_interfaces.incoming.size[i]):
                element_index = mesh.mpi_interfaces.incoming.elements[i][j]
                element_node0_x = mesh.nodes.x[mesh.elements.nodes[4 * element_index]]
                element_node1_x = mesh.nodes.x[mesh.elements.nodes[4 * element_index + 1]]
                element_node0_y = mesh.nodes.y[mesh.elements.nodes[4 * element_index]]
                element_node1_y = mesh.nodes.y[mesh.elements.nodes[4 * element_index + 1]]
                x_avg = (element_node0_x + element_node1_x)/2
                y_avg = (element_node0_y + element_node1_y)/2
                dx = element_node1_x - element_node0_x
                dy = element_node1_y - element_node0_y
                normal_x = -dy
                normal_y = dx
                ax.text(x_avg + normal_x * boundaries_text_offset[0] + abs(dx) * boundaries_text_offset[1], y_avg + normal_y * boundaries_text_offset[0] + abs(dy) * boundaries_text_offset[1], f"mpi in{i} {j}", fontfamily="Fira Code", fontsize=boundaries_font_size, horizontalalignment="center", verticalalignment="center", color=boundaries_colour)
            for j in range(mesh.mpi_interfaces.outgoing.size[i]):
                element_index = mesh.mpi_interfaces.outgoing.elements[i][j]
                element_side = mesh.mpi_interfaces.outgoing.elements_side[i][j]
                next_side = element_side + 1 if element_side + 1 < 4 else 0
                element_node0_x = mesh.nodes.x[mesh.elements.nodes[4 * element_index + element_side]]
                element_node1_x = mesh.nodes.x[mesh.elements.nodes[4 * element_index + next_side]]
                element_node0_y = mesh.nodes.y[mesh.elements.nodes[4 * element_index + element_side]]
                element_node1_y = mesh.nodes.y[mesh.elements.nodes[4 * element_index + next_side]]
                x_avg = (element_node0_x + element_node1_x)/2
                y_avg = (element_node0_y + element_node1_y)/2
                dx = element_node1_x - element_node0_x
                dy = element_node1_y - element_node0_y
                normal_x = -dy
                normal_y = dx
                ax.text(x_avg + normal_x * outgoing_boundaries_text_offset[0] + abs(dx) * outgoing_boundaries_text_offset[1], y_avg + normal_y * outgoing_boundaries_text_offset[0] + abs(dy) * outgoing_boundaries_text_offset[1], f"mpi out{i} {j}", fontfamily="Fira Code", fontsize=boundaries_font_size, horizontalalignment="center", verticalalignment="center", color=boundaries_colour)

        for i in range(mesh.interfaces.n):
            element_index = mesh.interfaces.destination[i]
            element_node0_x = mesh.nodes.x[mesh.elements.nodes[4 * element_index]]
            element_node1_x = mesh.nodes.x[mesh.elements.nodes[4 * element_index + 1]]
            element_node0_y = mesh.nodes.y[mesh.elements.nodes[4 * element_index]]
            element_node1_y = mesh.nodes.y[mesh.elements.nodes[4 * element_index + 1]]
            x_avg = (element_node0_x + element_node1_x)/2
            y_avg = (element_node0_y + element_node1_y)/2
            dx = element_node1_x - element_node0_x
            dy = element_node1_y - element_node0_y
            normal_x = -dy
            normal_y = dx
            ax.text(x_avg + normal_x * boundaries_text_offset[0] + abs(dx) * boundaries_text_offset[1], y_avg + normal_y * boundaries_text_offset[0] + abs(dy) * boundaries_text_offset[1], f"self in {i} ", fontfamily="Fira Code", fontsize=boundaries_font_size, horizontalalignment="center", verticalalignment="center", color=boundaries_colour)

            element_index = mesh.interfaces.origin[i]
            element_side = mesh.interfaces.origin_side[i]
            next_side = element_side + 1 if element_side + 1 < 4 else 0
            element_node0_x = mesh.nodes.x[mesh.elements.nodes[4 * element_index + element_side]]
            element_node1_x = mesh.nodes.x[mesh.elements.nodes[4 * element_index + next_side]]
            element_node0_y = mesh.nodes.y[mesh.elements.nodes[4 * element_index + element_side]]
            element_node1_y = mesh.nodes.y[mesh.elements.nodes[4 * element_index + next_side]]
            x_avg = (element_node0_x + element_node1_x)/2
            y_avg = (element_node0_y + element_node1_y)/2
            dx = element_node1_x - element_node0_x
            dy = element_node1_y - element_node0_y
            normal_x = -dy
            normal_y = dx
            ax.text(x_avg + normal_x * outgoing_boundaries_text_offset[0] + abs(dx) * outgoing_boundaries_text_offset[1], y_avg + normal_y * outgoing_boundaries_text_offset[0] + abs(dy) * outgoing_boundaries_text_offset[1], f"self out {i}", fontfamily="Fira Code", fontsize=boundaries_font_size, horizontalalignment="center", verticalalignment="center", color=boundaries_colour)

        for i in range(mesh.wall.n):
            element_index = mesh.wall.elements[i]
            element_node0_x = mesh.nodes.x[mesh.elements.nodes[4 * element_index]]
            element_node1_x = mesh.nodes.x[mesh.elements.nodes[4 * element_index + 1]]
            element_node0_y = mesh.nodes.y[mesh.elements.nodes[4 * element_index]]
            element_node1_y = mesh.nodes.y[mesh.elements.nodes[4 * element_index + 1]]
            x_avg = (element_node0_x + element_node1_x)/2
            y_avg = (element_node0_y + element_node1_y)/2
            dx = element_node1_x - element_node0_x
            dy = element_node1_y - element_node0_y
            normal_x = -dy
            normal_y = dx
            ax.text(x_avg + normal_x * boundaries_text_offset[0] + abs(dx) * boundaries_text_offset[1], y_avg + normal_y * boundaries_text_offset[0] + abs(dy) * boundaries_text_offset[1], f"wall {i}", fontfamily="Fira Code", fontsize=boundaries_font_size, horizontalalignment="center", verticalalignment="center", color=boundaries_colour)
        
        for i in range(mesh.symmetry.n):
            element_index = mesh.symmetry.elements[i]
            element_node0_x = mesh.nodes.x[mesh.elements.nodes[4 * element_index]]
            element_node1_x = mesh.nodes.x[mesh.elements.nodes[4 * element_index + 1]]
            element_node0_y = mesh.nodes.y[mesh.elements.nodes[4 * element_index]]
            element_node1_y = mesh.nodes.y[mesh.elements.nodes[4 * element_index + 1]]
            x_avg = (element_node0_x + element_node1_x)/2
            y_avg = (element_node0_y + element_node1_y)/2
            dx = element_node1_x - element_node0_x
            dy = element_node1_y - element_node0_y
            normal_x = -dy
            normal_y = dx
            ax.text(x_avg + normal_x * boundaries_text_offset[0] + abs(dx) * boundaries_text_offset[1], y_avg + normal_y * boundaries_text_offset[0] + abs(dy) * boundaries_text_offset[1], f"symm {i}", fontfamily="Fira Code", fontsize=boundaries_font_size, horizontalalignment="center", verticalalignment="center", color=boundaries_colour)
        
        for i in range(mesh.inflow.n):
            element_index = mesh.inflow.elements[i]
            element_node0_x = mesh.nodes.x[mesh.elements.nodes[4 * element_index]]
            element_node1_x = mesh.nodes.x[mesh.elements.nodes[4 * element_index + 1]]
            element_node0_y = mesh.nodes.y[mesh.elements.nodes[4 * element_index]]
            element_node1_y = mesh.nodes.y[mesh.elements.nodes[4 * element_index + 1]]
            x_avg = (element_node0_x + element_node1_x)/2
            y_avg = (element_node0_y + element_node1_y)/2
            dx = element_node1_x - element_node0_x
            dy = element_node1_y - element_node0_y
            normal_x = -dy
            normal_y = dx
            ax.text(x_avg + normal_x * boundaries_text_offset[0] + abs(dx) * boundaries_text_offset[1], y_avg + normal_y * boundaries_text_offset[0] + abs(dy) * boundaries_text_offset[1], f"inflow {i}", fontfamily="Fira Code", fontsize=boundaries_font_size, horizontalalignment="center", verticalalignment="center", color=boundaries_colour)
        
        for i in range(mesh.outflow.n):
            element_index = mesh.outflow.elements[i]
            element_node0_x = mesh.nodes.x[mesh.elements.nodes[4 * element_index]]
            element_node1_x = mesh.nodes.x[mesh.elements.nodes[4 * element_index + 1]]
            element_node0_y = mesh.nodes.y[mesh.elements.nodes[4 * element_index]]
            element_node1_y = mesh.nodes.y[mesh.elements.nodes[4 * element_index + 1]]
            x_avg = (element_node0_x + element_node1_x)/2
            y_avg = (element_node0_y + element_node1_y)/2
            dx = element_node1_x - element_node0_x
            dy = element_node1_y - element_node0_y
            normal_x = -dy
            normal_y = dx
            ax.text(x_avg + normal_x * boundaries_text_offset[0] + abs(dx) * boundaries_text_offset[1], y_avg + normal_y * boundaries_text_offset[0] + abs(dy) * boundaries_text_offset[1], f"outflow {i}", fontfamily="Fira Code", fontsize=boundaries_font_size, horizontalalignment="center", verticalalignment="center", color=boundaries_colour)

        ax.plot([], [], color=boundaries_colour, linestyle="None", linewidth=0, marker="$n$", markersize=boundaries_font_size, label="Boundary conditions")

    if show_legend:
        ax.legend()

def main(argv: list[str]):
    parser = argparse.ArgumentParser(description="Plots meshes with the data returned from the Mesh2D_t print() function.")
    parser.add_argument('meshes', metavar='mesh', type=Path, nargs='+', help='path to a mesh to display')
    parser.add_argument('--elements', type=bool, default=True, action=argparse.BooleanOptionalAction, help='show/hide elements')
    parser.add_argument('--faces', type=bool, default=True, action=argparse.BooleanOptionalAction, help='show/hide faces')
    parser.add_argument('--nodes', type=bool, default=True, action=argparse.BooleanOptionalAction, help='show/hide nodes')
    parser.add_argument('--ghosts', type=bool, default=True, action=argparse.BooleanOptionalAction, help='show/hide ghosts')
    parser.add_argument('--element-numbers', type=bool, default=True, action=argparse.BooleanOptionalAction, help='show/hide element numbers')
    parser.add_argument('--face-numbers', type=bool, default=True, action=argparse.BooleanOptionalAction, help='show/hide face numbers')
    parser.add_argument('--node-numbers', type=bool, default=True, action=argparse.BooleanOptionalAction, help='show/hide node numbers')
    parser.add_argument('--ghost-numbers', type=bool, default=True, action=argparse.BooleanOptionalAction, help='show/hide ghost numbers')
    parser.add_argument('--boundary-numbers', type=bool, default=True, action=argparse.BooleanOptionalAction, help='show/hide boundary numbers')
    parser.add_argument('--title', type=bool, default=True, action=argparse.BooleanOptionalAction, help='show/hide title')
    parser.add_argument('--axis', type=bool, default=True, action=argparse.BooleanOptionalAction, help='show/hide axes')
    parser.add_argument('--legend', type=bool, default=True, action=argparse.BooleanOptionalAction, help='show/hide legend')
    parser.add_argument('--curve', type=bool, default=True, action=argparse.BooleanOptionalAction, help='show/hide Hilbert curve')
    parser.add_argument('-v', '--version', action='version', version='%(prog)s 1.0')
    args = parser.parse_args(argv)

    meshes = []
    for mesh_file in args.meshes:
        meshes.append((read_file(mesh_file), mesh_file))

    for mesh, inputfile in meshes:
        plot_mesh(mesh, inputfile, args.elements, args.faces, args.nodes, args.ghosts, args.element_numbers, args.face_numbers, args.node_numbers, args.ghost_numbers, args.boundary_numbers, args.title, args.axis, args.legend, args.curve)

    plt.show()

if __name__ == "__main__":
    main(sys.argv[1:])