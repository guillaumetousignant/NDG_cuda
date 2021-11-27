import matplotlib.pyplot as plt
import numpy as np
import re
from pathlib import Path
import sys
import getopt

def read_file(filename: Path):
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
        elements_nodes = np.zeros(4 * n_elements_total, dtype=np.uint64)
        faces_nodes = np.zeros(2 * n_faces, dtype=np.uint64)
        faces_elements = np.zeros(2 * n_faces, dtype=np.uint64)
        faces_elements_side = np.zeros(2 * n_faces, dtype=np.uint64)
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
        self_interfaces_destinations = np.zeros(n_interfaces, dtype=np.uint64)
        self_interfaces_origins = np.zeros(n_interfaces, dtype=np.uint64)
        self_interfaces_origins_side = np.zeros(n_interfaces, dtype=np.uint64)
        wall_boundaries = np.zeros(n_walls, dtype=np.uint64)
        symmetry_boundaries = np.zeros(n_symmetries, dtype=np.uint64)
        inflow_boundaries = np.zeros(n_inflows, dtype=np.uint64)
        outflow_boundaries = np.zeros(n_outflows, dtype=np.uint64)

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

            faces_elements[2 * i]     = int(words[3])
            faces_elements[2 * i + 1] = int(words[4])

        line_index += n_faces + 2

        for i in range(n_faces):
            line = lines[line_index + i]
            words = line.split()

            faces_elements_side[2 * i]     = int(words[3])
            faces_elements_side[2 * i + 1] = int(words[4])

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

            self_interfaces_destinations[i] = bool(words[3])
            self_interfaces_origins[i] = bool(words[4])
            self_interfaces_origins_side[i] = bool(words[5])

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
            mpi_interfaces_outgoing_index.append(np.zeros(outgoing_size, dtype=np.uint64))
            mpi_interfaces_outgoing_side.append(np.zeros(outgoing_size, dtype=np.uint64))
            mpi_interfaces_incoming_index.append(np.zeros(incoming_size, dtype=np.uint64))

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



    points_colour = np.array([197, 134, 192])/255
    elements_colour = np.array([37, 37, 37])/255
    faces_colour = np.array([86, 156, 214])/255
    ghosts_colour = np.array([244, 71, 71])/255
    points_width = 12
    elements_width = 3
    faces_width = 1
    ghosts_width = 5
    points_size = 12
    points_shape = "."
    faces_offset = 0.2
    ghost_offset = 0.3

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_aspect(1)

    ax.plot(nodes_x, nodes_y, color=points_colour, linestyle="None", linewidth=points_width, marker=points_shape, markersize=points_size)

    for i in range(n_elements, n_elements_total):
        x = [nodes_x[elements_nodes[4 * i]], nodes_x[elements_nodes[4 * i + 1]]]
        y = [nodes_y[elements_nodes[4 * i]], nodes_y[elements_nodes[4 * i + 1]]]
        x_avg = (x[0] + x[1])/2
        y_avg = (y[0] + y[1])/2
        x = [x[0] * (1 - ghost_offset) + x_avg * ghost_offset, x[1] * (1 - ghost_offset) + x_avg * ghost_offset]
        y = [y[0] * (1 - ghost_offset) + y_avg * ghost_offset, y[1] * (1 - ghost_offset) + y_avg * ghost_offset]
        
        ax.plot(x, y, color=ghosts_colour, linewidth=ghosts_width)

    for i in range(n_elements):
        ax.plot([nodes_x[elements_nodes[4 * i]], nodes_x[elements_nodes[4 * i + 1]], nodes_x[elements_nodes[4 * i + 2]], nodes_x[elements_nodes[4 * i + 3]], nodes_x[elements_nodes[4 * i]]], [nodes_y[elements_nodes[4 * i]], nodes_y[elements_nodes[4 * i + 1]], nodes_y[elements_nodes[4 * i + 2]], nodes_y[elements_nodes[4 * i + 3]], nodes_y[elements_nodes[4 * i]]], color=elements_colour, linewidth=elements_width)

    for i in range(n_faces):
        x = [nodes_x[faces_nodes[2 * i]], nodes_x[faces_nodes[2 * i + 1]]]
        y = [nodes_y[faces_nodes[2 * i]], nodes_y[faces_nodes[2 * i + 1]]]
        x_avg = (x[0] + x[1])/2
        y_avg = (y[0] + y[1])/2
        x = [x[0] * (1 - faces_offset) + x_avg * faces_offset, x[1] * (1 - faces_offset) + x_avg * faces_offset]
        y = [y[0] * (1 - faces_offset) + y_avg * faces_offset, y[1] * (1 - faces_offset) + y_avg * faces_offset]
        ax.plot(x, y, color=faces_colour, linewidth=faces_width)

    plt.show()

def main(argv):
    inputfile = Path.cwd() / "input.log"

    try:
        opts, args = getopt.getopt(argv,"hi:",["input=","help"])
    except getopt.error as err:
        print (str(err))
        exit(2)

    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print("connectivity_plot.py -i <inputfile>")
            exit()
        elif opt in ("-i", "--input"):
            inputfile = arg

    read_file(inputfile)


if __name__ == "__main__":
    main(sys.argv[1:])