import numpy as np
import numpy.typing as npt
from pathlib import Path
import sys
import argparse

def read_su2(filename: Path) -> tuple[npt.ArrayLike, npt.ArrayLike, npt.ArrayLike, npt.ArrayLike]:
    nodes = np.zeros((0, 2))
    elements = np.zeros((0, 4), dtype=np.uint64)
    nodes_lines = []
    marker_lines = []

    with open(filename, 'r') as file:
        lines = file.readlines()

        index = 0
        while len(lines[index]) == 0 and index < len(lines):
            index += 1

        words = lines[index].split()
        if (words[0].upper() != "NDIME="):
            sys.exit(f"Error: first token should be 'NDIME=', found '{words[0]}'. Exiting.")

        ndim = int(words[1])
        if ndim != 2:
            sys.exit(f"Error: program only works for 2 dimensions, found '{ndim}'. Exiting.")
        index += 1

        while index < len(lines):
            while len(lines[index]) == 0 and index < len(lines):
                index += 1

            words = lines[index].split()
            token = words[0].upper()

            if token == "NPOIN=":
                n_nodes = int(words[1])
                nodes = np.zeros((n_nodes, 2))
                nodes_lines = lines[index : index + n_nodes + 1]
                index += 1

                for i in range(n_nodes):
                    words = lines[index + i].split()
                    nodes[i][0] = float(words[0])
                    nodes[i][1] = float(words[1])
                    
                index += n_nodes

            elif token == "NELEM=":
                n_elements = int(words[1])
                elements = np.zeros((n_elements, 4), dtype=np.uint64)
                index += 1

                for i in range(n_elements):
                    words = lines[index + i].split()
                    if words[0] != "9":
                        sys.exit(f"Error: expected token '9', found '{words[0]}'. Exiting.")

                    elements[i][0] = int(words[1])
                    elements[i][1] = int(words[2])
                    elements[i][2] = int(words[3])
                    elements[i][3] = int(words[4])
                    
                index += n_elements

            elif token == "NMARK=":
                marker_start = index
                n_markers = int(words[1])
                for i in range(n_markers):
                    index += 2
                    words = lines[index].split()
                    if words[0].upper() != "MARKER_ELEMS=":
                        sys.exit(f"Error: expected token 'MARKER_ELEMS=', found '{words[0]}'. Exiting.")
                    n_marker_elements = int(words[1])
                    index += n_marker_elements

                marker_end = index
                marker_lines = lines[marker_start : marker_end + 1]
                index += 1

            else:
                sys.exit(f"Error: expected marker 'NPOIN=', 'NELEM=' or 'NMARK=', found '{token}'. Exiting.")
    
    return (nodes, elements, nodes_lines, marker_lines)

def compute_centers(nodes: npt.ArrayLike, elements: npt.ArrayLike) -> npt.ArrayLike:
    centers = np.zeros((elements.shape[0], 2))
    for i in range(elements.shape[0]):
        centers[i][0] = (nodes[elements[i][0]][0] + nodes[elements[i][1]][0] + nodes[elements[i][2]][0] + nodes[elements[i][3]][0])/4
        centers[i][1] = (nodes[elements[i][0]][1] + nodes[elements[i][1]][1] + nodes[elements[i][2]][1] + nodes[elements[i][3]][1])/4
    return centers

def compute_theta(centers: npt.ArrayLike) -> npt.ArrayLike:
    return np.arctan(centers[:, 1], centers[:, 0])

def compute_circular_order(centers: npt.ArrayLike) -> npt.ArrayLike:
    return np.argsort(compute_theta(centers))

def compute_circle_square_mapping(centers: npt.ArrayLike) -> npt.ArrayLike:
    radii = np.sqrt(np.power(centers[:, 0], 2) + np.power(centers[:, 1], 2))
    radius = np.max(radii)
    
    x = centers[:, 0]/radius
    y = centers[:, 1]/radius

    return np.concatenate((np.atleast_2d(np.sqrt(2 + np.power(x, 2) - np.power(y, 2) + x * 2 * np.sqrt(2))/2 - np.sqrt(2 + np.power(x, 2) - np.power(y, 2) - x * 2 * np.sqrt(2))/2).T, np.atleast_2d(np.sqrt(2 - np.power(x, 2) + np.power(y, 2) + y * 2 * np.sqrt(2))/2 - np.sqrt(2 - np.power(x, 2) + np.power(y, 2) - y * 2 * np.sqrt(2))/2).T), axis=1)

def compute_hilbert_circular_order(centers: npt.ArrayLike) -> npt.ArrayLike:
    centers_square = compute_circle_square_mapping(centers)

    return np.argsort(compute_theta(centers_square))

def write_su2(filename: Path, elements: npt.ArrayLike, nodes_lines: npt.ArrayLike, marker_lines: npt.ArrayLike, order: npt.ArrayLike):
    with open(filename, 'w') as file:
        file.write("NDIME= 2\n")
        file.write(f"NELEM= {elements.shape[0]}\n")
        for i in range(elements.shape[0]):
            file.write(f"9 {elements[order[i]][0]} {elements[order[i]][1]} {elements[order[i]][2]} {elements[order[i]][3]} {i}\n")
        file.writelines(nodes_lines)
        file.writelines(marker_lines)

def main(argv: list[str]):
    parser = argparse.ArgumentParser(description="Re-numbers SU2 meshes clockwise.")
    parser.add_argument('-i', '--input', type=Path, help='path to a mesh to re-number')
    parser.add_argument('-o', '--output', type=Path, help='path of the re-numbered mesh')
    parser.add_argument('-a', '--algorithm', type=str, default='circular', choices=['circular', 'hilbert-circular'], help='renumbering algorithm')
    parser.add_argument('-v', '--version', action='version', version='%(prog)s 1.0.0')
    args = parser.parse_args(argv)

    nodes, elements, nodes_lines, marker_lines = read_su2(args.input)
    centers = compute_centers(nodes, elements)
    order = []

    match args.algorithm:
        case "circular":
            order = compute_circular_order(centers)
        case "hilbert-circular":
            order = compute_hilbert_circular_order(centers)
        case _:
            sys.exit(f"Error: unknown algorithm '{args.algorithm}', only 'circular' and 'hilbert-circular' are supported. Exiting.")

    write_su2(args.output, elements, nodes_lines, marker_lines, order)

if __name__ == "__main__":
    main(sys.argv[1:])