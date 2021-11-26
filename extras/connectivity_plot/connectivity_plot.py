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
        node_x_finder = re.compile(r" :      [[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?")
        node_y_finder = re.compile(r", [[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?")

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

        line_index = 15

        for i in range(n_nodes):
            line = lines[line_index + i]
            words = line.split()

            nodes_x[i] = float(words[3][1:-1])
            nodes_y[i] = float(words[4][0:-1])

    



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