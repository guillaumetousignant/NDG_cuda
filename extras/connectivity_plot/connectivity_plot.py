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


def main(argv):
    inputfile = Path.cwd() / "input.log"

    try:
        opts, args = getopt.getopt(argv,"hi:",["input=","help"])
    except getopt.GetoptError:
        print("connectivity_plot.py -i <inputfile>")
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