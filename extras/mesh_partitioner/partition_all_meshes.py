from pathlib import Path
import subprocess

P_list = [2, 4, 8, 10, 12, 16, 20, 24, 32, 40, 48, 80, 96, 160, 320, 384]
folder = Path('.')

files = [x for x in folder.iterdir() if x.is_file() and x.suffix == ".cgns"]

for file in files:
    for P in P_list:
        new_filename = file.with_stem(file.stem + "_partitioned_N" + str(P))
        subprocess.run(["mesh_partitioner.exe", "--in_path", file, "--out_path", new_filename, "--n", str(P)])