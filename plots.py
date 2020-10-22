# MCG 5136 Assignment 2
# Guillaume Tousignant, 0300151859
# February 3rd, 2020

import matplotlib.pyplot as plt
import numpy as np
import os
import re

# Init
times = []
x_arrays = []
ux_arrays = []

t_finder = re.compile(r"SOLUTIONTIME = [-+]?\d*\.?\d+")
I_finder = re.compile(r"I= \d*")

# Input from all the output_tX.dat files
filenames = [f for f in os.listdir(os.path.join(os.getcwd(), 'data')) if os.path.isfile(os.path.join(os.getcwd(), 'data', f)) and "output_t" in f and f.endswith(".dat")]
for filename in filenames:
    with open(os.path.join(os.getcwd(), 'data', filename), 'r') as file:
        lines = file.readlines()
        t_match = t_finder.search(lines[2])
        times.append(float(t_match.group(0)[15:]))
        N_match = I_finder.search(lines[2])
        N = int(N_match.group(0)[3:])
        x_arrays.append(np.zeros(N))
        ux_arrays.append(np.zeros(N))

        for i in range(N):
            numbers = lines[i+3].split()
            x_arrays[-1][i] = float(numbers[0])
            ux_arrays[-1][i] = float(numbers[1])

# Plotting ux
legend_list = []
fig, ax = plt.subplots(1, 1)
for i in range(len(filenames)):
    ind = np.argsort(x_arrays[i], axis=0)
    ax.plot(np.take_along_axis(x_arrays[i], ind, axis=0), np.take_along_axis(ux_arrays[i], ind, axis=0))
    legend_list.append(f"t = {times[i]} s")

ax.grid()
ax.set_ylabel('$U_x$ [$m/s$]')
ax.set_xlabel('x [m]')
ax.set_title("$U_x$ along x")
ax.legend(legend_list, loc='upper right')

plt.show()