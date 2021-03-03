# MCG 5136 Assignment 2
# Guillaume Tousignant, 0300151859
# February 3rd, 2020

import matplotlib.pyplot as plt
import numpy as np
import re
from pathlib import Path

t_finder = re.compile(r"SOLUTIONTIME = [+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?")
I_finder = re.compile(r"I= \d*")

class Solution:
    def __init__(self, x, ux, ux_prime, intermediate):
        self.x = x
        self.ux = ux
        self.ux_prime = ux_prime
        self.intermediate = intermediate

class Solution_element:
    def __init__(self, x, x_L, x_R, N, sigma, refine, coarsen, error):
        self.x = x
        self.x_L = x_L
        self.x_R = x_R
        self.N = N
        self.sigma = sigma
        self.refine = refine
        self.coarsen = coarsen
        self.error = error

def read_file(filename: Path, timesteps: dict) -> dict:
    with open(filename, 'r') as file:
        lines = file.readlines()
        t_match = t_finder.search(lines[2])
        time = float(t_match.group(0)[15:])
        N_match = I_finder.search(lines[2])
        N = int(N_match.group(0)[3:])

        x = []
        ux = []
        ux_prime = []
        intermediate = []

        index = 2
        while index < len(lines):
            N_match = I_finder.search(lines[index])
            N = int(N_match.group(0)[3:])

            x.append(np.zeros(N))
            ux.append(np.zeros(N))
            ux_prime.append(np.zeros(N))
            intermediate.append(np.zeros(N))

            for i in range(N):
                numbers = lines[i + index + 1].split()
                x[-1][i] = float(numbers[0])
                ux[-1][i] = float(numbers[1])
                ux_prime[-1][i] = float(numbers[2])
                intermediate[-1][i] = float(numbers[3])
            
            index = index + N + 1

        if time in timesteps:
            timesteps[time].x.extend(x)
            timesteps[time].ux.extend(ux)
            timesteps[time].ux_prime.extend(ux_prime)
            timesteps[time].intermediate.extend(intermediate)
        else:
            timesteps[time] = Solution(x, ux, ux_prime, intermediate)
    
    return timesteps

def read_element_file(filename: Path, timesteps: dict) -> dict:
    with open(filename, 'r') as file:
        lines = file.readlines()
        t_match = t_finder.search(lines[2])
        time = float(t_match.group(0)[15:])
        N_match = I_finder.search(lines[2])
        N = int(N_match.group(0)[3:])

        x = np.zeros(N)
        x_L = np.zeros(N)
        x_R = np.zeros(N)
        N_array = np.zeros(N)
        sigma = np.zeros(N)
        refine = np.zeros(N)
        coarsen = np.zeros(N)
        error = np.zeros(N)

        for i in range(N):
            numbers = lines[i+3].split()
            x[i] = float(numbers[0])
            x_L[i] = float(numbers[1])
            x_R[i] = float(numbers[2])
            N_array[i] = float(numbers[3])
            sigma[i] = float(numbers[4])
            refine[i] = float(numbers[5])
            coarsen[i] = float(numbers[6])
            error[i] = float(numbers[7])

        if time in timesteps:
            timesteps[time].x = np.append(timesteps[time].x, x)
            timesteps[time].x_L = np.append(timesteps[time].x_L, x_L)
            timesteps[time].x_R = np.append(timesteps[time].x_R, x_R)
            timesteps[time].N = np.append(timesteps[time].N, N_array)
            timesteps[time].sigma = np.append(timesteps[time].sigma, sigma)
            timesteps[time].refine = np.append(timesteps[time].refine, refine)
            timesteps[time].coarsen = np.append(timesteps[time].coarsen, coarsen)
            timesteps[time].error = np.append(timesteps[time].error, error)
        else:
            timesteps[time] = Solution_element(x, x_L, x_R, N_array, sigma, refine, coarsen, error)
    
    return timesteps

timesteps = dict()
timesteps_element = dict()

# Input from all the output_tX.dat files
data_path = Path.cwd() / "data"
for filename in data_path.glob("output_t*.dat"):
    timesteps = read_file(filename, timesteps)

# Input from all the output_element_tX.dat files
for filename in data_path.glob("output_element_t*.dat"):
    timesteps_element = read_element_file(filename, timesteps_element)

# Plotting 
times = sorted(timesteps.keys())
times_element = sorted(timesteps_element.keys())

vline_alpha = 0.2
vline_linestyle = '--'
color_map = plt.get_cmap("rainbow")

ux_fig, ux_ax = plt.subplots(1, 1)
ux_prime_fig, ux_prime_ax = plt.subplots(1, 1)
intermediate_fig, intermediate_ax = plt.subplots(1, 1)

for time in times:
    timestep = timesteps[time]
    normalised_time = (time - times[0])/(times[-1] - times[0])

    ux_ax.plot(timestep.x[0], timestep.ux[0], color=color_map(normalised_time), label=f"t = {time} s")
    ux_prime_ax.plot(timestep.x[0], timestep.ux_prime[0], color=color_map(normalised_time), label=f"t = {time} s")
    intermediate_ax.semilogy(timestep.x[0], timestep.intermediate[0], color=color_map(normalised_time), label=f"t = {time} s")

    for j in range(len(timestep.x) - 1):
        ux_ax.plot(timestep.x[j+1], timestep.ux[j+1], color=color_map(normalised_time))
        ux_prime_ax.plot(timestep.x[j+1], timestep.ux_prime[j+1], color=color_map(normalised_time))
        intermediate_ax.semilogy(timestep.x[j+1], timestep.intermediate[j+1], color=color_map(normalised_time))

    if time in timesteps_element:
        for x_L in timesteps_element[time].x_L:
            ux_ax.axvline(x=x_L, color=color_map(normalised_time), alpha=vline_alpha, linestyle=vline_linestyle)
            ux_prime_ax.axvline(x=x_L, color=color_map(normalised_time), alpha=vline_alpha, linestyle=vline_linestyle)
            intermediate_ax.axvline(x=x_L, color=color_map(normalised_time), alpha=vline_alpha, linestyle=vline_linestyle)
        ux_ax.axvline(x=timesteps_element[time].x_R[-1], color=color_map(normalised_time), alpha=vline_alpha, linestyle=vline_linestyle)
        ux_prime_ax.axvline(x=timesteps_element[time].x_R[-1], color=color_map(normalised_time), alpha=vline_alpha, linestyle=vline_linestyle)
        intermediate_ax.axvline(x=timesteps_element[time].x_R[-1], color=color_map(normalised_time), alpha=vline_alpha, linestyle=vline_linestyle)

ux_ax.grid()
ux_ax.set_ylabel('$U_x$ [$m/s$]')
ux_ax.set_xlabel('x [m]')
ux_ax.set_title("$U_x$ along x")
ux_ax.legend(loc='best')

ux_prime_ax.grid()
ux_prime_ax.set_ylabel('$U_x,prime$ [$1/s$]')
ux_prime_ax.set_xlabel('x [m]')
ux_prime_ax.set_title("$U_x,prime$ along x")
ux_prime_ax.legend(loc='best')

intermediate_ax.grid()
intermediate_ax.set_ylabel('Intermediate [?]')
intermediate_ax.set_xlabel('x [m]')
intermediate_ax.set_title("Intermediate along x")
intermediate_ax.legend(loc='best')

N_fig, N_ax = plt.subplots(1, 1)
sigma_fig, sigma_ax = plt.subplots(1, 1)
refine_fig, refine_ax = plt.subplots(1, 1)
coarsen_fig, coarsen_ax = plt.subplots(1, 1)
error_fig, error_ax = plt.subplots(1, 1)

for time in times_element:
    timestep = timesteps_element[time]
    normalised_time = (time - times_element[0])/(times_element[-1] - times_element[0])

    N_ax.plot(timestep.x, timestep.N, color=color_map(normalised_time), marker='+', markeredgewidth=2, markersize=16, label=f"t = {time} s", linestyle='')
    sigma_ax.plot(timestep.x, timestep.sigma, color=color_map(normalised_time), marker='+', markeredgewidth=2, markersize=16, label=f"t = {time} s", linestyle='')
    refine_ax.plot(timestep.x, timestep.refine, color=color_map(normalised_time), marker='+', markeredgewidth=2, markersize=16, label=f"t = {time} s", linestyle='')
    coarsen_ax.plot(timestep.x, timestep.coarsen, color=color_map(normalised_time), marker='+', markeredgewidth=2, markersize=16, label=f"t = {time} s", linestyle='')
    error_ax.semilogy(timestep.x, timestep.error, color=color_map(normalised_time), marker='+', markeredgewidth=2, markersize=16, label=f"t = {time} s", linestyle='')
    
    for x_L in timestep.x_L:
        N_ax.axvline(x=x_L, color=color_map(normalised_time), alpha=vline_alpha, linestyle=vline_linestyle)
        sigma_ax.axvline(x=x_L, color=color_map(normalised_time), alpha=vline_alpha, linestyle=vline_linestyle)
        refine_ax.axvline(x=x_L, color=color_map(normalised_time), alpha=vline_alpha, linestyle=vline_linestyle)
        coarsen_ax.axvline(x=x_L, color=color_map(normalised_time), alpha=vline_alpha, linestyle=vline_linestyle)
        error_ax.axvline(x=x_L, color=color_map(normalised_time), alpha=vline_alpha, linestyle=vline_linestyle)
    N_ax.axvline(x=timestep.x_R[-1], color=color_map(normalised_time), alpha=vline_alpha, linestyle=vline_linestyle)
    sigma_ax.axvline(x=timestep.x_R[-1], color=color_map(normalised_time), alpha=vline_alpha, linestyle=vline_linestyle)
    refine_ax.axvline(x=timestep.x_R[-1], color=color_map(normalised_time), alpha=vline_alpha, linestyle=vline_linestyle)
    coarsen_ax.axvline(x=timestep.x_R[-1], color=color_map(normalised_time), alpha=vline_alpha, linestyle=vline_linestyle)
    error_ax.axvline(x=timestep.x_R[-1], color=color_map(normalised_time), alpha=vline_alpha, linestyle=vline_linestyle)

N_ax.grid()
N_ax.set_ylabel('N [-]')
N_ax.set_xlabel('x [m]')
N_ax.set_title("N along x")
N_ax.legend(loc='best')

sigma_ax.grid()
sigma_ax.set_ylabel('$sigma$ [?]')
sigma_ax.set_xlabel('x [m]')
sigma_ax.set_title("$sigma$ along x")
sigma_ax.legend(loc='best')

refine_ax.grid()
refine_ax.set_ylabel('Refine flag [bool]')
refine_ax.set_xlabel('x [m]')
refine_ax.set_title("Refine flag along x")
refine_ax.legend(loc='best')

coarsen_ax.grid()
coarsen_ax.set_ylabel('Coarsen flag [bool]')
coarsen_ax.set_xlabel('x [m]')
coarsen_ax.set_title("Coarsen flag along x")
coarsen_ax.legend(loc='best')

error_ax.grid()
error_ax.set_ylabel('Error estimation [?]')
error_ax.set_xlabel('x [m]')
error_ax.set_title("Error estimation along x")
error_ax.legend(loc='best')

plt.show()