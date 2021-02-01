# MCG 5136 Assignment 2
# Guillaume Tousignant, 0300151859
# February 3rd, 2020

import matplotlib.pyplot as plt
import numpy as np
import os
import re

# Init
times = []
times_element = []
x_arrays = []
ux_arrays = []
ux_prime_arrays = []
intermediate_arrays = []
x_element_arrays = []
x_L_arrays = []
x_R_arrays = []
sigma_arrays = []
refine_arrays = []
coarsen_arrays = []
error_arrays = []

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
        ux_prime_arrays.append(np.zeros(N))
        intermediate_arrays.append(np.zeros(N))

        for i in range(N):
            numbers = lines[i+3].split()
            x_arrays[-1][i] = float(numbers[0])
            ux_arrays[-1][i] = float(numbers[1])
            ux_prime_arrays[-1][i] = float(numbers[2])
            intermediate_arrays[-1][i] = float(numbers[3])

# Input from all the output_element_tX.dat files
filenames_element = [f for f in os.listdir(os.path.join(os.getcwd(), 'data')) if os.path.isfile(os.path.join(os.getcwd(), 'data', f)) and "output_element_t" in f and f.endswith(".dat")]
for filename in filenames_element:
    with open(os.path.join(os.getcwd(), 'data', filename), 'r') as file:
        lines = file.readlines()
        t_match = t_finder.search(lines[2])
        times_element.append(float(t_match.group(0)[15:]))
        N_match = I_finder.search(lines[2])
        N = int(N_match.group(0)[3:])
        x_element_arrays.append(np.zeros(N))
        x_L_arrays.append(np.zeros(N))
        x_R_arrays.append(np.zeros(N))
        sigma_arrays.append(np.zeros(N))
        refine_arrays.append(np.zeros(N))
        coarsen_arrays.append(np.zeros(N))
        error_arrays.append(np.zeros(N))

        for i in range(N):
            numbers = lines[i+3].split()
            x_element_arrays[-1][i] = float(numbers[0])
            x_L_arrays[-1][i] = float(numbers[1])
            x_R_arrays[-1][i] = float(numbers[2])
            sigma_arrays[-1][i] = float(numbers[3])
            refine_arrays[-1][i] = float(numbers[4])
            coarsen_arrays[-1][i] = float(numbers[5])
            error_arrays[-1][i] = float(numbers[6])

# Plotting 
N_timesteps = len(filenames)
vline_alpha = 0.2
vline_linestyle = '--'
color_map = plt.get_cmap("rainbow")

ux_fig, ux_ax = plt.subplots(1, 1)
for i in range(N_timesteps):
    ux_ax.plot(x_arrays[i], ux_arrays[i], color=color_map(i/N_timesteps), label=f"t = {times[i]} s")
    for x_L in x_L_arrays[i]:
        ux_ax.axvline(x=x_L, color=color_map(i/N_timesteps), alpha=vline_alpha, linestyle=vline_linestyle)
    ux_ax.axvline(x=x_R_arrays[i][-1], color=color_map(i/N_timesteps), alpha=vline_alpha, linestyle=vline_linestyle)

ux_ax.grid()
ux_ax.set_ylabel('$U_x$ [$m/s$]')
ux_ax.set_xlabel('x [m]')
ux_ax.set_title("$U_x$ along x")
ux_ax.legend(loc='best')

ux_prime_fig, ux_prime_ax = plt.subplots(1, 1)
for i in range(N_timesteps):
    ux_prime_ax.plot(x_arrays[i], ux_prime_arrays[i], color=color_map(i/N_timesteps), label=f"t = {times[i]} s")
    for x_L in x_L_arrays[i]:
        ux_prime_ax.axvline(x=x_L, color=color_map(i/N_timesteps), alpha=vline_alpha, linestyle=vline_linestyle)
    ux_prime_ax.axvline(x=x_R_arrays[i][-1], color=color_map(i/N_timesteps), alpha=vline_alpha, linestyle=vline_linestyle)

ux_prime_ax.grid()
ux_prime_ax.set_ylabel('$U_x,prime$ [$1/s$]')
ux_prime_ax.set_xlabel('x [m]')
ux_prime_ax.set_title("$U_x,prime$ along x")
ux_prime_ax.legend(loc='best')

intermediate_fig, intermediate_ax = plt.subplots(1, 1)
for i in range(N_timesteps):
    intermediate_ax.semilogy(x_arrays[i], intermediate_arrays[i], color=color_map(i/N_timesteps), label=f"t = {times[i]} s")
    for x_L in x_L_arrays[i]:
        intermediate_ax.axvline(x=x_L, color=color_map(i/N_timesteps), alpha=vline_alpha, linestyle=vline_linestyle)
    intermediate_ax.axvline(x=x_R_arrays[i][-1], color=color_map(i/N_timesteps), alpha=vline_alpha, linestyle=vline_linestyle)

intermediate_ax.grid()
intermediate_ax.set_ylabel('Intermediate [?]')
intermediate_ax.set_xlabel('x [m]')
intermediate_ax.set_title("Intermediate along x")
intermediate_ax.legend(loc='best')

sigma_fig, sigma_ax = plt.subplots(1, 1)
for i in range(N_timesteps):
    sigma_ax.plot(x_element_arrays[i], sigma_arrays[i], color=color_map(i/N_timesteps), marker='+', markeredgewidth=2, markersize=16, label=f"t = {times[i]} s", linestyle='')
    for x_L in x_L_arrays[i]:
        sigma_ax.axvline(x=x_L, color=color_map(i/N_timesteps), alpha=vline_alpha, linestyle=vline_linestyle)
    sigma_ax.axvline(x=x_R_arrays[i][-1], color=color_map(i/N_timesteps), alpha=vline_alpha, linestyle=vline_linestyle)

sigma_ax.grid()
sigma_ax.set_ylabel('$sigma$ [?]')
sigma_ax.set_xlabel('x [m]')
sigma_ax.set_title("$sigma$ along x")
sigma_ax.legend(loc='best')

refine_fig, refine_ax = plt.subplots(1, 1)
for i in range(N_timesteps):
    refine_ax.plot(x_element_arrays[i], refine_arrays[i], color=color_map(i/N_timesteps), marker='+', markeredgewidth=2, markersize=16, label=f"t = {times[i]} s", linestyle='')
    for x_L in x_L_arrays[i]:
        refine_ax.axvline(x=x_L, color=color_map(i/N_timesteps), alpha=vline_alpha, linestyle=vline_linestyle)
    refine_ax.axvline(x=x_R_arrays[i][-1], color=color_map(i/N_timesteps), alpha=vline_alpha, linestyle=vline_linestyle)

refine_ax.grid()
refine_ax.set_ylabel('Refine flag [bool]')
refine_ax.set_xlabel('x [m]')
refine_ax.set_title("Refine flag along x")
refine_ax.legend(loc='best')

coarsen_fig, coarsen_ax = plt.subplots(1, 1)
for i in range(N_timesteps):
    coarsen_ax.plot(x_element_arrays[i], coarsen_arrays[i], color=color_map(i/N_timesteps), marker='+', markeredgewidth=2, markersize=16, label=f"t = {times[i]} s", linestyle='')
    for x_L in x_L_arrays[i]:
        coarsen_ax.axvline(x=x_L, color=color_map(i/N_timesteps), alpha=vline_alpha, linestyle=vline_linestyle)
    coarsen_ax.axvline(x=x_R_arrays[i][-1], color=color_map(i/N_timesteps), alpha=vline_alpha, linestyle=vline_linestyle)

coarsen_ax.grid()
coarsen_ax.set_ylabel('Coarsen flag [bool]')
coarsen_ax.set_xlabel('x [m]')
coarsen_ax.set_title("Coarsen flag along x")
coarsen_ax.legend(loc='best')

error_fig, error_ax = plt.subplots(1, 1)
for i in range(N_timesteps):
    error_ax.semilogy(x_element_arrays[i], error_arrays[i], color=color_map(i/N_timesteps), marker='+', markeredgewidth=2, markersize=16, label=f"t = {times[i]} s", linestyle='')
    for x_L in x_L_arrays[i]:
        error_ax.axvline(x=x_L, color=color_map(i/N_timesteps), alpha=vline_alpha, linestyle=vline_linestyle)
    error_ax.axvline(x=x_R_arrays[i][-1], color=color_map(i/N_timesteps), alpha=vline_alpha, linestyle=vline_linestyle)

error_ax.grid()
error_ax.set_ylabel('Error estimation [?]')
error_ax.set_xlabel('x [m]')
error_ax.set_title("Error estimation along x")
error_ax.legend(loc='best')

plt.show()