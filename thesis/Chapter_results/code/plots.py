import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

# Data
# Strong scaling
N = np.array([4, 6, 6, 6])
K = np.array([256 * 256, 256 * 256, 256 * 256, 256 * 256])
W = np.array([32, 32, 64, 128])

times_beluga_cpu = np.array([[1479.98, 1048.17, 569.601, 295.771, 146.801, 95.4677],
                             [5355.63, 3527.07, 1885.91, 1040.37, 513.924, 330.297],
                             [5355.63, 3527.07, 1885.91, 1040.37, 513.924, 330.297],
                             [5355.63, 3527.07, 1885.91, 1040.37, 513.924, 330.297]])
nodes_beluga_cpu = np.array([[0.25, 0.5, 1, 2, 4, 8],
                             [0.25, 0.5, 1, 2, 4, 8],
                             [0.25, 0.5, 1, 2, 4, 8],
                             [0.25, 0.5, 1, 2, 4, 8]])
total_time_beluga_cpu = times_beluga_cpu * nodes_beluga_cpu
min_total_time_beluga_cpu = np.amin(total_time_beluga_cpu, 1)

times_beluga_gpu = np.array([[4036.09, 1536.88, 492.492, 161.317, 60.6248, 32.8288],
                             [18194.3, 6660.81, 2186.03, 677.623, 256.302, 130.453],
                             [18345.7, 6666.96, 2247.11, 682.221, 264.406, 192.805],
                             [17968.7, 6666.6666666, 2146.23, 726.463, 362.465, 248.342]])
nodes_beluga_gpu = np.array([[0.25, 0.5, 1, 2, 4, 8],
                             [0.25, 0.5, 1, 2, 4, 8],
                             [0.25, 0.5, 1, 2, 4, 8],
                             [0.25, 0.5, 1, 2, 4, 8]])
total_time_beluga_gpu = times_beluga_gpu * nodes_beluga_gpu
min_total_time_beluga_gpu = np.amin(total_time_beluga_gpu, 1)

nodes_beluga_cpu_ideal = nodes_beluga_cpu[:, [0,-1]]
nodes_beluga_gpu_ideal = nodes_beluga_gpu[:, [0,-1]]

times_beluga_cpu_ideal = np.stack((min_total_time_beluga_cpu/nodes_beluga_cpu_ideal[:, 0], min_total_time_beluga_cpu/nodes_beluga_cpu_ideal[:, -1]), axis=-1)
times_beluga_gpu_ideal = np.stack((min_total_time_beluga_gpu/nodes_beluga_gpu_ideal[:, 0], min_total_time_beluga_gpu/nodes_beluga_gpu_ideal[:, -1]), axis=-1)

node_strings_beluga_gpu = np.array([["¼", "½", "1", "2", "4", "8"],
                                     ["¼", "½", "1", "2", "4", "8"],
                                     ["¼", "½", "1", "2", "4", "8"],
                                     ["¼", "½", "1", "2", "4", "8"]])

# Weak scaling
N_weak = np.array([4])
K_weak = np.array([64 * 64])
W_weak = np.array([32])

times_beluga_cpu_weak = np.array([[125.489, 248.753, 300.733, 373.963]])
nodes_beluga_cpu_weak = np.array([[0.25, 1, 4, 16]])
min_total_time_beluga_cpu_weak = np.amin(times_beluga_cpu_weak, 1)

times_beluga_gpu_weak = np.array([[113.873, 130.662, 132.527, 129.103]])
nodes_beluga_gpu_weak = np.array([[0.25, 1, 4, 16]])
min_total_time_beluga_gpu_weak = np.amin(times_beluga_gpu_weak, 1)

nodes_beluga_cpu_ideal_weak = nodes_beluga_cpu_weak[:, [0,-1]]
nodes_beluga_gpu_ideal_weak = nodes_beluga_gpu_weak[:, [0,-1]]

node_strings_beluga_gpu_weak = np.array([["¼", "1", "4", "16"]])

# Adaptivity efficiency
# Iterative results are from Narval, sequential results are from desktop
adaptivity_interval = np.array([5, 20, 100, 500])
adaptivity_N = np.array([4, 4, 4, 4])
adaptivity_K = np.array([4 * 4, 4 * 4, 4 * 4, 4 * 4])
adaptivity_s = np.array([5, 5, 5, 5])
adaptivity_C = np.array([[0, 1, 2, 3, 4, 5],
                         [0, 1, 2, 3, 4, 5],
                         [0, 1, 2, 3, 4, 5],
                         [0, 1, 2, 3, 4, 5]])
adaptivity_solving_t_iterative =  np.array([[568.537, 50.2221, 27.0194, 24.8199, 24.9935, 24.8425],
                                            [361.345, 87.5917, 39.5757, 51.7299, 51.6251, 51.6749],
                                            [0.414409, 39.018, 34.6826, 15.8131, 15.9021, 15.7603],
                                            [0.152319, 0.467167, 1.56873, 9.2288, 13.8472, 18.0576]])
adaptivity_solving_t_sequential = np.array([[756.998, 78.5987, 376.426, 71.308, 120.742, 151.535],
                                            [402.506, 158.75, 121.156, 168.402, 72.085, 114.967],
                                            [0.599613, 85.0527, 33.4027, 54.1339, 87.9368, 147.392],
                                            [0.358333, 1.17821, 1.14392, 1.14803, 1.14488, 1.13517]])
adaptivity_condition_t_iterative =  np.array([[0, 0.0825008, 0.19362, 0.212391, 0.229018, 0.24586],
                                              [0, 0.131457, 0.317237, 0.416562, 0.502704, 0.574876],
                                              [0, 0.309875, 0.597094, 1.01954, 1.55069, 2.07073],
                                              [0, 0.583054, 1.8537, 3.28107, 5.33752, 8.6036]])
adaptivity_condition_t_sequential = np.array([[0, 0.109984, 0.246979, 0.573679, 1.01607, 1.61849],
                                              [0, 0.129859, 0.405687, 0.833624, 1.85428, 3.86756],
                                              [0, 0.376936, 0.972618, 2.71825, 6.66742, 13.7539],
                                              [0, 1.23078, 3.31111, 5.39603, 7.4745, 9.56906]])
adaptivity_t_iterative = adaptivity_solving_t_iterative + adaptivity_condition_t_iterative
adaptivity_t_sequential = adaptivity_solving_t_sequential + adaptivity_condition_t_sequential

adaptivity_max_error_iterative = np.array([[2.9e-5, 1.3e-6, 7.1e-7, 6.7e-7, 6.7e-7, 6.7e-7],
                                           [1.2e-4, 2.7e-6, 3.5e-7, 9.5e-8, 9.5e-8, 9.5e-8],
                                           [9.3e-3, 5.8e-5, 2.9e-6, 4.6e-7, 4.6e-7, 4.6e-7],
                                           [1.2e-2, 2.1e-3, 4.9e-5, 8.1e-5, 7.0e-7, 8.1e-7]])
adaptivity_max_error_sequential = np.array([[2.9e-5, 1.3e-6, 4.8e-7, 5.9e-7, 6.5e-8, 6.5e-8],
                                            [1.2e-4, 2.7e-6, 2.9e-7, 7.9e-8, 7.6e-8, 6.3e-8],
                                            [9.3e-3, 5.8e-5, 3.1e-7, 2.6e-7, 5.7e-7, 6.4e-8],
                                            [1.2e-2, 2.1e-3, 2.1e-3, 2.1e-3, 2.3e-3, 2.1e-3]])

adaptivity_baseline_C = adaptivity_C[:, [0,-1]]
adaptivity_baseline_t = np.array([1136.14, 1136.14, 1136.14, 1136.14])
adaptivity_baseline_max_error = np.array([5.1e-9, 5.1e-9, 5.1e-9, 5.1e-9])
adaptivity_same_error_t = np.array([48.3641, 48.3641, 48.3641, 48.3641])
adaptivity_same_error_max_error = np.array([8.3e-8, 8.3e-8, 8.3e-8, 8.3e-8])
best_pre_condition = 4

# Load balancing efficiency interval
load_balancing_interval_A = np.array([20, 20, 20])                      # Adaptivity interval
load_balancing_interval_N = np.array([4, 4, 4])                         # Initial N
load_balancing_interval_K = np.array([128 * 128, 128 * 128, 128 * 128]) # Initial number of elements
load_balancing_interval_P = np.array([16, 16, 16])                      # Number of processes
load_balancing_interval_S = np.array([3, 5, 7])                         # Max split level
load_balancing_interval_max_N = np.array([12, 12, 12])                  # Max N

load_balancing_interval_L = np.array([[20, 100, 200, 500, 1000],
                                      [20, 100, 200, 500, 1000],
                                      [20, 100, 200, 500, 1000]]) # Load balancing interval
load_balancing_interval_t = np.array([[346.838, 284.198, 266.794, 248.574, 254.721],
                                      [2928.21, 2107.16, 1989.26, 1966.79, 2112.57],
                                      [20480.5, 26116.9, 24765.6, 13990.2, 24976.3]]) # Simulation time

load_balancing_interval_baseline_L = load_balancing_interval_L[:, [0,-1]]
load_balancing_interval_baseline_t = np.array([274.415, 4962.06, 107196])

# Load balancing efficiency threshold
load_balancing_threshold_A = np.array([20, 20, 20])                         # Adaptivity interval
load_balancing_threshold_L = np.array([20, 20, 20])                         # Load balancing interval
load_balancing_threshold_N = np.array([4, 4, 4])                            # Initial N
load_balancing_threshold_K = np.array([128 * 128, 128 * 128, 128 * 128])    # Initial number of elements
load_balancing_threshold_P = np.array([16, 16, 16])                         # Number of processes
load_balancing_threshold_S = np.array([3, 5, 7])                            # Max split level
load_balancing_threshold_I = np.array([2.16, 6.81, 12.07])                  # Load imbalance
load_balancing_threshold_max_N = np.array([12, 12, 12])                     # Max N

load_balancing_threshold_T = np.array([[1, 1.01, 1.1, 1.5, 2],
                                       [1, 1.01, 1.1, 1.5, 2],
                                       [1, 1.01, 1.1, 1.5, 2]]) # Load balancing threshold
load_balancing_threshold_t = np.array([[344.277, 295.591, 243.566, 241.076, 252.407],
                                       [2925.25, 2248.53, 1869.24, 1865.98, 2043.47],
                                       [20480.5, 14096.5, 13855.5, 14416.9, 15507.3]]) # Simulation time

load_balancing_threshold_baseline_T = load_balancing_threshold_T[:, [0,-1]]
load_balancing_threshold_baseline_t = np.array([274.359, 4962.06, 59061.9])
best_T = 2

# N scaling
N_scaling_N_cpu = np.array([4, 6, 8, 10, 12, 14, 16])
N_scaling_N_gpu = np.array([4, 6, 8, 10, 12, 14, 16])
N_scaling_t_cpu = np.array([40.1137, 71.5033, 108.358, 192.504, 257.458, 349.042, 490.11])
N_scaling_t_gpu = np.array([2.14944, 4.13552, 8.07533, 12.7138, 18.2212, 26.0583, 33.3919])

# Plots
save_path = Path(__file__).parent.parent / "media"
save_path.mkdir(parents=True, exist_ok=True)

data_width = 3
data_width_small = 2
ideal_width = 3
cpu_colour = np.array([244, 71, 71])/255
gpu_colour = np.array([79, 193, 255])/255
gpu_colour_dark = np.array([3, 125, 190])/255
error_colour = np.array([197, 134, 192])/255
error_colour_dark = np.array([168, 80, 161])/255
solving_colour = np.array([215, 186, 125])/255
pre_condition_colour = np.array([181, 206, 168])/255
N_colour_cpu = np.array([209, 105, 105])/255
N_colour_gpu = np.array([86, 156, 214])/255
imbalance_colour = np.array([78, 201, 176])/255
data_size = 12
data_size_small = 8
data_shape = "o"
solving_shape = "X"
pre_condition_shape = ">"
ideal_style = "--"
ideal_style2 = "-."
same_error_style = ":"
adaptivity_style = ""

# Strong scaling
for i in range(N.shape[0]):
    fig = plt.figure(figsize=(5, 4.5))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel("Number of nodes [-]")
    ax.set_ylabel("Time [s]")
    title = f"Strong scaling, N = {N[i]} K = {K[i]} W = {W[i]}"
    fig.canvas.manager.set_window_title(title)
    ax.grid()

    ax.loglog(nodes_beluga_cpu[i, :], times_beluga_cpu[i, :], color=cpu_colour, linewidth=data_width, marker=data_shape, markersize=data_size, label="CPU time")
    ax.loglog(nodes_beluga_cpu_ideal[i, :], times_beluga_cpu_ideal[i, :], color=cpu_colour, linewidth=data_width, linestyle=ideal_style, label="CPU ideal time")
    
    ax.loglog(nodes_beluga_gpu[i, :], times_beluga_gpu[i, :], color=gpu_colour, linewidth=data_width, marker=data_shape, markersize=data_size, label="GPU time")
    ax.loglog(nodes_beluga_gpu_ideal[i, :], times_beluga_gpu_ideal[i, :], color=gpu_colour, linewidth=data_width, linestyle=ideal_style, label="GPU ideal time")

    if i == 0:
        time = times_beluga_gpu[i, 0]*nodes_beluga_gpu[i, 0]
        times_beluga_gpu_ideal2 = time/nodes_beluga_gpu_ideal

        ax.loglog(nodes_beluga_gpu_ideal[i, :], times_beluga_gpu_ideal2[i, :], color=gpu_colour, linewidth=data_width, linestyle=ideal_style2, label="GPU ideal time first point")

    ax.set_xticks(nodes_beluga_gpu[i, :], node_strings_beluga_gpu[i, :])

    ax.legend()
    fig.tight_layout()

    fig.savefig(save_path / f"strong_scaling_N{N[i]}_K{K[i]}_W{W[i]}.svg", format='svg', transparent=True)

# Weak scaling
for i in range(N_weak.shape[0]):
    fig = plt.figure(figsize=(5, 4.5))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel("Number of nodes [-]")
    ax.set_ylabel("Time [s]")
    title = f"Weak scaling, N = {N_weak[i]} K = {K_weak[i]}/proc W = {W_weak[i]}"
    fig.canvas.manager.set_window_title(title)
    ax.grid()

    ax.semilogx(nodes_beluga_cpu_weak[i, :], times_beluga_cpu_weak[i, :], color=cpu_colour, linewidth=data_width, marker=data_shape, markersize=data_size, label="CPU time")
    ax.semilogx(nodes_beluga_cpu_ideal_weak[i, :], [min_total_time_beluga_cpu_weak[i], min_total_time_beluga_cpu_weak[i]], color=cpu_colour, linewidth=data_width, linestyle=ideal_style, label="CPU ideal time")
    
    ax.semilogx(nodes_beluga_gpu_weak[i, :], times_beluga_gpu_weak[i, :], color=gpu_colour, linewidth=data_width, marker=data_shape, markersize=data_size, label="GPU time")
    ax.semilogx(nodes_beluga_gpu_ideal_weak[i, :], [min_total_time_beluga_gpu_weak[i], min_total_time_beluga_gpu_weak[i]], color=gpu_colour, linewidth=data_width, linestyle=ideal_style, label="GPU ideal time")

    ax.set_ylim([0, 1.2 * max(max(times_beluga_cpu_weak[i, :]), max(times_beluga_gpu_weak[i, :]))])

    ax.set_xticks(nodes_beluga_gpu_weak[i, :], node_strings_beluga_gpu_weak[i, :])

    ax.legend()
    fig.tight_layout()

    fig.savefig(save_path / f"weak_scaling_N{N_weak[i]}_K{K_weak[i]}_W{W_weak[i]}.svg", format='svg', transparent=True)

# Adaptivity efficiency
for i in range(adaptivity_interval.shape[0]):
    fig = plt.figure(figsize=(5.5, 4.5))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel("Number of pre-condition refinement steps [-]")
    ax.set_ylabel("Time [s]")
    title = f"Adaptivity performance time, N = {adaptivity_N[i]} K = {adaptivity_K[i]} A = {adaptivity_interval[i]}"
    fig.canvas.manager.set_window_title(title)
    ax.grid()

    ax.semilogy(adaptivity_C[i, :], adaptivity_solving_t_iterative[i, :], color=solving_colour, linewidth=data_width, marker=solving_shape, markersize=data_size, linestyle=adaptivity_style, label="Solving time")
    ax.semilogy(adaptivity_C[i, 1:], adaptivity_condition_t_iterative[i, 1:], color=pre_condition_colour, linewidth=data_width, marker=pre_condition_shape, markersize=data_size, linestyle=adaptivity_style, label="Pre-condition time")
    ax.semilogy(adaptivity_C[i, :], adaptivity_t_iterative[i, :], color=gpu_colour, linewidth=data_width_small, marker=data_shape, markersize=data_size_small, linestyle=adaptivity_style, label="Total time")
    ax.semilogy(adaptivity_baseline_C[i, :], [adaptivity_baseline_t[i], adaptivity_baseline_t[i]], color=gpu_colour, linewidth=data_width, linestyle=ideal_style, label="Non-adaptive fully refined total time")
    ax.semilogy(adaptivity_baseline_C[i, :], [adaptivity_same_error_t[i], adaptivity_same_error_t[i]], color=gpu_colour, linewidth=data_width, linestyle=same_error_style, label="Non-adaptive similar error total time")

    ax.legend()
    fig.tight_layout()

    fig.savefig(save_path / f"adaptivity_time_N{adaptivity_N[i]}_K{adaptivity_K[i]}_A{adaptivity_interval[i]}.svg", format='svg', transparent=True)

    error_fig = plt.figure(figsize=(5.5, 4.5))
    error_ax = error_fig.add_subplot(1, 1, 1)
    error_ax.set_xlabel("Number of pre-condition refinement steps [-]")
    error_ax.set_ylabel("Analytical solution error [-]")
    error_title = f"Adaptivity performance error, N = {adaptivity_N[i]} K = {adaptivity_K[i]} A = {adaptivity_interval[i]}"
    error_fig.canvas.manager.set_window_title(error_title)
    error_ax.grid()

    # For the legend entries from the other axes
    error_ax.semilogy(adaptivity_C[i, :], adaptivity_max_error_iterative[i, :], color=error_colour, linewidth=data_width, marker=data_shape, markersize=data_size, linestyle=adaptivity_style, label="Max error")
    error_ax.semilogy(adaptivity_baseline_C[i, :], [adaptivity_baseline_max_error[i], adaptivity_baseline_max_error[i]], color=error_colour, linewidth=data_width, linestyle=ideal_style, label="Non-adaptive fully refined max error")
    error_ax.semilogy(adaptivity_baseline_C[i, :], [adaptivity_same_error_max_error[i], adaptivity_same_error_max_error[i]], color=error_colour, linewidth=data_width, linestyle=same_error_style, label="Non-adaptive similar error max error")

    error_ax.legend()
    error_fig.tight_layout()

    error_fig.savefig(save_path / f"adaptivity_error_N{adaptivity_N[i]}_K{adaptivity_K[i]}_A{adaptivity_interval[i]}.svg", format='svg', transparent=True)

amr_fig = plt.figure(figsize=(5.5, 4.5))
amr_ax = amr_fig.add_subplot(1, 1, 1)
amr_ax.set_xlabel("Refinement interval [-]")
amr_ax.set_ylabel("Time [s]")
amr_title = f"Adaptivity performance time, N = {adaptivity_N[0]} K = {adaptivity_K[0]} C = {adaptivity_C[0][best_pre_condition]}"
amr_fig.canvas.manager.set_window_title(amr_title)
amr_ax.grid()

amr_ax.loglog(adaptivity_interval, adaptivity_solving_t_iterative[:, best_pre_condition], color=solving_colour, linewidth=data_width, marker=solving_shape, markersize=data_size, linestyle=adaptivity_style, label="Solving time")
amr_ax.loglog(adaptivity_interval, adaptivity_condition_t_iterative[:, best_pre_condition], color=pre_condition_colour, linewidth=data_width, marker=pre_condition_shape, markersize=data_size, linestyle=adaptivity_style, label="Pre-condition time")
amr_ax.loglog(adaptivity_interval, adaptivity_t_iterative[:, best_pre_condition], color=gpu_colour, linewidth=data_width_small, marker=data_shape, markersize=data_size_small, linestyle=adaptivity_style, label="Total time")
amr_ax.loglog(adaptivity_interval, adaptivity_baseline_t, color=gpu_colour, linewidth=data_width, linestyle=ideal_style, label="Non-adaptive fully refined total time")
amr_ax.loglog(adaptivity_interval, adaptivity_same_error_t, color=gpu_colour, linewidth=data_width, linestyle=same_error_style, label="Non-adaptive similar error total time")

amr_ax.set_xticks(adaptivity_interval, adaptivity_interval.astype(str))
amr_ax.legend()
amr_fig.tight_layout()

amr_fig.savefig(save_path / f"adaptivity_time_N{adaptivity_N[0]}_K{adaptivity_K[0]}_C{adaptivity_C[0][best_pre_condition]}.svg", format='svg', transparent=True)

amr_error_fig = plt.figure(figsize=(5.5, 4.5))
amr_error_ax = amr_error_fig.add_subplot(1, 1, 1)
amr_error_ax.set_xlabel("Refinement interval [-]")
amr_error_ax.set_ylabel("Analytical solution error [-]")
amr_error_title = f"Adaptivity performance error, N = {adaptivity_N[0]} K = {adaptivity_K[0]} C = {adaptivity_C[0][best_pre_condition]}"
amr_error_fig.canvas.manager.set_window_title(amr_error_title)
amr_error_ax.grid()

# For the legend entries from the other axes
amr_error_ax.loglog(adaptivity_interval, adaptivity_max_error_iterative[:, best_pre_condition], color=error_colour, linewidth=data_width, marker=data_shape, markersize=data_size, linestyle=adaptivity_style, label="Max error")
amr_error_ax.loglog(adaptivity_interval, adaptivity_baseline_max_error, color=error_colour, linewidth=data_width, linestyle=ideal_style, label="Non-adaptive fully refined max error")
amr_error_ax.loglog(adaptivity_interval, adaptivity_same_error_max_error, color=error_colour, linewidth=data_width, linestyle=same_error_style, label="Non-adaptive similar error max error")

amr_error_ax.set_xticks(adaptivity_interval, adaptivity_interval.astype(str))
amr_error_ax.legend()
amr_error_fig.tight_layout()

amr_error_fig.savefig(save_path / f"adaptivity_error_N{adaptivity_N[0]}_K{adaptivity_K[0]}_C{adaptivity_C[0][best_pre_condition]}.svg", format='svg', transparent=True)

# Load balancing efficiency interval
for i in range(load_balancing_interval_N.shape[0]):
    fig = plt.figure(figsize=(5, 4.5))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel("Load balancing interval [-]")
    ax.set_ylabel("Time [s]")
    title = f"Load balancing performance, interval test, N = {load_balancing_interval_N[i]} K = {load_balancing_interval_K[i]} A = {load_balancing_interval_A[i]} P = {load_balancing_interval_P[i]} S = {load_balancing_interval_S[i]} N_max = {load_balancing_interval_max_N[i]}"
    fig.canvas.manager.set_window_title(title)
    ax.grid()

    ax.plot(load_balancing_interval_L[i, :], load_balancing_interval_t[i, :], color=gpu_colour, linewidth=data_width, marker=data_shape, markersize=data_size, label="Time with load balancing")
    ax.plot(load_balancing_interval_baseline_L[i, :], [load_balancing_interval_baseline_t[i], load_balancing_interval_baseline_t[i]], color=gpu_colour, linewidth=data_width, linestyle=ideal_style, label="Time without load balancing")

    ax.set_ylim([0, 1.2 * max(max(load_balancing_interval_t[i, :]), load_balancing_interval_baseline_t[i])])

    ax.legend()
    fig.tight_layout()

    fig.savefig(save_path / f"load_balancing_interval_N{load_balancing_interval_N[i]}_K{load_balancing_interval_K[i]}_A{load_balancing_interval_A[i]}_P{load_balancing_interval_P[i]}_S{load_balancing_interval_S[i]}.svg", format='svg', transparent=True)

# Load balancing efficiency threshold
for i in range(load_balancing_threshold_N.shape[0]):
    fig = plt.figure(figsize=(5, 4.5))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel("Load balancing threshold [-]")
    ax.set_ylabel("Time [s]")
    title = f"Load balancing performance, threshold test, N = {load_balancing_threshold_N[i]} K = {load_balancing_threshold_K[i]} A = {load_balancing_threshold_A[i]} L = {load_balancing_threshold_L[i]} P = {load_balancing_threshold_P[i]} S = {load_balancing_threshold_S[i]} N_max = {load_balancing_threshold_max_N[i]}"
    fig.canvas.manager.set_window_title(title)
    ax.grid()

    ax.plot(load_balancing_threshold_T[i, :], load_balancing_threshold_t[i, :], color=gpu_colour, linewidth=data_width, marker=data_shape, markersize=data_size, label="Time with load balancing")
    ax.plot(load_balancing_threshold_baseline_T[i, :], [load_balancing_threshold_baseline_t[i], load_balancing_threshold_baseline_t[i]], color=gpu_colour, linewidth=data_width, linestyle=ideal_style, label="Time without load balancing")

    ax.set_ylim([0, 1.2 * max(max(load_balancing_threshold_t[i, :]), load_balancing_threshold_baseline_t[i])])

    ax.legend()
    fig.tight_layout()

    fig.savefig(save_path / f"load_balancing_threshold_N{load_balancing_threshold_N[i]}_K{load_balancing_threshold_K[i]}_A{load_balancing_threshold_A[i]}_L{load_balancing_threshold_L[i]}_P{load_balancing_threshold_P[i]}_S{load_balancing_threshold_S[i]}.svg", format='svg', transparent=True)

lb_fig = plt.figure(figsize=(5, 4.5))
lb_ax = lb_fig.add_subplot(1, 1, 1)
lb_ax.set_xlabel("Load imbalance [-]")
lb_ax.set_ylabel("Speedup [-]")
lb_title = f"Load balancing performance, threshold test, N = {load_balancing_threshold_N[0]} K = {load_balancing_threshold_K[0]} A = {load_balancing_threshold_A[0]} L = {load_balancing_threshold_L[0]} P = {load_balancing_threshold_P[0]} N_max = {load_balancing_threshold_max_N[0]} T = {load_balancing_threshold_T[0][best_T]}"
lb_fig.canvas.manager.set_window_title(lb_title)

lb_ax.plot(load_balancing_threshold_I, load_balancing_threshold_baseline_t/load_balancing_threshold_t[:, best_T], color=imbalance_colour, linewidth=data_width, marker=data_shape, markersize=data_size, label="Speedup")

lb_ax.set_ylim([0, 1.2 * max(load_balancing_threshold_baseline_t/load_balancing_threshold_t[:, best_T])])
lb_fig.tight_layout()

lb_fig.savefig(save_path / f"load_balancing_threshold_N{load_balancing_threshold_N[0]}_K{load_balancing_threshold_K[0]}_A{load_balancing_threshold_A[0]}_L{load_balancing_threshold_L[0]}_P{load_balancing_threshold_P[0]}_T{load_balancing_threshold_T[0][best_T]}.svg", format='svg', transparent=True)

# N scaling
N_fig = plt.figure(figsize=(5, 4.5))
N_ax = N_fig.add_subplot(1, 1, 1)
N_ax.set_xlabel("N [-]")
N_ax.set_ylabel("Iteration time [s]")
N_title = "N iteration time"
N_fig.canvas.manager.set_window_title(N_title)
N_ax.grid()

N_ax.plot(N_scaling_N_cpu, N_scaling_t_cpu, color=N_colour_cpu, linewidth=data_width, marker=data_shape, markersize=data_size, label="CPU iteration time")
N_ax.plot(N_scaling_N_gpu, N_scaling_t_gpu, color=N_colour_gpu, linewidth=data_width, marker=data_shape, markersize=data_size, label="GPU iteration time")

N_ax.set_ylim([0, 1.2 * max(max(N_scaling_t_cpu), max(N_scaling_t_gpu))])

N_ax.legend()
N_fig.tight_layout()

N_fig.savefig(save_path / f"N_iteration_time.svg", format='svg', transparent=True)

plt.show()
