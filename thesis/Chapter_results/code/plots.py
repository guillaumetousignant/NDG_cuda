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

# Adaptivity efficiency
# Those were run on my computer, may need to be run again on Narval
adaptivity_interval = np.array([5, 20, 100, 500])
adaptivity_N = np.array([4, 4, 4, 4])
adaptivity_K = np.array([4 * 4, 4 * 4, 4 * 4, 4 * 4])
adaptivity_C = np.array([[0, 1, 2, 3],
                         [0, 1, 2, 3],
                         [0, 1, 2, 3],
                         [0, 1, 2, 3]])
adaptivity_t = np.array([[428.886 + 0, 252.003 + 0.103265, 365.731 + 0.25112, 308.074 + 0.628907],
                         [350.965 + 0, 189.985 + 0.13161, 264.603 + 0.439747, 500.48 + 0.822952],
                         [133.571 + 0, 334.156 + 0.393835, 86.4947 + 0.977535, 130.43 + 2.66017],
                         [0.692123 + 0, 2.07892 + 1.53437, 114.021 + 2.82137, 192.083 + 10.0328]])
adaptivity_max_error = np.array([[7.9e-7, 4.3e-8, 7.3e-9, 2.0e-8],
                                 [6.2e-6, 5.7e-8, 1.3e-8, 8.2e-9],
                                 [2.3e-4, 2.2e-6, 3.6e-8, 2.6e-8],
                                 [1.1e-3, 2.2e-5, 1.3e-6, 1.6e-6]])

adaptivity_baseline_C = adaptivity_C[:, [0,-1]]
adaptivity_baseline_t = np.array([293.175, 292.261, 289.285, 295.575])
adaptivity_baseline_max_error = np.array([1.2e-9, 1.2e-9, 1.2e-9, 1.2e-9])

# Load balancing efficiency interval
load_balancing_interval_A = np.array([20, 20])                  # Adaptivity interval
load_balancing_interval_N = np.array([4, 4])                    # Initial N
load_balancing_interval_K = np.array([128 * 128, 128 * 128])    # Initial number of elements
load_balancing_interval_P = np.array([16, 16])                  # Number of processes
load_balancing_interval_S = np.array([3, 5])                    # Max split level
load_balancing_interval_max_N = np.array([12, 12])              # Max N

load_balancing_interval_L = np.array([[20, 100, 200, 500, 1000],
                                      [20, 100, 200, 500, 1000]]) # Load balancing interval
load_balancing_interval_t = np.array([[346.838, 284.198, 266.794, 248.574, 254.721],
                                      [2928.21, 2107.16, 1989.26, 1966.79, 2112.57]]) # Simulation time

load_balancing_interval_baseline_L = load_balancing_interval_L[:, [0,-1]]
load_balancing_interval_baseline_t = np.array([274.415, 4962.06])

# Load balancing efficiency threshold
load_balancing_threshold_A = np.array([20, 20])                 # Adaptivity interval
load_balancing_threshold_L = np.array([20, 20])                 # Load balancing interval
load_balancing_threshold_N = np.array([4, 4])                   # Initial N
load_balancing_threshold_K = np.array([128 * 128, 128 * 128])   # Initial number of elements
load_balancing_threshold_P = np.array([16, 16])                 # Number of processes
load_balancing_threshold_S = np.array([3, 5])                   # Max split level
load_balancing_threshold_max_N = np.array([12, 12])             # Max N

load_balancing_threshold_T = np.array([[1, 1.01, 1.1, 1.5, 2],
                                       [1, 1.01, 1.1, 1.5, 2]]) # Load balancing threshold
load_balancing_threshold_t = np.array([[344.277, 295.591, 243.566, 241.076, 252.407],
                                       [2925.25, 2248.53, 1869.24, 1865.98, 2043.47]]) # Simulation time

load_balancing_threshold_baseline_T = load_balancing_threshold_T[:, [0,-1]]
load_balancing_threshold_baseline_t = np.array([274.359, 4962.06])

# N scaling
N_scaling_N = np.array([6, 8, 10, 12])
N_scaling_t = np.array([0.0346, 0.0594, 0.0856, 0.1196])

# Plots
save_path = Path(__file__).parent.parent / "media"
save_path.mkdir(parents=True, exist_ok=True)

data_width = 3
ideal_width = 3
cpu_colour = np.array([244, 71, 71])/255
gpu_colour = np.array([79, 193, 255])/255
gpu_colour_dark = np.array([3, 125, 190])/255
error_colour = np.array([197, 134, 192])/255
error_colour_dark = np.array([168, 80, 161])/255
N_colour = np.array([78, 201, 176])/255
data_size = 12
data_shape = "o"
ideal_style = "--"

# Strong scaling
for i in range(N.shape[0]):
    fig = plt.figure(figsize=(5, 4.5))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel("Number of nodes [-]")
    ax.set_ylabel("Time [s]")
    title = f"Strong scaling, N = {N[i]} K = {K[i]} W = {W[i]}"
    fig.canvas.manager.set_window_title(title)
    ax.grid()

    ax.loglog(nodes_beluga_cpu[i, :], times_beluga_cpu[i, :], color=cpu_colour, linewidth=data_width, marker=data_shape, markersize=data_size, label="Beluga CPU time")
    ax.loglog(nodes_beluga_cpu_ideal[i, :], times_beluga_cpu_ideal[i, :], color=cpu_colour, linewidth=data_width, linestyle=ideal_style, label="Beluga ideal CPU time")
    
    ax.loglog(nodes_beluga_gpu[i, :], times_beluga_gpu[i, :], color=gpu_colour, linewidth=data_width, marker=data_shape, markersize=data_size, label="Beluga GPU time")
    ax.loglog(nodes_beluga_gpu_ideal[i, :], times_beluga_gpu_ideal[i, :], color=gpu_colour, linewidth=data_width, linestyle=ideal_style, label="Beluga ideal GPU time")

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

    ax.semilogx(nodes_beluga_cpu_weak[i, :], times_beluga_cpu_weak[i, :], color=cpu_colour, linewidth=data_width, marker=data_shape, markersize=data_size, label="Beluga CPU time")
    ax.semilogx(nodes_beluga_cpu_ideal_weak[i, :], [min_total_time_beluga_cpu_weak[i], min_total_time_beluga_cpu_weak[i]], color=cpu_colour, linewidth=data_width, linestyle=ideal_style, label="Beluga ideal CPU time")
    
    ax.semilogx(nodes_beluga_gpu_weak[i, :], times_beluga_gpu_weak[i, :], color=gpu_colour, linewidth=data_width, marker=data_shape, markersize=data_size, label="Beluga GPU time")
    ax.semilogx(nodes_beluga_gpu_ideal_weak[i, :], [min_total_time_beluga_gpu_weak[i], min_total_time_beluga_gpu_weak[i]], color=gpu_colour, linewidth=data_width, linestyle=ideal_style, label="Beluga ideal GPU time")

    ax.set_ylim([0, 1.2 * max(max(times_beluga_cpu_weak[i, :]), max(times_beluga_gpu_weak[i, :]))])

    ax.legend()
    fig.tight_layout()

    fig.savefig(save_path / f"weak_scaling_N{N_weak[i]}_K{K_weak[i]}_W{W_weak[i]}.svg", format='svg', transparent=True)

# Adaptivity efficiency
for i in range(adaptivity_interval.shape[0]):
    fig = plt.figure(figsize=(5.5, 4.5))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel("Number of pre-condition adaptivity steps [-]")
    ax.set_ylabel("Time [s]", color=gpu_colour_dark)
    ax.tick_params(axis='y', labelcolor=gpu_colour_dark)
    title = f"Adaptivity performance, N = {adaptivity_N[i]} K = {adaptivity_K[i]} A = {adaptivity_interval[i]}"
    fig.canvas.manager.set_window_title(title)
    ax.grid(axis='x')

    ax.plot(adaptivity_C[i, :], adaptivity_t[i, :], color=gpu_colour, linewidth=data_width, marker=data_shape, markersize=data_size, label="GPU time")
    ax.plot(adaptivity_baseline_C[i, :], [adaptivity_baseline_t[i], adaptivity_baseline_t[i]], color=gpu_colour, linewidth=data_width, linestyle=ideal_style, label="GPU non adaptive time")

    ax.set_ylim([0, 1.2 * max(max(adaptivity_t[i, :]), adaptivity_baseline_t[i])])

    error_ax = ax.twinx()
    error_ax.set_ylabel("Analytical solution error [-]", color=error_colour_dark)
    error_ax.tick_params(axis='y', labelcolor=error_colour_dark)

    # For the legend entries from the other axes
    error_ax.plot([], [], color=gpu_colour, linewidth=data_width, marker=data_shape, markersize=data_size, label="GPU time")
    error_ax.plot([], [], color=gpu_colour, linewidth=data_width, linestyle=ideal_style, label="GPU non adaptive time")

    error_ax.semilogy(adaptivity_C[i, :], adaptivity_max_error[i, :], color=error_colour, linewidth=data_width, marker=data_shape, markersize=data_size, label="GPU max error")
    error_ax.semilogy(adaptivity_baseline_C[i, :], [adaptivity_baseline_max_error[i], adaptivity_baseline_max_error[i]], color=error_colour, linewidth=data_width, linestyle=ideal_style, label="GPU non adaptive max error")

    error_ax.legend()
    fig.tight_layout()

    fig.savefig(save_path / f"adaptivity_N{adaptivity_N[i]}_K{adaptivity_K[i]}_C{adaptivity_interval[i]}.svg", format='svg', transparent=True)

# Load balancing efficiency interval
for i in range(load_balancing_interval_N.shape[0]):
    fig = plt.figure(figsize=(5, 4.5))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel("Load balancing interval [-]")
    ax.set_ylabel("Time [s]")
    title = f"Load balancing performance, interval test, N = {load_balancing_interval_N[i]} K = {load_balancing_interval_K[i]} A = {load_balancing_interval_A[i]} P = {load_balancing_interval_P[i]} S = {load_balancing_interval_S[i]} N_max = {load_balancing_interval_max_N[i]}"
    fig.canvas.manager.set_window_title(title)
    ax.grid()

    ax.plot(load_balancing_interval_L[i, :], load_balancing_interval_t[i, :], color=gpu_colour, linewidth=data_width, marker=data_shape, markersize=data_size, label="Narval GPU time")
    ax.plot(load_balancing_interval_baseline_L[i, :], [load_balancing_interval_baseline_t[i], load_balancing_interval_baseline_t[i]], color=gpu_colour, linewidth=data_width, linestyle=ideal_style, label="Narval non load balancing time")

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

    ax.plot(load_balancing_threshold_T[i, :], load_balancing_threshold_t[i, :], color=gpu_colour, linewidth=data_width, marker=data_shape, markersize=data_size, label="Narval GPU time")
    ax.plot(load_balancing_threshold_baseline_T[i, :], [load_balancing_threshold_baseline_t[i], load_balancing_threshold_baseline_t[i]], color=gpu_colour, linewidth=data_width, linestyle=ideal_style, label="Narval non load balancing time")

    ax.set_ylim([0, 1.2 * max(max(load_balancing_threshold_t[i, :]), load_balancing_threshold_baseline_t[i])])

    ax.legend()
    fig.tight_layout()

    fig.savefig(save_path / f"load_balancing_threshold_N{load_balancing_threshold_N[i]}_K{load_balancing_threshold_K[i]}_A{load_balancing_threshold_A[i]}_L{load_balancing_threshold_L[i]}_P{load_balancing_threshold_P[i]}_S{load_balancing_threshold_S[i]}.svg", format='svg', transparent=True)

# N scaling
N_fig = plt.figure(figsize=(5, 4.5))
N_ax = N_fig.add_subplot(1, 1, 1)
N_ax.set_xlabel("N [-]")
N_ax.set_ylabel("Iteration time [s]")
N_title = "N iteration time"
N_fig.canvas.manager.set_window_title(N_title)
N_ax.grid()

N_ax.plot(N_scaling_N, N_scaling_t, color=N_colour, linewidth=data_width, marker=data_shape, markersize=data_size, label="Iteration time")

N_ax.set_ylim([0, 1.2 * max(N_scaling_t)])

#N_ax.legend()
N_fig.tight_layout()

N_fig.savefig(save_path / f"N_iteration_time.svg", format='svg', transparent=True)

plt.show()
