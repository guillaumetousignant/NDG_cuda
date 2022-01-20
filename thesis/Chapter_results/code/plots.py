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

# Plots
# Strong scaling
save_path = Path(__file__).parent.parent / "media"
save_path.mkdir(parents=True, exist_ok=True)

data_width = 3
ideal_width = 3
cpu_colour = np.array([244, 71, 71])/255
gpu_colour = np.array([79, 193, 255])/255
data_size = 12
data_shape = "o"
ideal_style = "--"

for i in range(N.shape[0]):
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel("Number of nodes [-]")
    ax.set_ylabel("Time [s]")
    title = f"Strong scaling, N = {N[i]} K = {K[i]} W = {W[i]}"
    ax.set_title(title)
    fig.canvas.manager.set_window_title(title)
    ax.grid()

    ax.loglog(nodes_beluga_cpu[i, :], times_beluga_cpu[i, :], color=cpu_colour, linewidth=data_width, marker=data_shape, markersize=data_size, label="Beluga CPU time")
    ax.loglog(nodes_beluga_cpu_ideal[i, :], times_beluga_cpu_ideal[i, :], color=cpu_colour, linewidth=data_width, linestyle=ideal_style, label="Beluga ideal CPU time")
    
    ax.loglog(nodes_beluga_gpu[i, :], times_beluga_gpu[i, :], color=gpu_colour, linewidth=data_width, marker=data_shape, markersize=data_size, label="Beluga GPU time")
    ax.loglog(nodes_beluga_gpu_ideal[i, :], times_beluga_gpu_ideal[i, :], color=gpu_colour, linewidth=data_width, linestyle=ideal_style, label="Beluga ideal GPU time")

    ax.legend()

    fig.savefig(save_path / f"strong_scaling_N{N[i]}_K{K[i]}_W{W[i]}.svg", format='svg', transparent=True)

# Weak scaling
for i in range(N_weak.shape[0]):
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel("Number of nodes [-]")
    ax.set_ylabel("Time [s]")
    title = f"Weak scaling, N = {N_weak[i]} K = {K_weak[i]}/proc W = {W_weak[i]}"
    ax.set_title(title)
    fig.canvas.manager.set_window_title(title)
    ax.grid()

    ax.semilogx(nodes_beluga_cpu_weak[i, :], times_beluga_cpu_weak[i, :], color=cpu_colour, linewidth=data_width, marker=data_shape, markersize=data_size, label="Beluga CPU time")
    ax.semilogx(nodes_beluga_cpu_ideal_weak[i, :], [min_total_time_beluga_cpu_weak[i], min_total_time_beluga_cpu_weak[i]], color=cpu_colour, linewidth=data_width, linestyle=ideal_style, label="Beluga ideal CPU time")
    
    ax.semilogx(nodes_beluga_gpu_weak[i, :], times_beluga_gpu_weak[i, :], color=gpu_colour, linewidth=data_width, marker=data_shape, markersize=data_size, label="Beluga GPU time")
    ax.semilogx(nodes_beluga_gpu_ideal_weak[i, :], [min_total_time_beluga_gpu_weak[i], min_total_time_beluga_gpu_weak[i]], color=gpu_colour, linewidth=data_width, linestyle=ideal_style, label="Beluga ideal GPU time")

    ax.set_ylim([0, 1.2 * max(max(times_beluga_cpu_weak[i, :]), max(times_beluga_gpu_weak[i, :]))])

    ax.legend()

    fig.savefig(save_path / f"weak_scaling_N{N_weak[i]}_K{K_weak[i]}_W{W_weak[i]}.svg", format='svg', transparent=True)


plt.show()