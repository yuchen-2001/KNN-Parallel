import matplotlib.pyplot as plt

# Data
small_proc = [1, 3, 5, 9, 15, 27, 45, 135]
small_time = [0.002030, 0.001392, 0.002309, 0.004003, 0.007632, 0.018543, 0.051692, 0.260916]

medium_proc = [1, 2, 4, 5, 8, 10, 20, 25, 50, 100, 125, 200, 250, 500, 1000]
medium_time = [0.010396, 0.008198, 0.008210, 0.008847, 0.007573, 0.010862, 0.013399, 0.021768, 
               0.053428, 0.093966, 0.114770, 0.483382, 0.634540, 1.034356, 1.409305]

large_proc = [1, 2, 4, 5, 8, 10, 20, 25, 40, 50, 100, 125, 200, 250, 500, 1000, 2000, 2500]
large_time = [0.114225, 0.086839, 0.073729, 0.080400, 0.076383, 0.072293, 0.090847, 0.081255,
              0.111994, 0.114434, 0.175275, 0.242287, 0.384580, 0.483579, 0.901695, 1.243396,
              1.602698, 2.024593]

if __name__ == '__main__':
    # Plotting
    plt.figure(figsize=(12, 7))
    plt.plot(small_proc, small_time, marker='o', label='Small Dataset (135 rows)')
    plt.plot(medium_proc, medium_time, marker='s', label='Medium Dataset (1000 rows)')
    plt.plot(large_proc, large_time, marker='^', label='Large Dataset (10,000 rows)')

    plt.xlabel('Number of MPI Processes')
    plt.ylabel('Execution Time (seconds)')
    plt.title('OpenMPI kNN Execution Time vs Number of Processes')
    plt.xscale('log')  # Log scale helps visualize wide range of processors
    plt.yscale('log')  # Optional: Also make y log-scale to highlight small diffs
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()
