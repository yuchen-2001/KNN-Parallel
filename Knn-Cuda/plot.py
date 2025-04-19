import matplotlib.pyplot as plt

### BASELINE DATA ###
# Data for 2 kernels
# small_calcDistance_time = [2.461664, 1.288704, 1.270944, 1.198080, 2.115584, 1.968416, 4.562752, 4.564128, 4.558752, 4.561984, 4.557344]

# medium_calcDistance_time = [10.501408, 3.528032, 1.699616, 1.517600, 4.563424, 4.561440, 4.565248, 4.563296, 4.564992, 4.562432, 1.357664]

# large_calcDistance_time = [5.764096, 1.507104, 1.235200, 1.492992, 1.636032, 1.802848, 1.761248, 1.358656, 1.581824, 1.441312, 1.838752]

# small_sortArray_time = [0.032640, 0.041760, 0.027072, 0.027328, 0.028704, 0.059360, 0.010656, 0.010528, 0.004288, 0.004512, 0.004192]

# medium_sortArray_time = [0.065760, 0.117344, 0.064160, 0.056416, 0.033312, 0.033824, 0.032768, 0.032928, 0.038912, 0.042592, 0.018720]

# large_sortArray_time = [13.488224, 6.930464, 3.534848, 1.880064, 0.774432, 0.740928, 0.740288, 0.743456, 0.911360, 0.594048, 1.089792]

# Total calculation time
small_total_time_b = [6.326368, 5.522592, 5.004736, 4.750944, 4.958048]

medium_total_time_b = [346.325287, 363.448425, 350.728943, 423.357330, 390.299744]

large_total_time_b = [11943.401367, 12525.944336, 12575.667969, 13425.613281, 13132.176758]

### OUR IMPLEMENTATION ###
# Data for 2 kernels
small_batchCalcDistance_time = [1.483328, 59.324923, 91.421234, 73.120894, 66.051105]

medium_batchCalcDistance_time = [62.671841, 62.695393, 66.213470, 68.143105, 62.576256]

large_batchCalcDistance_time = [68.118530, 72.527809, 7.690240, 58.762337, 75.376640]

small_batchCalcDistance_time = [0.132320, 0.154752, 0.134496, 0.284768, 0.385024]

medium_batchCalcDistance_time = [1.400064, 2.219072, 2.421504, 2.729536, 3.713696]

large_batchCalcDistance_time = [14.411488, 13.509408, 23.905151, 27.309216, 39.715199]

# Total calculation time
small_total_time = [2.402080, 61.827202, 92.805054, 73.999870, 66.288033]

medium_total_time = [178.772995, 119.263741, 96.276993, 94.415779, 103.645119]

large_total_time = [811.623535, 406.533508, 410.199371, 423.760254, 427.272095]

def plotTotalExeTime():
    # set x
    x = []
    # 2**9 = 512
    for i in range(5, 10):
        x.append(2**i)

    # Plotting
    plt.figure(figsize=(12, 7))
    plt.plot(x, small_total_time, marker='o', label='Small Dataset (135 train points, 15 test points)')
    plt.plot(x, medium_total_time, marker='s', label='Medium Dataset (10,000 train points, 500 test points)')
    plt.plot(x, large_total_time, marker='^', label='Large Dataset (50,000 train points, 2,000 test points)')

    plt.xlabel('Number of Kernels per block')
    plt.ylabel('Execution Time (ms)')
    plt.title('CUDA kNN Total Execution Time vs Number of Kernels per block (baseline)')
    plt.xticks(x, labels=[str(v) for v in x])
    # plt.xscale()  # Log scale helps visualize wide range of processors
    plt.yscale('log')  # Optional: Also make y log-scale to highlight small diffs
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    # plt.legend()
    plt.legend(loc='upper left', bbox_to_anchor=(0.5, 0.5))
    plt.tight_layout()

def plotCompareTotalExeTime():
    # set x
    x = []
    # 2**9 = 512
    for i in range(5, 10):
        x.append(2**i)

    plt.figure(figsize=(12, 7))
    plt.plot(x, large_total_time_b, marker='o', label='Baseline')
    plt.plot(x, large_total_time, marker='s', label='Optimized')

    plt.xlabel('Number of Kernels')
    plt.ylabel('Execution Time (ms)')
    plt.title('CUDA kNN Execution Time: Baseline vs Optimized implementation (large size dataset)')
    plt.xticks(x, labels=[str(v) for v in x])
    plt.yscale('log')  # Optional: Also make y log-scale to highlight small diffs
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.tight_layout()

    plt.show()

def plotOptbatchCalcDisTime():
    # set x
    x = []
    # 2**9 = 512
    for i in range(5, 10):
        x.append(2**i)

    # Plotting
    plt.figure(figsize=(12, 7))
    plt.plot(x, small_batchCalcDistance_time, marker='o', label='Small Dataset (135 train points, 15 test points)')
    plt.plot(x, medium_batchCalcDistance_time, marker='s', label='Medium Dataset (10,000 train points, 500 test points)')
    plt.plot(x, large_batchCalcDistance_time, marker='^', label='Large Dataset (50,000 train points, 2,000 test points)')

    plt.xlabel('Number of Kernels per block')
    plt.ylabel('Execution Time (ms)')
    plt.title('Kernel batchCalcDistance Execution Time vs Number of Kernels per block (optimized)')
    plt.xticks(x, labels=[str(v) for v in x])
    # plt.xscale()  # Log scale helps visualize wide range of processors
    plt.yscale('log')  # Optional: Also make y log-scale to highlight small diffs
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    # plt.legend()
    plt.legend(loc='upper left', bbox_to_anchor=(0.5, 0.5))
    plt.tight_layout()

if __name__ == '__main__':
    # plotTotalExeTime()
    # plotCompareTotalExeTime()
