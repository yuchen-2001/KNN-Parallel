import matplotlib.pyplot as plt

# Data for 2 kernels
small_calcDistance_time = [2.461664, 1.288704, 1.270944, 1.198080, 2.115584, 1.968416, 4.562752, 4.564128, 4.558752, 4.561984, 4.557344]

medium_calcDistance_time = [10.501408, 3.528032, 1.699616, 1.517600, 4.563424, 4.561440, 4.565248, 4.563296, 4.564992, 4.562432, 1.357664]

large_calcDistance_time = [5.764096, 1.507104, 1.235200, 1.492992, 1.636032, 1.802848, 1.761248, 1.358656, 1.581824, 1.441312, 1.838752]

small_sortArray_time = [0.032640, 0.041760, 0.027072, 0.027328, 0.028704, 0.059360, 0.010656, 0.010528, 0.004288, 0.004512, 0.004192]

medium_sortArray_time = [0.065760, 0.117344, 0.064160, 0.056416, 0.033312, 0.033824, 0.032768, 0.032928, 0.038912, 0.042592, 0.018720]

large_sortArray_time = [13.488224, 6.930464, 3.534848, 1.880064, 0.774432, 0.740928, 0.740288, 0.743456, 0.911360, 0.594048, 1.089792]

small_total_time = [5.216960, 5.270112, 4.909120, 5.264768, 5.124832, 6.326368, 5.522592, 5.004736, 4.750944, 4.958048, 4.768608]

medium_total_time = [44.359135, 39.309696, 29.600321, 29.052320, 39.694847, 27.991167, 28.390079, 28.596031, 30.050560, 28.951391, 22.690176]

large_total_time = [5333.633301, 2970.058350, 1721.422241, 1125.911255, 800.460022, 656.855591, 675.420532, 695.952515, 763.381897, 765.337952, 1021.554138]

if __name__ == '__main__':
    # set x
    x = []
    # 2**11 = 2048
    for i in range(12):
        x.append(2**i)

    # Plotting
    plt.figure(figsize=(12, 7))
    plt.plot(x, small_total_time, marker='o', label='Small Dataset (135 rows)')
    plt.plot(x, medium_total_time, marker='s', label='Medium Dataset (1000 rows)')
    plt.plot(x, large_total_time, marker='^', label='Large Dataset (10,000 rows)')

    plt.xlabel('Number of Kernels')
    plt.ylabel('Execution Time (ms)')
    plt.title('CUDA kNN Execution Time vs Number of Kernels (calcDistance)')
    plt.xscale('log')  # Log scale helps visualize wide range of processors
    plt.yscale('log')  # Optional: Also make y log-scale to highlight small diffs
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.tight_layout()

    # plt.figure(figsize=(12, 7))
    # plt.plot(x, small_sortArray_time, marker='o', label='Small Dataset (135 rows)')
    # plt.plot(x, medium_sortArray_time, marker='s', label='Medium Dataset (1000 rows)')
    # plt.plot(x, large_sortArray_time, marker='^', label='Large Dataset (10,000 rows)')

    # plt.xlabel('Number of Kernels')
    # plt.ylabel('Execution Time (ms)')
    # plt.title('CUDA kNN Execution Time vs Number of Kernels (sortArray)')
    # plt.xscale('log')  # Log scale helps visualize wide range of processors
    # plt.yscale('log')  # Optional: Also make y log-scale to highlight small diffs
    # plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    # plt.legend()
    # plt.tight_layout()

    plt.show()
