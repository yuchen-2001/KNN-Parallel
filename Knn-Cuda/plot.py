import matplotlib.pyplot as plt

### BASELINE DATA ###
# Data for 2 kernels
small_calcDistance_time = [2.461664, 1.288704, 1.270944, 1.198080, 2.115584, 1.968416, 4.562752, 4.564128, 4.558752, 4.561984, 4.557344]

medium_calcDistance_time = [10.501408, 3.528032, 1.699616, 1.517600, 4.563424, 4.561440, 4.565248, 4.563296, 4.564992, 4.562432, 1.357664]

large_calcDistance_time = [5.764096, 1.507104, 1.235200, 1.492992, 1.636032, 1.802848, 1.761248, 1.358656, 1.581824, 1.441312, 1.838752]

small_sortArray_time = [0.032640, 0.041760, 0.027072, 0.027328, 0.028704, 0.059360, 0.010656, 0.010528, 0.004288, 0.004512, 0.004192]

medium_sortArray_time = [0.065760, 0.117344, 0.064160, 0.056416, 0.033312, 0.033824, 0.032768, 0.032928, 0.038912, 0.042592, 0.018720]

large_sortArray_time = [13.488224, 6.930464, 3.534848, 1.880064, 0.774432, 0.740928, 0.740288, 0.743456, 0.911360, 0.594048, 1.089792]

# Total calculation time
small_total_time = [5.216960, 5.270112, 4.909120, 5.264768, 5.124832, 6.326368, 5.522592, 5.004736, 4.750944, 4.958048, 4.768608]

medium_total_time = [2735.677490, 1480.564453, 915.711487, 613.092163, 429.029694, 346.325287, 363.448425, 350.728943, 423.357330, 390.299744, 523.808105]

large_total_time = [269571.312500, 136642.406250, 70499.570312, 37300.902344, 20990.832031, 11943.401367, 12525.944336, 12575.667969, 13425.613281, 13132.176758, 15961.399414]

### OUR IMPLEMENTATION ###
# Data for 2 kernels

# Total calculation time
small_total_time = []

medium_total_time = []

large_total_time = []


if __name__ == '__main__':
    # set x
    x = []
    # 2**11 = 2048
    for i in range(12):
        x.append(2**i)

    # Plotting
    plt.figure(figsize=(12, 7))
    plt.plot(x, small_total_time, marker='o', label='Small Dataset (135 train points, 15 test points)')
    plt.plot(x, medium_total_time, marker='s', label='Medium Dataset (10,000 train points, 500 test points)')
    plt.plot(x, large_total_time, marker='^', label='Large Dataset (50,000 train points, 2,000 test points)')

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
