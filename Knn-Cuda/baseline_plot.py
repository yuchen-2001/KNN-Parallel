import matplotlib.pyplot as plt

# Data for 2 kernels
small_calcDistance_time = [2.461664, 1.288704, 1.270944, 1.198080, 2.115584, 1.968416, 4.562752, 4.564128, 4.558752, 4.561984, 4.557344]

medium_calcDistance_time = [7.726720, 1.679520, 4.811456, 1.146464, 3.309568]

large_calcDistance_time = []

small_sortArray_time = [0.032640, 0.041760, 0.027072, 0.027328, 0.028704, 0.059360, 0.010656, 0.010528, 0.004288, 0.004512, 0.004192]

medium_sortArray_time = [0.129088, 0.081760, 0.112480, 0.067392]

large_sortArray_time = []

def plot(y1, y2, y3, title, xlabel, ylabel):
    plt.figure(figsize=(12, 7))
    plt.plot(x, y1, marker='o', label='Small Dataset (135 rows)')
    plt.plot(x, y2, marker='s', label='Medium Dataset (1000 rows)')
    plt.plot(x, y3, marker='^', label='Large Dataset (10,000 rows)')

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xscale('log')  # Log scale helps visualize wide range of processors
    plt.yscale('log')  # Optional: Also make y log-scale to highlight small diffs
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # set x
    x = []
    # 2**11 = 2048
    for i in range(12):
        x.append(2**i)

    # Plotting
    plot()
