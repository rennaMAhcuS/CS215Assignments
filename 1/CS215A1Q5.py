import numpy as np
import matplotlib.pyplot as plt


def simulate_queue(sq_traders=200, sq_trials=10000):
    sq_win_count = np.zeros(sq_traders)

    for _ in range(sq_trials):
        queue = np.random.randint(1, 201, size=sq_traders)
        seen_ids = set()

        for j in range(sq_traders):
            if queue[j] in seen_ids:
                sq_win_count[j] += 1
                break
            seen_ids.add(queue[j])

    return sq_win_count


def find_optimal_position(fop_win_count):
    fop_optimal_position = np.argmax(fop_win_count)
    fop_max_wins = fop_win_count[fop_optimal_position]
    return fop_optimal_position, fop_max_wins


for i in range(5):
    traders = 365
    trials = 1000000

    win_count = simulate_queue(traders, trials)
    optimal_position, max_wins = find_optimal_position(win_count)

    print(f"Optimal position in the queue: {optimal_position + 1}")
    print(f"Number of wins at this position: {max_wins} out of {trials} trials")
    plt.plot(np.arange(1, traders + 1), win_count * 100 / trials)
    plt.xlabel('Position in Queue')
    plt.ylabel('Number of Wins')
    plt.title('Wins by Position in Queue')
    plt.grid(True)
    plt.show()
