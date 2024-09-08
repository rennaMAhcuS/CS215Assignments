import matplotlib.pyplot as plt
import numpy as np

def simulate_galton_board(depth: int, number_of_balls: int) -> np.ndarray:
    """
    Simulates the Galton board with depth h and N balls.
    """

    # Random movement sequence, less than 0.5 = left, greater than 0.5 = right.
    random_movement = np.random.uniform(0, 1, (number_of_balls, depth))
    movement_array = np.where(random_movement < 0.5, -1, 1)

    final_positions_galton_board = np.sum(movement_array, axis=1)
    return final_positions_galton_board

# Parameters
N = 1000
h_values = [10, 50, 100]
file_names = ['2d1.png', '2d2.png', '2d3.png']

# Run simulations and create histograms
for h, file_name in zip(h_values, file_names):
    final_positions = simulate_galton_board(h, N)
    # Params
    plt.figure(figsize=(10, 6), dpi=500)
    plt.rcParams['font.family'] = 'Cambria'
    plt.hist(final_positions, bins=h + 1, range=(-h / 2, h / 2), density=True)
    plt.xlabel('Pocket')
    plt.ylabel('Normalized count')
    plt.title(f'Galton Board Simulation for h = {h}, N = {N}')

    plt.show()

    # plt.savefig(file_name)
    # plt.close()
