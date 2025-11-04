import matplotlib.pyplot as plt
import numpy as np
from model import TrustGrid

bwr = plt.get_cmap("bwr")
# Set the random seed
seed = 21


def draw_grid(model: TrustGrid, ax=None):
    """
    Draw the current state of the grid, with Defecting agents in red
    and Cooperating agents in blue.
    """
    if not ax:
        fig, ax = plt.subplots(figsize=(6, 6))
    grid = np.zeros((model.grid.width, model.grid.height))
    for cell in model.grid.all_cells.cells:
        agent = cell.agents[0]
        (x, y) = cell.coordinate
        if agent.move == "D":
            grid[y][x] = 1
        else:
            grid[y][x] = 0
    ax.pcolormesh(grid, cmap=bwr, vmin=0, vmax=1)
    ax.axis("off")
    ax.set_title(f"Steps: {model.steps}")


def run_model(model):
    """
    Run an experiment with a given model, and plot the results.
    """
    fig = plt.figure(figsize=(12, 8))

    ax1 = fig.add_subplot(231)
    ax2 = fig.add_subplot(232)
    ax3 = fig.add_subplot(233)
    ax4 = fig.add_subplot(212)

    draw_grid(model, ax1)
    model.run(10)
    # draw_grid(model, ax2)
    # model.run(10)
    # draw_grid(model, ax3)
    model.datacollector.get_model_vars_dataframe().plot(ax=ax4)


m = TrustGrid(50, 50, "Random", seed=seed)
run_model(m)
