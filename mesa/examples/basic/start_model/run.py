from model import PdGrid

# Set the random seed
seed = 21


def run_model(model):
    model.run(5)


m = PdGrid(10, 10, "Random", seed=seed)
run_model(m)
