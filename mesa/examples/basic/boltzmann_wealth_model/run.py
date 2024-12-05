from mesa.examples.basic.boltzmann_wealth_model.model import BoltzmannWealthModel

# Set the random seed
seed = 21


def run_model(model):
    for _ in range(10):
        model.step()


m = BoltzmannWealthModel(50, 10, 10)
run_model(m)
