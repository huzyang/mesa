from model import PdGrid

# Data visualization tools.
import seaborn as sns
import matplotlib.pyplot as plt
import mesa

# Set the random seed
seed = 21


def run_model(model):
    model.run(10)


# m = PdGrid(10, 10, "Random", seed=seed)
# run_model(m)
params = {"width": range(10, 30, 10), "height": 10}
results = mesa.batch_run(
    PdGrid,
    parameters=params,
    iterations=5,
    max_steps=10,
    number_processes=1,
    data_collection_period=1,
    display_progress=True,
)



# Agent data
# agent_score = [a.score for a in m.agents]
# g1 = sns.histplot(agent_score, discrete=True)
# g1.set(title="Score distribution", xlabel="Score", ylabel="number of agents")

# Model data
# fra_c = m.datacollector.get_model_vars_dataframe()
# g2 = sns.lineplot(data=fra_c)
# g2.set(title="fra_c over Time", ylabel="fra_c")
# g2.set_ylim(0.3, 1.01)
# plt.show()
