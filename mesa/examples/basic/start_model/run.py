from model import PdGrid

# Data visualization tools.
import seaborn as sns
import matplotlib.pyplot as plt

# Set the random seed
seed = 21


def run_model(model):
    model.run(5)


m = PdGrid(10, 10, "Random", seed=seed)
run_model(m)

# agent-score data
# agent_score = m.datacollector.get_agent_vars_dataframe()
# agent_score.head()

# Model data
fra_c = m.datacollector.get_model_vars_dataframe()
# Plot the fra_c over time
g2 = sns.lineplot(data=fra_c)
g2.set(title="fra_c over Time", ylabel="fra_c")
g2.set_ylim(0.3, 1)
plt.show()
