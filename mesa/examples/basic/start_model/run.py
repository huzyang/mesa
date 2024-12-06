from model import PdGrid
# Data visualization tools.
import seaborn as sns

# Set the random seed
seed = 21


def run_model(model):
    model.run(5)


m = PdGrid(10, 10, "Random", seed=seed)
run_model(m)

agent_score = [a.score for a in m.agents]
# Create a histogram with seaborn
g1 = sns.histplot(agent_score, discrete=True)
g1.set(
    title="score distribution", xlabel="score", ylabel="number of agents"
);  # The semicolon is just to avoid printing the object representation

fra_c = m.datacollector.get_model_vars_dataframe()
# Plot the fra_c over time
g2 = sns.lineplot(data=fra_c)
g2.set(title="fra_c over Time", ylabel="fra_c")
