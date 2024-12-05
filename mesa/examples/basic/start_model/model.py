import mesa
from agents import PDAgent
from mesa.experimental.cell_space import OrthogonalMooreGrid


class PdGrid(mesa.Model):
    """Model class for iterated, spatial prisoner's dilemma model."""

    activation_regimes = ["Sequential", "Random", "Simultaneous"]

    # This dictionary holds the payoff for this agent,

    payoff = {("C", "C"): 1, ("C", "D"): 0, ("D", "C"): 1.6, ("D", "D"): 0}

    def __init__(
        self, width=50, height=50, activation_order="Random", payoffs=None, seed=None
    ):
        """
        Create a new Spatial Prisoners' Dilemma Model.

        Args:
            width, height: Grid size. There will be one agent per grid cell.
            activation_order: Can be "Sequential", "Random", or "Simultaneous".
                           Determines the agent activation regime.
            payoffs: (optional) Dictionary of (strategy, neighbor_strategy) payoffs.
        """
        super().__init__(seed=seed)
        self.activation_order = activation_order
        self.grid = OrthogonalMooreGrid((width, height), torus=True)

        if payoffs is not None:
            self.payoff = payoffs

        # Create agents
        for x in range(width):
            for y in range(height):
                agent = PDAgent(self)
                agent.cell = self.grid[(x, y)]

        # Collect data: set measure
        self.datacollector = mesa.DataCollector(
            model_reporters={"Cooperating_Agents": self.compute_fra_c},
            agent_reporters={"Score": "score"},
        )
        self.running = True
        self.datacollector.collect(self)

    def step(self):
        # Activate all agents, based on the activation regime
        match self.activation_order:
            case "Sequential":
                self.agents.do("step")
            case "Random":
                self.agents.shuffle_do("step")
            case "Simultaneous":
                self.agents.do("step")
                self.agents.do("advance")
            case _:
                raise ValueError(f"Unknown activation order: {self.activation_order}")

        # Collect data
        self.datacollector.collect(self)
        print(f"fra_c list is {self.datacollector.model_vars["Cooperating_Agents"]}")

    def run(self, n):
        """Run the model for n steps."""
        for _ in range(n):
            self.step()

    def compute_fra_c(self):
        num_agents = len(self.agents)
        return len([a for a in self.agents if a.strategy == "C"]) / num_agents
