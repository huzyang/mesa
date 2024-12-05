from mesa.experimental.cell_space import CellAgent


class PDAgent(CellAgent):
    """Agent member of the iterated, spatial prisoner's dilemma model."""

    def __init__(self, model, starting_strategy=None):
        """
        Create a new Prisoner's Dilemma agent.

        Args:
            model: model instance
            starting_strategy: If provided, determines the agent's initial state:
                           C(ooperating) or D(efecting). Otherwise, random.
        """
        super().__init__(model)
        self.score = 0
        # 指定策略或随机选择策略
        if starting_strategy:
            self.strategy = starting_strategy
        else:
            self.strategy = self.random.choice(["C", "D"])

        self.next_strategy = None

    @property
    def is_cooroperating(self):
        return self.strategy == "C"

    def step(self):
        """Get the best neighbor's move, and change own move accordingly
        if better than own score."""

        # neighbors = self.model.grid.get_neighbors(self.pos, True, include_center=True)
        neighbors = [*list(self.cell.neighborhood.agents), self]
        best_neighbor = max(neighbors, key=lambda a: a.score)
        self.next_strategy = best_neighbor.strategy

        if self.model.activation_order != "Simultaneous":
            self.advance()

    def advance(self):
        self.strategy = self.next_strategy
        self.score += self.caculate_score()

    def caculate_score(self):
        neighbors = self.cell.neighborhood.agents
        if self.model.activation_order == "Simultaneous":
            strategies = [neighbor.next_strategy for neighbor in neighbors]
        else:
            strategies = [neighbor.strategy for neighbor in neighbors]
        return sum(
            self.model.payoff[(self.strategy, strategy)] for strategy in strategies
        )
