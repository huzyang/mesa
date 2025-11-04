"""
Solara-based visualization for the Spatial Prisoner's Dilemma Model.
"""

from model import TrustGrid
from mesa.visualization import (
    Slider,
    SolaraViz,
    SpaceRenderer,
    make_plot_component,
)
from mesa.visualization.components import AgentPortrayalStyle


def pd_agent_portrayal(agent):
    """
    Portrayal function for rendering PD agents in the visualization.
    """
    match agent.move:
        case "I":
            return AgentPortrayalStyle(
                color="blue", marker="s", size=25
            )
        case "T":
            return AgentPortrayalStyle(
                color="green", marker="s", size=25
            )
        case "U":
            return AgentPortrayalStyle(
                color="red", marker="s", size=25
            )
        case _:
            raise ValueError(f"Invalid move: {agent.move}")


# Model parameters
model_params = {
    "seed": {
        "type": "InputText",
        "value": 42,
        "label": "Random Seed",
    },
    "width": Slider("Grid Width", value=50, min=10, max=100, step=1),
    "height": Slider("Grid Height", value=50, min=10, max=100, step=1),
    "activation_order": {
        "type": "Select",
        "value": "Random",
        "values": TrustGrid.activation_regimes,
        "label": "Activation Regime",
    },
}


# Create plot for tracking all agent types over time
plot_component = make_plot_component(
    ["Investors", "Trustworthy_Trustees", "Untrustworthy_Trustees"],
    backend="altair",
    grid=True
)

# Initialize model
initial_model = TrustGrid()
# Create grid and agent visualization component using Altair
renderer = SpaceRenderer(initial_model, backend="altair").render(pd_agent_portrayal)

# Create visualization with all components
page = SolaraViz(
    model=initial_model,
    renderer=renderer,
    components=[plot_component],
    model_params=model_params,
    name="N-Player Trust Game",
)
page  # noqa B018
