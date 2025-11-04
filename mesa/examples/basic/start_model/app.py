"""
Solara-based visualization for the Spatial Prisoner's Dilemma Model.
"""

import mesa

print(f"Mesa version: {mesa.__version__}")

from mesa.visualization import SolaraViz, make_space_component, make_plot_component
from model import PdGrid


def pd_agent_portrayal(agent):
    """
    Portrayal function for rendering PD agents in the visualization.
    """
    return {
        "color": "blue" if agent.strategy == "C" else "red",
        "marker": "s",  # square marker
        "size": 25,
    }


# Model parameters
model_params = {
    "seed": {
        "type": "InputText",
        "value": 21,
        "label": "Random Seed",
    },
    "width": {
        "type": "SliderInt",
        "value": 50,
        "min": 10,
        "max": 100,
        "step": 1,
        "label": "Grid Width",
    },
    "height": {
        "type": "SliderInt",
        "value": 50,
        "min": 10,
        "max": 100,
        "step": 1,
        "label": "Grid Height",
    },
    "activation_order": {
        "type": "Select",
        "value": "Random",
        "values": PdGrid.activation_regimes,
        "label": "Activation Regime",
    },
}

# 创建网格可视化组件
grid_viz = make_space_component(agent_portrayal=pd_agent_portrayal)

# 创建代理可视化组件
plot_component = make_plot_component("Cooperating_Agents")

# Initialize model
initial_model = PdGrid(50, 50, "Random")

# Create visualization with all components

page = SolaraViz(
    model=initial_model,
    components=[grid_viz, plot_component],
    model_params=model_params,
    name="Spatial Prisoner's Dilemma On Grid",
)
page  # http://localhost:8765
