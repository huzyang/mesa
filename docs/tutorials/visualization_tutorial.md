# Visualization Tutorial

*This version of the visualisation tutorial is updated for Mesa 3.0, and works with Mesa `3.0.0a4` and above. If you are using Mesa 2.3.x, check out the [stable version](https://mesa.readthedocs.io/stable/tutorials/visualization_tutorial.html) of this tutorial on Readthedocs.*

**Important:** 
- If you are just exploring Mesa and want the fastest way to the the dashboard and code checkout [![py.cafe](https://img.shields.io/badge/launch-py.cafe-blue)](https://py.cafe/app/tpike3/boltzmann-wealth-model) (click "Editor" to see the code)
- If you want to see the dashboard in an interactive notebook try [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/projectmesa/mesa/main?labpath=docs%2Ftutorials%2Fvisualization_tutorial.ipynb)
- If you have installed mesa and are running locally, please ensure that your [Mesa version](https://pypi.org/project/Mesa/) is up-to-date in order to run this tutorial.

### Adding visualization

So far, we've built a model, run it, and analyzed some output afterwards. However, one of the advantages of agent-based models is that we can often watch them run step by step, potentially spotting unexpected patterns, behaviors or bugs, or developing new intuitions, hypotheses, or insights. Other times, watching a model run can explain it to an unfamiliar audience better than static explanations. Like many ABM frameworks, Mesa allows you to create an interactive visualization of the model. In this section we'll walk through creating a visualization using built-in components, and (for advanced users) how to create a new visualization element.

First, a quick explanation of how Mesa's interactive visualization works. The visualization is done in a browser window, using the [Solara](https://solara.dev/) framework, a pure Python, React-style web framework. Running `solara run app.py` will launch a web server, which runs the model, and displays model detail at each step via the Matplotlib plotting library. Alternatively, you can execute everything inside a notebook environment and display it inline.

#### Grid Visualization

To start with, let's have a visualization where we can watch the agents moving around the grid. Let us use the same `MoneyModel` created in the [Introductory Tutorial](https://mesa.readthedocs.io/stable/tutorials/intro_tutorial.html).


Mesa's grid visualizer works by looping over every cell in a grid, and generating a portrayal for every agent it finds. A portrayal is a dictionary (which can easily be turned into a JSON object) which tells Matplotlib the color and size of the scatterplot markers (each signifying an agent). The only thing we need to provide is a function which takes an agent, and returns a portrayal dictionary. Here's the simplest one: it'll draw each agent as a blue, filled circle, with a radius size of 50.

## Part 1 - Basic Dashboard

**Note:**  Due to the computational cost of running multiple dashboards it is recommended that at the end of each part you restart your kernel and then only run the the cells in that portion of the tutorial (e.g. Part 1). Each portion is entirely self contained.


```python
import mesa
print(f"Mesa version: {mesa.__version__}")

from mesa.visualization import SolaraViz, make_plot_measure, make_space_matplotlib
# Import the local MoneyModel.py
from MoneyModel import MoneyModel

```

    Mesa version: 3.0.0b1
    


```python
def agent_portrayal(agent):
    return {
        "color": "tab:blue",
        "size": 50,
    }
```

In addition to the portrayal method, we instantiate the model parameters, some of which are modifiable by user inputs. In this case, the number of agents, N, is specified as a slider of integers.


```python
model_params = {
    "n": {
        "type": "SliderInt",
        "value": 50,
        "label": "Number of agents:",
        "min": 10,
        "max": 100,
        "step": 1,
    },
    "width": 10,
    "height": 10,
}
```

Next, we instantiate the visualization object which (by default) displays the grid containing the agents, and timeseries of values computed by the model's data collector. In this example, we specify the Gini coefficient.

There are 3 buttons:
- the step button, which advances the model by 1 step
- the play button, which advances the model indefinitely until it is paused, or until `model.running` is False (you may specify the stopping condition)
- the pause button, which pauses the model

To reset the model, simply change the model parameter from the user input (e.g. the "Number of agents" slider).


```python
# Create initial model instance
model1 = MoneyModel(50, 10, 10)

SpaceGraph = make_space_matplotlib(agent_portrayal)
GiniPlot = make_plot_measure("Gini")

page = SolaraViz(
    model1,
    components=[SpaceGraph, GiniPlot],
    model_params=model_params,
    name="Boltzmann Wealth Model",
)
# This is required to render the visualization in the Jupyter notebook
page
```

## Part 2 - Dynamic Agent Representation 

Due to the computational cost of running multiple dashboards it is recommended that at the end of each part you restart your kernel and then only run the import cell and the cells in that portion of the part of the tutorial (e.g. Part 2)


In the visualization above, all we could see is the agents moving around -- but not how much money they had, or anything else of interest. Let's change it so that agents who are broke (wealth 0) are drawn in red, smaller. (TODO: Currently, we can't predict the drawing order of the circles, so a broke agent may be overshadowed by a wealthy agent. We should fix this by doing a hollow circle instead)
In addition to size and color, an agent's shape can also be customized when using the default drawer. The allowed values for shapes can be found [here](https://matplotlib.org/stable/api/markers_api.html).

To do this, we go back to our `agent_portrayal` code and add some code to change the portrayal based on the agent properties and launch the server again.


```python
import mesa
print(f"Mesa version: {mesa.__version__}")

from mesa.visualization import SolaraViz, make_plot_measure, make_space_matplotlib
# Import the local MoneyModel.py
from MoneyModel import MoneyModel

```


```python
def agent_portrayal(agent):
    size = 10
    color = "tab:red"
    if agent.wealth > 0:
        size = 50
        color = "tab:blue"
    return {"size": size, "color": color}

model_params = {
    "n": {
        "type": "SliderInt",
        "value": 50,
        "label": "Number of agents:",
        "min": 10,
        "max": 100,
        "step": 1,
    },
    "width": 10,
    "height": 10,
}
```


```python
# Create initial model instance
model1 = MoneyModel(50, 10, 10)

SpaceGraph = make_space_matplotlib(agent_portrayal)
GiniPlot = make_plot_measure("Gini")

page = SolaraViz(
    model1,
    components=[SpaceGraph, GiniPlot],
    model_params=model_params,
    name="Boltzmann Wealth Model",
)
# This is required to render the visualization in the Jupyter notebook
page
```

## Part 3 - Custom Components 

Due to the computational cost of running multiple dashboards it is recommended that at the end of each part you restart your kernel and then only run the import cell and the cells in that portion of the part of the tutorial (e.g. Part 3)

**Note:** This section is for users who have a basic familiarity with Python's Matplotlib plotting library.

If the visualization elements provided by Mesa aren't enough for you, you can build your own and plug them into the model server.

For this example, let's build a simple histogram visualization, which can count the number of agents with each value of wealth.

**Note:** Due to the way solara works we need to trigger an update whenever the underlying model changes. For this you need to register an update counter with every component.


```python
import mesa
print(f"Mesa version: {mesa.__version__}")
import solara
from matplotlib.figure import Figure

from mesa.visualization.utils import update_counter
from mesa.visualization import SolaraViz, make_plot_measure, make_space_matplotlib
# Import the local MoneyModel.py
from MoneyModel import MoneyModel

```


```python
def agent_portrayal(agent):
    size = 10
    color = "tab:red"
    if agent.wealth > 0:
        size = 50
        color = "tab:blue"
    return {"size": size, "color": color}

model_params = {
    "n": {
        "type": "SliderInt",
        "value": 50,
        "label": "Number of agents:",
        "min": 10,
        "max": 100,
        "step": 1,
    },
    "width": 10,
    "height": 10,
}
```

Next, we update our solara frontend to use this new component


```python
@solara.component
def Histogram(model):
    update_counter.get() # This is required to update the counter
    # Note: you must initialize a figure using this method instead of
    # plt.figure(), for thread safety purpose
    fig = Figure()
    ax = fig.subplots()
    wealth_vals = [agent.wealth for agent in model.agents]
    # Note: you have to use Matplotlib's OOP API instead of plt.hist
    # because plt.hist is not thread-safe.
    ax.hist(wealth_vals, bins=10)
    solara.FigureMatplotlib(fig)
```


```python
# Create initial model instance
model1 = MoneyModel(50, 10, 10)

SpaceGraph = make_space_matplotlib(agent_portrayal)
GiniPlot = make_plot_measure("Gini")
```


```python
page = SolaraViz(
    model1,
    components=[SpaceGraph, GiniPlot, Histogram],
    model_params=model_params,
    name="Boltzmann Wealth Model",
)
# This is required to render the visualization in the Jupyter notebook
page
```

You can even run the visuals independently by calling it with the model instance


```python
Histogram(model1)
```

### Happy Modeling!

This document is a work in progress.  If you see any errors, exclusions or have any problems please contact [us](https://github.com/projectmesa/mesa/issues).
