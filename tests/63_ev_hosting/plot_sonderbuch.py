import json
import powerflow.plotting as plt

with open("sonderbuch.json") as f:
    grid = json.load(f)

plt.plotgraph(grid, shape="force", filename="sonderbuch")
