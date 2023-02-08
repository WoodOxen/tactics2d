import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import pandas as pd
import numpy as np


def get_corners(x, y, width, length, heading):
    coords = np.array([
        [x-0.5*width, y+0.5*length],
        [x+0.5*width, y+0.5*length],
        [x+0.5*width, y-0.5*length],
        [x-0.5*width, y-0.5*length]
    ])
    
    return coords

fig, ax = plt.subplots()

df = pd.read_csv("../data/trajectory_processed/DLP/0001_obstacles.csv")
patches = []
for row in df.iterrows():
    info = row[1]
    coords = get_corners(info["xCenter"], info["yCenter"], info["width"], info["length"], info["heading"])

    polygon = Polygon(coords, color='y', closed=True)

    ax.add_patch(polygon)

ax.set_xlim(0, 200)
ax.set_ylim(0, 100)
plt.show()
