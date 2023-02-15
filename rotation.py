import pandas as pd
from code_functions import DF_to_segmented_DF,print_df
import numpy as np
import matplotlib.pyplot as plt

TDF0 = DF_to_segmented_DF(pd.read_excel('data/221005_eksempelsegment001.xlsx'))

# Define some sample data
x = np.linspace(0, 2*np.pi, 100)
y = np.sin(x)

# Plot the original graph
plt.plot(x, y)
plt.title('Original Graph')

# Rotate the graph
angle = np.pi/2  # 45 degrees
rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
xy = np.column_stack((x, y))
rotated_xy = np.dot(xy, rotation_matrix)
rotated_x, rotated_y = rotated_xy[:,0], rotated_xy[:,1]

# Plot the rotated graph
plt.figure()
plt.plot(rotated_x, rotated_y)
plt.title('Rotated Graph')

plt.show()