# -*- coding: utf-8 -*-
"""
Created on Wed May 21 10:42:33 2025

@author: chypu
"""


#%% Import libraries
import numpy as np
from pykrige.ok import OrdinaryKriging
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import cluster
import scipy.cluster
from mpl_toolkits.axes_grid1 import make_axes_locatable

#%% Update plot parameters for paper
plt.rcdefaults()
params = {
    'lines.linewidth': 0.3,
    'lines.markersize': 1,
    'axes.labelsize': 8,
    'axes.titlesize': 8,
    'axes.titleweight': 'normal',
    'font.size': 8,
    'font.family': 'Times New Roman',
    'font.weight': 'normal',
    'mathtext.fontset': 'stix',
    'legend.shadow': False,
    'legend.fontsize': 8,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'text.usetex': False,
    'figure.figsize': [3.0, 3.0],  # Slightly larger to prevent label crowding
    'figure.autolayout': False
}
plt.rcParams.update(params)

#%% conductivity 2D x and y
#%% Input data
x = np.array([0.85, 0.85, 0.85, 1.15, 1.15, 1.15, 1.45, 1.45, 1.45]) # x-coordinate
y = np.array([0.2, 0.5, 0.8, 0.2, 0.5, 0.8, 0.2, 0.5, 0.8])# y-coordinate
z=np.array([0.25,0.25,0.25,0.35, 0.35,0.35,0.45,0.45,0.45]) # z-coordinate (not directly used in Kriging for XY plot)
# R has been renamed to R to signify Resistance
# R=np.array([1.065185,0, 0, 0, 0, 2.359383,0, 0, 0,]) # 1st time stamp TS1: 1397.96805 sec
# R=np.array([1.065185,0, 0, 0, 0,3.517124, 4.737214, 18.79436,25.83256]) # 2nd time stamp TS2: 1853.32298 sec
# R=np.array([11.5888,0, 10.55272,7.427341, 8.472001,5.822115,9.416244, 25.83256,26.85826]) # 3rd time stamp TS3: 2247.93314 sec
# R=np.array([23.8352,24.83523, 19.80799,22.83248,	23.8352,5.822115,9.416244,26.85826,26.85826]) # 4th time stamp TS4: 4706.0852 sec
R=np.array([22.8324756,23.83520094,19.80799096,22.8324756,	24.83522648,8.145853945,10.58514135,28.83942145,27.85016969]) # 5th time stamp TS5: 6567.4691 sec


#%% Ordinary Kriging interpolation
OK = OrdinaryKriging(
    x, y, R, # Changed V to R
    variogram_model='gaussian',
    verbose=False,
    enable_plotting=False
)

gridx = np.arange(0.0, 2.0, 0.01)
gridy = np.arange(0.0, 1.0, 0.005)
zstar, ss = OK.execute("grid", gridx, gridy)

#%% Kriging Plot
fig, ax = plt.subplots()
cax = ax.imshow(zstar, extent=(0, 2, 0, 1), origin='lower', aspect='equal', cmap='viridis')
ax.scatter(x, y, marker='.', s=40, facecolors='none', edgecolors='k')

ax.set_xlabel('x-coordinate (m)')
ax.set_ylabel('y-coordinate (m)')
ax.set_xticks(np.arange(0, 2.1, 0.25))
ax.set_yticks(np.arange(0, 1.1, 0.25))

divider = make_axes_locatable(ax)
cbar_ax = divider.append_axes("right", size="5%", pad=0.05)
cbar = fig.colorbar(cax, cax=cbar_ax)
cbar.set_label(r'resistance ($\Omega$)') # Label changed to resistance

plt.tight_layout()
# Save kriging plots
# plt.savefig('plots/XY/spatial_krigtest_2Dxy_TS1.png', dpi=400)
# plt.savefig('plots/XY/spatial_krigtest_2Dxy_TS1.pdf', dpi=400)
# plt.savefig('plots/XY/spatial_krigtest_2Dxy_TS2.png', dpi=400)
# plt.savefig('plots/XY/spatial_krigtest_2Dxy_TS2.pdf', dpi=400)
# plt.savefig('plots/XY/spatial_krigtest_2Dxy_TS3.png', dpi=400)
# plt.savefig('plots/XY/spatial_krigtest_2Dxy_TS3.pdf', dpi=400)
# plt.savefig('plots/XY/spatial_krigtest_2Dxy_TS4.png', dpi=400)
# plt.savefig('plots/XY/spatial_krigtest_2Dxy_TS4.pdf', dpi=400)
# plt.savefig('plots/XY/spatial_krigtest_2Dxy_TS5.png', dpi=400)
# plt.savefig('plots/XY/spatial_krigtest_2Dxy_TS5.pdf', dpi=400)
plt.close()

#%% Prepare data for clustering
all_x = np.tile(gridx, gridy.shape[0])
all_y = np.repeat(gridy, gridx.shape[0])
all_res = zstar.flatten() # Renamed all_v to all_res
df = pd.DataFrame({"x_cord": all_x, "y_cord": all_y, "resistance": all_res}) # Column renamed to 'resistance'

## KMeans Clustering with Fixed Color Assignment

#To ensure consistent color assignment based on resistance values, we'll sort the cluster centroids and then map them to your predefined colors and labels.

k = 3
model = cluster.KMeans(n_clusters=k, init='k-means++', n_init='auto', random_state=0)
df_X = df[["resistance"]].copy() # Using 'resistance' for clustering
df["cluster"] = model.fit_predict(df_X)

# Find real centroids (for display purposes, not directly for coloring)
closest, distances = scipy.cluster.vq.vq(model.cluster_centers_, df_X.values)
df["centroids"] = 0
df.loc[closest, "centroids"] = 1

# Get the centroid values and sort them to ensure consistent color assignment
centroid_values = model.cluster_centers_.flatten()
# Sort by resistance (lowest to highest) to map to Dry -> Partially Saturated -> Saturated
sorted_indices = np.argsort(centroid_values)

# Define the colors and labels in the desired order based on SORTED resistance
# Lowest resistance = Dry (cyan)
# Middle resistance = Partially Saturated (orange)
# Highest resistance = Saturated (green)
fixed_colors = np.array(['c', 'orange', 'g'])
fixed_labels = ['Dry', 'Partially Saturated', 'Saturated']

# Create a mapping from KMeans' arbitrary cluster ID to your fixed color and label
color_label_map = {}
for i, original_cluster_id in enumerate(sorted_indices):
    color_label_map[original_cluster_id] = {'color': fixed_colors[i], 'label': fixed_labels[i]}

# Apply the consistent color mapping to the DataFrame
df["Color"] = df['cluster'].apply(lambda x: color_label_map[x]['color'])
df["Label"] = df['cluster'].apply(lambda x: color_label_map[x]['label'])

## plot clustering
fig, ax = plt.subplots()
ax.scatter(df["x_cord"], df["y_cord"], c=df["Color"], s=1)

# Add legend using the fixed labels and colors
for i in range(k):
    color_to_use = fixed_colors[i]
    label_to_use = fixed_labels[i]
    ax.scatter([], [], c=color_to_use, label=label_to_use)

ax.set_xlabel('x-coordinate (m)')
ax.set_ylabel('y-coordinate (m)')
ax.set_xticks(np.arange(0, 2.1, 0.25))
ax.set_yticks(np.arange(0, 1.1, 0.25))
ax.set_aspect('equal')
# plt.legend(loc='upper right', fontsize=6)

plt.tight_layout()

# Display Centroid Values with Colors and Labels
print("\nCentroid Values with Colors and Labels:")
for original_cluster_id in range(k):
    centroid_value = model.cluster_centers_[original_cluster_id][0]
    color = color_label_map[original_cluster_id]['color']
    label = color_label_map[original_cluster_id]['label']
    print(f"Cluster (KMeans ID {original_cluster_id}): Centroid [{centroid_value:.6f}], Color: {color}, Label: {label}")


# save clustering plots
# plt.savefig('plots/XY/clustering_2Dxy_TS1.png',dpi=400)
# plt.savefig('plots/XY/clustering_2Dxy_TS1.pdf',dpi=50)
# plt.savefig('plots/XY/clustering_2Dxy_TS2.png',dpi=400)
# plt.savefig('plots/XY/clustering_2Dxy_TS2.pdf',dpi=50)
# plt.savefig('plots/XY/clustering_2Dxy_TS3.png',dpi=400)
# plt.savefig('plots/XY/clustering_2Dxy_TS3.pdf',dpi=50)
# plt.savefig('plots/XY/clustering_2Dxy_TS4.png',dpi=400)
# plt.savefig('plots/XY/clustering_2Dxy_TS4.pdf',dpi=50)
plt.savefig('plots/XY/clustering_2Dxy_TS5.png', dpi=400)
plt.savefig('plots/XY/clustering_2Dxy_TS5.pdf', dpi=50)
plt.close()