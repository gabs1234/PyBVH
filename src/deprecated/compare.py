import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Open bvh.csv file
bvh = pd.read_csv("data.csv")

# open trig_times.csv file
trig = pd.read_csv("trig_times.csv")

# # Extract bvh
# columns_id_bvh = [bvh.columns.get_loc(col) for col in bvh.columns]
# columns_bvh = [bvh.iloc[:, i].values for i in columns_id_bvh]
# column_headers = bvh.columns
# x = columns_bvh[0]
# y1 = columns_bvh[1]
# y2 = columns_bvh[2]
# y3 = columns_bvh[3]

# Extract trig
columns_id_trig = [trig.columns.get_loc(col) for col in trig.columns]
columns_trig = [trig.iloc[:, i].values for i in columns_id_trig]
column_headers_trig = trig.columns
x_trig = columns_trig[1]
y1_trig = columns_trig[3]
y2_trig = columns_trig[5]

# # Plot bvh
# fig = plt.figure(figsize=(12, 8), dpi=300)
# markers = ['o', 's', 'D']
# for i in range(1, len(column_headers)):
#     plt.loglog(x, columns_bvh[i], label=column_headers[i], marker=markers[i-1])

# plt.xlabel("Number of primitives")
# plt.ylabel("Time (ms)")
# plt.legend()

# # Add grid + tight layout
# plt.grid(True)
# plt.tight_layout()

# # Save plot
# plt.savefig("plot.png")

# Plot trig
fig = plt.figure(figsize=(12, 8), dpi=300)
markers = ['o', 's']
for j, i in enumerate([3, 5]):
    plt.loglog(x_trig, columns_trig[i], label=column_headers_trig[i], marker=markers[j])

plt.xlabel("Number of primitives")
plt.ylabel("Time (ms)")
plt.legend()

# Add grid + tight layout
plt.grid(True)
plt.tight_layout()

# Save plot
plt.savefig("plot_trig.png")

plt.show()