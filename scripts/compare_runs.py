#!/usr/bin/env python
import sys

### A helper function for reading data from the energies files
### and skipping the summary/timing lines at their ends.
def skip_last(iterator):
    prev = next(iterator)
    for item in iterator:
        yield prev
        prev = item

### Lists to hold the energy data
run_1_energies = []
run_2_energies = []

### Open first run's file
with open(sys.argv[1]) as run_1_energies_file:
    ### Skip first line and get the column headers
    headers = run_1_energies_file.next()
    energy_types = headers.split()
    energy_types[1:3] = [reduce(lambda x,y: x+y, energy_types[1:3])]
    energy_types[2:4] = [reduce(lambda x,y: x+y, energy_types[2:4])]
    energy_types[13:15] = [reduce(lambda x,y: x+y, energy_types[13:15])]
    ### Read energy data from first run's file
    for row in skip_last(run_1_energies_file):
        energies = [float(x) for x in row.split()]
        run_1_energies.append(energies)

### Open second run's file
with open(sys.argv[2]) as run_2_energies_file:
    # Skip first line
    headers = run_2_energies_file.next()
    ### Read energy data from second run's file
    for row in skip_last(run_2_energies_file):
        energies = [float(x) for x in row.split()]
        run_2_energies.append(energies)

### Compute the differences between the two runs' energies at each time step 
diffs = []
for t in xrange(len(run_1_energies)):
    d = [ abs(x - y) for x, y in zip(run_1_energies[t], run_2_energies[t]) ]
    diffs.append(d)

### Find the max difference to enable the log scale color mapping in the plotting stage
max_diff = max([max(x) for x in diffs])

### Modules used for plotting
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

### Set up the grid for the plot
energy_type = range(len(diffs[0])+1)
run = range(len(diffs)+1)
energy_type, run = np.meshgrid(energy_type, run)
diffs = np.array(diffs)

### Make the plot 
### Note: If there is no difference between the runs' energy at a time step, the cell 
### corresponding to (energy, time step) should be white. If there is a difference, 
### the cell will be shaded red in proportion to the size of the difference. 
### The intensity of the shading is done on a log scale for visibility.
### See the color bar on the plot to map shades of red to numerical values. 
fig = plt.figure()
ax = fig.add_subplot(111)
heatmap = plt.pcolormesh(energy_type, run, diffs, cmap="Reds", norm=LogNorm(vmin=np.finfo(float).eps, vmax=max_diff))
#heatmap = plt.pcolormesh(energy_type, run, diffs, cmap="Reds")
plt.colorbar()

### Annotate the plot's axes
ax.set_xticks(range(len(diffs[0])))
ax.set_xticklabels(energy_types, rotation=90, ha="left")
ax.set_xlabel("Energy Type")
ax.set_ylabel("Time Step")
ax.grid(b=True, which='major', color='k',linestyle='-')

### Save the plot and display it
plt.savefig(sys.argv[3])
plt.show()
