import sys

def skip_last(iterator):
    prev = next(iterator)
    for item in iterator:
        yield prev
        prev = item

run_1_energies = []
run_2_energies = []

with open(sys.argv[1]) as run_1_energies_file:
    headers = run_1_energies_file.next()
    
    keys = headers.split()
    keys[1:3] = [reduce(lambda x,y: x+y, keys[1:3])]
    keys[2:4] = [reduce(lambda x,y: x+y, keys[2:4])]
    keys[13:15] = [reduce(lambda x,y: x+y, keys[13:15])]
    energy_types = keys
   # run_1_energies = {k:[] for k in keys}
    
    for row in skip_last(run_1_energies_file):
        energies = [float(x) for x in row.split()]
        run_1_energies.append(energies)

with open(sys.argv[2]) as run_2_energies_file:
    headers = run_2_energies_file.next()

   # keys = headers.split()
   # keys[1:3] = [reduce(lambda x,y: x+y, keys[1:3])]
   # keys[2:4] = [reduce(lambda x,y: x+y, keys[2:4])]
   # keys[13:15] = [reduce(lambda x,y: x+y, keys[13:15])]
   # run_2_energies = {k:[] for k in keys}
    
    for row in skip_last(run_2_energies_file):
        energies = [float(x) for x in row.split()]
        run_2_energies.append(energies)

diffs = []
for t in xrange(len(run_1_energies)):
    d = [ abs(x - y) for x, y in zip(run_1_energies[t], run_2_energies[t]) ]
    diffs.append(d)

max_diff = max([max(x) for x in diffs])

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

energy_type = range(len(diffs[0])+1)
run = range(len(diffs)+1)

energy_type, run = np.meshgrid(energy_type, run)
diffs = np.array(diffs)

fig = plt.figure()
ax = fig.add_subplot(111)

#heatmap = plt.pcolormesh(energy_type, run, diffs, cmap="Reds", norm=LogNorm(vmin=np.finfo(float).eps, vmax=max_diff))
heatmap = plt.pcolormesh(energy_type, run, diffs, cmap="Reds")
plt.colorbar()

ax.set_xticks(range(len(diffs[0])))
ax.set_xticklabels(energy_types, rotation=90, ha="left")
ax.set_xlabel("Energy Type")

ax.set_ylabel("Time Step")

ax.grid(b=True, which='major', color='k',linestyle='-')

plt.show()
