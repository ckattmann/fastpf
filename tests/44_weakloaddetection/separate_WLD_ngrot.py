import numpy as np
import functools
import powerflow
import powerflow.plotting as plt

# grid = powerflow.mockgrids.radial(150, voltage_level='l')
# S = powerflow.mockloads.beta(grid, maxload=30000, numberofloads=2000)
# filename='sumSoverminU_radial100'

grid = powerflow.mockgrids.ngrot()
S = powerflow.mockloads.ngrot(grid)
filename='sumSoverminU_ngrot'
U0 = 110_000 / 1000
safety_factor = 0.99

print(S.shape)

U, iters, runtime = powerflow.zbusjacobi(grid,S)

minUs = np.min(np.abs(U),axis=1)
sumSs = np.max(np.abs(S), axis=1)

plt.plot()
sumSs /= 1e9  # Powers in GW

volt_load_tuples = [(u,s) for u,s in zip(minUs, sumSs)]

sorted_volt_load_tuples = sorted(volt_load_tuples, key=lambda x:x[0])
sortedU = np.array([x[0] / 1000 for x in sorted_volt_load_tuples])
sortedS = np.array([x[1] for x in sorted_volt_load_tuples])

fig, ax = plt.subplots()
plt.setsize(fig)

U_above390 = sortedU[sortedU > U0 * safety_factor]
U_below390 = sortedU[sortedU <= U0 * safety_factor]

S_above390 = sortedS[sortedU > U0 * safety_factor]
S_below390 = sortedS[sortedU <= U0 * safety_factor]

# Theoretically perfect cutoff:
max_safe_S = min(S_below390)

# Positives: Points that have minU > 390 and could be detected by their sumS if its perfectly known
S_WL_positive = S_above390[S_above390 < max_safe_S]
U_WL_positive = U_above390[S_above390 < max_safe_S]

# False negative: Points that have minU > 390, but their sumS is > smallest sumS of a point with minU < 390:
S_WL_falsenegative = S_above390[S_above390 >= max_safe_S]
U_WL_falsenegative = U_above390[S_above390 >= max_safe_S]

ax.axvline(U0 * safety_factor, color='black', linestyle='--')
ax.axhline(max_safe_S, color='red', linestyle='--')

ax.plot(U_WL_positive, S_WL_positive, '.', color='green', alpha=0.7, markeredgewidth=0)
ax.plot(U_WL_falsenegative, S_WL_falsenegative, '.', color='lightgreen', alpha=0.7, markeredgewidth=0)
ax.plot(U_below390, S_below390, '.', alpha=0.7, markeredgewidth=0)

xmin, xmax = ax.get_xlim()
x_extension = (xmax-xmin) * 0.25
ax.set_xlim(xmin-x_extension, xmax+x_extension)

ymin, ymax = ax.get_ylim()
y_extension = (ymax-ymin) * 0.25
ax.set_ylim(ymin-y_extension, ymax+y_extension)


# Set transform and bbox for all 'text'-functions:
bbox = {'boxstyle': 'round', 'linewidth': 0, 'facecolor': 'lightgray'}
text = functools.partial(ax.text, fontsize=8, transform=ax.transAxes, bbox=bbox)

text(0.03, 0.95, f'Negatives: {round(U_below390.size / sortedU.size*100)}\%', verticalalignment='top', horizontalalignment='left')
text(0.97, 0.95, f'False Negatives: {round(U_WL_falsenegative.size / sortedU.size*100)}\%', verticalalignment='top', horizontalalignment='right')
text(0.03, 0.05, 'False Positives: 0', verticalalignment='bottom', horizontalalignment='left')
text(0.97, 0.05, f'Positives: {round(U_WL_positive.size / sortedU.size*100)}\%' , verticalalignment='bottom', horizontalalignment='right')

ax.grid()
ax.set_xlabel('Minimum Voltage in Grid / kV')
ax.set_ylabel('Sum of Loads in Grid / GVA')

plt.tight_layout()
# plt.save(fig, filename)
plt.show()
