import time
import numpy as np
import functools
import powerflow
import powerflow.calc_grid_parameters
import powerflow.plotting as plt


def zbusjacobi(Zred, S, slack_voltage, eps_s=1.0, max_iters=20, u_check_pu=1):
    numberofnodes = S.shape[1]
    numberofloads = S.shape[0]
    iters_all = np.zeros(numberofloads, dtype=np.int32)
    U_all = np.zeros((numberofloads, numberofnodes), dtype=np.complex128)
    short_curcuited = np.zeros((numberofloads), dtype=np.bool)
    u = np.ones(numberofnodes, dtype=np.complex128) * slack_voltage
    u_check = slack_voltage * u_check_pu

    for i in range(numberofloads):
        s = S[i, :]
        iters = 0
        while True:
            iters += 1
            Ibus = np.conj(s / u)
            u = np.dot(Zred, Ibus) + slack_voltage
            if np.max(np.abs(s - u * np.conj(Ibus))) < eps_s or iters > max_iters:
                break
            if iters == 1 and u.min() > u_check:
                short_curcuited[i] = True
        U_all[i, :] = u
        iters_all[i] = iters

    return U_all, iters_all, short_curcuited


# grid = powerflow.mockgrids.radial(150, voltage_level='l')
# S = powerflow.mockloads.beta(grid, maxload=30000, numberofloads=2000)
# filename='sumSoverminU_radial100'

grid = powerflow.mockgrids.ngrot()
S = powerflow.mockloads.ngrot(grid, n=None)
filename = "modified_cc_ngrot"

print(S.shape)

starttime = time.time()
grid_parameters = powerflow.calc_grid_parameters.prepdata(grid)
Zred = np.linalg.inv(grid_parameters["Yred"])
# U0 = grid_parameters['u0']
slack_voltage = grid_parameters["u0"][0]
deleted_nodes = grid_parameters["node_ids_to_delete"]
slack_index = grid_parameters["slack_index"]
S = np.delete(S, deleted_nodes + [slack_index], axis=1)
U, iters, sc = zbusjacobi(Zred, S, slack_voltage, 1.0, 1000, u_check_pu=0.99)
print(f"Runtime: {round(time.time() - starttime) / 1000} ms")

saved_iters = sum(iters[sc]) - len(iters[sc])
print(
    f"Saved {saved_iters} or {round(saved_iters/sum(iters)*100,1)}% of a total of {sum(iters)}"
)

U0 = 110_000 / 1000
safety_factor = 0.99

U /= 1000
S /= 1e9
minUs = np.min(np.abs(U), axis=1)
sumSs = np.sum(np.abs(S), axis=1)

volt_load_tuples = [(u, s, sc) for u, s, sc in zip(minUs, sumSs, sc)]

# sorted_volt_load_tuples = sorted(volt_load_tuples, key=lambda x:x[0])  # Sort by voltage
# sortedU = np.array([x[0] for x in sorted_volt_load_tuples])
# sortedS = np.array([x[1] for x in sorted_volt_load_tuples])
# sortedSC = np.array([x[2] for x in sorted_volt_load_tuples])

fig, ax = plt.subplots()
plt.setsize(fig)

# This should def be refactored for speed:
no_total = len(volt_load_tuples)
no_positives = 0
no_falsenegatives = 0
no_negatives = 0
for u, s, sc in volt_load_tuples:
    if sc:
        (positives,) = ax.plot(u, s, ".", color="green", markeredgewidth=0)
        no_positives += 1
    elif u > U0 * safety_factor:
        (falsenegatives,) = ax.plot(u, s, ".", color="lightgreen", markeredgewidth=0)
        no_falsenegatives += 1
    else:
        (negatives,) = ax.plot(u, s, ".", color="tab:blue", markeredgewidth=0)
        no_negatives += 1

# Percentages
p_negatives = round(no_negatives / no_total * 100)
p_falsenegatives = round(no_falsenegatives / no_total * 100)
p_positives = round(no_positives / no_total * 100)

# Strings
s_negatives = f"{p_negatives}\% Negatives"
s_falsenegatives = f"{p_falsenegatives}\% False Negatives"
s_positives = f"{p_positives}\% Positives"
ax.legend(
    (negatives, falsenegatives, positives),
    (s_negatives, s_falsenegatives, s_positives),
    fontsize=8,
    facecolor="gainsboro",
    edgecolor="gainsboro",
    markerscale=2.0,
)


# U_abovecrit = sortedU[sortedU > U0 * safety_factor]
# U_belowcrit = sortedU[sortedU <= U0 * safety_factor]

# S_above390 = sortedS[sortedU > U0 * safety_factor]
# S_below390 = sortedS[sortedU <= U0 * safety_factor]

# SC_abovecrit = sortedSC[sortedU > U0 * safety_factor]
# SC_belowcrit = sortedSC[sortedU <= U0 * safety_factor]

# # Theoretically perfect cutoff:
# max_safe_S = min(S_below390)

# # Positives: Points that have minU > 390 and could be detected by their sumS if its perfectly known
# S_WL_positive = S_above390[sortedSC]
# U_WL_positive = U_abovecrit[sortedSC]

# # False negative: Points that have minU > 390, but their sumS is > smallest sumS of a point with minU < 390:
# S_WL_falsenegative = S_above390[not sortedSC]
# U_WL_falsenegative = U_abovecrit[not sortedSC]

ax.axvline(U0 * safety_factor, color="black", linestyle="--")
# ax.axhline(max_safe_S, color='red', linestyle='--')

# ax.plot(U_WL_positive, S_WL_positive, '.', color='green', alpha=0.7, markeredgewidth=0)
# ax.plot(U_WL_falsenegative, S_WL_falsenegative, '.', color='lightgreen', alpha=0.7, markeredgewidth=0)
# ax.plot(U_belowcrit, S_below390, '.', alpha=0.7, markeredgewidth=0)

xmin, xmax = ax.get_xlim()
x_extension = (xmax - xmin) * 0.25
ax.set_xlim(xmin - x_extension, xmax + x_extension)

ymin, ymax = ax.get_ylim()
y_extension = (ymax - ymin) * 0.25
ax.set_ylim(ymin - y_extension, ymax + y_extension)


# Set transform and bbox for all 'text'-functions:
bbox = {"boxstyle": "round", "linewidth": 0, "facecolor": "lightgray"}
text = functools.partial(ax.text, fontsize=8, transform=ax.transAxes, bbox=bbox)

# text(0.03, 0.95, f'Negatives: {round(U_belowcrit.size / sortedU.size*100)}\%', verticalalignment='top', horizontalalignment='left')
# text(0.97, 0.95, f'False Negatives: {round(U_WL_falsenegative.size / sortedU.size*100)}\%', verticalalignment='top', horizontalalignment='right')
# text(0.03, 0.05, 'False Positives: 0', verticalalignment='bottom', horizontalalignment='left')
# text(0.97, 0.05, f'Positives: {round(U_WL_positive.size / sortedU.size*100)}\%' , verticalalignment='bottom', horizontalalignment='right')

ax.grid()
ax.set_xlabel("Minimum Voltage in Grid / kV")
ax.set_ylabel("Sum of Loads in Grid / GVA")

plt.tight_layout()
plt.save(fig, filename)
plt.show()
