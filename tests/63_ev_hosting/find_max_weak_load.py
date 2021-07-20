import json
import time

import numpy as np

import powerflow
import powerflow.plotting as plt
import makeloads


with open("sonderbuch.json") as f:
    grid = json.load(f)

## The deterministic way:
## ======================

loads = []
Umins = []
for load in np.arange(-10000, 10001, 500):
    S = powerflow.mockloads.fixed(grid, load=load)
    U, iters, runtime = powerflow.zbusjacobi(grid, S)
    # Usign = np.sign(U[np.argmax(np.abs(U-400))]-400)
    U_ext_index = np.unravel_index(np.argmax(np.abs(U - 400), axis=None), U.shape)
    Umin = U[U_ext_index]

    # print(load, Umin)
    loads.append(load)
    Umins.append(Umin)

loads = np.array(loads)
Umins = np.array(Umins)
print([np.abs(Umins - 400) < 40])
loads_ok = loads[np.abs(Umins - 400) < 40]
Umins_ok = Umins[np.abs(Umins - 400) < 40]


# colors = ['green' if (U_ext - 400) < 40 else 'red' for U_ext in Umins]
# print(colors)
fig, ax = plt.subplots()
plt.setsize(fig, size=1)
ax.axvline(360, color="red", linestyle="--")
ax.axvline(440, color="red", linestyle="--")
ax.axhline(5950, color="black", linestyle="--")
ax.axhline(-7010, color="black", linestyle="--")
ax.set_xlim([330, 470])
ax.plot(Umins_ok, loads_ok, "o", color="green", markersize=3)
ax.plot(
    Umins[np.abs(Umins - 400) > 40],
    loads[np.abs(Umins - 400) > 40],
    "o",
    color="red",
    markersize=3,
)
ax.grid(True)
ax.set_ylabel("Load applied at every node / W")
ax.set_xlabel("Voltage with max. deviation from slack / V")
plt.tight_layout()
plt.save(fig, "deterministic_WLD")
plt.show()

## The heuristic way:
## ==================

# starttime = time.time()
# S = powerflow.mockloads.once_at_each_node(grid, load=30000)
# S = powerflow.mockloads.fixed(grid, load=load)
# S = makeloads.construct_S(grid, days=365, timesteps_per_day=1440, number_of_evs=10, charging_power_kW=22)
# print(f'Data Setup Time: {time.time()-starttime} s')
# print(f'{S.shape[0]} PFs')

# U, iters, runtime = powerflow.zbusjacobi(grid, S)

# plt.plot(U.T)
# plt.grid(True)
# plt.show()

# starttime = time.time()
# grid_parameters = powerflow.calc_grid_parameters.calc_grid_parameters(grid, S)
# Zred = np.linalg.inv(grid_parameters['Yred'])
# slack_index = grid_parameters['slack_index']
# S = np.delete(S, [slack_index], axis=1)
# U2 = powerflow.powerflow_methods_cc.zbusjacobi_1iter(Zred, S, 400)
# # Zred_f32 = (np.sign(np.real(Zred))*np.abs(Zred)).astype(np.float32)
# # S_f32 = (np.sign(np.real(S))*np.abs(S)).astype(np.float32)
# # U2 = powerflow.powerflow_methods_cc.zbusjacobi_1iter_f32(Zred_f32, S_f32, 400)
# print(f'Runtime: {time.time()-starttime} s')
# print(np.mean(np.abs(U-U2)))
# print(np.max(np.abs(U-U2)))


# fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2)

# # plt.plot(np.abs(S) - S_f32)
# ax1.plot(S)
# ax1.grid(True)
# # ax2.plot(S_f32)
# # ax2.grid(True)

# ax3.plot(U)
# ax3.grid(True)
# ax4.plot(U2)
# ax4.grid(True)
# # plt.plot(np.sign(np.real(U-U2)) * np.abs(U-U2))
# plt.show()
