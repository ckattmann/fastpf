import os
import functools
import numpy as np
# import powerflow.powerflow
# import powerflow.calc_grid_parameters
# import powerflow.data.mockgrids as mockgrids
# import powerflow.data.mockloads as mockloads

import powerflow

import matplotlib.pyplot as plt
import matplotlib.font_manager
from matplotlib import rcParams
rcParams['font.family'] = 'serif'

to_images = functools.partial(os.path.join, os.getenv('HOME'), 'Dropbox', 'diss', 'images')


plt.style.use('~/Dropbox/diss/diss.mplstyle')
cmap = plt.get_cmap('Blues')


# fig, (ax1,ax2) = plt.subplots(2,1)
fig1, ax1 = plt.subplots()
fig2, ax2 = plt.subplots()

fig_width = 4.33
fig1.set_size_inches((fig_width,2.5))
fig2.set_size_inches((fig_width,2.5))

nodenumbers = [5,10,20,30]
acc_factors = np.arange(1.0, 2.0, 0.05)

c_iterator = iter(cmap(np.linspace(0.5,1,len(nodenumbers))))

for i,non in enumerate(nodenumbers):

    grid = powerflow.mockgrids.feeder(non)

    S = powerflow.mockloads.beta(grid, load=5000, num_loads=100, seed=0)

    # grid_parameters = powerflow.calc_grid_parameters.calc_grid_parameters(grid, S)
    # U, iters, runtime = powerflow.ybusgaussseidel(grid_parameters, S, acc_factor=1.0, verbose=False)

    runtimes = []
    all_iters = []

    print(f'Grid: Feeder, {non} Nodes')
    for acc_factor in acc_factors:
        # acc_factor = 1.0
        # grid_parameters = powerflow.calc_grid_parameters.calc_grid_parameters(grid, S)
        U, iters, runtime = powerflow.ybusgaussseidel(grid, S, acc_factor=acc_factor, verbose=False)
        runtimes.append(runtime)
        all_iters.append(np.mean(iters))
        print(f'{acc_factor:3.2f}    {np.min(np.abs(U)):6.3f}    {np.mean(iters):5.1f}    {runtime*1000:8.3f} ms')

    c = next(c_iterator)

    ax1.plot(acc_factors, runtimes, '.-', markersize=6, clip_on=False, color=c, zorder=5, label=str(non).rjust(2,' ') + ' Nodes')
    ax1.grid(True)
    ax1.set_xlabel('Acceleration Factor / -')
    ax1.set_ylabel('Runtime / s')

    ax2.plot(acc_factors, all_iters, '.-', markersize=6, clip_on=False, color=c, zorder=5, label=str(non).rjust(2,' ')+' Nodes')
    ax2.grid(True)
    ax2.set_xlabel('Acceleration Factor / -')
    ax2.set_ylabel('Iterations / -')

    # Big dots for minimal runtime and iterations:
    min_runtime = min(runtimes)
    acc_factor_min_runtime = acc_factors[np.argmin(runtimes)]
    ax1.plot(acc_factor_min_runtime, min_runtime, 'o', color=c, markersize=6, clip_on=False, zorder=10)

    min_iters = min(all_iters)
    acc_factor_min_iters = acc_factors[np.argmin(all_iters)]
    ax2.plot(acc_factor_min_iters, min_iters, 'o', color=c, markersize=6, clip_on=False, zorder=10)

x1,x2,y1,y2 = ax1.axis()
ax1.axis((1.0,x2,0,y2))

x1,x2,y1,y2 = ax2.axis()
ax2.axis((1.0,x2,0,y2))

legend1 = ax1.legend(facecolor='gainsboro')
legend1.get_frame().set_linewidth(0.0)
legend2 = ax2.legend(facecolor='gainsboro')
legend2.get_frame().set_linewidth(0.0)

plt.tight_layout()

fig1.savefig(to_images('gs_accfactors_runtime.eps'), dpi=600, bbox_inches='tight', pad_inches=0.04)
fig2.savefig(to_images('gs_accfactors_iterations.eps'), dpi=600, bbox_inches='tight', pad_inches=0.04)
plt.show()

