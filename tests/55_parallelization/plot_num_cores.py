import time
import matplotlib.ticker as ticker
import powerflow
import powerflow.plotting as plt


simulations = {
    'ybusjacobi': {
        'alg': powerflow.ybusjacobi,
        'name': 'Y\\textsubscript{BUS} Jacobi',
        'filename': 'parallel_numcores_ybusjacobi'
        },
    'ybusgaussseidel': {
        'alg': powerflow.ybusgaussseidel,
        'name': 'Y\\textsubscript{BUS} Gauss-Seidel',
        'filename': 'parallel_numcores_ybusgaussseidel'
        },
    'ybusnewton': {
        'alg': powerflow.ybusnewton,
        'name': 'Y\\textsubscript{BUS} Newton',
        'filename': 'parallel_numcores_ybusnewton'
        },
    'zbusjacobi': {
        'alg': powerflow.zbusjacobi,
        'name': 'Z\\textsubscript{BUS} Jacobi',
        'filename': 'parallel_numcores_zbusjacobi'
        }, 
}


nodes = 200
loads = 1000
methods = ['ybusnewton']
# methods = ['zbusjacobi']


# Warmup Run:
grid = powerflow.grids.radial(nodes)
S = powerflow.loads.beta(grid, maxload=10000, numberofloads=loads)
for method in methods:
    U, iters, runtime = simulations[method]['alg'](grid, S)


all_runtimes = []
for method in methods:
    runtimes = []

    for n in [1,2,3,4]:
        starttime = time.time()
        U, iters, runtime = simulations[method]['alg'](grid, S, num_processes=n)

        mt_runtime = time.time() - starttime
        runtimes.append(mt_runtime)
        print(f'Runtime {simulations[method]["name"]} using {n} Processes: {mt_runtime * 1000:.3f}ms - {mt_runtime/runtimes[0]*100:.1f}%')
    all_runtimes.append(runtimes)
    

## Plotting:
fig, ax = plt.subplots()
plt.setsize(fig,0.75)
for method, runtimes in zip(methods, all_runtimes):
    # ax.plot([1,2,3,4],[r / runtimes[0] for r in runtimes], 'o-', label=simulations[method]['name'], clip_on=False, zorder=10)
    ax.plot([1,2,3,4],[r for r in runtimes], 'o-', label=simulations[method]['name'], clip_on=False, zorder=10)
# ax.set_yscale('log')
ax.set_ylim(0,None)
ax.grid(True)
ax.set_ylabel('Runtime / s')
# ax.set_ylabel('Runtime Factor')
ax.set_xlabel('Number of Processes')
ax.xaxis.set_major_locator(ticker.FixedLocator((1,2,3,4)))
legend = plt.legend(facecolor='lightgray', edgecolor='k')
legend.get_frame().set_linewidth(0.0)
plt.tight_layout()
plt.save(fig, simulations[method]['filename'])
plt.show()
