import time
import functools
import concurrent.futures
import numpy as np
import matplotlib.ticker as ticker

import mkl

import powerflow
import powerflow.plotting as plt

numberofloads = 10000


grid = powerflow.mockgrids.feeder(20)
S = powerflow.mockloads.beta(grid, maxload=5000, numberofloads=numberofloads)


all_runtimes = []
# for func in [powerflow.ybusjacobi, powerflow.ybusgaussseidel, powerflow.ybusnewton, powerflow.zbusjacobi]:
# for func in [powerflow.ybusjacobi, powerflow.ybusgaussseidel]:
# for func in [powerflow.ybusnewton, powerflow.zbusjacobi]:
# for func in [powerflow.ybusgaussseidel, powerflow.zbusjacobi]:
for func in [powerflow.ybusgaussseidel]:
# for func in [powerflow.zbusjacobi]:
    mkl.set_num_threads(2)
    pf = functools.partial(func, grid, verbose=True)
    runtimes = []

    ## Single-threaded version:
    starttime = time.time()
    U, iters, runtime = pf(S)
    st_runtime = time.time() - starttime
    runtimes.append(st_runtime)
    print(f'Runtime ST: {st_runtime * 1000:.3f}ms')

    mkl.set_num_threads(1)

    for n in [2,3,4]:
        starttime = time.time()
        # U, iters, runtime = powerflow.ybusgaussseidel(grid, S, processes=2)
        borders = np.linspace(0,numberofloads, n+1, dtype=np.int)
        S_chunks = [S[borders[i]:borders[i+1],:] for i in range(n)]
        print([s.shape[0] for s in S_chunks])

        with concurrent.futures.ProcessPoolExecutor() as executor:
            all_results = executor.map(pf, S_chunks)
        # U, iters, _ = (result_set for result_set in zip(*all_results))
        # U = np.vstack(U)
        # iters = np.hstack(iters)
        # runtimes = list(runtimes)

        mt_runtime = time.time() - starttime
        runtimes.append(mt_runtime)
        print(f'Runtime MT using {n} Processes: {mt_runtime * 1000:.3f}ms - {mt_runtime/st_runtime*100:.1f}%')
    all_runtimes.append(runtimes)

# method_names = ['Y\\textsubscript{BUS} Jacobi', 'Y\\textsubscript{BUS} Gauss-Seidel', 'Y\\textsubscript{BUS} Newton', 'Z\\textsubscript{BUS} Jacobi']
method_names = ['Y\\textsubscript{BUS} Newton', 'Z\\textsubscript{BUS} Jacobi']

fig, ax = plt.subplots()
plt.setsize(fig,0.75)
for method_name, runtimes in zip(method_names,all_runtimes):
    # ax.plot([1,2,3,4],runtimes, 'o-', label=method_name, clip_on=False)
    ax.plot([1,2,3,4],[r / runtimes[0] for r in runtimes], 'o-', label=method_name, clip_on=False, zorder=10)
    # ax2.plot([r / runtimes[0] for r in runtimes], 'o-')
# ax.set_yscale('log')
ax.set_ylim(0,None)
ax.grid(True)
# ax.set_ylabel('Runtime / s')
ax.set_ylabel('Runtime Factor')
ax.set_xlabel('Number of Processes')
ax.xaxis.set_major_locator(ticker.FixedLocator((1,2,3,4)))
legend = plt.legend(facecolor='lightgray', edgecolor='k')
legend.get_frame().set_linewidth(0.0)
plt.tight_layout()
plt.save(fig, 'parallel_gaussseidel')
plt.show()

