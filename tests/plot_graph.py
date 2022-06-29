import fastpf

fastpf.set_loglevel("debug")
grid = fastpf.testgrids.meshed(40)
fastpf.plot_grid(grid, shape="force")
