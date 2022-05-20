import fastpf

grid = fastpf.testgrids.radial(30)
S = fastpf.testloads.random_uniform(grid, max_load_W=1000, number_of_scenarios=10000)

fastpf.compare_methods(grid, S)
