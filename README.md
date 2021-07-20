# fastpf
Performance-first Power Flow implemented in Python with AOT Compilation

## Installation
fastpf should work best with Anaconda3 5.0.0+  
The parallelization requires Pathos and the MKL  
The AOT compilation requires Numba 0.47+

## Minimal example
```python
import powerflow
grid = powerflow.grids.feeder(10)
S = powerflow.loads.random(grid)
U = powerflow.zbusjacobi(grid, S)
```
