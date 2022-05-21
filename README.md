# fastpf
Performance-first Power Flow implemented in Python with AOT Compilation

## Installation

### Quickstart
```bash
sudo apt install python3.8 python3.8-venv python3.8-dev
mkdir fastpf
python3.8 -m venv .venv
source .venv/bin/activate
pip install fastpf
```

### Requirements
The Ahead-of-time compilation feature of numba requires python3.x-dev, which can be installed on Linux with 
```bash
sudo apt|yum|pacman|... install python3.x-dev
```
where x is the version of python used. 
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
