fastpf is a power flow library written in Python with a focus on maximum performance and a simple-to-use API. It was born from a PhD project that set out to find ways to solve the sort of large-scale power flow problems that occur in modern distribution grid plannung, often involving millions of individual scenarios in a comparatively simple grid.

## What fastpf is not
- **A solver for dynamic/EMC problems** - fastpf is made for steady-state computations and has no provisions for the kind of component models that are required for millisecond-scale simulations.
- **A power flow solver made for massive grids -** Although is certainly possible to compute problems with thousands of nodes with fastpf, it is not specifically optimized for it and will simply run out of memory if the space for the resulting admittance matrix can not be allocated.
- **An Optimal Power Flow solver** - Although you are welcome to implement the optimization part on top of fastpf

### Design
 The performance is achieved in three main ways:

- fastpf implements four power flow algorithms - YBus Jacobi, YBus Gauss-Seidel, YBus Newton-Raphson, and ZBus Jacobi. ZBus Jacobi is the fastest of these in almost all scenarios.
- The code is vectorized as much as possible - all performace-critical code uses numpy as much as possible and explicit loops as little as possible
- The performance critical part - the power flow methods themselves - can be compiled to a binary library using Numba. If the compilation fails, the _JIT_ compiler of numba can still achieve a nice speed up.
