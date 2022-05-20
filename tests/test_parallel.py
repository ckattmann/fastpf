import pytest
import fastpf
import numpy as np


@pytest.mark.parametrize(
    "pf_func",
    [fastpf.ybusjacobi, fastpf.ybusgaussseidel, fastpf.ybusnewton, fastpf.zbusjacobi],
)
@pytest.mark.parametrize(
    "load_W",
    [0, 10, 10000],
)
def test_fixed_load_parallel(pf_func, load_W):
    grid = fastpf.testgrids.feeder(3)
    S = fastpf.testloads.fixed(grid, load_W=load_W, number_of_scenarios=1)

    U, all_converged, iters, runtime = pf_func(grid, S, num_processes=1)
    assert all_converged
    assert U.size == 3
    assert U[0, 0] == 400, f"with {pf_func.__name__}"

    U2, all_converged, iters, runtime = pf_func(grid, S, num_processes=2)
    assert all_converged
    assert U2.size == 3
    assert U2[0, 0] == 400, f"with {pf_func.__name__}"

    U4, all_converged, iters, runtime = pf_func(grid, S, num_processes=4)
    assert all_converged
    assert U4.size == 3
    assert U4[0, 0] == 400, f"with {pf_func.__name__}"

    assert np.allclose(U, U2, U4)
