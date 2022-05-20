import pytest
import fastpf


@pytest.mark.parametrize("pf_func", [fastpf.ybusnewton, fastpf.zbusjacobi])
def test_multiple_slacks_zero_load(pf_func):
    grid = fastpf.testgrids.feeder(3, number_of_slacks=2)
    S = fastpf.testloads.fixed(grid, load_W=0, number_of_scenarios=1)
    U, all_converged, iters, runtime = pf_func(grid, S)
    assert all_converged
    assert U.size == len(grid["nodes"])
    assert U[0, 0] == 400
    assert abs(U[0, 1]) == 400
    assert abs(U[0, 2]) == 400


@pytest.mark.parametrize("pf_func", [fastpf.ybusnewton, fastpf.zbusjacobi])
def test_multiple_slacks_low_load(pf_func):
    grid = fastpf.testgrids.feeder(3, number_of_slacks=2)
    S = fastpf.testloads.fixed(grid, load_W=10, number_of_scenarios=1)
    U, all_converged, iters, runtime = pf_func(grid, S)
    assert all_converged
    assert U.size == len(grid["nodes"])
    assert U[0, 0] == 400
    assert abs(U[0, 1]) < 400
    assert abs(U[0, 2]) == 400


@pytest.mark.parametrize("pf_func", [fastpf.ybusnewton, fastpf.zbusjacobi])
def test_multiple_slacks_medium_load(pf_func):
    grid = fastpf.testgrids.feeder(3, number_of_slacks=2)
    S = fastpf.testloads.fixed(grid, load_W=10000, number_of_scenarios=1)
    U, all_converged, iters, runtime = pf_func(grid, S)
    assert all_converged
    assert U.size == len(grid["nodes"])
    assert U[0, 0] == 400
    assert abs(U[0, 1]) < 400
    assert abs(U[0, 2]) == 400


@pytest.mark.parametrize("pf_func", [fastpf.ybusnewton, fastpf.zbusjacobi])
def test_multiple_slacks_10nodes(pf_func):
    grid = fastpf.testgrids.feeder(10, number_of_slacks=2)
    S = fastpf.testloads.fixed(grid, load_W=10000, number_of_scenarios=1)
    U, all_converged, iters, runtime = pf_func(grid, S)
    assert all_converged
    assert U.size == len(grid["nodes"])
    assert U[0, 0] == 400
    assert abs(U[0, 1]) < 400
    assert abs(U[0, 9]) == 400
