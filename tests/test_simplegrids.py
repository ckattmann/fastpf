import pytest
import fastpf


@pytest.fixture
def build_feeder3():
    return fastpf.testgrids.feeder(3)


@pytest.fixture
def build_meshed10():
    return fastpf.testgrids.meshed(10)


@pytest.mark.parametrize(
    "pf_func",
    [fastpf.ybusjacobi, fastpf.ybusgaussseidel, fastpf.ybusnewton, fastpf.zbusjacobi],
)
def test_zero_load(pf_func):
    grid = fastpf.testgrids.feeder(3)
    S = fastpf.testloads.fixed(grid, load_W=0, number_of_scenarios=1)
    U, all_converged, iters, runtime = pf_func(grid, S)
    assert U.size == 3
    assert U[0, 0] == 400, f"with {pf_func.__name__}"
    assert abs(U[0, 1]) == 400, f"with {pf_func.__name__}"
    assert abs(U[0, 2]) == 400, f"with {pf_func.__name__}"


@pytest.mark.parametrize(
    "pf_func",
    [fastpf.ybusjacobi, fastpf.ybusgaussseidel, fastpf.ybusnewton, fastpf.zbusjacobi],
)
def test_low_load(pf_func, build_feeder3):
    grid = build_feeder3
    S = fastpf.testloads.fixed(grid, load_W=10, number_of_scenarios=1)
    U, all_converged, iters, runtime = pf_func(grid, S)
    assert U.size == 3
    assert U[0, 0] == 400
    assert abs(U[0, 1]) > 399.9, f"with {pf_func.__name__}"
    assert abs(U[0, 1]) < 400, f"with {pf_func.__name__}"
    assert abs(U[0, 2]) > 399.9, f"with {pf_func.__name__}"
    assert abs(U[0, 2]) < 400, f"with {pf_func.__name__}"


@pytest.mark.parametrize(
    "pf_func",
    [fastpf.ybusjacobi, fastpf.ybusgaussseidel, fastpf.ybusnewton, fastpf.zbusjacobi],
)
def test_medium_load(pf_func, build_feeder3):
    grid = build_feeder3
    S = fastpf.testloads.fixed(grid, load_W=10000, number_of_scenarios=1)
    U, all_converged, iters, runtime = pf_func(grid, S)
    assert U.size == 3, f"with {pf_func.__name__}"
    assert U[0, 0] == 400, f"with {pf_func.__name__}"
    assert abs(U[0, 1]) > abs(U[0, 2]), f"with {pf_func.__name__}"

    # assert abs(U[0, 1]) > 399.77, f"with {pf_func.__name__}"
    # assert abs(U[0, 1]) < 399.78, f"with {pf_func.__name__}"
    # assert abs(U[0, 2]) > 399.66, f"with {pf_func.__name__}"
    # assert abs(U[0, 2]) < 399.67, f"with {pf_func.__name__}"


@pytest.mark.parametrize(
    "pf_func",
    [fastpf.ybusjacobi, fastpf.ybusgaussseidel, fastpf.ybusnewton, fastpf.zbusjacobi],
)
def test_meshed(pf_func):
    grid = fastpf.testgrids.meshed(10)
    S = fastpf.testloads.fixed(grid, load_W=1000, number_of_scenarios=1)

    U, all_converged, iters, runtime = fastpf.ybusjacobi(grid, S)
    assert all_converged
    assert U.size == 10
    assert U[0, 0] == 400
    for i in range(1, 10):
        assert abs(U[0, i]) < 400
        assert abs(U[0, i]) > 399


# def test_allclose():
#     grid = fastpf.testgrids.feeder(10)
#     S = fastpf.testestds.random(grid, maxload=1000, number_of_loads=1000)
