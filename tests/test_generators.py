import pytest
import fastpf


@pytest.fixture
def build_feeder3():
    # Tiny grid: Slack - PQ Node - Slack
    return fastpf.mockgrids.feeder(3, num_slacks=2)


@pytest.fixture
def build_meshed10():
    return fastpf.mockgrids.meshed(10)


# @pytest.mark.parametrize("pf_func", [fastpf.ybusnewton, fastpf.zbusjacobi])
# @pytest.mark.parametrize("pf_func", [fastpf.ybusnewton])
# def test_multiple_slacks_zero_load(pf_func, build_feeder3):
#     grid = build_feeder3
#     S = fastpf.mockloads.fixed(grid, load=0, num_loads=1)
#     U, iters, runtime = pf_func(grid, S)
#     assert U.size == len(grid["nodes"])
#     assert U[0, 0] == 400
#     assert abs(U[0, 1]) == 400
#     assert abs(U[0, 2]) == 401


# @pytest.mark.parametrize("pf_func", [fastpf.ybusnewton, fastpf.zbusjacobi])
# def test_multiple_slacks_low_load(pf_func, build_feeder3):
#     grid = fastpf.mockgrids.feeder(3, num_slacks=2)
#     S = fastpf.mockloads.fixed(grid, load=10, num_loads=1)
#     U, iters, runtime = pf_func(grid, S)
#     assert U.size == len(grid["nodes"])
#     assert U[0, 0] == 400
#     assert abs(U[0, 1]) < 400
#     assert abs(U[0, 2]) == 400


# @pytest.mark.parametrize("pf_func", [fastpf.ybusnewton, fastpf.zbusjacobi])
# def test_multiple_slacks_medium_load(pf_func):
#     grid = fastpf.mockgrids.feeder(3, num_slacks=2)
#     S = fastpf.mockloads.fixed(grid, load=10000, num_loads=1)
#     U, iters, runtime = pf_func(grid, S)
#     print(U)
#     assert U.size == len(grid["nodes"])
#     assert U[0, 0] == 400
#     assert abs(U[0, 1]) < 400
#     assert abs(U[0, 2]) == 400


# @pytest.mark.parametrize("pf_func", [fastpf.ybusnewton, fastpf.zbusjacobi])
# def test_multiple_slacks_10nodes(pf_func):
#     grid = fastpf.mockgrids.feeder(10, num_slacks=2)
#     S = fastpf.mockloads.fixed(grid, load=10000, num_loads=1)
#     U, iters, runtime = pf_func(grid, S)
#     print(U)
#     assert U.size == len(grid["nodes"])
#     assert U[0, 0] == 400
#     assert abs(U[0, 1]) < 400
#     assert abs(U[0, 9]) == 400
