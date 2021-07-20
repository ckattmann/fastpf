import json
import time
import copy
import functools

import numpy as np
import numba

# import matplotlib.pyplot as plt
import matplotlib.colors
import matplotlib.cm

# import seaborn
from pathos.multiprocessing import ProcessingPool as Pool

# import powerflow
# import powerflow.plotting as plt
import matplotlib.pyplot as plt


def noop_decorator(*args, **kwargs):
    def decorator(func):
        return func

    return decorator


def noop(*args, **kwargs):
    return None


# plt.plot = plt.grid = plt.show = noop
# numba.jit = noop_decorator

H0_base = np.array(
    [
        70.0,
        62.4,
        55.8,
        50.6,
        47.1,
        44.9,
        43.5,
        42.5,
        41.6,
        40.8,
        39.9,
        39.2,
        38.5,
        38.0,
        37.7,
        37.8,
        38.2,
        38.9,
        40.0,
        41.2,
        42.8,
        45.1,
        49.1,
        55.3,
        64.3,
        74.6,
        84.9,
        93.9,
        100.2,
        104.3,
        107.1,
        109.4,
        111.8,
        114.1,
        116.2,
        117.5,
        117.9,
        117.6,
        116.7,
        115.4,
        114.2,
        113.2,
        112.4,
        112.1,
        112.4,
        113.7,
        115.7,
        118.9,
        122.9,
        127.1,
        130.4,
        131.7,
        130.2,
        126.6,
        121.9,
        116.8,
        112.3,
        108.4,
        105.0,
        102.0,
        99.3,
        97.0,
        95.2,
        93.9,
        93.0,
        92.6,
        93.0,
        93.9,
        95.4,
        97.6,
        100.5,
        104.0,
        108.1,
        112.7,
        117.7,
        122.9,
        128.1,
        132.9,
        136.5,
        138.4,
        138.2,
        136.5,
        134.1,
        131.7,
        129.9,
        128.5,
        127.2,
        125.6,
        123.2,
        120.0,
        115.6,
        110.1,
        103.2,
        95.3,
        86.9,
        78.3,
    ],
    dtype=np.float32,
)

PV_base = np.array(
    [
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0.001362314483783,
        0.007948583113342,
        0.017125322103546,
        0.028803474844718,
        0.042869746159648,
        0.059187700116883,
        0.077599082074454,
        0.097925352143648,
        0.119969415213763,
        0.143517530774196,
        0.16834138402795,
        0.194200298227799,
        0.22084356679789,
        0.248012882642887,
        0.275444841105002,
        0.302873492315651,
        0.33003291820981,
        0.35665980923221,
        0.382496015767457,
        0.407291049571109,
        0.430804510963219,
        0.452808418265203,
        0.473089416908128,
        0.491450846806512,
        0.507714647965076,
        0.52172308585341,
        0.533340279829936,
        0.542453519804965,
        0.548974358384575,
        0.552839467912611,
        0.554011254106194,
        0.55247822033869,
        0.548255079040319,
        0.541382609136921,
        0.531927260908292,
        0.519980512094953,
        0.50565798149257,
        0.489098308623176,
        0.470461810339002,
        0.44992892737623,
        0.427698475911331,
        0.403985721062139,
        0.379020291001144,
        0.353043951892954,
        0.326308265216419,
        0.299072150171633,
        0.271599374791725,
        0.244156000070182,
        0.217007801869724,
        0.190417695594065,
        0.164643188577082,
        0.139933884875379,
        0.116529066642415,
        0.094655375520331,
        0.074524616516432,
        0.056331705644441,
        0.040252781217565,
        0.026443497094666,
        0.015037514417716,
        0.006145206455392,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    ],
    dtype=np.float32,
)
PV_base *= 1.5


@functools.lru_cache()
# @numba.jit(nopython=True, cache=True)
def interpolate_H0(number_of_timesteps_per_day):
    t_base = np.arange(0, H0_base.size)
    t = np.linspace(0, H0_base.size, number_of_timesteps_per_day)
    H0 = np.interp(t, t_base, H0_base)
    return H0


@functools.lru_cache()
# @numba.jit(nopython=True, cache=True)
def interpolate_PV(number_of_timesteps_per_day):
    t_base = np.arange(0, PV_base.size)
    t = np.linspace(0, PV_base.size, number_of_timesteps_per_day)
    PV = np.interp(t, t_base, PV_base)
    return PV


# @numba.jit(nopython=True, cache=True)
def make_noise(number_of_timesteps, noise_magnitude=20, base_timesteps=24):
    """Generate a noise signal of correct size
    number_of_timesteps [int]   :   Desired size of the output array
    noise_magnitude [int]       :   Standard deviation of the normal distribution of the noise
    base_timesteps [int]        :   Number of 'real' noise timesteps, the rest is interpolated
    """
    t_base = np.arange(0, base_timesteps)
    noise_base_timesteps = np.random.normal(0, noise_magnitude, size=base_timesteps)
    noise_base_timesteps[-1] = noise_base_timesteps[
        0
    ]  # Make the noise 'smooth' at the edges
    t = np.linspace(0, base_timesteps, number_of_timesteps)
    noise = np.interp(t, t_base, noise_base_timesteps)
    return noise


# @numba.jit(nopython=True, cache=True)
def simulate_ev(installed_ev_charging_power_kW, number_of_timesteps_per_day):
    """Create one ev charging profile with the given charging power"""
    # EV Charge Start Probability Distribution:
    t_start_minute = np.random.normal(
        1440 * 18 / 24, 120
    )  # Normal distribution, Mean 18:00, stddev 2h
    t_start_minute = t_start_minute % 1440  # If time > midnight, wrap around to morning

    # EV Charge Time Probability Distribution:
    driven_km = np.random.gamma(18, 2)  # Mean km / day: around 35
    ev_demand_kW = driven_km / 100 * 17  # 17 kWh / 100 km
    charging_duration_minutes = int(ev_demand_kW / installed_ev_charging_power_kW * 60)

    t_start = int(t_start_minute / 1440 * number_of_timesteps_per_day)
    t_stop = int(
        (t_start_minute + charging_duration_minutes)
        / 1440
        * number_of_timesteps_per_day
    )

    t_stop = min(number_of_timesteps_per_day, t_stop)  # Limit t_stop to end of day

    # EV Charging Curve:
    ev_charging = np.zeros(number_of_timesteps_per_day, dtype=np.float32)
    ev_charging[t_start:t_stop] = (
        np.ones(t_stop - t_start) * installed_ev_charging_power_kW * 1000
    )

    return ev_charging


# @numba.jit(nopython=True, cache=True)
def power_over_ndays(
    days,
    number_of_timesteps_per_day,
    installed_pv_load=0,
    load_multiplier=3.8 * 3,
    noise_magnitude=1000,
):

    H0_day = interpolate_H0(number_of_timesteps_per_day) * load_multiplier
    # H0 = np.zeros(days * number_of_timesteps_per_day, np.float32)
    # H0 = np.tile(H0_day, reps=days)
    H0 = np.zeros(days * number_of_timesteps_per_day)
    for i in np.arange(
        0, days * number_of_timesteps_per_day, number_of_timesteps_per_day
    ):
        H0[i : i + number_of_timesteps_per_day] = H0_day

    noise = make_noise(
        H0.size, noise_magnitude=noise_magnitude, base_timesteps=24 * days
    )
    noise = np.roll(noise, shift=np.random.randint(number_of_timesteps_per_day))

    load = H0 + noise
    load[load < 0] = 0

    total_load = load

    # PV Generation
    # =============
    if installed_pv_load:
        pv_generation_day = installed_pv_load * interpolate_PV(
            number_of_timesteps_per_day
        )
        # pv_generation = np.tile(pv_generation_day, reps=days)
        pv_generation = np.zeros(days * number_of_timesteps_per_day)
        for i in np.arange(
            0, days * number_of_timesteps_per_day, number_of_timesteps_per_day
        ):
            pv_generation[i : i + number_of_timesteps_per_day] = pv_generation_day
        total_load -= pv_generation

    return total_load


# @numba.jit(nopython=True, cache=True)
def ev_power_over_ndays(
    days, number_of_timesteps_per_day, installed_ev_charging_powers_kW=np.array([])
):

    # np.random.seed()

    ev_charging = np.zeros(days * number_of_timesteps_per_day, dtype=np.float32)
    if installed_ev_charging_powers_kW.size > 0:
        for charging_power_kW in installed_ev_charging_powers_kW:

            installed_ev_charging_power_kW = charging_power_kW

            t_starts_minute = (np.random.normal(1440 * 18 / 24, 120, size=days)).astype(
                np.int32
            )  # Normal distribution, Mean 18:00, stddev 2h
            t_starts_minute[t_starts_minute > 1440] = 1439

            t_starts_minute += np.arange(0, days * number_of_timesteps_per_day, 1440)
            # t_start_minute %= 1440  # If time > midnight, wrap around to morning

            # EV Charge Time Probability Distribution:
            driven_km = np.random.gamma(18, 2, size=days)  # Mean km / day: around 35
            ev_demands_kWh = driven_km / 100 * 17  # 17 kWh / 100 km
            charging_durations_minutes = (
                ev_demands_kWh / installed_ev_charging_power_kW * 60
            ).astype(np.int32)

            t_starts = (t_starts_minute / 1440 * number_of_timesteps_per_day).astype(
                np.int32
            )
            t_stops = (
                (t_starts_minute + charging_durations_minutes)
                / 1440
                * number_of_timesteps_per_day
            ).astype(np.int32)
            t_stops[-1] = min(t_stops[-1], days * number_of_timesteps_per_day)

            t_stops[t_stops - t_starts < 0] = t_starts[t_stops - t_starts < 0]
            # print(t_stops-t_starts)
            # EV Charging Curve:
            for t_start, t_stop in zip(t_starts, t_stops):
                ev_charging[t_start:t_stop] = (
                    np.ones(t_stop - t_start) * installed_ev_charging_power_kW * 1000
                )

    return ev_charging


# @numba.jit(nopython=True, cache=True)
def power_over_day(
    number_of_timesteps_per_day,
    installed_pv_load=0,
    installed_ev_charging_powers_kW=[],
    load_multiplier=3.8 * 3,
    noise_magnitude=1000,
):
    np.random.seed()
    H0 = interpolate_H0(number_of_timesteps_per_day) * load_multiplier

    noise = make_noise(number_of_timesteps_per_day, noise_magnitude=noise_magnitude)
    noise = np.roll(noise, shift=np.random.randint(number_of_timesteps_per_day))

    load = H0 + noise
    load[load < 0] = 0
    # load = np.clip(load, 0, None)

    total_load = load

    # PV Generation
    # =============
    if installed_pv_load:
        pv_generation = installed_pv_load * interpolate_PV(number_of_timesteps_per_day)
        total_load -= pv_generation

        # plt.plot(pv_generation)

    # EV Load
    # =======
    if installed_ev_charging_powers_kW.size > 0:
        for charging_power_kW in installed_ev_charging_powers_kW:
            ev_charging = simulate_ev(charging_power_kW, number_of_timesteps_per_day)
            total_load += ev_charging

        # plt.plot(ev_charging)

        # plt.plot(total_load)
        # plt.plot(load)
        # plt.grid(True)
        # plt.show()

    return total_load


def ndays(grid, number_of_days, number_of_timesteps_per_day):

    nodes = grid["nodes"]
    S = np.zeros(
        (number_of_days * number_of_timesteps_per_day, len(nodes)), dtype=np.complex128
    )

    for i, node in enumerate(nodes):
        installed_pv_load = 0
        ev_powers = []
        for el in node["elements"]:
            if el["type"] == "PV":
                installed_pv_load += el["kWp"] * 1000
            if el["type"] == "EV":
                ev_powers.append(el["kWcharge"])
        ev_powers = np.array(ev_powers)
        # S[:,i] = power_over_day(number_of_timesteps, installed_pv_load=installed_pv_load, installed_ev_charging_power_kW=installed_ev_load, noise_magnitude=1000)
        for start in range(
            0, number_of_days * number_of_timesteps_per_day, number_of_timesteps_per_day
        ):
            S[start : start + number_of_timesteps_per_day, i] = power_over_day(
                number_of_timesteps_per_day,
                installed_pv_load=installed_pv_load,
                installed_ev_charging_powers_kW=ev_powers,
                noise_magnitude=1000,
            )
    return S


def distribute_evs_in_grid(org_grid, number_of_evs, charging_power_kW):
    grid = copy.deepcopy(org_grid)
    nodes = grid["nodes"]
    load_node_ids = [n["id"] for n in nodes if n["is_slack"] == False]

    node_ids_with_evs = np.random.choice(
        load_node_ids, size=number_of_evs, replace=True
    )
    for ev_node_id in node_ids_with_evs:
        for node in nodes:
            if node["id"] == ev_node_id:
                ev_dict = {"type": "EV", "kWcharge": charging_power_kW}
                node["elements"].append(ev_dict)
    grid["nodes"] = nodes
    return grid


def voltages_evs(grid, days, timesteps_per_day, number_of_evs, charging_power_kW):
    grid = copy.deepcopy(grid)
    modified_grid = distribute_evs_in_grid(grid, number_of_evs, charging_power_kW)
    # with open('sonderbuch_evs.json','w') as f:
    #     json.dump(grid,f)

    starttime = time.time()
    S = ndays(modified_grid, days, timesteps_per_day)
    # print(f'Time for Data Simulation: {time.time() - starttime}')

    U, iters, runtime = powerflow.zbusjacobi(
        modified_grid, S, num_processes=4, verbose=False
    )

    return U, runtime


def plot_day_distribution(U, timesteps_per_day):
    U_day = U.reshape(timesteps_per_day, U.size // timesteps_per_day, order="F")
    U_24 = U_day.reshape(24, U_day.size // 24, order="C")
    seaborn.boxplot(data=np.real(U_24.T[:, :]), whis=3.0, fliersize=1.0)
    plt.grid(True)
    plt.show()


def analyse_voltages(U, verbose=False):
    U_under_360 = (np.abs(U) < 360).sum()
    U_under_360_percent = U_under_360 / U.size * 100

    if verbose:
        print(f"{U.shape[0]} timesteps at {U.shape[1]} nodes")
        print(f"\tMax. voltage {np.max(np.abs(U)):.2f} V")
        print(f"\tMean voltage {np.mean(np.abs(U)):.2f} V")
        print(f"\tMin. voltage {np.min(np.abs(U)):.2f} V")
        print(f"\t<380 V : {(np.abs(U)<380).sum() / U.size * 100:.3f}%")
        print(f"\t<360 V : {U_under_360_percent:.3f}%")

    return U_under_360


def simulation_run5(grid, ev_counts, ev_powers, reduce_to=None):
    """Lossy grid reduction + parallelized execution"""

    runs = 36
    days = 30
    timesteps_per_day = 1440

    number_of_power_flows = (
        runs * days * timesteps_per_day * len(ev_counts) * len(ev_powers)
    )

    grid = copy.deepcopy(grid)

    # Power without EVs:
    nodes = grid["nodes"]
    S = np.zeros((days * timesteps_per_day, len(nodes)), dtype=np.complex128)
    for i, node in enumerate(nodes):
        installed_pv_load = 0
        for el in node["elements"]:
            if el["type"] == "PV":
                installed_pv_load += el["kWp"] * 1000
        S[:, i] = power_over_ndays(
            days,
            timesteps_per_day,
            installed_pv_load=installed_pv_load,
            noise_magnitude=1000,
        )

    # plt.plotgraph(grid, shape='force')
    if reduce_to:
        grid, S, nodes_reduced = powerflow.grid_reduction.lossy.reduce_to_n_nodes(
            grid, S, reduce_to
        )
    # plt.plotgraph(grid, shape='force')

    func = lambda parameters: check_ev_voltage_fails_process(
        grid, S, days, timesteps_per_day, runs, parameters
    )

    parameter_list = []
    for charging_power_kW in ev_powers:
        fail_percentages = []
        for number_of_evs in ev_counts:
            parameter_list.append((number_of_evs, charging_power_kW))

    with Pool(4) as pool:
        fail_percentages = pool.map(func, parameter_list)

    # for parameters in parameter_list:
    #     fail_percentages = func(parameters)

    # print(fail_percentages)

    return fail_percentages


def simulation_run6(
    grid,
    ev_counts,
    ev_powers,
    runs,
    days,
    timesteps_per_day,
    S_limit,
    num_processes,
    num_pf_processes=1,
):
    """Weak Load Detection & Parallelization"""

    grid = copy.deepcopy(grid)

    # Power without EVs:
    nodes = grid["nodes"]
    S = np.zeros((days * timesteps_per_day, len(nodes)), dtype=np.complex128)
    for i, node in enumerate(nodes):
        installed_pv_load = 0
        for el in node["elements"]:
            if el["type"] == "PV":
                installed_pv_load += el["kWp"] * 1000
        S[:, i] = power_over_ndays(
            days,
            timesteps_per_day,
            installed_pv_load=installed_pv_load,
            noise_magnitude=1000,
        )

    func = lambda parameters: check_ev_voltage_fails_process2(
        grid, S, days, timesteps_per_day, runs, S_limit, num_pf_processes, parameters
    )

    parameter_list = []
    for charging_power_kW in ev_powers:
        fail_percentages = []
        for number_of_evs in ev_counts:
            parameter_list.append((number_of_evs, charging_power_kW))

    if num_processes == 1:
        fail_percentages = list(map(func, parameter_list))

    elif num_processes > 1:
        with Pool(num_processes) as pool:
            fail_percentages = pool.map(func, parameter_list)

    # for parameters in parameter_list:
    #     fail_percentages = func(parameters)

    # print(fail_percentages)

    return fail_percentages


def construct_S(grid, days, timesteps_per_day, number_of_evs=0, charging_power_kW=11):

    grid_plus_evs = distribute_evs_in_grid(grid, number_of_evs, charging_power_kW)

    S = np.zeros(
        (days * timesteps_per_day, len(grid_plus_evs["nodes"])), dtype=np.complex128
    )

    for i, node in enumerate(grid_plus_evs["nodes"]):
        installed_pv_load = 0
        for el in node["elements"]:
            if el["type"] == "PV":
                installed_pv_load += el["kWp"] * 1000
        S[:, i] = power_over_ndays(
            days,
            timesteps_per_day,
            installed_pv_load=installed_pv_load,
            noise_magnitude=1000,
        )

    for i, node in enumerate(grid_plus_evs["nodes"]):
        ev_powers = []
        for el in node["elements"]:
            if el["type"] == "EV":
                ev_powers.append(el["kWcharge"])
        S[:, i] += ev_power_over_ndays(
            days, timesteps_per_day, installed_ev_charging_powers_kW=np.array(ev_powers)
        )

    return S


def check_ev_voltage_fails_process2(
    grid, S, days, timesteps_per_day, runs, S_limit, num_pf_processes, parameters
):

    number_of_evs, charging_power_kW = parameters

    total_runtime = 0
    total_pfs = 0
    fails = 0
    S_ev = np.zeros_like(S)

    np.random.seed(number_of_evs * charging_power_kW)

    for run in range(runs):

        # Build S_ev according to EV count and power:
        mod_grid = copy.deepcopy(grid)
        modified_grid = distribute_evs_in_grid(
            mod_grid, number_of_evs, charging_power_kW
        )
        for i, node in enumerate(modified_grid["nodes"]):
            ev_powers = []
            for el in node["elements"]:
                if el["type"] == "EV":
                    ev_powers.append(el["kWcharge"])
            ev = ev_power_over_ndays(
                days,
                timesteps_per_day,
                installed_ev_charging_powers_kW=np.array(ev_powers),
            )
            S_ev[:, i] = S[:, i] + ev

        # Prune S for weak-load situations:
        if S_limit:
            S_rest = S_ev[np.max(np.abs(S_ev), axis=1) > S_limit, :]
        else:
            S_rest = S_ev
        total_pfs += S_rest.shape[0]

        # PF:
        # U, iters, runtime = powerflow.zbusjacobi(modified_grid, S_rest, num_processes=num_pf_processes, verbose=False)
        # U, iters, runtime = powerflow.ybusnewton(modified_grid, S_rest, num_processes=1, verbose=False)
        # print(f'Mean iters: {np.mean(iters)}')

        ## 1 Iteration only:
        starttime = time.time()
        grid_parameters = powerflow.calc_grid_parameters.calc_grid_parameters(
            grid, S_rest
        )
        Yred = grid_parameters["Yred"]
        Zred = np.linalg.inv(Yred)
        slack_index = grid_parameters["slack_index"]
        S_rest = np.delete(S_rest, [slack_index], axis=1)
        U = powerflow.powerflow_methods_cc.zbusjacobi_2iter(Zred, S_rest, 400)
        runtime = time.time() - starttime

        # plot_day_distribution(U_dev.T, timesteps_per_day)

        # if U.size == 0:
        #     U_fails = 0
        # else:
        #     U_fails = (np.min(np.abs(U), axis=0) < 360).sum()

        fails += U_fails
        total_runtime += runtime

    fails_per_year = fails / (runs * days) * 365
    print(
        f"{number_of_evs:2d} cars @ {charging_power_kW:6.1f} kW: {fails_per_year:6.1f} Fails per Year [{runs*days*timesteps_per_day:7d} -> {total_pfs:7d} PFs ({total_pfs/(runs*days*timesteps_per_day)*100:3.0f}%) in {total_runtime:7.3f} s]"
    )

    return fails_per_year


if __name__ == "__main__":

    starttime = time.time()

    np.random.seed(90827349)

    with open("sonderbuch.json", "r") as f:
        grid = json.load(f)

    ev_counts = [10, 20, 30, 40, 50, 60, 70]
    ev_powers = [11, 22, 43, 80, 100]
    # ev_counts = [10,50]
    # ev_powers = [43,100]

    days = 120
    runs = int(360 / days * 10)
    # runs = 1
    timesteps_per_day = 1440

    number_of_power_flows = (
        runs * days * timesteps_per_day * len(ev_counts) * len(ev_powers)
    )
    print(f"Number of Power Flows: {number_of_power_flows}")

    # Experiment 1: No optimizations:
    # ===============================
    # label = 'noopt'
    # S_limit = None
    # num_processes = 1
    # num_pf_processes = 1
    # fails_per_year_list = simulation_run6(grid, ev_counts, ev_powers, runs, days, timesteps_per_day, S_limit, num_processes, num_pf_processes)

    # Experiment 2.1: Parallelization inside of power flow:
    # ===================================================
    # label = 'parpf'
    # S_limit = None
    # num_processes = 1
    # num_pf_processes = 4
    # fails_per_year_list = simulation_run6(grid, ev_counts, ev_powers, runs, days, timesteps_per_day, S_limit, num_processes, num_pf_processes)

    # Experiment 2.2: Parallelization over everything:
    # ==============================================
    # label = 'par'
    # S_limit = None
    # num_processes = 4
    # num_pf_processes = 1
    # fails_per_year_list = simulation_run6(grid, ev_counts, ev_powers, runs, days, timesteps_per_day, S_limit, num_processes, num_pf_processes)

    # Experiment 3.1: Weak Load Detection with no Parallelization:
    # ============================================================
    # label = 'wld_max_nopar'
    # S_limit = 6000
    # num_processes = 1
    # num_pf_processes = 1
    # fails_per_year_list = simulation_run6(grid, ev_counts, ev_powers, runs, days, timesteps_per_day, S_limit, num_processes, num_pf_processes)

    # Experiment 3.2: Weak Load Detection with Parallelization over everything:
    # =======================================================================
    # label = 'wld_max_par'
    # S_limit = 6000
    # num_processes = 4
    # num_pf_processes = 1
    # fails_per_year_list = simulation_run6(grid, ev_counts, ev_powers, runs, days, timesteps_per_day, S_limit, num_processes, num_pf_processes)

    # Experiment 4.1: Test of MatMul-PowerFlow
    # ========================================
    # label = 'MM_nopar'
    # S_limit = None
    # num_processes = 1
    # num_pf_processes = 1
    # fails_per_year_list = simulation_run6(grid, ev_counts, ev_powers, runs, days, timesteps_per_day, S_limit, num_processes, num_pf_processes)

    # Experiment 4.2: Test of MatMul-PowerFlow parallelized
    # =====================================================
    label = "MM_par"
    # S_limit = None
    # num_processes = 4
    # num_pf_processes = 1
    # fails_per_year_list = simulation_run6(
    #     grid,
    #     ev_counts,
    #     ev_powers,
    #     runs,
    #     days,
    #     timesteps_per_day,
    #     S_limit,
    #     num_processes,
    #     num_pf_processes,
    # )

    # result = {'ev_powers': ev_powers, 'ev_counts': ev_counts, 'S_limit':S_limit, 'fails_per_year_list':fails_per_year_list, 'label':label, 'number_of_power_flows': number_of_power_flows}
    # with open(f'result_{label}.json','w') as f:
    #     json.dump(result, f)
    label = "wld_par"
    with open(f"result_{label}.json") as f:
        result = json.load(f)
    globals().update(result)

    # fails_per_year_list, number_of_power_flows = simulation_run5(grid, ev_counts, ev_powers, reduce_to=30)

    runtime = time.time() - starttime
    time_string = f"{int(runtime // 60)}m {int(runtime%60)}s"
    print(
        f"Total Runtime : {runtime:.3f} ({time_string}) s, Total Power Flows: {number_of_power_flows} -> {number_of_power_flows/runtime:.0f} PF/s"
    )

    fig, ax = plt.subplots()
    plt.setsize(fig, size=1)
    for i, ev_power in enumerate(ev_powers):
        fails_for_power = []
        for _ in ev_counts:
            fails_for_power.append(fails_per_year_list.pop(0))
        color = matplotlib.cm.jet(0.5 + (i / len(ev_powers) / 2))
        color = matplotlib.cm.hsv_r(0.6 + (i / len(ev_powers) / 2))
        color = matplotlib.cm.rainbow(0.5 + (i / len(ev_powers) / 2))
        # print(0.5 + (i / len(ev_powers) / 2), color)
        ax.plot(
            ev_counts,
            fails_for_power,
            "-o",
            clip_on=False,
            zorder=10,
            color=color,
            label=f"{ev_power} kW",
        )
    ax.set_xlabel("Number of electric vehicles")
    ax.set_ylabel("Voltage band violations per year")
    # ax.set_xlim([0,None])
    ax.set_ylim([0, 400])
    ax.grid()
    legend = ax.legend(frameon=1)
    legend.get_frame().set_facecolor("lightgray")
    legend.get_frame().set_linewidth(0)
    # legend.get_frame().set_alpha(0)

    plotname = f"ev_influence_{label}_{S_limit}B"
    plt.savefig(fig, plotname)
    plt.show()

    # ============================================
    # Plot typical day curve:
    # days = 1
    # number_of_timesteps_per_day = 1440
    # installed_pv_load = 12000
    # installed_ev_charging_powers_kW = np.array([11])
    # S_ev = power_over_ndays(days, number_of_timesteps_per_day, installed_pv_load, load_multiplier=3.8*3, noise_magnitude=0)
    # ev_power = ev_power_over_ndays(days, number_of_timesteps_per_day, installed_ev_charging_powers_kW)
    # plt.plot(S_ev + ev_power)
    # plt.show()
    # ============================================

    # simulation_run3(grid)
    # fails, number_of_power_flows = simulation_run4(grid, ev_counts, ev_powers)

    # print([fail * 365

# def power_over_ndays(days, number_of_timesteps_per_day, installed_pv_load=0, installed_ev_charging_powers_kW=[], load_multiplier=3.8*3, noise_magnitude=1000):
# simulation_run2(grid)
# starttime = time.time()
# S = power_over_ndays(365, 1440, installed_pv_load=10000,  noise_magnitude=1000)
# S_ev = S + ev_power_over_ndays(365, 1440, installed_ev_charging_powers_kW=np.array([11]))
# print(time.time() - starttime)
# plt.plot(S)
# plt.grid(True)
# plt.show()


# plot_day_distribution(U, timesteps_per_day)

# U, runtime = voltages_evs(grid, days=30, timesteps_per_day=1440, number_of_evs=20, charging_power_kW=11)
# analyse_voltages(U, verbose=True)
# plt.plot(np.abs(U))
# plt.grid(True)
# plt.show()
