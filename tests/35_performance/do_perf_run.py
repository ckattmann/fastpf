import os
import json
import subprocess
import pprint
import numpy as np

import powerflow.plotting as plt

dyn = False


def record_performance_singlefeeder(methodnumber, nodes, loads):
    perfstats = {}
    events = [
        "task-clock",
        "cycles",
        "instructions",
        "branches",
        "branch-misses",
        "duration_time",
    ]
    events_string = "-e " + ",".join(events)
    flags = '-d -d -d -D 1000 -x " | "'
    if dyn:
        py_filename = "perftest_dyn.py"
    else:
        py_filename = "perftest.py"
    output = subprocess.run(
        f"perf stat {events_string} {flags} -o perf.txt python {py_filename} {methodnumber} {nodes} {loads}",
        shell=True,
        capture_output=True,
    )
    o = output.stdout.decode("utf-8")
    runtime, iters = o.strip().split("\n")
    # print(runtime, iters)
    perfstats["runtime"] = float(runtime)
    perfstats["iterations"] = int(iters)
    with open("perf.txt", "r") as f:
        for line in f:
            if not "|" in line:
                continue
            fields = [field.strip().replace(",", ".") for field in line.split("|")]
            if fields[0] == "<not supported>":
                continue
            perfstats[fields[2]] = float(fields[0])
    return perfstats


if __name__ == "__main__":
    all_methods = [
        "YBus\nJacobi",
        "YBus\nGauss-\nSeidel",
        "YBus\nRelaxation",
        "YBus\nNewton",
        "ZBus\nJacobi",
        "BFS",
    ]

    if dyn:
        filename = "all_perfstats_dyn.json"
    else:
        filename = "all_perfstats.json"

    if not os.path.exists(filename):
        all_perfstats = []
        for methodnumber, method in enumerate(all_methods, 1):
            all_perfstats.append(
                record_performance_singlefeeder(methodnumber, 20, 10000)
            )
            # print('-----------------------------')

        with open(filename, "w") as f:
            json.dump(all_perfstats, f)
    else:
        with open(filename, "r") as f:
            all_perfstats = json.load(f)

    # pprint.pprint(all_perfstats)

    def plot_metric(
        metric, ylabel="", unit="", filename="", stackmetric=[], log=True, show=False
    ):
        fig, ax = plt.subplots()
        plt.setsize(fig, 0.8)
        ax.set_axisbelow(True)
        ax.grid(
            True,
            axis="y",
            which="minor",
            linestyle="--",
            color="gainsboro",
            linewidth=0.5,
        )
        ax.grid(True, axis="y", which="major", linestyle="-", color="lightgray")
        if stackmetric:
            ax.bar(range(len(metric)), metric, tick_label=all_methods, color="red")
            ax.bar(
                range(len(stackmetric)),
                stackmetric,
                tick_label=all_methods,
                bottom=metric,
            )
        else:
            ax.bar(range(len(metric)), metric, tick_label=all_methods)

        if log:
            ax.set_yscale("log")

        for x, y in enumerate(metric, 0):
            ymin, ymax = ax.get_ylim()
            if y > 1e13:  # > 10T
                text = f"{y/1e12:.0f}T"
            elif y > 1e12:  # > 1T
                text = f"{y/1e12:.1f}T"
            elif y > 1e10:  # > 10B
                text = f"{y/1e9:.0f}B"
            elif y > 1e9:  # > 1B
                text = f"{y/1e9:.1f}B"
            elif y > 1e7:  # > 10M
                text = f"{y/1e6:.0f}M"
            elif y > 1e6:
                text = f"{y/1e6:.1f}M"
            elif y > 1e4:
                text = f"{y/1e3:.0f}k"
            elif y > 1e3:
                text = f"{y/1e3:.1f}k"
            elif y > 1e1:
                text = f"{y/1e0:.0f}"
            elif y > 1e0:
                text = f"{y/1e0:.1f}"
            else:
                text = f"{y:.3f}"

            if unit == "s":
                if y > 999:
                    text = f"{y/60:.0f}min"
                elif y > 10:
                    text = f"{y:.0f} s"
                elif y > 1:
                    text = f"{y:.1f} s"
                elif y > 1e-2:  # > 10 ms
                    text = f"{y/1e-3:.0f} ms"
                elif y > 1e-3:  # > 1 ms
                    text = f"{y/1e-3:.1f} ms"
                elif y > 1e-5:  # > 10 us
                    text = f"{y/1e-6:.0f} $\\upmu$s"
                elif y > 1e-6:  # > 1 us
                    text = f"{y/1e-6:.1f} $\\upmu$s"
                elif y > 1e-8:  # > 10 us
                    text = f"{y/1e-9:.0f} ns"
                elif y > 1e-9:  # > 1 us
                    text = f"{y/1e-9:.1f} ns"
            else:
                text += unit

            if log:
                if (np.log10(y) - np.log10(ymin)) / (
                    np.log10(ymax) - np.log10(ymin)
                ) > 0.2:
                    y -= 0.3 * y
                    va = "top"
                    color = "white"
                else:
                    va = "bottom"
                    color = "black"
            else:
                if (y - ymin) / (ymax - ymin) > 0.2:
                    y -= 0.05 * (ymax - ymin)
                    va = "top"
                    color = "white"
                else:
                    va = "bottom"
                    color = "black"
            ax.text(x, y, text, ha="center", va=va, color=color, fontsize=10)

        ax.tick_params(axis="x", labelsize=7)
        if ylabel:
            ax.set_ylabel(ylabel)
        plt.tight_layout()
        if filename:
            if dyn:
                suffix = "_dyn"
            else:
                suffix = ""
            plt.save(fig, filename + suffix)
        if show:
            plt.show()
        plt.close()

    runtimes = [stats["runtime"] for stats in all_perfstats]
    iterations = [stats["iterations"] for stats in all_perfstats]
    seconds_per_iteration = [
        stats["runtime"] / stats["iterations"] for stats in all_perfstats
    ]

    instructions = [stats["instructions"] for stats in all_perfstats]
    instructions_per_second = [
        stats["instructions"] / stats["runtime"] for stats in all_perfstats
    ]
    instructions_per_cycle = [
        stats["instructions"] / stats["cycles"] for stats in all_perfstats
    ]

    branches = [stats["branches"] for stats in all_perfstats]
    branch_misses = [stats["branch-misses"] for stats in all_perfstats]
    branches_per_instruction = [
        stats["branches"] / stats["instructions"] for stats in all_perfstats
    ]
    L1_cache_loads = [stats["L1-dcache-loads"] for stats in all_perfstats]
    L1_cache_load_per_instruction = [
        stats["L1-dcache-loads"] / stats["instructions"] for stats in all_perfstats
    ]
    LLC_cache_loads = [stats["LLC-loads"] for stats in all_perfstats]
    LLC_cache_misses = [stats["LLC-load-misses"] for stats in all_perfstats]

    branch_misses_percent = [
        stats["branch-misses"] / stats["branches"] * 100 for stats in all_perfstats
    ]
    L1_cache_misses_percent = [
        stats["L1-dcache-load-misses"] / stats["L1-dcache-loads"] * 100
        for stats in all_perfstats
    ]
    L1i_cache_misses_percent = [
        stats["L1-icache-load-misses"] / stats["L1-dcache-loads"] * 100
        for stats in all_perfstats
    ]
    LLC_cache_misses_percent = [
        stats["LLC-load-misses"] / stats["LLC-loads"] * 100 for stats in all_perfstats
    ]

    # TLB:
    dTLB_loads = [stats["dTLB-loads"] for stats in all_perfstats]
    dTLB_load_misses = [stats["dTLB-load-misses"] for stats in all_perfstats]
    dTLB_load_misses_percent = [
        stats["dTLB-load-misses"] / stats["dTLB-loads"] * 100 for stats in all_perfstats
    ]
    iTLB_loads = [stats["iTLB-loads"] for stats in all_perfstats]
    iTLB_load_misses = [stats["iTLB-load-misses"] for stats in all_perfstats]
    iTLB_load_misses_percent = [
        stats["iTLB-load-misses"] / stats["iTLB-loads"] * 100 for stats in all_perfstats
    ]
    iTLB_load_misses_per_instruction = [
        stats["instructions"] / stats["iTLB-load-misses"] for stats in all_perfstats
    ]

    if not dyn:
        pass
        # plot_metric([i/10000 for i in iterations], ylabel='Iterations per Load / -', filename='perf_iterations_per_load')
        # plot_metric(seconds_per_iteration, ylabel='Time per Iteration / s', filename='perf_seconds_per_iteration', unit='s')
        # plot_metric(runtimes, ylabel='Wall Time / s', filename='perf_walltime', unit='s', show=False)

        # plot_metric(instructions, ylabel='CPU Instructions / -', filename='perf_instructions', show=False)
        plot_metric(
            instructions_per_second,
            ylabel="Instructions per second / -",
            filename="perf_instructions_per_second",
            log=False,
            show=True,
        )

        # plot_metric(branches, ylabel='Branches / -', filename='perf_branches')
        # plot_metric(L1_cache_loads, ylabel='L1 Cache Loads / -', filename='perf_L1_cache_loads', log=True)
        # plot_metric([l/1e6 for l in LLC_cache_loads], ylabel='LLC Cache Loads / $10^6$', unit='M', filename='perf_LLC_cache_loads', log=False)
        # plot_metric(LLC_cache_misses, ylabel='LLC Cache Misses / \%', filename='perf_LLC_cache_misses', log=False)

        # plot_metric(iTLB_loads, ylabel='iTLB Cache Loads / -', filename='perf_iTLB_cache_loads', log=False, show=True)
        # plot_metric(iTLB_load_misses, ylabel='iTLB Cache Misses / -', filename='perf_iTLB_cache_load_misses', log=False, show=True)
        # plot_metric(iTLB_load_misses_percent, ylabel='iTLB Cache Misses / \%', filename='perf_iTLB_cache_misses_percent', log=False, show=True)
        plot_metric(
            iTLB_load_misses_per_instruction,
            ylabel="iTLB Cache Misses Per Instr. / ",
            filename="perf_iTLB_cache_misses_per_instruction",
            log=False,
            show=True,
        )

        # plot_metric(dTLB_loads, ylabel='dTLB Cache Loads / -', filename='perf_dTLB_cache_loads', log=False, show=True)
        # plot_metric(dTLB_load_misses, ylabel='dTLB Cache Load Misses / -', filename='perf_dTLB_cache_load_misses', log=False, show=True)
        # plot_metric(dTLB_load_misses_percent, ylabel='dTLB Cache Misses / \%', filename='perf_dTLB_cache_misses_percent', log=False, show=True)

    if dyn:
        pass
        # plot_metric(L1_cache_load_per_instruction, ylabel='L1 Cache Loads per Instr. / -', filename='perf_L1_cache_loads_per_instruction', log=False, show=False)
        # plot_metric(branches_per_instruction, ylabel='Branches per Instruction / -', filename='perf_branches_per_instruction', log=False)
        # plot_metric(instructions_per_cycle, ylabel='Instructions per Cycle / -', filename='perf_instructions_per_cycle', log=False)
        # plot_metric(branch_misses_percent, ylabel='Branch Misses / \%', filename='perf_branch_misses_percent', log=False)
        # plot_metric(L1_cache_misses_percent, ylabel='L1 Cache Misses / \%', filename='perf_L1_cache_misses_percent', log=False)
        # plot_metric(LLC_cache_misses_percent, ylabel='LLC Cache Misses / \%', filename='perf_LLC_cache_misses_percent', log=False)

    # # Stacked Plot doesnt work:
    # plot_metric(branch_misses, ylabel='Branch Misses / \%', filename='perf_branches_and_misses', stackmetric=branches, log=False, show=True)

    # # Not Used:
    # plot_metric(iterations, ylabel='Iterations / -', filename='perf_iterations')
