import os
import numpy as np
import powerflow.plotting as plt


def plot_voltage_comparison(U, U_org, filename=None):
    # Plot a comparison of minimal voltage at all 1440 times:
    import matplotlib.ticker as ticker

    linewidth = 0.5

    minU = np.min(abs(U), axis=1)
    minUorg = np.min(abs(U_org), axis=1)
    fig, (ax1, ax2) = plt.subplots(2, 1)
    plt.setsize(fig, size=1.5)
    ax1.set_xlim([0, 1440])
    ax1.plot(minU, linewidth=linewidth, label="Reduced Grid")
    ax1.plot(minUorg, linewidth=linewidth, label="Original Grid")
    ax1.set_ylabel("Min. abs. Voltage / V")
    ax1.set_xlabel("Time of day / hh:mm")
    ax1.xaxis.set_major_locator(
        ticker.FixedLocator([t * 60 for t in (0, 6, 12, 18, 24)])
    )
    ax1.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{x//60}:00"))
    ax1.legend()
    ax1.grid()

    ax2.plot(abs(minU - minUorg), linewidth=linewidth)
    ax2.set_xlim([0, 1440])
    ax2.set_ylabel("Difference / V")
    ax2.set_xlabel("Time of day / hh:mm")
    ax2.xaxis.set_major_locator(
        ticker.FixedLocator([t * 60 for t in (0, 6, 12, 18, 24)])
    )
    ax2.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{x//60}:00"))
    ax2.grid()
    plt.tight_layout()
    if filename:
        # plt.savefig(filename + ".png")
        # plt.savefig(os.path.join("home", "kipfer", "diss", "images", f"{filename}.eps"))
        plt.savefig(f"{filename}.eps")
        # plt.save(fig, filename + ".eps")
    plt.show()


if __name__ == "__main__":
    n = 50
    U_red = np.load(f"U_red_{n}.npy")
    U_org = np.load(f"U_org_{n}.npy")

    plot_voltage_comparison(U_red, U_org, f"european_lv_feeder_voltages_red_to_{n}")
