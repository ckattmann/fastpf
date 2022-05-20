import matplotlib.pyplot as plt


def plot_phasor_points(U):
    plt.plot(U.real, U.imag, ".")
    plt.savefig("U.png")


def plot_phasor_points_radial(U, save_as=None):
    plt.plot(np.abs(U), np.angle(U), ".")
    if save_as:
        plt.savefig(f"{save_as}.png")
