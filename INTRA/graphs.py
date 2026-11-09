#!/usr/bin/env python3
# Programme bonus : affiche les donnees et la droite de regression

import signal
import sys

from predict import load_thetas, estimate_price
from train import read_data, DATA_FILE

try:
    import matplotlib.pyplot as plt
except ImportError:
    print("Error: matplotlib is not installed.", file=sys.stderr)
    print("       Run: pip install --user matplotlib", file=sys.stderr)
    sys.exit(1)


def main():
    mileages, prices = read_data(DATA_FILE)
    theta0, theta1 = load_thetas()

    # un point par voiture
    plt.scatter(mileages, prices, color="blue", label="Data")

    # 2 points suffisent (km min et km max)
    x = [min(mileages), max(mileages)]
    y = [estimate_price(x[0], theta0, theta1),
         estimate_price(x[1], theta0, theta1)]
    plt.plot(x, y, color="red", label="Regression")

    plt.xlabel("Mileage (km)")
    plt.ylabel("Price")
    plt.title("ft_linear_regression")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # Ctrl+C Traceback
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    main()
