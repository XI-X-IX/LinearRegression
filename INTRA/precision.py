#!/usr/bin/env python3
# Programme de precision : calcule le R2 du modele

import sys

from predict import load_thetas, estimate_price
from train import read_data, DATA_FILE


def r_squared(mileages, prices, theta0, theta1):
    mean_price = sum(prices) / len(prices)
    ss_res = 0.0
    ss_tot = 0.0
    for i in range(len(prices)):
        prediction = estimate_price(mileages[i], theta0, theta1)
        ss_res += (prices[i] - prediction) ** 2
        ss_tot += (prices[i] - mean_price) ** 2
    if ss_tot == 0:
        print("Error: all prices are the same, R2 is undefined.", file=sys.stderr)
        sys.exit(1)
    return 1 - ss_res / ss_tot


if __name__ == "__main__":
    mileages, prices = read_data(DATA_FILE)
    theta0, theta1 = load_thetas()
    print(f"R2 = {r_squared(mileages, prices, theta0, theta1):.4f}")
