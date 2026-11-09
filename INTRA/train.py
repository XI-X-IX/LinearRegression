#!/usr/bin/env python3
# Programme d'entrainement : trouve theta0 et theta1 a partir de data.csv

import math
import sys

DATA_FILE = "data.csv"
THETAS_FILE = "thetas.txt"
LEARNING_RATE = 0.5
ITERATIONS = 10000


def error_exit(message):
    '''Affiche une erreur et quitte proprement.'''
    print(f"Error: {message}", file=sys.stderr)
    sys.exit(1)


def estimate_price(mileage, theta0, theta1):
    return theta0 + theta1 * mileage


def read_data(filename):
    '''
    Lit le fichier CSV (on saute la premiere ligne "km,price").
    Renvoie deux listes : les kilometrages et les prix.
    '''
    try:
        with open(filename) as file:
            lines = file.read().split("\n")
    except FileNotFoundError:
        error_exit(f"'{filename}' not found.")
    except IsADirectoryError:
        error_exit(f"'{filename}' is a directory.")
    except PermissionError:
        error_exit(f"permission denied for '{filename}'.")
    except (OSError, UnicodeDecodeError):
        error_exit(f"cannot read '{filename}'.")

    mileages = []
    prices = []
    for line in lines[1:]:
        if line.strip() == "":
            continue
        parts = line.split(",")
        if len(parts) != 2:
            error_exit("expected 'km,price'.")
        try:
            mileage = float(parts[0])
            price = float(parts[1])
        except ValueError:
            error_exit("values must be numbers.")
        if not (math.isfinite(mileage) and math.isfinite(price)):
            error_exit("values must be finite.")
        if mileage < 0 or price < 0:
            error_exit("values must be positive.")
        mileages.append(mileage)
        prices.append(price)

    if len(mileages) == 0:
        error_exit(f"no data in '{filename}'.")
    return mileages, prices


def normalize(mileages):
    '''
    Met les kilometrages entre 0 et 1 (sinon l'apprentissage explose).
    Renvoie la liste normalisee et le kilometrage max.
    '''
    max_mileage = max(mileages)
    if max_mileage == 0:
        error_exit("all mileages are 0, cannot train.")
    small_mileages = []
    for mileage in mileages:
        small_mileages.append(mileage / max_mileage)
    return small_mileages, max_mileage


def gradient_descent(mileages, prices, learning_rate, iterations):
    '''Descente de gradient avec les formules du sujet.'''
    m = len(mileages)
    theta0 = 0.0
    theta1 = 0.0

    for _ in range(iterations):
        sum0 = 0.0
        sum1 = 0.0
        for i in range(m):
            error = estimate_price(mileages[i], theta0, theta1) - prices[i]
            sum0 = sum0 + error
            sum1 = sum1 + error * mileages[i]

        tmp_theta0 = learning_rate * sum0 / m
        tmp_theta1 = learning_rate * sum1 / m

        # mise a jour des deux thetas en meme temps
        theta0 = theta0 - tmp_theta0
        theta1 = theta1 - tmp_theta1

    return theta0, theta1


def save_thetas(filename, theta0, theta1):
    '''Sauvegarde les thetas dans le fichier.'''
    try:
        with open(filename, "w") as file:
            file.write(f"{theta0},{theta1}")
    except IsADirectoryError:
        error_exit(f"'{filename}' is a directory.")
    except PermissionError:
        error_exit(f"permission denied for '{filename}'.")
    except OSError:
        error_exit(f"cannot write '{filename}'.")


def main():
    mileages, prices = read_data(DATA_FILE)
    small_mileages, max_mileage = normalize(mileages)

    theta0, theta1 = gradient_descent(small_mileages, prices,
                                      LEARNING_RATE, ITERATIONS)

    # On a appris avec mileage / max_mileage, donc on divise theta1
    # par max_mileage pour qu'il marche avec les vrais kilometrages.
    theta1 = theta1 / max_mileage

    if not math.isfinite(theta0) or not math.isfinite(theta1):
        error_exit("training failed (values too large).")

    save_thetas(THETAS_FILE, theta0, theta1)

    print("theta0 =", theta0)
    print("theta1 =", theta1)
    print(f"Thetas saved in {THETAS_FILE}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(1)
