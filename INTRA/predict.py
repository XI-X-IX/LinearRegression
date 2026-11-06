#!/usr/bin/env python3
# Programme de prediction : donne le prix d'une voiture selon son kilometrage

import math
import sys

THETAS_FILE = "thetas.txt"


def error_exit(message):
    '''Affiche une erreur et quitte proprement.'''
    print(f"Error: {message}", file=sys.stderr)
    sys.exit(1)


def estimate_price(mileage, theta0, theta1):
    return theta0 + theta1 * mileage


def load_thetas():
    '''
    1. Charger les thetas (0 et 0 si train.py n'a pas encore ete lance)
    '''
    try:
        with open(THETAS_FILE) as file:
            content = file.read()
    except FileNotFoundError:
        print(f"[warning] '{THETAS_FILE}' not found.")
        print("          Run train.py first to train the model.")
        print("          For now, theta0=0 and theta1=0.\n")
        return 0.0, 0.0
    except IsADirectoryError:
        error_exit(f"'{THETAS_FILE}' is a directory.")
    except PermissionError:
        error_exit(f"permission denied for '{THETAS_FILE}'.")
    except (OSError, UnicodeDecodeError):
        error_exit(f"cannot read '{THETAS_FILE}'.")

    parts = content.strip().split(",")
    if len(parts) != 2:
        error_exit(f"'{THETAS_FILE}' is invalid, run train.py again.")
    try:
        theta0 = float(parts[0])
        theta1 = float(parts[1])
    except ValueError:
        error_exit(f"'{THETAS_FILE}' is invalid, run train.py again.")
    if not math.isfinite(theta0) or not math.isfinite(theta1):
        error_exit(f"'{THETAS_FILE}' is invalid, run train.py again.")
    return theta0, theta1


def ask_mileage():
    '''
    2. Demander le kilometrage jusqu'a avoir un nombre valide
    '''
    while True:
        try:
            text = input("Mileage: ")
        except (EOFError, KeyboardInterrupt):
            print()
            sys.exit(0)
        try:
            mileage = float(text)
        except ValueError:
            print("Error: please enter a number.")
            continue
        if mileage < 0 or mileage != mileage or mileage == float("inf"):
            print("Error: please enter a positive number.")
            continue
        break

    return mileage


if __name__ == "__main__":
    theta0, theta1 = load_thetas()
    mileage = ask_mileage()

    # 3. Calculer et afficher le prix
    print("Estimated price:", estimate_price(mileage, theta0, theta1))
    