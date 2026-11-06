#!/usr/bin/env python3
# Programme de prediction : donne le prix d'une voiture selon son kilometrage

import os


def estimate_price(mileage, theta0, theta1):
    return theta0 + theta1 * mileage


def load_thetas():
    '''
    1. Charger les thetas (0 et 0 si train.py n'a pas encore ete lance
    '''
    theta0 = 0.0
    theta1 = 0.0
    if os.path.exists("thetas.txt"):
        file = open("thetas.txt")
        parts = file.read().split(",")
        file.close()
        theta0 = float(parts[0])
        theta1 = float(parts[1])
        return theta0, theta1
    else:
        print("[warning] '{}' not found.".format("thetas.txt"))
        print("          Run train.py first to train the model.")
        print("          For now, theta0=0 and theta1=0.\n")
        return 0.0, 0.0


def ask_mileage():
    '''
    2. Demander le kilometrage jusqu'a avoir un nombre valide
    '''
    while True:
        try:
            text = input("Mileage: ")
        except (EOFError, KeyboardInterrupt):
            print()
            exit()
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
    