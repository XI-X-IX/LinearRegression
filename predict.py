#!/usr/bin/env python3
# =============================================================================
#  predict.py  -  Prediction program (ft_linear_regression)
# =============================================================================
#
#  This program asks the user for a mileage and shows
#  the estimated price for this car.
#
#  It uses theta0 and theta1 saved by train.py in thetas.json.
#  If thetas.json does not exist, the two thetas are 0 (price = 0).
#
# =============================================================================

import json   # to read the file thetas.json
import os     # to check if the file exists


# -----------------------------------------------------------------------------
# Load theta0 and theta1
# -----------------------------------------------------------------------------

def load_thetas(file_name):
    """
    Reads thetas.json and returns (theta0, theta1).
    If the file does not exist, returns (0, 0) and shows a warning.
    """
    if not os.path.exists(file_name):
        print("[warning] " + file_name + " not found.")
        print("          Run train.py first to train the model.")
        print("          For now, theta0 = 0 and theta1 = 0.")
        print()
        return 0.0, 0.0

    file = open(file_name)
    data = json.load(file)
    file.close()

    theta0 = data["theta0"]
    theta1 = data["theta1"]
    return theta0, theta1


# -----------------------------------------------------------------------------
# Hypothesis : the prediction formula
# -----------------------------------------------------------------------------
#
# This is the same formula as in train.py:
#     estimated_price = theta0 + theta1 * km

def estimate_price(km, theta0, theta1):
    return theta0 + theta1 * km


# -----------------------------------------------------------------------------
# Ask the user for the mileage
# -----------------------------------------------------------------------------

def ask_mileage():
    """
    Asks again and again until the user gives a good positive number.
    Returns the mileage as a float.
    """
    while True:
        text = input("Enter the mileage (km) : ")

        # Check that it is a number
        try:
            km = float(text)
        except ValueError:
            print("  [error] Please enter a number (ex: 50000).")
            print()
            continue

        # Check that it is positive
        if km < 0:
            print("  [error] The mileage can not be negative.")
            print()
            continue

        return km


# -----------------------------------------------------------------------------
# MAIN PROGRAM
# -----------------------------------------------------------------------------

def main():
    print("==================================================")
    print("   ft_linear_regression  -  Prediction")
    print("==================================================")
    print()

    # 1. Load the trained thetas
    theta0, theta1 = load_thetas("thetas.json")

    # 2. Ask for the mileage
    km = ask_mileage()

    # 3. Compute the estimated price
    price = estimate_price(km, theta0, theta1)

    # A price can not be negative
    if price < 0:
        price = 0.0

    # 4. Show the result
    print()
    print("  Mileage         :", km, "km")
    print("  Estimated price :", round(price, 2), "CHF")


# Python starts here
if __name__ == "__main__":
    main()
