#!/usr/bin/env python3
# =============================================================================
#  train.py  -  Training program (ft_linear_regression)
# =============================================================================
#
#  This program reads a CSV file (km, price) and finds two numbers,
#  theta0 and theta1, to guess the price of a car from its mileage.
#
#  The formula is:
#      estimated_price = theta0 + theta1 * km
#
#  To find good thetas, we use "gradient descent":
#  we start with theta0 = 0 and theta1 = 0, and we change them a little
#  at each step, until the estimated prices are close to the real prices.
#
# =============================================================================

import json     # to save theta0 and theta1 in a file
import math     # for the square root (sqrt)


# -----------------------------------------------------------------------------
# STEP 1 : Read the data
# -----------------------------------------------------------------------------

def read_data(file_name):
    """
    Opens the CSV file and returns two lists:
      - kms    : list of mileages
      - prices : list of prices
    """
    kms = []
    prices = []

    file = open(file_name)
    lines = file.readlines()
    file.close()

    # The first line is "km,price", so we skip it
    for line in lines[1:]:
        line = line.strip()
        if line == "":
            continue
        parts = line.split(",")
        kms.append(float(parts[0]))
        prices.append(float(parts[1]))

    if len(kms) == 0:
        print("Error : the file is empty !")
        exit(1)

    return kms, prices


# -----------------------------------------------------------------------------
# STEP 2 : Normalize the data
# -----------------------------------------------------------------------------
#
# Problem : the mileages are very big (ex: 150 000) and the prices are
# smaller (ex: 6 000). If we do gradient descent with these numbers,
# theta1 goes crazy or learns very slowly.
#
# Solution : we "normalize" each list, so it has a mean of 0 and a
# standard deviation of 1. After the training, we "denormalize" to get
# the real theta0 and theta1.
#
# Formula : normalized_value = (value - mean) / std

def get_mean(values):
    total = 0
    for v in values:
        total = total + v
    return total / len(values)


def get_std(values, mean):
    # Variance = mean of the squared differences
    total = 0
    for v in values:
        total = total + (v - mean) ** 2
    variance = total / len(values)
    return math.sqrt(variance)


def normalize(values):
    """
    Returns (normalized_values, mean, std).
    We keep the mean and the std to denormalize later.
    """
    mean = get_mean(values)
    std = get_std(values, mean)

    if std == 0:
        std = 1  # to not divide by zero

    normalized = []
    for v in values:
        normalized.append((v - mean) / std)

    return normalized, mean, std


# -----------------------------------------------------------------------------
# STEP 3 : The hypothesis function
# -----------------------------------------------------------------------------
#
# This is the main formula of the project:
#     estimated_price = theta0 + theta1 * km

def estimate_price(km, theta0, theta1):
    return theta0 + theta1 * km


# -----------------------------------------------------------------------------
# STEP 4 : Gradient descent
# -----------------------------------------------------------------------------
#
# Goal : find theta0 and theta1 with the smallest errors.
#
# We repeat "iterations" times:
#   1. We compute the error for each car (estimated_price - real_price)
#   2. We compute how much to change theta0 and theta1
#   3. We change theta0 and theta1 AT THE SAME TIME (with tmp0 and tmp1)
#
# Formulas from the subject:
#   tmp0 = learning_rate * (1/m) * sum(estimate_price(km[i]) - price[i])
#   tmp1 = learning_rate * (1/m) * sum((estimate_price(km[i]) - price[i]) * km[i])
#   theta0 = theta0 - tmp0
#   theta1 = theta1 - tmp1

def gradient_descent(kms, prices, learning_rate, iterations):
    """
    Trains the model on normalized data.
    Returns (theta0, theta1, cost_history).
    """
    theta0 = 0.0
    theta1 = 0.0
    m = len(kms)   # m = number of cars (lines in the CSV)

    cost_history = []

    for i in range(iterations):

        # --- Compute the errors for each car ---
        sum_errors = 0
        sum_errors_km = 0
        sum_squared_errors = 0

        for j in range(m):
            error = estimate_price(kms[j], theta0, theta1) - prices[j]
            sum_errors = sum_errors + error
            sum_errors_km = sum_errors_km + error * kms[j]
            sum_squared_errors = sum_squared_errors + error ** 2

        # --- Compute the changes (formulas from the subject) ---
        tmp0 = learning_rate * (1 / m) * sum_errors
        tmp1 = learning_rate * (1 / m) * sum_errors_km

        # --- Change the thetas AT THE SAME TIME ---
        theta0 = theta0 - tmp0
        theta1 = theta1 - tmp1

        # --- Save the cost (global error) for the graph ---
        cost = sum_squared_errors / (2 * m)
        cost_history.append(cost)

    return theta0, theta1, cost_history


# -----------------------------------------------------------------------------
# STEP 5 : Denormalize the thetas
# -----------------------------------------------------------------------------
#
# We found theta0_n and theta1_n with normalized data.
# We must convert them so they work with real km and real prices.
#
# Math:
#   price_norm = theta0_n + theta1_n * km_norm
#   (price - price_mean) / price_std = theta0_n + theta1_n * (km - km_mean) / km_std
#
# So:
#   theta1 = price_std * theta1_n / km_std
#   theta0 = price_mean + price_std * theta0_n - theta1 * km_mean

def denormalize_thetas(theta0_n, theta1_n, km_mean, km_std, price_mean, price_std):
    theta1 = price_std * theta1_n / km_std
    theta0 = price_mean + price_std * theta0_n - theta1 * km_mean
    return theta0, theta1


# -----------------------------------------------------------------------------
# BONUS : Precision of the model
# -----------------------------------------------------------------------------
#
# We check how good our model is with three numbers:
#
#  - MAE  (Mean Absolute Error) : the mean error, always positive
#          "on average, we are wrong by X CHF"
#
#  - RMSE (Root Mean Squared Error) : big errors count more
#          useful to see when we are really wrong
#
#  - R2   (Coefficient of determination) : between 0 and 1
#          1 = perfect, 0 = bad model (same as always guessing the mean)

def get_precision(kms, prices, theta0, theta1):
    m = len(kms)
    price_mean = get_mean(prices)

    sum_abs_errors = 0
    sum_squared_errors = 0
    sum_squared_diff_mean = 0

    for i in range(m):
        prediction = estimate_price(kms[i], theta0, theta1)
        error = prediction - prices[i]
        sum_abs_errors = sum_abs_errors + abs(error)
        sum_squared_errors = sum_squared_errors + error ** 2
        sum_squared_diff_mean = sum_squared_diff_mean + (prices[i] - price_mean) ** 2

    mae = sum_abs_errors / m
    rmse = math.sqrt(sum_squared_errors / m)

    if sum_squared_diff_mean == 0:
        r2 = 0.0
    else:
        r2 = 1 - sum_squared_errors / sum_squared_diff_mean

    return mae, rmse, r2


# -----------------------------------------------------------------------------
# BONUS : Show the graphs
# -----------------------------------------------------------------------------

def show_graphs(kms, prices, theta0, theta1, cost_history, mae, rmse, r2):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[info] matplotlib is not installed. No graphs.")
        print("       To install it : pip install matplotlib")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("ft_linear_regression")

    # --- Graph 1 : the points + the regression line ---
    ax1.scatter(kms, prices, color="blue", label="Real data")

    # We draw the line from the smallest km to the biggest km
    x_min = min(kms)
    x_max = max(kms)
    y_min = estimate_price(x_min, theta0, theta1)
    y_max = estimate_price(x_max, theta0, theta1)
    ax1.plot([x_min, x_max], [y_min, y_max], color="red", label="Regression line")

    ax1.set_xlabel("Mileage (km)")
    ax1.set_ylabel("Price (CHF)")
    ax1.set_title("Data and regression line")
    ax1.legend()
    ax1.grid(True)

    # Show the precision numbers in a box
    text = "MAE  = " + str(round(mae)) + " CHF\n"
    text = text + "RMSE = " + str(round(rmse)) + " CHF\n"
    text = text + "R2   = " + str(round(r2, 4))
    ax1.text(0.97, 0.97, text, transform=ax1.transAxes,
             verticalalignment="top", horizontalalignment="right",
             bbox=dict(facecolor="lightyellow"))

    # --- Graph 2 : the cost during the training ---
    # We only show the first 50 iterations to see the descent
    zoom = 50
    ax2.plot(range(zoom), cost_history[:zoom], color="green")
    ax2.set_yscale("log")   # log scale to see the descent better
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Cost (log scale)")
    ax2.set_title("Gradient descent - cost (first 50 iterations)")
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig("regression_plot.png")
    print("[graph] Saved in regression_plot.png")
    plt.show()


# -----------------------------------------------------------------------------
# MAIN PROGRAM
# -----------------------------------------------------------------------------

def main():
    print("==================================================")
    print("   ft_linear_regression  -  Training")
    print("==================================================")

    # 1. Read the file
    print()
    print("[1/5] Reading the data...")
    kms, prices = read_data("data.csv")
    print("      Number of cars :", len(kms))

    # 2. Normalize
    print("[2/5] Normalizing the data...")
    kms_norm, km_mean, km_std = normalize(kms)
    prices_norm, price_mean, price_std = normalize(prices)

    # 3. Gradient descent on the normalized data
    learning_rate = 0.1
    iterations = 10000
    print("[3/5] Gradient descent (" + str(iterations) + " iterations, learning rate = " + str(learning_rate) + ")...")
    theta0_n, theta1_n, cost_history = gradient_descent(kms_norm, prices_norm, learning_rate, iterations)

    # 4. Denormalize the thetas
    print("[4/5] Denormalizing the thetas...")
    theta0, theta1 = denormalize_thetas(theta0_n, theta1_n, km_mean, km_std, price_mean, price_std)
    print("      theta0 =", theta0)
    print("      theta1 =", theta1)

    # 5. Save in thetas.json
    file = open("thetas.json", "w")
    json.dump({"theta0": theta0, "theta1": theta1}, file, indent=2)
    file.close()
    print("[5/5] Thetas saved in thetas.json")

    # --- BONUS : precision ---
    mae, rmse, r2 = get_precision(kms, prices, theta0, theta1)
    print()
    print("----------------------------------------")
    print("  Precision :")
    print("    MAE  :", round(mae, 2), "CHF  (mean absolute error)")
    print("    RMSE :", round(rmse, 2), "CHF  (root mean squared error)")
    print("    R2   :", round(r2, 4), "      (1 = perfect, 0 = bad)")
    print("----------------------------------------")

    # --- BONUS : graphs ---
    show_graphs(kms, prices, theta0, theta1, cost_history, mae, rmse, r2)


# Python starts here
if __name__ == "__main__":
    main()
