#!/usr/bin/env python3
# =============================================================================
#  train.py  -  Programme d'entrainement (ft_linear_regression)
# =============================================================================
#
#  Ce programme lit un fichier CSV (km, price) et cherche les deux valeurs
#  theta0 et theta1 qui permettent de predire le prix d'une voiture
#  a partir de son kilometrage.
#
#  La formule utilisee est :
#      estimated_price = theta0 + theta1 * km
#
#  Pour trouver les bons thetas, on utilise la "descente de gradient" :
#  on part de theta0 = 0 et theta1 = 0, et on les ajuste petit a petit
#  jusqu'a ce que l'estimation soit la plus proche possible des vrais prix.
#
# =============================================================================

import csv      # pour lire le fichier CSV
import json     # pour sauvegarder theta0 et theta1
import math     # pour la racine carree (sqrt)


# -----------------------------------------------------------------------------
# ETAPE 1 : Lire les donnees
# -----------------------------------------------------------------------------

def read_dataset(path):
    """
    Ouvre le fichier CSV et retourne deux listes :
      - mileages : liste des kilometrages
      - prices   : liste des prix correspondants
    """
    mileages = []
    prices = []

    with open(path, newline="") as file:
        reader = csv.DictReader(file)   # lit les colonnes par leur nom
        for row in reader:
            mileages.append(float(row["km"]))
            prices.append(float(row["price"]))

    if len(mileages) == 0:
        print("Error: the file is empty!")
        exit(1)

    return mileages, prices


# -----------------------------------------------------------------------------
# ETAPE 2 : Normaliser les donnees
# -----------------------------------------------------------------------------
#
# Probleme : les kilometrages sont dans les centaines de milliers (ex: 150 000)
# et les prix dans les milliers (ex: 6 000). Si on fait la descente de gradient
# directement, les valeurs sont trop grandes et theta1 diverge ou converge
# tres lentement.
#
# Solution : on "normalise" chaque liste pour qu'elle ait une moyenne de 0
# et un ecart-type de 1. Apres l'entrainement, on "denormalise" pour retrouver
# les vraies valeurs de theta0 et theta1.
#
# Formule : normalized_value = (value - mean) / std

def compute_mean(values):
    return sum(values) / len(values)

def compute_std(values, mean):
    # Variance = moyenne des carres des ecarts
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    return math.sqrt(variance)

def normalize(values):
    """
    Retourne (valeurs_normalisees, moyenne, ecart_type).
    On garde la moyenne et l'ecart_type pour pouvoir denormaliser ensuite.
    """
    mean = compute_mean(values)
    std = compute_std(values, mean)

    if std == 0:
        std = 1  # evite la division par zero

    normalized = [(v - mean) / std for v in values]
    return normalized, mean, std


# -----------------------------------------------------------------------------
# ETAPE 3 : La fonction hypothesis (hypothese)
# -----------------------------------------------------------------------------
#
# C'est la formule centrale du projet :
#     estimated_price = theta0 + theta1 * km

def estimate_price(km, theta0, theta1):
    return theta0 + (theta1 * km)


# -----------------------------------------------------------------------------
# ETAPE 4 : La descente de gradient
# -----------------------------------------------------------------------------
#
# Objectif : trouver theta0 et theta1 qui minimisent les erreurs.
#
# On repete "iterations" fois :
#   1. On calcule l'erreur de chaque point (estimated_price - real_price)
#   2. On calcule comment ajuster theta0 et theta1 pour reduire l'erreur
#   3. On met a jour theta0 et theta1 SIMULTANEMENT (via tmp0 et tmp1)
#
# Formules exactes du sujet :
#   tmp0 = learning_rate * (1/m) * somme(estimate_price(km[i]) - price[i])
#   tmp1 = learning_rate * (1/m) * somme((estimate_price(km[i]) - price[i]) * km[i])
#   theta0 = theta0 - tmp0
#   theta1 = theta1 - tmp1

def gradient_descent(kms_norm, prices_norm, learning_rate, iterations):
    """
    Entraine le modele sur des donnees normalisees.
    Retourne (theta0, theta1, historique_du_cout).
    """
    theta0 = 0.0
    theta1 = 0.0
    m = len(kms_norm)   # m = nombre d'exemples (lignes du CSV)

    cost_history = []

    for i in range(iterations):

        # --- Calcul des erreurs pour chaque voiture ---
        errors = []
        for j in range(m):
            estimate = estimate_price(kms_norm[j], theta0, theta1)
            error = estimate - prices_norm[j]
            errors.append(error)

        # --- Calcul des ajustements (formules du sujet) ---
        tmp0 = learning_rate * (1 / m) * sum(errors)
        tmp1 = learning_rate * (1 / m) * sum(errors[j] * kms_norm[j] for j in range(m))

        # --- Mise a jour SIMULTANEE des thetas ---
        theta0 = theta0 - tmp0
        theta1 = theta1 - tmp1

        # --- On enregistre le cout (erreur globale) pour le graphique ---
        cost = sum(e ** 2 for e in errors) / (2 * m)
        cost_history.append(cost)

    return theta0, theta1, cost_history


# -----------------------------------------------------------------------------
# ETAPE 5 : Denormaliser les thetas
# -----------------------------------------------------------------------------
#
# On a trouve theta0_n et theta1_n sur des donnees normalisees.
# Il faut les convertir pour qu'ils fonctionnent sur des km reels et des prix reels.
#
# Mathematiquement :
#   price_norm = theta0_n + theta1_n * km_norm
#   (price - price_mean) / price_std = theta0_n + theta1_n * (km - km_mean) / km_std
#
# En developpant :
#   theta1_real = price_std * theta1_n / km_std
#   theta0_real = price_mean + price_std * theta0_n - theta1_real * km_mean

def denormalize_thetas(theta0_n, theta1_n, km_mean, km_std, price_mean, price_std):
    theta1_real = price_std * theta1_n / km_std
    theta0_real = price_mean + price_std * theta0_n - theta1_real * km_mean
    return theta0_real, theta1_real


# -----------------------------------------------------------------------------
# BONUS : Calcul de la precision du modele
# -----------------------------------------------------------------------------
#
# On mesure a quel point notre modele est bon avec trois metriques :
#
#  - MAE  (Mean Absolute Error) : erreur moyenne en valeur absolue
#          "en moyenne, on se trompe de X CHF"
#
#  - RMSE (Root Mean Squared Error) : penalise plus les grosses erreurs
#          utile pour detecter les cas ou on se plante vraiment
#
#  - R2   (Coefficient de determination) : entre 0 et 1
#          1 = prediction parfaite, 0 = modele nul (predit toujours la moyenne)

def compute_accuracy(real_kms, real_prices, theta0, theta1):
    m = len(real_kms)
    predictions = [estimate_price(real_kms[i], theta0, theta1) for i in range(m)]
    errors      = [predictions[i] - real_prices[i] for i in range(m)]

    mae  = sum(abs(e) for e in errors) / m
    rmse = math.sqrt(sum(e ** 2 for e in errors) / m)

    price_mean = compute_mean(real_prices)
    ss_res     = sum((real_prices[i] - predictions[i]) ** 2 for i in range(m))
    ss_tot     = sum((real_prices[i] - price_mean)     ** 2 for i in range(m))

    if ss_tot == 0:
        r2 = 0.0
    else:
        r2 = 1 - ss_res / ss_tot

    return mae, rmse, r2


# -----------------------------------------------------------------------------
# BONUS : Affichage des graphiques
# -----------------------------------------------------------------------------

def show_plots(real_kms, real_prices, theta0, theta1, cost_history, mae, rmse, r2):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[info] matplotlib is not installed. No plots.")
        print("       To install it: pip install matplotlib")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("ft_linear_regression", fontsize=14, fontweight="bold")

    # --- Graphique 1 : nuage de points + droite de regression ---
    ax1.scatter(real_kms, real_prices,
                color="#4C72B0", alpha=0.85,
                edgecolors="white", linewidths=0.6,
                zorder=3, label="Actual data")

    # On dessine la droite entre le min et le max du kilometrage
    x_min = min(real_kms)
    x_max = max(real_kms)
    y_min = estimate_price(x_min, theta0, theta1)
    y_max = estimate_price(x_max, theta0, theta1)
    ax1.plot([x_min, x_max], [y_min, y_max],
             color="#DD4444", linewidth=2, label="Regression line")

    ax1.set_xlabel("Mileage (km)")
    ax1.set_ylabel("Price (CHF)")
    ax1.set_title("Data distribution and regression line")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Affiche les metriques dans un encadre
    info = "MAE  = {:,.0f} CHF\nRMSE = {:,.0f} CHF\nR2   = {:.4f}".format(mae, rmse, r2)
    ax1.text(0.97, 0.97, info,
             transform=ax1.transAxes,
             fontsize=9,
             verticalalignment="top", horizontalalignment="right",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", alpha=0.8))

    # --- Graphique 2 : evolution du cout pendant l'entrainement ---
    # On affiche seulement les 50 premieres iterations pour voir la descente
    zoom = min(50, len(cost_history))
    ax2.plot(range(zoom), cost_history[:zoom], color="#2ca02c", linewidth=1.5)
    ax2.set_yscale("log")   # echelle logarithmique pour mieux voir la descente
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Cost (log scale)")
    ax2.set_title("Gradient descent - cost (first 50 iterations)")
    ax2.grid(True, alpha=0.3, which="both")

    plt.tight_layout()
    plt.savefig("regression_plot.png", dpi=150)
    print("[plot] Saved to regression_plot.png")
    plt.show()


# -----------------------------------------------------------------------------
# PROGRAMME PRINCIPAL
# -----------------------------------------------------------------------------

def main():
    print("=" * 50)
    print("   ft_linear_regression  -  Training")
    print("=" * 50)

    # 1. Lecture du fichier
    print("\n[1/5] Reading dataset...")
    real_kms, real_prices = read_dataset("data.csv")
    print("      {} cars loaded.".format(len(real_kms)))

    # 2. Normalisation
    print("[2/5] Normalizing data...")
    kms_norm,    km_mean,    km_std    = normalize(real_kms)
    prices_norm, price_mean, price_std = normalize(real_prices)

    # 3. Descente de gradient sur les donnees normalisees
    learning_rate = 0.1
    iterations    = 10000
    print("[3/5] Gradient descent ({} iterations, lr={})...".format(iterations, learning_rate))
    theta0_n, theta1_n, cost_history = gradient_descent(
        kms_norm, prices_norm, learning_rate, iterations
    )

    # 4. Denormalisation des thetas
    print("[4/5] Denormalizing thetas...")
    theta0, theta1 = denormalize_thetas(
        theta0_n, theta1_n, km_mean, km_std, price_mean, price_std
    )
    print("      theta0 = {:.4f}".format(theta0))
    print("      theta1 = {:.6f}".format(theta1))

    # 5. Sauvegarde dans thetas.json
    with open("thetas.json", "w") as f:
        json.dump({"theta0": theta0, "theta1": theta1}, f, indent=2)
    print("[5/5] Thetas saved to thetas.json")

    # --- BONUS : precision ---
    mae, rmse, r2 = compute_accuracy(real_kms, real_prices, theta0, theta1)
    print("\n" + "-" * 40)
    print("  Accuracy metrics:")
    print("    MAE  : {:,.2f} CHF  (mean absolute error)".format(mae))
    print("    RMSE : {:,.2f} CHF  (root mean squared error)".format(rmse))
    print("    R2   : {:.4f}         (1 = perfect, 0 = useless)".format(r2))
    print("-" * 40)

    # --- BONUS : graphiques ---
    show_plots(real_kms, real_prices, theta0, theta1, cost_history, mae, rmse, r2)


# Point d'entree Python
if __name__ == "__main__":
    main()
