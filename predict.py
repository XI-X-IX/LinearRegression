#!/usr/bin/env python3
# =============================================================================
#  predict.py  -  Programme de prediction (ft_linear_regression)
# =============================================================================
#
#  Ce programme demande un kilometrage a l'utilisateur et retourne
#  le prix estime pour cette voiture.
#
#  Il utilise theta0 et theta1 sauvegardes par train.py dans thetas.json.
#  Si thetas.json n'existe pas, les deux thetas valent 0 (prediction nulle).
#
# =============================================================================

import json   # pour lire le fichier thetas.json
import os     # pour verifier si le fichier existe


# -----------------------------------------------------------------------------
# Charger theta0 et theta1
# -----------------------------------------------------------------------------

def load_thetas(path):
    """
    Lit thetas.json et retourne (theta0, theta1).
    Si le fichier n'existe pas, retourne (0, 0) et affiche un avertissement.
    """
    if not os.path.exists(path):
        print("[warning] '{}' not found.".format(path))
        print("          Run train.py first to train the model.")
        print("          For now, theta0=0 and theta1=0.\n")
        return 0.0, 0.0

    with open(path) as f:
        data = json.load(f)

    theta0 = float(data["theta0"])
    theta1 = float(data["theta1"])
    return theta0, theta1


# -----------------------------------------------------------------------------
# Hypothese : formule de prediction
# -----------------------------------------------------------------------------
#
# C'est la meme formule que dans train.py :
#     estimated_price = theta0 + theta1 * km

def estimate_price(km, theta0, theta1):
    return theta0 + (theta1 * km)


# -----------------------------------------------------------------------------
# Demander le kilometrage a l'utilisateur
# -----------------------------------------------------------------------------

def ask_mileage():
    """
    Boucle jusqu'a ce que l'utilisateur entre un nombre valide et positif.
    Retourne le kilometrage sous forme de float.
    """
    while True:
        user_input = input("Enter the mileage (km): ").strip()

        # Verifie que c'est bien un nombre
        try:
            km = float(user_input)
        except ValueError:
            print("  [error] Please enter a valid number (e.g. 50000).\n")
            continue

        # Verifie que c'est positif
        if km < 0:
            print("  [error] Mileage cannot be negative.\n")
            continue

        return km


# -----------------------------------------------------------------------------
# PROGRAMME PRINCIPAL
# -----------------------------------------------------------------------------

def main():
    print("=" * 50)
    print("   ft_linear_regression  -  Prediction")
    print("=" * 50)
    print()

    # 1. Charger les thetas entraines
    theta0, theta1 = load_thetas("thetas.json")

    # 2. Demander le kilometrage
    km = ask_mileage()

    # 3. Calculer le prix estime
    price = estimate_price(km, theta0, theta1)

    # Les prix ne peuvent pas etre negatifs
    if price < 0:
        price = 0.0

    # 4. Afficher le resultat
    print()
    print("  Mileage entered : {:,.0f} km".format(km))
    print("  Estimated price : {:,.2f} CHF".format(price))


# Point d'entree Python
if __name__ == "__main__":
    main()
