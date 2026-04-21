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

def charger_thetas(chemin):
    """
    Lit thetas.json et retourne (theta0, theta1).
    Si le fichier n'existe pas, retourne (0, 0) et affiche un avertissement.
    """
    if not os.path.exists(chemin):
        print("[avertissement] '{}' introuvable.".format(chemin))
        print("                Lancez d'abord train.py pour entrainer le modele.")
        print("                Pour l'instant, theta0=0 et theta1=0.\n")
        return 0.0, 0.0

    with open(chemin) as f:
        donnees = json.load(f)

    theta0 = float(donnees["theta0"])
    theta1 = float(donnees["theta1"])
    return theta0, theta1


# -----------------------------------------------------------------------------
# Hypothese : formule de prediction
# -----------------------------------------------------------------------------
#
# C'est la meme formule que dans train.py :
#     prix_estime = theta0 + theta1 * km

def estimer_prix(km, theta0, theta1):
    return theta0 + (theta1 * km)


# -----------------------------------------------------------------------------
# Demander le kilometrage a l'utilisateur
# -----------------------------------------------------------------------------

def demander_kilometrage():
    """
    Boucle jusqu'a ce que l'utilisateur entre un nombre valide et positif.
    Retourne le kilometrage sous forme de float.
    """
    while True:
        saisie = input("Entrez le kilometrage (km) : ").strip()

        # Verifie que c'est bien un nombre
        try:
            km = float(saisie)
        except ValueError:
            print("  [erreur] Veuillez entrer un nombre valide (ex: 50000).\n")
            continue

        # Verifie que c'est positif
        if km < 0:
            print("  [erreur] Le kilometrage ne peut pas etre negatif.\n")
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
    theta0, theta1 = charger_thetas("thetas.json")

    # 2. Demander le kilometrage
    km = demander_kilometrage()

    # 3. Calculer le prix estime
    prix = estimer_prix(km, theta0, theta1)

    # Les prix ne peuvent pas etre negatifs
    if prix < 0:
        prix = 0.0

    # 4. Afficher le resultat
    print()
    print("  Kilometrage entre  : {:,.0f} km".format(km))
    print("  Prix estime        : {:,.2f} CHF".format(prix))


# Point d'entree Python
if __name__ == "__main__":
    main()