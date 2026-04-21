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
#      prix_estime = theta0 + theta1 * km
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

def lire_dataset(chemin):
    """
    Ouvre le fichier CSV et retourne deux listes :
      - kilometrages : liste des kilometrages
      - prix         : liste des prix correspondants
    """
    kilometrages = []
    prix = []

    with open(chemin, newline="") as fichier:
        lecteur = csv.DictReader(fichier)   # lit les colonnes par leur nom
        for ligne in lecteur:
            kilometrages.append(float(ligne["km"]))
            prix.append(float(ligne["price"]))

    if len(kilometrages) == 0:
        print("Erreur : le fichier est vide !")
        exit(1)

    return kilometrages, prix


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
# Formule : valeur_normalisee = (valeur - moyenne) / ecart_type

def calculer_moyenne(valeurs):
    return sum(valeurs) / len(valeurs)

def calculer_ecart_type(valeurs, moyenne):
    # Variance = moyenne des carres des ecarts
    variance = sum((v - moyenne) ** 2 for v in valeurs) / len(valeurs)
    return math.sqrt(variance)

def normaliser(valeurs):
    """
    Retourne (valeurs_normalisees, moyenne, ecart_type).
    On garde la moyenne et l'ecart_type pour pouvoir denormaliser ensuite.
    """
    moyenne = calculer_moyenne(valeurs)
    ecart_type = calculer_ecart_type(valeurs, moyenne)

    if ecart_type == 0:
        ecart_type = 1  # evite la division par zero

    normalises = [(v - moyenne) / ecart_type for v in valeurs]
    return normalises, moyenne, ecart_type


# -----------------------------------------------------------------------------
# ETAPE 3 : La fonction hypothesis (hypothese)
# -----------------------------------------------------------------------------
#
# C'est la formule centrale du projet :
#     prix_estime = theta0 + theta1 * km

def estimer_prix(km, theta0, theta1):
    return theta0 + (theta1 * km)


# -----------------------------------------------------------------------------
# ETAPE 4 : La descente de gradient
# -----------------------------------------------------------------------------
#
# Objectif : trouver theta0 et theta1 qui minimisent les erreurs.
#
# On repete "iterations" fois :
#   1. On calcule l'erreur de chaque point (prix_estime - prix_reel)
#   2. On calcule comment ajuster theta0 et theta1 pour reduire l'erreur
#   3. On met a jour theta0 et theta1 SIMULTANEMENT (via tmp0 et tmp1)
#
# Formules exactes du sujet :
#   tmp0 = learning_rate * (1/m) * somme(estimer_prix(km[i]) - prix[i])
#   tmp1 = learning_rate * (1/m) * somme((estimer_prix(km[i]) - prix[i]) * km[i])
#   theta0 = theta0 - tmp0
#   theta1 = theta1 - tmp1

def descente_gradient(kms_norm, prix_norm, learning_rate, iterations):
    """
    Entraine le modele sur des donnees normalisees.
    Retourne (theta0, theta1, historique_du_cout).
    """
    theta0 = 0.0
    theta1 = 0.0
    m = len(kms_norm)   # m = nombre d'exemples (lignes du CSV)

    historique_cout = []

    for i in range(iterations):

        # --- Calcul des erreurs pour chaque voiture ---
        erreurs = []
        for j in range(m):
            estimation = estimer_prix(kms_norm[j], theta0, theta1)
            erreur = estimation - prix_norm[j]
            erreurs.append(erreur)

        # --- Calcul des ajustements (formules du sujet) ---
        tmp0 = learning_rate * (1 / m) * sum(erreurs)
        tmp1 = learning_rate * (1 / m) * sum(erreurs[j] * kms_norm[j] for j in range(m))

        # --- Mise a jour SIMULTANEE des thetas ---
        theta0 = theta0 - tmp0
        theta1 = theta1 - tmp1

        # --- On enregistre le cout (erreur globale) pour le graphique ---
        cout = sum(e ** 2 for e in erreurs) / (2 * m)
        historique_cout.append(cout)

    return theta0, theta1, historique_cout


# -----------------------------------------------------------------------------
# ETAPE 5 : Denormaliser les thetas
# -----------------------------------------------------------------------------
#
# On a trouve theta0_n et theta1_n sur des donnees normalisees.
# Il faut les convertir pour qu'ils fonctionnent sur des km reels et des prix reels.
#
# Mathematiquement :
#   prix_norm = theta0_n + theta1_n * km_norm
#   (prix - prix_moy) / prix_std = theta0_n + theta1_n * (km - km_moy) / km_std
#
# En developpant :
#   theta1_reel = prix_std * theta1_n / km_std
#   theta0_reel = prix_moy + prix_std * theta0_n - theta1_reel * km_moy

def denormaliser_thetas(theta0_n, theta1_n, km_moy, km_std, prix_moy, prix_std):
    theta1_reel = prix_std * theta1_n / km_std
    theta0_reel = prix_moy + prix_std * theta0_n - theta1_reel * km_moy
    return theta0_reel, theta1_reel


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

def calculer_precision(kms_reels, prix_reels, theta0, theta1):
    m = len(kms_reels)
    predictions = [estimer_prix(kms_reels[i], theta0, theta1) for i in range(m)]
    erreurs     = [predictions[i] - prix_reels[i] for i in range(m)]

    mae  = sum(abs(e) for e in erreurs) / m
    rmse = math.sqrt(sum(e ** 2 for e in erreurs) / m)

    prix_moy = calculer_moyenne(prix_reels)
    ss_res   = sum((prix_reels[i] - predictions[i]) ** 2 for i in range(m))
    ss_tot   = sum((prix_reels[i] - prix_moy)       ** 2 for i in range(m))

    if ss_tot == 0:
        r2 = 0.0
    else:
        r2 = 1 - ss_res / ss_tot

    return mae, rmse, r2


# -----------------------------------------------------------------------------
# BONUS : Affichage des graphiques
# -----------------------------------------------------------------------------

def afficher_graphiques(kms_reels, prix_reels, theta0, theta1, historique_cout, mae, rmse, r2):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[info] matplotlib n'est pas installe. Pas de graphiques.")
        print("       Pour l'installer : pip install matplotlib")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("ft_linear_regression", fontsize=14, fontweight="bold")

    # --- Graphique 1 : nuage de points + droite de regression ---
    ax1.scatter(kms_reels, prix_reels,
                color="#4C72B0", alpha=0.85,
                edgecolors="white", linewidths=0.6,
                zorder=3, label="Donnees reelles")

    # On dessine la droite entre le min et le max du kilometrage
    x_min = min(kms_reels)
    x_max = max(kms_reels)
    y_min = estimer_prix(x_min, theta0, theta1)
    y_max = estimer_prix(x_max, theta0, theta1)
    ax1.plot([x_min, x_max], [y_min, y_max],
             color="#DD4444", linewidth=2, label="Droite de regression")

    ax1.set_xlabel("Kilometrage (km)")
    ax1.set_ylabel("Prix (CHF)")
    ax1.set_title("Distribution des donnees et droite de regression")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Affiche les metriques dans un encadre
    infos = "MAE  = {:,.0f} CHF\nRMSE = {:,.0f} CHF\nR2   = {:.4f}".format(mae, rmse, r2)
    ax1.text(0.97, 0.97, infos,
             transform=ax1.transAxes,
             fontsize=9,
             verticalalignment="top", horizontalalignment="right",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", alpha=0.8))

    # --- Graphique 2 : evolution du cout pendant l'entrainement ---
    # On affiche seulement les 50 premieres iterations pour voir la descente
    zoom = min(50, len(historique_cout))
    ax2.plot(range(zoom), historique_cout[:zoom], color="#2ca02c", linewidth=1.5)
    ax2.set_yscale("log")   # echelle logarithmique pour mieux voir la descente
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Cout (echelle log)")
    ax2.set_title("Descente de gradient - cout (50 premieres iterations)")
    ax2.grid(True, alpha=0.3, which="both")

    plt.tight_layout()
    plt.savefig("regression_plot.png", dpi=150)
    print("[graphique] Sauvegarde dans regression_plot.png")
    plt.show()


# -----------------------------------------------------------------------------
# PROGRAMME PRINCIPAL
# -----------------------------------------------------------------------------

def main():
    print("=" * 50)
    print("   ft_linear_regression  -  Entrainement")
    print("=" * 50)

    # 1. Lecture du fichier
    print("\n[1/5] Lecture du dataset...")
    kms_reels, prix_reels = lire_dataset("data.csv")
    print("      {} voitures chargees.".format(len(kms_reels)))

    # 2. Normalisation
    print("[2/5] Normalisation des donnees...")
    kms_norm,  km_moy,   km_std   = normaliser(kms_reels)
    prix_norm, prix_moy, prix_std = normaliser(prix_reels)

    # 3. Descente de gradient sur les donnees normalisees
    learning_rate = 0.1
    iterations    = 10000
    print("[3/5] Descente de gradient ({} iterations, lr={})...".format(iterations, learning_rate))
    theta0_n, theta1_n, historique_cout = descente_gradient(
        kms_norm, prix_norm, learning_rate, iterations
    )

    # 4. Denormalisation des thetas
    print("[4/5] Denormalisation des thetas...")
    theta0, theta1 = denormaliser_thetas(
        theta0_n, theta1_n, km_moy, km_std, prix_moy, prix_std
    )
    print("      theta0 = {:.4f}".format(theta0))
    print("      theta1 = {:.6f}".format(theta1))

    # 5. Sauvegarde dans thetas.json
    with open("thetas.json", "w") as f:
        json.dump({"theta0": theta0, "theta1": theta1}, f, indent=2)
    print("[5/5] Thetas sauvegardes dans thetas.json")

    # --- BONUS : precision ---
    mae, rmse, r2 = calculer_precision(kms_reels, prix_reels, theta0, theta1)
    print("\n" + "-" * 40)
    print("  Metriques de precision :")
    print("    MAE  : {:,.2f} CHF  (erreur absolue moyenne)".format(mae))
    print("    RMSE : {:,.2f} CHF  (erreur quadratique moyenne)".format(rmse))
    print("    R2   : {:.4f}         (1 = parfait, 0 = nul)".format(r2))
    print("-" * 40)

    # --- BONUS : graphiques ---
    afficher_graphiques(kms_reels, prix_reels, theta0, theta1, historique_cout, mae, rmse, r2)


# Point d'entree Python
if __name__ == "__main__":
    main()