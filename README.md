# ft_linear_regression

> Un premier projet de machine learning — régression linéaire avec descente de gradient.

---

## C'est quoi ce projet ?

On veut prédire le **prix d'une voiture** à partir de son **kilométrage**.

L'idée : plus une voiture a roulé, moins elle vaut. On cherche donc une droite qui représente au mieux cette relation.

```
prix_estimé = theta0 + theta1 * km
```

- `theta0` = ordonnée à l'origine (le prix "de base")
- `theta1` = pente (combien le prix baisse par km supplémentaire, valeur négative)

Pour trouver les bons `theta0` et `theta1`, on utilise la **descente de gradient** : on part de zéro et on les ajuste petit à petit, des milliers de fois, jusqu'à minimiser les erreurs.

---

## Structure des fichiers

```
ft_linear_regression/
├── train.py          ← entraîne le modèle, sauvegarde thetas.json
├── predict.py        ← prédit un prix à partir d'un kilométrage
├── data.csv          ← le dataset (24 voitures : km + prix)
├── thetas.json       ← créé automatiquement par train.py
└── regression_plot.png  ← créé automatiquement par train.py (bonus)
```

---

## Installation

Python 3.x suffit. Pour les graphiques (bonus), il faut `matplotlib` :

```bash
pip install matplotlib
```

---

## Utilisation

### 1. Entraîner le modèle

```bash
python3 train.py
```

Ce programme :
1. Lit `data.csv`
2. Normalise les données (pour stabiliser la descente de gradient)
3. Fait 10 000 itérations de descente de gradient
4. Sauvegarde `theta0` et `theta1` dans `thetas.json`
5. Affiche les métriques de précision (MAE, RMSE, R²)
6. Génère `regression_plot.png` avec deux graphiques

Exemple de sortie :

```
==================================================
   ft_linear_regression  -  Entrainement
==================================================

[1/5] Lecture du dataset...
      24 voitures chargees.
[2/5] Normalisation des donnees...
[3/5] Descente de gradient (10000 iterations, lr=0.1)...
[4/5] Denormalisation des thetas...
      theta0 = 8499.5996
      theta1 = -0.021449
[5/5] Thetas sauvegardes dans thetas.json

----------------------------------------
  Metriques de precision :
    MAE  : 557.84 s  (erreur absolue moyenne)
    RMSE : 667.57 CHF  (erreur quadratique moyenne)
    R2   : 0.7330         (1 = parfait, 0 = nul)
----------------------------------------
```

### 2. Prédire un prix

```bash
python3 predict.py
```

```
==================================================
   ft_linear_regression  -  Prediction
==================================================

Entrez le kilometrage (km) : 80000

  Kilometrage entre  : 80,000 km
  Prix estime        : 6,783.68 CHF
```

> **Important** : lancez `train.py` avant `predict.py`. Sans `thetas.json`, les deux thetas valent 0 et la prédiction sera toujours 0.

---

## Comment ça marche — la descente de gradient

### Le problème

On a 24 paires `(km, prix)`. On veut trouver `theta0` et `theta1` tels que :

```
estimer_prix(km) = theta0 + theta1 * km
```

soit le plus proche possible des vrais prix.

### La solution : répéter ces étapes

Pour chaque itération :

1. **Calculer l'erreur** pour chaque voiture :
   ```
   erreur[i] = estimer_prix(km[i]) - prix[i]
   ```

2. **Calculer les ajustements** (formules du sujet) :
   ```
   tmp0 = learning_rate * (1/m) * somme(erreur[i])
   tmp1 = learning_rate * (1/m) * somme(erreur[i] * km[i])
   ```

3. **Mettre à jour les thetas simultanément** :
   ```
   theta0 = theta0 - tmp0
   theta1 = theta1 - tmp1
   ```

> `m` = nombre de voitures dans le dataset  
> `learning_rate` = taille du pas (0.1 dans ce projet)

### Pourquoi normaliser ?

Les km sont dans les centaines de milliers et les prix dans les milliers. Cette différence d'échelle rend la descente de gradient très lente ou instable. On normalise donc les données avant l'entraînement :

```
valeur_normalisee = (valeur - moyenne) / ecart_type
```

Après l'entraînement, on "dénormalise" les thetas pour qu'ils fonctionnent avec les vraies valeurs.

---

## Les graphiques (bonus)

`regression_plot.png` contient deux graphiques :

| Graphique | Ce qu'il montre |
|-----------|-----------------|
| Gauche | Nuage de points (données réelles) + droite de régression |
| Droite | Evolution du coût sur les 50 premières itérations (échelle log) |

Le graphique de coût est en **échelle logarithmique** car la descente est très rapide au début et le coût devient minuscule ensuite — sans log, la courbe paraît plate.

---

## Les métriques de précision (bonus)

| Métrique | Formule | Interprétation |
|----------|---------|----------------|
| **MAE** | `moyenne(\|erreur\|)` | En moyenne, on se trompe de X CHF |
| **RMSE** | `racine(moyenne(erreur²))` | Comme MAE, mais pénalise les grosses erreurs |
| **R²** | `1 - SS_res / SS_tot` | 1 = parfait, 0 = le modèle ne sert à rien |

---

## Ce qui est vérifié à la peer-review

- [ ] Pas de librairie qui fait le travail à ta place (`numpy.polyfit`, `sklearn`, etc.)
- [ ] Hypothèse correcte : `theta0 + theta1 * mileage`
- [ ] Formules de gradient correctes (celles du sujet)
- [ ] Mise à jour **simultanée** de theta0 et theta1 (via des temporaires)