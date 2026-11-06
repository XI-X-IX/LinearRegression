# ft_linear_regression

Prédire le prix d'une voiture à partir de son kilométrage, avec une régression linéaire codée à la main (aucune librairie).

## Fichiers

- `train.py` : apprend `theta0` et `theta1` à partir de `data.csv`, puis les sauvegarde dans `thetas.txt`
- `predict.py` : demande un kilométrage et affiche le prix estimé
- `precision.py` (bonus) : calcule de combien le modèle se trompe en moyenne
- `data.csv` : le dataset (24 voitures : kilométrage et prix)

## Utilisation

```bash
python3 train.py
python3 predict.py
python3 precision.py   # bonus
```

Si on lance `predict.py` avant `train.py`, les deux thetas valent 0, donc le prix estimé est 0.

## Comment ça marche

La formule de prédiction :

```
estimate_price = theta0 + theta1 * mileage
```

- `theta0` : le prix d'une voiture à 0 km
- `theta1` : de combien le prix baisse par km (un nombre négatif)

Pour trouver les bons thetas, on utilise la **descente de gradient**. On part de 0 et on répète 10 000 fois :

1. Pour chaque voiture, on calcule l'erreur : `estimate_price - vrai_prix`
2. `tmp_theta0 = learning_rate * (somme des erreurs) / m`
3. `tmp_theta1 = learning_rate * (somme des erreurs * mileage) / m`
4. On met à jour les deux en même temps : `theta0 -= tmp_theta0` et `theta1 -= tmp_theta1`

`m` est le nombre de voitures.

### L'idée en une phrase

On part de 0, on regarde de combien on se trompe, on corrige un peu, et on recommence jusqu'à ce que l'erreur ne baisse plus.

### Exemple à la main (3 voitures, un seul theta)

On cherche **un seul nombre** le plus proche des prix 4000, 6000 et 8000. On part de 0 et on corrige de la moitié de l'erreur moyenne à chaque tour (`learning_rate = 0.5`).

| Tour | Réponse | Erreurs (réponse - vrai prix) | Erreur moyenne | Correction |
|---|---|---|---|---|
| 1 | 0 | -4000, -6000, -8000 | -6000 (trop bas) | +3000 |
| 2 | 3000 | -1000, -3000, -5000 | -3000 (trop bas) | +1500 |
| 3 | 4500 | +500, -1500, -3500 | -1500 (trop bas) | +750 |
| 4 | 5250 | +1250, -750, -2750 | -750 (trop bas) | +375 |
| ... | ... | ... | ... | ... |
| fin | **6000** | -2000, 0, +2000 | 0 (parfait) | 0 |

La réponse se rapproche de 6000 à chaque tour, et les corrections deviennent de plus en plus petites.

### Ce qui se passe dans train.py

Le programme fait la même chose avec deux nombres (`theta0` et `theta1`) sur les 24 voitures :

| Tour | theta0 | theta1 | Erreur moyenne | Prix prédit pour 80 000 km |
|---|---|---|---|---|
| 0 | 0 | 0 | 6 332 CHF | 0 CHF |
| 1 | 3 166 | +0.005 | 2 721 CHF | 3 571 CHF |
| 2 | 4 493 | +0.007 | 1 734 CHF | 5 035 CHF |
| 10 | 5 820 | +0.004 | 1 218 CHF | 6 142 CHF |
| 50 | 7 280 | -0.010 | 704 CHF | 6 492 CHF |
| 100 | 8 044 | -0.017 | 577 CHF | 6 675 CHF |
| 500 | 8 499 | -0.021 | 558 CHF | 6 784 CHF |
| 10 000 | 8 500 | -0.021 | 558 CHF | 6 784 CHF |

L'erreur moyenne à chaque tour :

```
Tour     0 | ################################ 6332 CHF
Tour     1 | ##############                   2721 CHF
Tour     2 | #########                        1734 CHF
Tour    10 | ######                           1218 CHF
Tour    50 | ####                              704 CHF
Tour   100 | ###                               577 CHF
Tour   500 | ###                               558 CHF  <- plus rien ne bouge
Tour 10000 | ###                               558 CHF
```

L'erreur baisse très vite au début, puis se stabilise vers le tour 500 : on a trouvé la meilleure droite. Elle ne tombe pas à 0, car aucune droite ne passe exactement par les 24 voitures.

### Pourquoi diviser les kilométrages par le kilométrage maximum ?

Les kilométrages sont de très grands nombres (jusqu'à 240 000). Avec des nombres aussi grands, les thetas font des sauts énormes et l'apprentissage explose. On divise donc chaque kilométrage par le plus grand, ce qui les ramène entre 0 et 1.

À la fin, `theta1` a été appris pour des kilométrages divisés par `max_mileage`. Pour qu'il fonctionne avec les vrais kilométrages, on le divise aussi par `max_mileage`. `theta0` ne change pas.

## Résultat

```
theta0 = 8499.60
theta1 = -0.021449
```

Par exemple, pour 80 000 km, le prix estimé est 8499.60 - 0.021449 × 80 000 ≈ 6783.68.
