#!/usr/bin/env python3
# Programme d'entrainement : trouve theta0 et theta1 a partir de data.csv


def estimate_price(mileage, theta0, theta1):
    return theta0 + theta1 * mileage


# 1. Lire data.csv (on saute la premiere ligne "km,price")
mileages = []
prices = []
file = open("data.csv")
lines = file.read().split("\n")
file.close()
for line in lines[1:]:
    if line != "":
        parts = line.split(",")
        mileages.append(float(parts[0]))
        prices.append(float(parts[1]))

m = len(mileages)  # nombre de voitures

# 2. Mettre les kilometrages a une petite echelle (entre 0 et 1)
#    Sinon les nombres sont trop grands et l'apprentissage explose.
max_mileage = max(mileages)
small_mileages = []
for mileage in mileages:
    small_mileages.append(mileage / max_mileage)

# 3. Descente de gradient (formules du sujet)
learning_rate = 0.5
theta0 = 0.0
theta1 = 0.0

for step in range(10000):
    sum0 = 0.0
    sum1 = 0.0
    for i in range(m):
        error = estimate_price(small_mileages[i], theta0, theta1) - prices[i]
        sum0 = sum0 + error
        sum1 = sum1 + error * small_mileages[i]

    tmp_theta0 = learning_rate * sum0 / m
    tmp_theta1 = learning_rate * sum1 / m

    # mise a jour des deux thetas en meme temps
    theta0 = theta0 - tmp_theta0
    theta1 = theta1 - tmp_theta1

# 4. On a appris avec mileage / max_mileage, donc on divise theta1 par max_mileage
#    pour qu'il marche avec les vrais kilometrages.
theta1 = theta1 / max_mileage

# 5. Sauvegarder les thetas dans thetas.txt
file = open("thetas.txt", "w")
file.write(str(theta0) + "," + str(theta1))
file.close()

print("theta0 =", theta0)
print("theta1 =", theta1)
print("Thetas saved in thetas.txt")
