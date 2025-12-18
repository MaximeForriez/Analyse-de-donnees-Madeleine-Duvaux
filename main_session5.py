# coding:utf8
import os
import pandas as pd
import math
import numpy as np
from scipy import stats

def ouvrirUnFichier(nom):
    with open(nom, "r") as fichier:
        contenu = pd.read_csv(fichier)
    return contenu


N_population = 2185
pop_counts = np.array([852, 911, 422])  # Pour, Contre, Sans opinion

donnees = pd.DataFrame(ouvrirUnFichier("./src/data/Echantillonnage-100-Echantillons.csv"))

# 1) Théorie de l’échantillonnage
# Moyennes par colonne (arrondies à 0 décimale avec round())
means_raw = donnees.mean(axis=0).values
means_rounded = [round(m) for m in means_raw]

# fréquence moyenne par catégorie (à partir des moyennes arrondies)
sum_means = sum(means_rounded)
freq_means = [round(m / sum_means, 2) for m in means_rounded]

# fréquences population mère (arrondies à 2 décimales)
freq_population = [round(c / N_population, 2) for c in pop_counts]

print("Moyennes (arrondies) par catégorie :", means_rounded)
print("Somme des moyennes (taille d'échantillon moyenne) :", sum_means)
print("Fréquences moyennes (arrondies 2 déc) :", freq_means)
print("Fréquences population mère (arrondies 2 déc) :", freq_population)

# Intervalle de fluctuation à 95% (zC=1.96) pour chaque fréquence
zC = 1.96
n = sum_means  
fpc = math.sqrt(max(0.0, (N_population - n) / (N_population - 1)))
intervals_fluct = []
for p_hat in [m / sum_means for m in means_rounded]:
    se = math.sqrt(p_hat * (1 - p_hat) / n) * fpc
    lower = p_hat - zC * se
    upper = p_hat + zC * se
    intervals_fluct.append((round(lower, 4), round(upper, 4)))

print("Intervalle de fluctuation 95% (avec correction population finie) :")
print("Pour, Contre, Sans opinion :", intervals_fluct)

# 2) Théorie de l’estimation
# Prendre le premier échantillon (première ligne)
premier = list(donnees.iloc[0].astype(int).values)
n1 = sum(premier)
freqs_premier = [c / n1 for c in premier]

# Intervalles de confiance (approx normale) pour chaque opinion du 1er échantillon
ic_premier = []
fpc1 = math.sqrt(max(0.0, (N_population - n1) / (N_population - 1)))
for p in freqs_premier:
    se = math.sqrt(p * (1 - p) / n1) * fpc1
    lower = p - zC * se
    upper = p + zC * se
    ic_premier.append((round(lower, 4), round(upper, 4)))

print("Premier échantillon (comptages) :", premier)
print("Taille du premier échantillon :", n1)
print("Fréquences premier échantillon :", [round(p, 4) for p in freqs_premier])
print("IC 95% (approx. normale, correction finie) pour le 1er échantillon :", ic_premier)

# 3) Théorie de la décision - test de Shapiro-Wilk sur deux fichiers
f1 = pd.Series(ouvrirUnFichier("./src/data/Loi-normale-Test-1.csv").iloc[:, 0].dropna().astype(float).values)
f2 = pd.Series(ouvrirUnFichier("./src/data/Loi-normale-Test-2.csv").iloc[:, 0].dropna().astype(float).values)

sh1 = stats.shapiro(f1)
sh2 = stats.shapiro(f2)

print("Shapiro Test - fichier 1: statistic={:.4f}, p={:.4g}".format(sh1.statistic, sh1.pvalue))
print("Shapiro Test - fichier 2: statistic={:.4f}, p={:.4g}".format(sh2.statistic, sh2.pvalue))

if sh1.pvalue > 0.05:
    print("Distribution 1: conforme à la normalité (alpha=0.05)")
else:
    print("Distribution 1: non normale (alpha=0.05)")

if sh2.pvalue > 0.05:
    print("Distribution 2: conforme à la normalité (alpha=0.05)")
else:
    print("Distribution 2: non normale (alpha=0.05)")

# Bonus : identification de la loi la mieux ajustée pour la série non normale

def best_fit_distribution(data, distributions=("norm", "expon", "uniform")):
    best = {"dist": None, "pvalue": -1, "params": None}
    for dist_name in distributions:
        dist = getattr(stats, dist_name)
        try:
            params = dist.fit(data)
            ks = stats.kstest(data, dist_name, args=params)
            if ks.pvalue > best["pvalue"]:
                best.update({"dist": dist_name, "pvalue": ks.pvalue, "params": params})
        except Exception:
            continue
    return best


for idx, (series, sh) in enumerate([(f1, sh1), (f2, sh2)], start=1):
    if sh.pvalue <= 0.05:
        best = best_fit_distribution(series.values)
        print(f"Meilleure loi ajustée pour la série {idx} (non normale) : {best['dist']} with KS p-value={best['pvalue']:.4g} and params={best['params']}")

