
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import scipy.stats

def ouvrirUnFichier(nom):
    with open(nom, "r", encoding="utf-8") as fichier:
        contenu = pd.read_csv(fichier, low_memory=False)
    return contenu

data = pd.DataFrame(ouvrirUnFichier("./src/data/pib-vs-energie.csv"))

colonnes = ["PIB_2022", "Utilisation_d_energie_2022"]
donnees = data[colonnes].copy()

# Question 1 et 2 : sélection des colonnes 2022 puis suppression des couples incomplets
donnees["PIB_2022"] = pd.to_numeric(donnees["PIB_2022"], errors="coerce")
donnees["Utilisation_d_energie_2022"] = pd.to_numeric(donnees["Utilisation_d_energie_2022"], errors="coerce")
donnees = donnees.dropna(subset=colonnes)

# Question 3 : régression linéaire simple (x = énergie, y = PIB)
x = donnees["Utilisation_d_energie_2022"].values
y = donnees["PIB_2022"].values

resultat_reg = scipy.stats.linregress(x, y)
print("Régression linéaire (PIB ~ énergie 2022)")
print(f"Slope: {resultat_reg.slope:.6f}")
print(f"Intercept: {resultat_reg.intercept:.6f}")
print(f"R (corrélation): {resultat_reg.rvalue:.6f}")
print(f"p-value: {resultat_reg.pvalue:.6e}")
print(f"Std err: {resultat_reg.stderr:.6f}")

# Question 4 : corrélation simple
corr_pearson = donnees.corr(method="pearson").loc["PIB_2022", "Utilisation_d_energie_2022"]
print(f"Corrélation de Pearson: {corr_pearson:.6f}")

# Question 5 : graphique avec droite de régression
output_dir = os.path.join("src", "output", "img", "session7")
os.makedirs(output_dir, exist_ok=True)

plt.figure(figsize=(8, 5))
plt.scatter(x, y, alpha=0.6, s=20, label="Observations")

x_line = np.linspace(x.min(), x.max(), 200)
y_line = resultat_reg.intercept + resultat_reg.slope * x_line
plt.plot(x_line, y_line, color="red", label="Droite de régression")

plt.xlabel("Consommation d'énergie 2022 (kg équivalent pétrole)")
plt.ylabel("PIB 2022 (USD courants)")
plt.title("PIB vs consommation d'énergie (2022)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "pib_vs_energie_regression.png"), dpi=150)
plt.close()

# Question 6 : commentaire d'interprétation
"""
 Interprétation : la pente indique la variation moyenne du PIB associée à 1 kg équivalent pétrole
 supplémentaire consommé. L'ordonnée à l'origine donne le niveau de PIB théorique quand la
 consommation est nulle (utile surtout pour positionner la droite). Le coefficient r (et R² = r²)
 mesure la force et la part de variance expliquée par une relation linéaire ; la p-value teste si la
 pente diffère significativement de zéro.

 Limites : corrélation ne signifie pas causalité. La dispersion peut masquer des effets de taille,
 de structure économique ou de mix énergétique. Des valeurs extrêmes peuvent influencer la pente ;
 un ajustement log-log ou une pondération par population/PIB par habitant peut être pertinent. Ici
 l'analyse porte sur 2022 uniquement : répliquer sur plusieurs années permettrait de tester la
 stabilité de la relation.
"""

# Bonus : généralisation 1962-2022
bonus_resultats = []
bonus_img_dir = os.path.join("src", "output", "img", "session7", "years")
bonus_csv_dir = os.path.join("src", "output", "session7")
os.makedirs(bonus_img_dir, exist_ok=True)
os.makedirs(bonus_csv_dir, exist_ok=True)

for annee in range(1962, 2023):
    col_pib = f"PIB_{annee}"
    col_energie = f"Utilisation_d_energie_{annee}"

    if col_pib not in data.columns or col_energie not in data.columns:
        continue

    df_year = data[[col_pib, col_energie]].copy()
    df_year[col_pib] = pd.to_numeric(df_year[col_pib], errors="coerce")
    df_year[col_energie] = pd.to_numeric(df_year[col_energie], errors="coerce")
    df_year = df_year.dropna(subset=[col_pib, col_energie])

    if len(df_year) < 2:
        continue

    x_year = df_year[col_energie].values
    y_year = df_year[col_pib].values

    reg_year = scipy.stats.linregress(x_year, y_year)
    corr_year = df_year.corr(method="pearson").loc[col_pib, col_energie]

    bonus_resultats.append({
        "annee": annee,
        "n_obs": len(df_year),
        "slope": reg_year.slope,
        "intercept": reg_year.intercept,
        "rvalue": reg_year.rvalue,
        "pvalue": reg_year.pvalue,
        "stderr": reg_year.stderr,
        "corr_pearson": corr_year,
    })

    # Graphique par année
    plt.figure(figsize=(8, 5))
    plt.scatter(x_year, y_year, alpha=0.6, s=18, label="Observations")
    x_line_year = np.linspace(x_year.min(), x_year.max(), 200)
    y_line_year = reg_year.intercept + reg_year.slope * x_line_year
    plt.plot(x_line_year, y_line_year, color="red", label="Droite de régression")
    plt.xlabel(f"Consommation d'énergie {annee} (kg équivalent pétrole)")
    plt.ylabel(f"PIB {annee} (USD courants)")
    plt.title(f"PIB vs consommation d'énergie ({annee})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(bonus_img_dir, f"pib_vs_energie_{annee}.png"), dpi=150)
    plt.close()

# Sauvegarde du tableau de synthèse des régressions par année
if bonus_resultats:
    df_bonus = pd.DataFrame(bonus_resultats)
    df_bonus.to_csv(os.path.join(bonus_csv_dir, "regressions_par_annee.csv"), index=False)


