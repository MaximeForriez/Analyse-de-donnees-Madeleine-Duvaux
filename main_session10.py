import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS


def ouvrirUnFichier(nom):
    with open(nom, "r", encoding="utf-8") as fichier:
        contenu = pd.read_csv(fichier, low_memory=False)
    return contenu

# Question 1 : Partie sur les températures
temperature = ouvrirUnFichier("./src/data/temperature.csv")

# Question 1a/b : corrélation et descriptives (on ignore la colonne Ville non numérique)
cols_num = [c for c in temperature.columns if c != "Ville"]
print("Corrélation (variables numériques) :")
print(temperature[cols_num].corr())
print("\nStatistiques descriptives :")
print(temperature.describe())

# Question 1b : isolation des variables
ville = temperature["Ville"] if "Ville" in temperature.columns else temperature[temperature.columns[0]]
y_temp = temperature["Température_en_janvier"]
X_temp = temperature[["Latitude", "Longitude", "Altitude"]]

# Question 1c/d/e : régression OLS statsmodels et paramètres
X_sm = sm.add_constant(X_temp)
modele_sm = sm.OLS(y_temp, X_sm).fit()
print("\nRégression OLS (statsmodels) :")
print(modele_sm.summary())

# Coeffs et R²
print("Coefficients statsmodels :", modele_sm.params)
print("R² statsmodels :", modele_sm.rsquared)
print("p-values statsmodels :", modele_sm.pvalues)

# Question 1f : régression scikit-learn
lr = LinearRegression()
lr.fit(X_temp, y_temp)
print("\nCoefficients sklearn :", lr.coef_)
print("Intercept sklearn :", lr.intercept_)

# Question 2 : Partie sur le géomarketing
geomarketing = ouvrirUnFichier("./src/data/geomarketing.csv")

variables_signif = [
    "surface_totale",
    "potentiel_Z20",
    "nb_primaire_Z10",
    "nb_primaire_Z20",
    "nb_gsa_Z10",
    "nb_pharmacie_Z5",
    "nb_conc2_Z10",
    "nb_conc2_Z20",
    "P10_POP_Z15",
    "P10_MEN_Z10",
]

y_ca = geomarketing["ca"]
X_ca = geomarketing[variables_signif]

X_ca_sm = sm.add_constant(X_ca)
modele_ca_sm = sm.OLS(y_ca, X_ca_sm).fit()
print("\nRégression géomarketing (variables sélectionnées) :")
print(modele_ca_sm.summary())

print("Coefficients statsmodels :", modele_ca_sm.params)
print("R² statsmodels :", modele_ca_sm.rsquared)
print("p-values statsmodels :", modele_ca_sm.pvalues)

# Bonus : régression avec toutes les variables (numériques uniquement)
X_ca_all = geomarketing.drop(columns=["ca"])
X_ca_all_numeric = X_ca_all.select_dtypes(exclude=["object"])
X_ca_all_sm = sm.add_constant(X_ca_all_numeric)
modele_ca_all_sm = sm.OLS(y_ca, X_ca_all_sm).fit()
print("\nRégression géomarketing (toutes variables numériques) :")
print(modele_ca_all_sm.summary())

# Sauvegarde des résumés dans un dossier output
output_dir = os.path.join("src", "output", "session10")
os.makedirs(output_dir, exist_ok=True)
with open(os.path.join(output_dir, "temperature_ols_summary.txt"), "w", encoding="utf-8") as f:
    f.write(str(modele_sm.summary()))
with open(os.path.join(output_dir, "geomarketing_sel_ols_summary.txt"), "w", encoding="utf-8") as f:
    f.write(str(modele_ca_sm.summary()))
with open(os.path.join(output_dir, "geomarketing_all_ols_summary.txt"), "w", encoding="utf-8") as f:
    f.write(str(modele_ca_all_sm.summary()))
