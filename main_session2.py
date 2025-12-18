# Session 2 : Les principes généraux de la statistique
# Script converti depuis le notebook main_session2.ipynb

import pandas as pd
import matplotlib.pyplot as plt

# Question 2 : charger le fichier CSV
with open("./src/data/resultats-elections-presidentielles-2022-1er-tour.csv", "r", encoding="utf-8") as fichier:
    contenu = pd.read_csv(fichier)

# Question 5 : afficher le DataFrame
df = pd.DataFrame(contenu)
print(df)

# Question 6 : nombre de lignes et de colonnes
print(f"Nombre de colonnes : {len(df.columns)}")
print(f"Nombre de lignes : {len(df)}")

# Question 7 : types des colonnes
print(df.dtypes)

# Question 8 : affichage de la première ligne (noms de colonnes)
print(df.head())

# Question 9 : somme des inscrits
print(f"Le nombre total d'inscrits est {df['Inscrits'].sum()}")

# Question 10 : effectifs des colonnes quantitatives
liste_effectifs = []
for colonne in contenu.columns:
    dtype = contenu[colonne].dtype
    if dtype == "int64" or dtype == "float64":
        somme_colonne = contenu[colonne].sum()
        liste_effectifs.append(somme_colonne)
print(f"Liste des effectifs {liste_effectifs}")
