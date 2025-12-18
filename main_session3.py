# Session 3 : Statistiques descriptives et boites a moustaches (conversion notebook)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Question 1-4 : dossier data present, fichier charge via with + read_csv
with open("./src/data/resultats-elections-presidentielles-2022-1er-tour.csv", "r", encoding="utf-8") as fichier:
    contenu = pd.read_csv(fichier)

output_dir = Path("src/output/session3")
output_dir.mkdir(parents=True, exist_ok=True)

# Question 5-6 : statistiques descriptives par colonne quantitative
stats_par_colonne = []
for colonne in contenu.columns:
    dtype = contenu[colonne].dtype
    if dtype == "int64" or dtype == "float64":
        serie = contenu[colonne].dropna()
        moyenne_colonne = serie.mean()
        mediane_colonne = serie.median()
        mode_colonne = serie.mode().iloc[0]
        ecart_type_colonne = serie.std()
        ecart_absolu_moyenne_colonne = np.abs(serie - moyenne_colonne).mean()
        etendue_colonne = serie.max() - serie.min()
        stats_par_colonne.append({
            "colonne": colonne,
            "dtype": str(dtype),
            "moyenne": round(moyenne_colonne, 2),
            "mediane": round(mediane_colonne, 2),
            "mode": round(mode_colonne, 2),
            "ecart_type": round(ecart_type_colonne, 2),
            "ecart_absolu_moyenne": round(ecart_absolu_moyenne_colonne, 2),
            "etendue": round(etendue_colonne, 2),
        })
        print(f"Statistiques pour la colonne {colonne} :\n{pd.DataFrame([stats_par_colonne[-1]])}")

stats_df = pd.DataFrame(stats_par_colonne)
stats_df.to_csv(output_dir / "stats_colonnes.csv", index=False)
stats_df.to_excel(output_dir / "stats_colonnes.xlsx", index=False)

# Question 7 : distances interquartile et interdecile
distances_par_colonne = []
for colonne in contenu.columns:
    dtype = contenu[colonne].dtype
    if dtype == "int64" or dtype == "float64":
        serie = contenu[colonne].dropna()
        q1 = serie.quantile(0.25)
        q3 = serie.quantile(0.75)
        q10 = serie.quantile(0.10)
        q90 = serie.quantile(0.90)
        distance_interquartile = round(q3 - q1, 2)
        distance_interdecile = round(q90 - q10, 2)
        distances_par_colonne.append({
            "colonne": colonne,
            "dtype": str(dtype),
            "IQR": distance_interquartile,
            "IDR": distance_interdecile,
        })
distances_df = pd.DataFrame(distances_par_colonne)
print(distances_df)
distances_df.to_csv(output_dir / "distances_colonnes.csv", index=False)
distances_df.to_excel(output_dir / "distances_colonnes.xlsx", index=False)

# Question 8 : boites a moustaches pour chaque colonne quantitative
IMG_DIR = Path("src/output/img/session3")
IMG_DIR.mkdir(parents=True, exist_ok=True)
resultats_boites = []

for colonne in contenu.columns:
    dtype = contenu[colonne].dtype
    if dtype == "int64" or dtype == "float64":
        serie = contenu[colonne].dropna()
    else:
        continue
    plt.figure(figsize=(6, 4))
    plt.boxplot(serie, vert=True)
    plt.title(f"Boite a moustaches - {colonne}")
    plt.ylabel(colonne)
    nom_fichier = colonne.lower().replace(" ", "_").replace("/", "-")
    img_path = IMG_DIR / f"boxplot_{nom_fichier}.png"
    plt.savefig(img_path, dpi=150, bbox_inches="tight")
    plt.close()
    resultats_boites.append({"colonne": colonne, "fichier": str(img_path)})

print(pd.DataFrame(resultats_boites))

# Question 9-10 : categorisation des surfaces d'iles
iles_path = Path("./src/data/island-index.csv")
if iles_path.exists():
    iles = pd.read_csv(iles_path, encoding="utf-8", low_memory=False)
    surface_cols = [c for c in iles.columns if "Surface" in c]
    surface_col = surface_cols[0]
    surfaces = pd.to_numeric(iles[surface_col], errors="coerce")
    bins = [0, 10, 25, 50, 100, 2500, 5000, 10000, np.inf]
    labels = [
        "]0,10]",
        "]10,25]",
        "]25,50]",
        "]50,100]",
        "]100,2500]",
        "]2500,5000]",
        "]5000,10000]",
        "]10000,+inf[",
    ]
    categories = pd.cut(surfaces, bins=bins, labels=labels, include_lowest=True, right=True)
    repartition = categories.value_counts().sort_index()
    print("Repartition des iles par tranche de surface :")
    print(repartition)
    repartition.to_csv(output_dir / "iles_repartition_surfaces.csv")
else:
    print("Fichier island-index.csv manquant dans src/data : question 9-10 non executee")

# Bonus : sauvegarde des listes (stats, distances) deja exportees en CSV/Excel ci-dessus
