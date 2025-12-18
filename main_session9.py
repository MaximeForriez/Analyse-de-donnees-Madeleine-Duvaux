#coding:utf8

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import scipy
import scipy.stats
import prince
import matplotlib.pyplot as plt

def ouvrirUnFichier(nom):
    with open(nom, "r", encoding="utf-8") as fichier:
        contenu = pd.read_csv(fichier, low_memory=False)
    return contenu

output_dir = os.path.join("src", "output", "img", "session9")
os.makedirs(output_dir, exist_ok=True)

# 1. ACP sur les températures françaises
temperature = ouvrirUnFichier("./src/data/france-temperatures.csv")

# a/b : isoler individus et variables numériques
villes = temperature["Villes"]
X_num = temperature.drop(columns=["Villes"])

# c : centrage-réduction
scaler = StandardScaler()
X_std = scaler.fit_transform(X_num)

# d : ACP avec 12 facteurs
pca = PCA(n_components=12)
pca.fit(X_std)
print("PCA fitted (12 composantes)")
print(pca)

# e : variances expliquées et valeurs propres
var_exp = pca.explained_variance_ratio_
var_exp_pct = var_exp * 100
eig_vals = pca.explained_variance_
print("Variance expliquée (ratio):", var_exp)
print("Variance expliquée (%):", var_exp_pct)
print("Valeurs propres:", eig_vals)

# f : charges (loadings) sous forme de DataFrame
loadings = pd.DataFrame(
    pca.components_.T * np.sqrt(eig_vals),
    index=X_num.columns,
    columns=[f"Dim{i+1}" for i in range(pca.n_components_)],
)
print("Loadings (variables x composantes):")
print(loadings)

# g : coordonnées individus + mapping (Dim1/Dim2)
scores = pca.transform(X_std)
coords_df = pd.DataFrame(scores, index=villes, columns=[f"Dim{i+1}" for i in range(pca.n_components_)])
plt.figure(figsize=(7, 6))
plt.scatter(scores[:, 0], scores[:, 1])
for i, label in enumerate(villes):
    plt.text(scores[i, 0], scores[i, 1], label, fontsize=8)
plt.axhline(0, color="grey", linewidth=0.8)
plt.axvline(0, color="grey", linewidth=0.8)
plt.xlabel("Dim1")
plt.ylabel("Dim2")
plt.title("ACP - Mapping des villes (Dim1/Dim2)")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "acp_individus_dim12.png"), dpi=150)
plt.close()

# h : contributions et cos² des individus
n = X_std.shape[0]
contrib = (scores ** 2) / (n * eig_vals)
cos2 = (scores ** 2) / np.sum(scores ** 2, axis=1, keepdims=True)
contrib_df = pd.DataFrame(contrib, index=villes, columns=[f"Dim{i+1}" for i in range(pca.n_components_)])
cos2_df = pd.DataFrame(cos2, index=villes, columns=[f"Dim{i+1}" for i in range(pca.n_components_)])
print("Contributions individus (premières lignes):")
print(contrib_df.head())
print("Cos² individus (premières lignes):")
print(cos2_df.head())

# i : coordonnées des variables (cercle de corrélation)
corvar = pca.components_.T * np.sqrt(var_exp)
plt.figure(figsize=(6, 6))
plt.axhline(0, color="grey", linewidth=0.8)
plt.axvline(0, color="grey", linewidth=0.8)
circle = plt.Circle((0, 0), 1, color="grey", fill=False, linestyle="--")
plt.gca().add_artist(circle)
plt.scatter(corvar[:, 0], corvar[:, 1])
for i, label in enumerate(X_num.columns):
    plt.text(corvar[i, 0], corvar[i, 1], label, fontsize=8)
plt.xlabel("Dim1")
plt.ylabel("Dim2")
plt.title("Cercle de corrélation (Dim1/Dim2)")
plt.axis("equal")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "acp_cercle_corr_dim12.png"), dpi=150)
plt.close()

# 2. ACM sur chiens.csv
chiens_path = os.path.join("./src/data", "chiens.csv")
if os.path.exists(chiens_path):
    chiens = ouvrirUnFichier(chiens_path)
    # a/b : individus et variables
    races = chiens["Race"]
    vars_cat = chiens[["Taille","Poids","Vitesse","Intelligence","Affection","Agressivité","Fonction","Origine"]]
    # c : TDC
    tdc = pd.get_dummies(vars_cat)
    # d : ACM 8 facteurs
    mca = prince.MCA(n_components=8, random_state=42)
    mca = mca.fit(tdc)
    # e : valeurs propres
    eig_mca = mca.eigenvalues_
    print("MCA eigenvalues:", eig_mca)
    # f : coordonnées lignes/colonnes
    rows_mca = mca.row_coordinates(tdc)
    cols_mca = mca.column_coordinates(tdc)
    # g : mapping Dim1/Dim2
    plt.figure(figsize=(7,6))
    plt.scatter(rows_mca[0], rows_mca[1], alpha=0.6)
    for i, label in enumerate(races):
        plt.text(rows_mca.iloc[i,0], rows_mca.iloc[i,1], label, fontsize=7)
    plt.axhline(0, color="grey", linewidth=0.8)
    plt.axvline(0, color="grey", linewidth=0.8)
    plt.xlabel("Dim1")
    plt.ylabel("Dim2")
    plt.title("ACM - mapping des races (Dim1/Dim2)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "acm_races_dim12.png"), dpi=150)
    plt.close()
    # h : cos² lignes/colonnes
    rows_cos2 = mca.row_cosine_similarities(tdc)
    cols_cos2 = mca.column_cosine_similarities(tdc)
    print("ACM cos² lignes (extrait):")
    print(rows_cos2.head())
    print("ACM cos² colonnes (extrait):")
    print(cols_cos2.head())
else:
    print("Fichier chiens.csv introuvable dans ./src/data, ACM non exécutée.")

# Bonus : C.A.H. sur les scores ACP (Ward)
from scipy.cluster.hierarchy import linkage, dendrogram
Z = linkage(scores, method="ward")
plt.figure(figsize=(10, 5))
dendrogram(Z, labels=villes.tolist(), leaf_rotation=90)
plt.title("CAH (Ward) sur les scores ACP")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "cah_acp.png"), dpi=150)
plt.close()


