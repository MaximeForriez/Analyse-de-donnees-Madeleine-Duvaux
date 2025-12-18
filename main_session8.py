import os
import numpy as np
import pandas as pd
import scipy
import scipy.stats

def ouvrirUnFichier(nom):
    with open(nom, "r", encoding="utf-8") as fichier:
        contenu = pd.read_csv(fichier, low_memory=False)
    return contenu

def tableauDeContingence(nom, donnees):
    indexValeurs = {}
    for element in range(0,len(nom)):
        indexValeurs.update({element: nom[element]})
    return pd.DataFrame(donnees).rename(index = indexValeurs)

def sommeDesColonnes(tableau):
    colonne = list(tableau.head(0))
    sommeColonne = []
    for element in colonne:
        sommeColonne.append(tableau[element].sum())
    return sommeColonne

def sommeDesLignes(tableau):
    colonne = list(tableau.head(0))
    sommeLigne = []
    for element1 in range(0,len(tableau)):
        ligne = []
        for element2 in range(0,len(colonne)):
            ligne.append(tableau.iloc[element1, element2])
        sommeLigne.append(np.sum(list(ligne)))
    return sommeLigne

data = pd.DataFrame(ouvrirUnFichier("./src/data/Socioprofessionnelle-vs-sexe.csv"))

# Question 1 : création du tableau de contingence (le fichier est déjà un tableau croisé) & calculer les marges
tab_cont = tableauDeContingence(data["Catégorie"], {"Femmes": data["Femmes"], "Hommes": data["Hommes"]})
print(tab_cont)


print("Calcul des marges")
margin_cols = sommeDesColonnes(tab_cont)
margin_rows = sommeDesLignes(tab_cont)
print(f"Somme colonnes (Femmes, Hommes): {margin_cols}")
print(f"Somme lignes (par catégorie): {margin_rows}")

# Question 2 : condition de cohérence des totaux
total_cols = sum(margin_cols)
total_rows = sum(margin_rows)
print(f"Total colonnes: {total_cols}, Total lignes: {total_rows}")
if np.isclose(total_cols, total_rows):
    print("Totaux identiques : OK")
else:
    print("Totaux non identiques : vérifier les données")

# Question 3 : test du chi2 avec Scipy.stats
print("Test du chi2")
chi2, p_value, dof, expected = scipy.stats.chi2_contingency(tab_cont.values)
print(f"Chi2: {chi2:.4f}, ddl: {dof}, p-value: {p_value:.6e}")
print("Effectifs attendus :")
print(expected)

# Question 4 : intensité de liaison phi2 de Pearson
print("Intensité de liaison phi2")
n_total = tab_cont.values.sum()
phi2 = chi2 / n_total
print(f"Phi2 de Pearson: {phi2:.6f}")

# Bonus : ANOVA sur Echantillonnage-100-Echantillons.csv & A.F.C. (analyse factorielle des correspondances)
bonus_data = pd.DataFrame(ouvrirUnFichier("./src/data/Echantillonnage-100-Echantillons.csv"))
anova_result = scipy.stats.f_oneway(bonus_data.iloc[:, 0], bonus_data.iloc[:, 1], bonus_data.iloc[:, 2])
print("ANOVA (Pour vs Contre vs Sans opinion)")
print(f"F = {anova_result.statistic:.4f}, p-value = {anova_result.pvalue:.6e}")

def correspondence_analysis(table):
    X = table.values.astype(float)
    n = X.sum()
    P = X / n
    r = P.sum(axis=1, keepdims=True)
    c = P.sum(axis=0, keepdims=True)
    expected = r @ c
    S = (P - expected) / np.sqrt(expected)
    U, singular_vals, Vt = np.linalg.svd(S, full_matrices=False)
    F = (1 / np.sqrt(r)) * (U * singular_vals)
    G = (1 / np.sqrt(c.T)) * (Vt.T * singular_vals)
    inertia = singular_vals ** 2
    return F, G, inertia

row_coords, col_coords, inertia = correspondence_analysis(tab_cont)
afc_dir = os.path.join("src", "output", "session8")
os.makedirs(afc_dir, exist_ok=True)

pd.DataFrame(row_coords, columns=["Dim1", "Dim2"] + [f"Dim{i}" for i in range(3, row_coords.shape[1] + 1)]).to_csv(
    os.path.join(afc_dir, "afc_coord_lignes.csv"), index_label="Categorie"
)
pd.DataFrame(col_coords, columns=["Dim1", "Dim2"] + [f"Dim{i}" for i in range(3, col_coords.shape[1] + 1)]).to_csv(
    os.path.join(afc_dir, "afc_coord_colonnes.csv"), index_label="Sexe"
)
pd.DataFrame({"inertie": inertia}).to_csv(
    os.path.join(afc_dir, "afc_inertie.csv"), index_label="dimension"
)

print("AFC : inerties principales (variance expliquée par dimension)")
print(inertia)


"""
χ² et liaison : la p-value (très faible, non affichée mais issue du test chi²) implique rejet de l'indépendance : la répartition Femmes/Hommes dépend de la catégorie socioprofessionnelle.

Intensité φ² : 0,0883 ≈ effet faible (entre 0 et 0,1 on parle généralement de liaison faible). Donc dépendance statistiquement significative mais de faible ampleur.

ANOVA (Pour/Contre/Sans opinion) : F = 14075,71 avec p ≈ 6e-295 ; les moyennes des trois groupes diffèrent très significativement. Il faudrait un post-hoc (Tukey) pour savoir quels groupes diffèrent, mais la significativité globale est sans ambiguïté.

AFC : inertie Dim1 ≈ 0,088 (≈ 8,8% de la variance du tableau), Dim2 quasi nulle. L'essentiel de la structure se projette sur l'axe 1 ; l'axe 2 n'apporte presque rien. Pour interpréter finement, il faut lire les coordonnées lignes/colonnes exportées (fichiers afc_coord_lignes/colonnes) et repérer quelles catégories et quel sexe tirent l'axe 1.

"""