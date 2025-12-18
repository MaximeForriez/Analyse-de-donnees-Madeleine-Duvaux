import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import scipy.stats
import math

#Fonction pour ouvrir les fichiers
def ouvrirUnFichier(nom):
    with open(nom, "r", encoding="utf-8") as fichier:
        contenu = pd.read_csv(fichier, low_memory=False)
    return contenu

#Fonction pour convertir les données en données logarithmiques
def conversionLog(liste):
    log = []
    for element in liste:
        log.append(math.log(element))
    return log

#Fonction pour trier par ordre décroissant les listes (îles et populations)
def ordreDecroissant(liste):
    liste.sort(reverse = True)
    return liste

#Fonction pour obtenir le classement des listes spécifiques aux populations
def ordrePopulation(pop, etat):
    ordrepop = []
    for element in range(0, len(pop)):
        if np.isnan(pop[element]) == False:
            ordrepop.append([float(pop[element]), etat[element]])
    ordrepop = ordreDecroissant(ordrepop)
    for element in range(0, len(ordrepop)):
        ordrepop[element] = [element + 1, ordrepop[element][1]]
    return ordrepop

#Fonction pour obtenir l'ordre défini entre deux classements (listes spécifiques aux populations)
def classementPays(ordre1, ordre2):
    classement = []
    dict1 = {nom: rang for rang, nom in ordre1}
    dict2 = {nom: rang for rang, nom in ordre2}
    for nom, rang1 in dict1.items():
        if nom in dict2:
            classement.append([rang1, dict2[nom], nom])
    return classement

# Q1-2 - Partie sur les îles (chargement fichier)
iles = pd.DataFrame(ouvrirUnFichier("./src/data/island-index.csv"))

# Certaines colonnes contiennent des caractères spéciaux (km²) : on détecte la colonne Surface dynamiquement.
surface_colonnes = [col for col in iles.columns if "Surface" in col]
surface_col = surface_colonnes[0]  # Q3 - colonne surface

#Attention ! Il va falloir utiliser des fonctions natives de Python dans les fonctions locales que je vous propose pour faire l'exercice. Vous devez caster l'objet Pandas en list().

surface_iles = list(iles[surface_col].astype(float))
surface_iles.extend([
    float(85545323),
    float(37856841),
    float(7768030),
    float(7605049),
])  # Q3 - ajouts continents
surface_iles = ordreDecroissant(surface_iles)  # Q4 - tri décroissant
rangs_iles = list(range(1, len(surface_iles) + 1))

output_dir = os.path.join("src", "output", "img", "session6")
os.makedirs(output_dir, exist_ok=True)

# Q5 - Visualisation rang-taille (échelle linéaire)
plt.figure(figsize=(8, 5))
plt.plot(rangs_iles, surface_iles, "o-", markersize=4)
plt.xlabel("Rang")
plt.ylabel("Surface (km2)")
plt.title("Loi rang-taille des surfaces (îles + continents)")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "rang_taille_lineaire.png"), dpi=150)
plt.close()

# Q6 - Visualisation rang-taille (échelle log-log)
log_rangs_iles = conversionLog(rangs_iles)
log_surface_iles = conversionLog(surface_iles)
plt.figure(figsize=(8, 5))
plt.plot(log_rangs_iles, log_surface_iles, "o-", markersize=4)
plt.xlabel("log(Rang)")
plt.ylabel("log(Surface)")
plt.title("Loi rang-taille des surfaces (échelle log-log)")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "rang_taille_loglog.png"), dpi=150)
plt.close()

# Q7 - Oui, on peut tester les rangs avec une corrélation de Spearman (scipy.stats.spearmanr).






# Q8 - Partie sur les populations des États du monde
#Source. Depuis 2007, tous les ans jusque 2025, M. Forriez a relevé l'intégralité du nombre d'habitants dans chaque États du monde proposé par un numéro hors-série du monde intitulé États du monde. Vous avez l'évolution de la population et de la densité par année.
monde = pd.DataFrame(ouvrirUnFichier("./src/data/Le-Monde-HS-Etats-du-monde-2007-2025.csv"))

#Attention ! Il va falloir utiliser des fonctions natives de Python dans les fonctions locales que je vous propose pour faire l'exercice. Vous devez caster l'objet Pandas en list().

# Q9-10 - colonnes utiles
colonnes_population = ["État", "Pop 2007", "Pop 2025", "Densité 2007", "Densité 2025"]
monde_selection = monde[colonnes_population]
etat = list(monde_selection["État"])
pop_2007 = list(monde_selection["Pop 2007"])
pop_2025 = list(monde_selection["Pop 2025"])
densite_2007 = list(monde_selection["Densité 2007"])
densite_2025 = list(monde_selection["Densité 2025"])


# Q11 - classements décroissants pour population et densité (2007/2025)
ordre_pop_2007 = ordrePopulation(pop_2007, etat)
ordre_pop_2025 = ordrePopulation(pop_2025, etat)
ordre_dens_2007 = ordrePopulation(densite_2007, etat)
ordre_dens_2025 = ordrePopulation(densite_2025, etat)


# Q12 - rapprochement population/densité (2007) puis tri selon rang population 2007
classement_2007 = classementPays(ordre_pop_2007, ordre_dens_2007)
classement_2007.sort(key=lambda x: x[0])


# Q13 - extraction des deux colonnes de rangs (pop vs densité)
rang_pop_2007 = []
rang_dens_2007 = []
for ligne in classement_2007:
    rang_pop_2007.append(ligne[0])
    rang_dens_2007.append(ligne[1])


# Q14 - corrélations de rangs (Spearman / Kendall)
spearman_2007 = scipy.stats.spearmanr(rang_pop_2007, rang_dens_2007)
kendall_2007 = scipy.stats.kendalltau(rang_pop_2007, rang_dens_2007)
print("Spearman population vs densite 2007: rho={:.4f}, p={:.3e}".format(spearman_2007.statistic, spearman_2007.pvalue))
print("Kendall population vs densite 2007: tau={:.4f}, p={:.3e}".format(kendall_2007.statistic, kendall_2007.pvalue))


# Bonus - helpers génériques pour analyser les concordances de rangs
def correlations_rangs(rang_a, rang_b):
    return {
        "spearman": scipy.stats.spearmanr(rang_a, rang_b),
        "kendall": scipy.stats.kendalltau(rang_a, rang_b),
    }


def print_concordance(title, resultats):
    print(title)
    for annee in sorted(resultats.keys()):
        res = resultats[annee]
        sp = res["spearman"]
        kd = res["kendall"]
        print(
            "  {}: Spearman rho={:.4f}, p={:.3e} | Kendall tau={:.4f}, p={:.3e}".format(
                annee,
                float(sp.statistic),
                float(sp.pvalue),
                float(kd.statistic),
                float(kd.pvalue),
            )
        )


# Bonus - comparaison surfaces vs traits de côte & factorisation pour les classements annuels 2007-2025
coast_cols = [col for col in iles.columns if "coast" in col.lower() or "trait" in col.lower()]
if coast_cols:
    trait_cote = list(iles[coast_cols[0]].astype(float))
    # Align ranks par index (ordre décroissant)
    rang_surface = pd.Series(surface_iles).rank(ascending=False, method="min").tolist()
    rang_cote = pd.Series(trait_cote).rank(ascending=False, method="min").tolist()
    corr_iles = correlations_rangs(rang_surface[: len(rang_cote)], rang_cote)
    print(
        "Correlations rangs surfaces vs traits de cote: Spearman rho={:.4f}, p={:.3e} | Kendall tau={:.4f}, p={:.3e}".format(
            float(corr_iles["spearman"].statistic),
            float(corr_iles["spearman"].pvalue),
            float(corr_iles["kendall"].statistic),
            float(corr_iles["kendall"].pvalue),
        )
    )


def classements_annuels(df, prefix):
    colonnes = [c for c in df.columns if c.startswith(prefix)]
    rangs = {}
    for col in colonnes:
        valeurs = list(df[col])
        rangs[col] = ordrePopulation(valeurs, etat)
    return rangs


classements_pop = classements_annuels(monde, "Pop ")
classements_dens = classements_annuels(monde, "Densité ")

# Concordance des rangs de chaque année avec 2007 (population et densité)
def concordance_vs_ref(classements, annee_ref):
    ref = classements.get(annee_ref, [])
    ref_dict = {etat: rang for rang, etat in ref}
    resultats = {}
    for annee, classement in classements.items():
        if annee == annee_ref:
            continue
        rang_ref = []
        rang_cmp = []
        for rang, nom in classement:
            if nom in ref_dict:
                rang_cmp.append(rang)
                rang_ref.append(ref_dict[nom])
        resultats[annee] = correlations_rangs(rang_ref, rang_cmp)
    return resultats


concordance_pop_vers_2007 = concordance_vs_ref(classements_pop, "Pop 2007")
concordance_dens_vers_2007 = concordance_vs_ref(classements_dens, "Densité 2007")
print_concordance("Concordance des classements population vs 2007:", concordance_pop_vers_2007)
print_concordance("Concordance des classements densite vs 2007:", concordance_dens_vers_2007)


