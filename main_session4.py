#coding:utf8

import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st

dist_names = [
    "norm",
    "beta",
    "gamma",
    "pareto",
    "t",
    "lognorm",
    "invgamma",
    "invgauss",
    "loggamma",
    "alpha",
    "chi",
    "chi2",
    "bradford",
    "burr",
    "burr12",
    "cauchy",
    "dweibull",
    "erlang",
    "expon",
    "exponnorm",
    "exponweib",
    "exponpow",
    "f",
    "genpareto",
    "gausshyper",
    "gibrat",
    "gompertz",
    "gumbel_r",
    "pareto",
    "pearson3",
    "powerlaw",
    "triang",
    "weibull_min",
    "weibull_max",
    "bernoulli",
    "betabinom",
    "betanbinom",
    "binom",
    "geom",
    "hypergeom",
    "logser",
    "nbinom",
    "poisson",
    "poisson_binom",
    "randint",
    "zipf",
    "zipfian",
]
print(dist_names)

output_dir = os.path.join("src", "output", "img", "session4")
os.makedirs(output_dir, exist_ok=True)


def mean_std(dist, *args, **kwargs):
    m, v = dist.stats(*args, **kwargs, moments="mv")
    return float(m), float(np.sqrt(v))


def plot_discrete(name, x, pmf):
    plt.figure(figsize=(6, 4))
    markerline, _, _ = plt.stem(x, pmf)
    plt.setp(markerline, markersize=6)
    plt.title(f"{name} (discrete)")
    plt.xlabel("x")
    plt.ylabel("PMF")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{name}_pmf.png"), dpi=150)
    plt.close()


def plot_continuous(name, x, pdf):
    plt.figure(figsize=(6, 4))
    plt.plot(x, pdf)
    plt.title(f"{name} (continue)")
    plt.xlabel("x")
    plt.ylabel("PDF")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{name}_pdf.png"), dpi=150)
    plt.close()


print("=== Distributions discretes ===")

# Dirac en 0
x_dirac = np.array([0])
pmf_dirac = np.array([1.0])
plot_discrete("dirac", x_dirac, pmf_dirac)
print("dirac", "moyenne=0", "ecart_type=0")

# Uniforme discrete sur {0,...,4}
x_uni = np.arange(0, 5)
pmf_uni = st.randint.pmf(x_uni, 0, 5)
plot_discrete("uniforme_discrete", x_uni, pmf_uni)
m, s = mean_std(st.randint, 0, 5)
print("uniforme_discrete", "moyenne=", round(m, 3), "ecart_type=", round(s, 3))

# Binomiale n=20, p=0.3
x_binom = np.arange(0, 21)
pmf_binom = st.binom.pmf(x_binom, n=20, p=0.3)
plot_discrete("binomiale", x_binom, pmf_binom)
m, s = mean_std(st.binom, 20, 0.3)
print("binomiale", "moyenne=", round(m, 3), "ecart_type=", round(s, 3))

# Poisson lambda=4
x_pois = np.arange(0, 20)
pmf_pois = st.poisson.pmf(x_pois, mu=4)
plot_discrete("poisson_discrete", x_pois, pmf_pois)
m, s = mean_std(st.poisson, 4)
print("poisson_discrete", "moyenne=", round(m, 3), "ecart_type=", round(s, 3))

# Zipf-Mandelbrot (zipf) a=2.0
x_zipf = np.arange(1, 20)
pmf_zipf = st.zipf.pmf(x_zipf, a=2.0)
pmf_zipf = pmf_zipf / pmf_zipf.sum()
plot_discrete("zipf_mandelbrot", x_zipf, pmf_zipf)
m, s = mean_std(st.zipf, 2.0)
print("zipf_mandelbrot", "moyenne=", round(m, 3), "ecart_type=", round(s, 3))


print("\n=== Distributions continues ===")

# Poisson (approx en continu via pmf lisse)
x_pois_cont = np.linspace(0, 20, 400)
pdf_pois_cont = st.poisson.pmf(np.round(x_pois_cont), mu=4)
plot_continuous("poisson_continue", x_pois_cont, pdf_pois_cont)
m, s = mean_std(st.poisson, 4)
print("poisson_continue", "moyenne=", round(m, 3), "ecart_type=", round(s, 3))

# Normale mu=0 sigma=1
x_norm = np.linspace(-4, 4, 400)
pdf_norm = st.norm.pdf(x_norm, loc=0, scale=1)
plot_continuous("normale", x_norm, pdf_norm)
m, s = mean_std(st.norm, 0, 1)
print("normale", "moyenne=", round(m, 3), "ecart_type=", round(s, 3))

# Log-normale s=0.5
x_logn = np.linspace(0.01, 5, 400)
pdf_logn = st.lognorm.pdf(x_logn, s=0.5)
plot_continuous("lognormale", x_logn, pdf_logn)
m, s = mean_std(st.lognorm, 0.5)
print("lognormale", "moyenne=", round(m, 3), "ecart_type=", round(s, 3))

# Uniforme continue [0,1]
x_unif = np.linspace(0, 1, 200)
pdf_unif = st.uniform.pdf(x_unif, loc=0, scale=1)
plot_continuous("uniforme_continue", x_unif, pdf_unif)
m, s = mean_std(st.uniform, 0, 1)
print("uniforme_continue", "moyenne=", round(m, 3), "ecart_type=", round(s, 3))

# Chi2 k=4
x_chi2 = np.linspace(0, 20, 400)
pdf_chi2 = st.chi2.pdf(x_chi2, df=4)
plot_continuous("chi2", x_chi2, pdf_chi2)
m, s = mean_std(st.chi2, 4)
print("chi2", "moyenne=", round(m, 3), "ecart_type=", round(s, 3))

# Pareto b=3
x_pareto = np.linspace(1, 5, 400)
pdf_pareto = st.pareto.pdf(x_pareto, b=3)
plot_continuous("pareto", x_pareto, pdf_pareto)
m, s = mean_std(st.pareto, 3)
print("pareto", "moyenne=", round(m, 3), "ecart_type=", round(s, 3))
