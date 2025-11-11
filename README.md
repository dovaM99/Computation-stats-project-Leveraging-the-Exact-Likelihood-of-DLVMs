# Project README — Leveraging the Exact Likelihood of DLVMs

> **Course:** Computational Statistics (M2)
> **Presentation date:** **7 January 2026**
> **Team:** 2 students • ~6h/week from now until the talk

---

## 📌 Paper Under Study (to cite)

**Mattei, P.-A., & Frellsen, J. (2018). *Leveraging the Exact Likelihood of Deep Latent Variable Models*.** arXiv:1802.04826v4.
Key results: ill-posedness of ML for Gaussian-output DLVMs; spectral constraints that ensure bounded likelihood; data-dependent nonparametric upper bounds (NPMLE) yielding a **parsimony gap**; exact missing-data imputation via **Metropolis-within-Gibbs** (MWG) with the same computational budget as pseudo-Gibbs.    

> **How to cite in slides/report:**
> Mattei, P.-A., & Frellsen, J. (2018). Leveraging the Exact Likelihood of Deep Latent Variable Models. *arXiv:1802.04826*.

---

## 🎯 Goals

1. Understand and present the **theory**: boundedness/ill-posedness, NPMLE bound, parsimony gap, convergence arguments.  
2. **Reproduce** minimal experiments: likelihood blow-up on Frey Faces; MWG vs pseudo-Gibbs for imputation (MNIST/OMNIGLOT/Caltech silhouettes).  
3. Deliver a clean **15-min talk** + Q&A with proof sketches and code-backed figures.

---

## 🗺️ Roadmap (Week-by-Week)

**Week 1 (Nov 11–17)** — *Big-picture pass + notes*

* Skim full paper; list main statements (Thm/Prop) and where they’re used in practice.
* Build a glossary: ELBO, amortization/approximation/parsimony gaps. 
* Output: **1-page plan** + glossary.

**Week 2 (Nov 18–24)** — *Ill-posedness & toy demo*

* Re-derive ill-posedness for Gaussian-output DLVMs; connect to **mode collapse**. 
* Toy 1D mixture: show likelihood blow-up as variance → 0.
* Output: `02_blowup_toy.ipynb`.

**Week 3 (Nov 25–Dec 1)** — *Spectral constraints & boundedness*

* Prove boundedness with **Σ(z) ⪰ ξI**; implement variance clamp (logσ² ≥ log ξ) in a VAE. 
* Output: minimal PyTorch VAE (Gaussian decoder).

**Week 4 (Dec 2–8)** — *Nonparametric mixture upper bound*

* Study NPMLE upper bound `(Ĝ)` and why the maximizer has **k ≤ n** components (Lindsay + compactification).  
* Approximate `(Ĝ)` with constrained GaussianMixture (sklearn `reg_covar=ξ`).
* Output: `03_sandwich_likelihood.ipynb` (ELBO ≤ log-lik ≤ NPMLE).

**Week 5 (Dec 9–15)** — *Parsimony gap & tightness (large-capacity)*

* Understand Theorem 3 and universal approximation requirement; make a clean proof sketch. 
* Ablation: vary decoder capacity `h` and observe parsimony gap.
* Output: figures for slides.

**Week 6 (Dec 16–22)** — *Missing-data imputation*

* Implement **pseudo-Gibbs** and **MWG**; re-derive acceptance ratio (eq. (15)).  
* Datasets: MNIST (bin), OMNIGLOT, Caltech silhouettes; MAR {40–80%}, top/bottom.
* Output: `04_imputation_mwg.ipynb` + F1 curves. 

**Week 7 (Dec 23–29)** — *Stabilize & document*

* Seed everything, freeze requirements, regenerate all figs.
* Write **“Notes de preuves & pièges”** (2–3 pp).

**Week 8 (Dec 30–Jan 5)** — *Slides & dry-runs*

* Final slides + 15’ talk timing; prepare backup proofs/lemmas.
* Final repo cleanup.

**Jan 6 (Eve of talk)** — *Full rehearsal & pack release*

---

## 👥 Task Split (2 people)

**A — Theory lead**

* Ill-posedness & boundedness proofs; NPMLE existence (k ≤ n); parsimony gap tightness.    
* Draft “Notes de preuves & pièges” with figure sketches.

**B — Experiments lead**

* VAE (Gaussian decoder; clamp on variance), Frey Faces blow-up replication. 
* Sandwich bound with sklearn GMM; MWG vs pseudo-Gibbs imputation (+ Wilcoxon as in paper). 

> Weekly 1h cross-review (A reviews code; B reviews proofs).

---

## 🧪 Mini-Project: “Sandwich the Likelihood”

**Goal:** Empirically bracket the **exact** log-likelihood between **ELBO** and **NPMLE** and study the **parsimony gap** across model capacities; then compare **MWG** vs **pseudo-Gibbs** for imputation at fixed compute.  

**Steps:**

1. Train VAE (Gaussian outputs) on Frey Faces with and without variance clamp; reproduce **likelihood blow-up** figure. 
2. Fit constrained GMM (reg_covar=ξ) on training data to estimate `(Ĝ)`; plot **ELBO**, **NPMLE bound**, gap. 
3. Do imputation on MNIST/OMNIGLOT/Caltech silhouettes with **pseudo-Gibbs** and **MWG** (same T, same VAE); report **F1 vs iterations** and **Wilcoxon** tests. 

**Deliverables:**

* `02_blowup_toy.ipynb`
* `03_sandwich_likelihood.ipynb`
* `04_imputation_mwg.ipynb`
* Figures mirroring paper’s plots (blow-up, F1).  

---

## 🧱 Repo Structure

```
.
├── README.md  (this file)
├── notes/
│   └── 01_theory_notes.pdf   # proofs & pitfalls (2–3 pp)
├── data/   # scripts to fetch/preprocess Frey Faces, MNIST, OMNIGLOT, Caltech silhouettes
├── notebooks/
│   ├── 02_blowup_toy.ipynb
│   ├── 03_sandwich_likelihood.ipynb
│   └── 04_imputation_mwg.ipynb
├── models/
│   └── vae.py                # Gaussian decoder with variance clamp (ξ)
├── train.py                  # train VAE
├── impute.py                 # pseudo-Gibbs & MWG samplers
├── requirements.txt
└── figs/                     # saved plots for slides
```

---

## 🧰 Implementation Notes (to match paper)

* **VAE training:** pathwise gradients (Roeder et al., 2017), Adam lr=1e-4, minibatch=10; Glorot init; TF Distributions or PyTorch equivalents. 
* **Frey Faces setup:** Gaussian DLVM, `d=5`, hidden `h=200`; same for inference net. 
* **Constrained GMM:** sklearn `GaussianMixture(reg_covar=ξ)` to proxy `(Ĝ)`. 
* **MWG acceptance (eq. 15):** ratio of importance ratios using the variational posterior as proposal; reduces to Gibbs if posterior is exact. 

---

## 🖥️ Slide Outline (15’)

1. **Setup & ELBO refresher** (1.5’) — DLVMs, ELBO, encoder/decoder. 
2. **Ill-posedness vs boundedness** (3’) — Gaussian blow-up; Bernoulli well-posed; spectral constraint.  
3. **Nonparametric upper bound & parsimony gap** (3’) — `(Ĝ)` with k ≤ n; gap definition; sandwiching inequality. 
4. **Tightness (large capacity)** (2’) — universal approximation ⇒ `(Ĝ) ≈ `(θ)`; proof sketch. 
5. **Imputation: pseudo-Gibbs vs MWG** (2.5’) — algorithm & acceptance ratio; results.  
6. **Our reproductions** (2.5’) — blow-up figure; sandwich curves; F1 curves.

---

## ❓ Anticipated Questions

* **Where exactly is compactness used for the NPMLE?** — Lindsay’s geometric likelihood with Alexandroff compactification for Gaussian outputs under spectral constraint. 
* **Why does blow-up relate to mode collapse?** — Infinite likelihood attained at near-singular covariances yields highly concentrated, poor generative models. 
* **Does MWG cost more?** — Essentially same compute as pseudo-Gibbs (plus cheap prior evals) but targets the **true** conditional; initialize with a few pseudo-Gibbs steps. 

---

## 📚 Prerequisites & Related Reading (to cite)

* **Variational Inference / VAEs:**
  Kingma & Welling (2014); Burda et al. (2016); Cremer et al. (2018).  
* **Nonparametric Mixtures / NPMLE:**
  Lindsay (1983/1995); Hathaway (1985); van der Vaart & Wellner (1992); Kiefer & Wolfowitz (1956).  
* **MCMC / Imputation:**
  Metropolis (1953), Hastings (1970), Geman & Geman (1984), Gelman (1993). 

---

## ✅ Success Criteria

* Clear **proof sketches** aligned with the paper’s arguments.
* Reproduced **blow-up** and **imputation** plots that qualitatively match the paper.  
* A compact **15-min** presentation with confident Q&A on convergence/compactness arguments.

---

## 🔗 Key Formulas/Objects (for quick reference)

* **ELBO:** `ELBO(θ,q) = ℓ(θ) − KL(q || p(·|X)) ≤ ℓ(θ)` (eq. (4)). 
* **NPMLE bound:** `ℓ(θ) ≤ ℓ(Ĝ)`; **parsimony gap** = `ℓ(Ĝ) − ℓ(θ)`. 
* **MWG acceptance (eq. 15):** ratio using variational posterior as proposal; equals 1 if posterior is exact. 

---

*This README provides the plan, division of labor, minimal code artifacts, and canonical citations to execute the project and deliver a rigorous talk on Mattei & Frellsen (2018).*

