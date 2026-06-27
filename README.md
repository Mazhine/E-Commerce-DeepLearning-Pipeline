# Pipeline d'IA Multimodal pour l'E-Commerce — Projet Deep Learning EMSI 2025-2026

> Conception, implémentation et évaluation d'architectures de Deep Learning (MLP, CNN, LSTM, Seq2Seq) appliquées à trois modalités de données e-commerce : tabulaire, image et texte.

---

## 📄 Rapport Scientifique

Le rapport académique complet (30+ pages, avec équations, tableaux de résultats, bibliographie) est disponible dans le dossier `rapport/` :

- [`rapport/rapport_final.tex`](rapport/rapport_final.tex) — Source LaTeX (compilable sur [Overleaf](https://overleaf.com))
- [`rapport/rapport_final (1).pdf`](<rapport/rapport_final (1).pdf>) — Version PDF compilée ⬇️

---

## 🗂️ Structure du Repository

```
.
├── src/                        # Code source modulaire PyTorch
│   ├── data/                   # Chargement et prétraitement des données
│   ├── models/                 # Architectures MLP, CNN, RNN/LSTM/GRU, Seq2Seq
│   ├── training/               # Boucles d'entraînement communes
│   └── utils/                  # Métriques, visualisation, seed, device
├── notebooks/
│   └── projet_deep_learning_emsi.ipynb   # Notebook de démonstration
├── rapport/
│   ├── rapport_final.tex       # Rapport LaTeX académique complet
│   └── rapport_final (1).pdf  # PDF compilé
├── data/
│   ├── download_datasets.py    # Script de téléchargement automatique
│   ├── tabular/                # UCI Online Shoppers (CSV)
│   ├── text/                   # Amazon Food Reviews (CSV)
│   └── images/                 # Fashion-MNIST (téléchargé automatiquement)
├── artifacts/                  # Résultats : modèles .pt, courbes .png, métriques .json
├── main.py                     # Point d'entrée principal
└── requirements.txt            # Dépendances Python
```

---

## 🧠 Architecture du Projet

| Partie | Données | Modèle | Meilleure Accuracy |
|--------|---------|--------|--------------------|
| Partie I | Tabulaire — UCI Online Shoppers (12 330 sessions, 29 features) | MLP (2 variantes × 3 initialisations) | **88.85%** |
| Partie II | Images — Fashion-MNIST (70 000 images 28×28, 10 classes) | CNN (LeNet adapté) vs MLP baseline | **88.76%** (CNN) vs 85.27% (MLP) |
| Partie III | Texte — Amazon Food Reviews | RNN / LSTM / GRU / Seq2Seq | Entraînement convergé ✅ |

---

## 🗃️ Jeux de Données

| Dataset | Modalité | Source | Taille |
|---------|----------|--------|--------|
| UCI Online Shoppers Purchasing Intention | Tabulaire | [UCI ML Repository](https://archive.ics.uci.edu/dataset/468) | 12 330 sessions |
| Fashion-MNIST | Images | [Torchvision](https://pytorch.org/vision/stable/datasets.html) | 70 000 images |
| Amazon Fine Food Reviews | Texte | [Hugging Face](https://huggingface.co/datasets/NiyatiC/amazon_food_reviews) | 568 454 avis |

---

## ⚡ Lancement Rapide

```bash
# 1. Installer les dépendances
pip install -r requirements.txt

# 2. Télécharger les données
python data/download_datasets.py

# 3. Tester le pipeline complet (mode rapide)
python main.py --task all --quick-run

# 4. Entraînement d'une partie spécifique
python main.py --task tabular   # Partie I  — MLP
python main.py --task image     # Partie II — CNN
python main.py --task sequence  # Partie III — Séquentiel
```

> **Note :** Le mode `--quick-run` réduit le nombre d'époques pour valider rapidement le pipeline. Supprimer ce flag pour un entraînement complet.

---

## 🛠️ Technologies

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange?logo=pytorch)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-yellow?logo=jupyter)
![Git](https://img.shields.io/badge/Git-GitHub-black?logo=github)
