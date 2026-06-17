# Livrable final - Projet Deep Learning EMSI

Ce dossier regroupe tous les elements necessaires a la remise du projet, organises de maniere claire.

## Contenu

### 01_RAPPORT

- `rapport_scientifique.md`
- Contient l'introduction, les objectifs, la methodologie, l'implementation, l'interpretation, les limites et la conclusion.

### 02_CODE_SOURCE

- Dossier `src/` copie dans cette section.
- Contient tout le code source modulaire PyTorch pour les parties MLP, CNN et modeles sequentiels.

### 03_NOTEBOOK_OU_SCRIPT

- `main.py` : script principal executable
- `projet_deep_learning_emsi.ipynb` : notebook de demonstration

### 04_ANNEXE_EXPERIMENTALE

- `annexe_experimentale.md`
- Contient les protocoles, tableaux comparatifs, figures attendues et points d'analyse.

### 05_DONNEES_SELECTIONNEES

- `online_shoppers_intention.csv` : dataset tabulaire
- `amazon_food_reviews.csv` : dataset textuel
- `download_datasets.py` : script de recuperation des datasets
- `README_DONNEES.md` : explication des sources et du mapping de colonnes

### 06_GUIDE_TEST

- `GUIDE_DE_TEST.md` : procedure de test du projet
- `requirements.txt` : dependances Python
- `README_PROJET.md` : vue d'ensemble du projet

## Remarque

Fashion-MNIST n'est pas stocke ici sous forme CSV car il est telecharge automatiquement par `torchvision` lors de la premiere execution de la partie image.
