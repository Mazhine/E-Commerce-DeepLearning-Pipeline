# Guide de test

## 1. Objectif

Ce guide explique comment verifier que le projet fonctionne et comment lancer chaque partie separement.

## 2. Prerequis

Il faut disposer d'un environnement Python avec les bibliotheques du fichier `requirements.txt`.

## 3. Installation

Depuis le dossier du projet, installer les dependances :

```bash
pip install -r requirements.txt
```

Si tu veux seulement verifier la structure des donnees :

```bash
python data/download_datasets.py
```

## 4. Jeux de donnees utilises

- Partie I : `data/tabular/online_shoppers_intention.csv`
- Partie II : `data/images/` pour Fashion-MNIST
- Partie III : `data/text/amazon_food_reviews.csv`

## 5. Tests recommandés

Tu peux lancer les tests de deux manieres :

- soit manuellement avec les commandes ci-dessous ;
- soit automatiquement avec `test_all.bat` ou `run_project.ps1`.

### Test rapide de l'ensemble du pipeline

```bash
python main.py --task all --quick-run
```

Ce mode lance les trois parties avec peu d'epoques pour verifier la coherence generale.

### Lanceur automatique Windows

```bash
test_all.bat
```

ou en PowerShell :

```powershell
.\run_project.ps1
```

### Test de la partie tabulaire uniquement

```bash
python main.py --task tabular --tabular-source local --tabular-csv data/tabular/online_shoppers_intention.csv
```

### Test de la partie image uniquement

```bash
python main.py --task image --image-root data/images
```

Lors du premier lancement, Fashion-MNIST sera telecharge automatiquement.

### Test de la partie sequences uniquement

```bash
python main.py --task text --text-source local_csv --text-csv data/text/amazon_food_reviews.csv --text-column text --summary-column summary
```

## 6. Sorties attendues

Les sorties d'experience doivent etre generees dans le dossier `artifacts/` :

- `artifacts/partie_1_mlp/`
- `artifacts/partie_2_cnn/`
- `artifacts/partie_3_sequences/`

On y attend notamment :

- des checkpoints `.pt`
- des fichiers `summary.json`
- des courbes d'apprentissage `.png`
- une matrice de confusion pour la partie MLP

## 7. Verification minimale

Le projet est considere comme correctement teste si :

- `main.py --help` affiche bien les options du script
- `python main.py --task all --quick-run` se lance sans erreur bloquante
- les dossiers `artifacts/partie_1_mlp`, `artifacts/partie_2_cnn` et `artifacts/partie_3_sequences` sont crees

## 8. Conseils avant rendu

- lancer au moins une execution reelle sans `--quick-run`
- remplir les tableaux de `annexe_experimentale.md` avec les vraies metriques obtenues
- ajouter les figures generees dans `artifacts/`
- harmoniser les resultats du rapport avec les sorties finales
