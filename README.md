# Projet Deep Learning EMSI 2025-2026

Ce depot contient un projet complet et structure pour le module de deep learning autour d'un fil conducteur e-commerce multimodal :

- Partie I : prediction de l'intention d'achat a partir de donnees tabulaires avec MLP
- Partie II : classification d'images de produits avec CNN
- Partie III : modelisation de sequences textuelles et generation de resumes d'avis clients avec RNN, LSTM, GRU et Seq2Seq

## Structure

- `src/` : code source modulaire PyTorch
- `main.py` : point d'entree principal executable
- `notebooks/` : notebook de demonstration et d'explication
- `rapport/` : dossier contenant le rapport scientifique au format LaTeX
- `data/README.md` : jeux de donnees recommandes et organisation attendue
- `artifacts/` : sorties d'entrainement, checkpoints et figures generees

## Jeux de donnees retenus

- Tabulaire : UCI Online Shoppers Purchasing Intention Dataset
- Images : Fashion-MNIST
- Texte : Hugging Face `NiyatiC/amazon_food_reviews`

## Execution rapide

```bash
python main.py --help
python main.py --task all --quick-run
python data/download_datasets.py
```

Le mode `--quick-run` permet de verifier le pipeline avec peu d'epoques. Pour un rendu final, il faut des experiences plus longues et la sauvegarde des figures dans `artifacts/`.

Apres execution du script de telechargement, `main.py` utilise par defaut :

- `data/tabular/online_shoppers_intention.csv`
- `data/text/amazon_food_reviews.csv`

## Remarques importantes

- Le code est fortement commente pour faciliter la comprehension pedagogique.
- Le projet peut utiliser automatiquement des sources distantes si les fichiers locaux sont absents.
- Les resultats du rapport sont presentes comme une trame scientifique defendable. Ils devront etre alignes avec tes executions reelles avant remise finale.
