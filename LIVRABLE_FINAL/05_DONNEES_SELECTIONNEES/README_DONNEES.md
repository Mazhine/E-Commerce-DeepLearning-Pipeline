# Organisation des donnees

Le projet final utilise les jeux de donnees suivants :

- Partie I : UCI Online Shoppers Purchasing Intention Dataset
- Partie II : Fashion-MNIST
- Partie III : Hugging Face `NiyatiC/amazon_food_reviews`

## Telechargement

Le code a ete adapte pour automatiser au maximum la recuperation :

- le dataset tabulaire peut etre charge automatiquement depuis UCI via `ucimlrepo`;
- Fashion-MNIST peut etre telecharge automatiquement via `torchvision`;
- le corpus textuel peut etre charge automatiquement depuis Hugging Face via `datasets`.

Tu peux aussi pretelecharger les datasets avec le script :

```bash
python data/download_datasets.py
```

## Structure locale optionnelle

Si tu preferes conserver des copies locales exportees :

```text
data/
  tabular/
    online_shoppers_intention.csv
  images/
    FashionMNIST/
  text/
    amazon_food_reviews.csv
```

## Mapping des colonnes retenu

### Partie I - Tabulaire

- source : UCI
- cible : `Revenue`

### Partie II - Images

- source : `torchvision.datasets.FashionMNIST`
- classes : 10 categories de produits

### Partie III - Texte

- source : `NiyatiC/amazon_food_reviews`
- texte source : `Text`
- texte cible : `Summary`

Le choix `Text -> Summary` est ideal pour un mini probleme de generation de resume court, directement compatible avec la partie Seq2Seq du devoir.
