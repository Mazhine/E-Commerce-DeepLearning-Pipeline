# Rapport scientifique

## Titre

Etude comparative de modeles de deep learning sur donnees multimodales e-commerce : MLP pour la prediction d'achat, CNN pour la classification visuelle de produits et architectures recursives pour la generation de resumes d'avis clients

## Introduction

Le deep learning ne se limite pas a l'application d'un meme schema d'apprentissage a tous les problemes. La structure d'un jeu de donnees conditionne fortement le choix architectural, la nature des representations apprises et les contraintes d'entrainement. Dans un contexte e-commerce, cette diversite est immediate : les plateformes manipulent des donnees tabulaires relatives au comportement utilisateur, des images de produits et des textes issus des avis clients. Le present projet se propose d'etudier comment des familles de modeles distinctes repondent a ces structures de donnees heterogenes.

Le fil conducteur retenu est celui d'un pipeline multimodal e-commerce. La premiere partie porte sur la prediction de l'intention d'achat a partir de donnees tabulaires de sessions utilisateurs. La deuxieme partie traite la classification d'images de produits, dans une logique de categorisation automatique. La troisieme partie porte sur la modelisation de sequences textuelles et la generation de resumes d'avis clients.

## Objectifs

Les objectifs du projet sont les suivants :

- justifier le choix d'un MLP pour la classification tabulaire tout en discutant ses limites;
- montrer la superiorite structurelle d'un CNN sur un MLP pour des images;
- comparer RNN, LSTM et GRU pour la modelisation de sequences;
- construire un mini systeme Seq2Seq pour generer un resume textuel;
- relier theorie, implementation PyTorch, experimentation et interpretation scientifique.

## Partie I - Donnees tabulaires et MLP

### Problematique

Dans quelle mesure un MLP bien parametre constitue-t-il une solution pertinente pour predire l'intention d'achat d'un utilisateur a partir de variables de session, de navigation et de contexte ?

### Jeu de donnees

Le dataset retenu est Online Shoppers Purchasing Intention, publie sur UCI. Il s'agit d'un jeu de donnees tabulaires binaire dans lequel la variable cible `Revenue` indique si la session s'est conclue par un achat.

### Methodologie

La preparation suit quatre etapes : nettoyage, imputation, encodage des variables categorielles et normalisation des variables numeriques. Le split retenu est apprentissage / validation / test. Deux architectures MLP sont comparees :

- une implementation basee sur `nn.Sequential`;
- une implementation personnalisee basee sur `nn.Module`.

Trois strategies d'initialisation sont testees :

- initialisation gaussienne;
- initialisation constante;
- initialisation Xavier.

Les metriques utilisees sont l'accuracy, la precision, le recall, le F1-score et la matrice de confusion.

### Interpretation attendue

Un MLP peut bien fonctionner sur des donnees tabulaires lorsque le preprocessing est soigne et que les interactions non lineaires entre variables sont significatives. En revanche, ses limites apparaissent rapidement face a des distributions heterogenes, a la rarete des classes positives et a l'absence de biais inductif specifique au tabulaire.

## Partie II - Images et CNN

### Problematique

Pourquoi un CNN est-il plus adapte qu'un MLP pour categoriser automatiquement des images de produits ?

### Jeu de donnees

Le projet s'appuie sur Fashion-MNIST, accessible via `torchvision`, choisi pour conserver une coherence metier avec l'e-commerce tout en gardant une charge de calcul raisonnable.

### Methodologie

La partie theorique rappelle la localite, le partage des poids et la hierarchie des representations. Une implementation manuelle de la correlation croisee 2D, du max-pooling et de l'average-pooling est fournie pour relier le formalisme mathematique aux couches PyTorch.

Le modele principal est un CNN de type LeNet avec variantes experimentales :

- modification du padding;
- modification du stride;
- comparaison max-pooling / average-pooling;
- variation du nombre de filtres;
- presence ou absence d'une convolution 1x1.

Un MLP applique aux memes images sert de baseline comparative.

### Interpretation attendue

Le CNN doit apprendre des filtres spatiaux plus robustes et plus econome en parametres qu'un MLP. Le padding permet de mieux preserver l'information en bordure, le stride controle la resolution spatiale, le pooling introduit une forme d'invariance locale, et la convolution 1x1 agit comme un mecanisme de recombinaison des canaux.

## Partie III - Sequences textuelles et Seq2Seq

### Problematique

Comment justifier le passage d'un RNN simple vers un LSTM ou un GRU, puis vers un schema encodeur-decodeur pour generer des resumes d'avis clients ?

### Jeu de donnees

Le corpus retenu est `NiyatiC/amazon_food_reviews`, accessible via Hugging Face. Le champ `Text` est utilise comme sequence source et `Summary` comme sequence cible. Ce choix permet de construire un probleme realiste de resume court de contenu client en contexte e-commerce.

### Methodologie

Les textes sont tokenises, transformes en sequences entieres, puis completes avec des tokens speciaux de debut, fin, inconnus et padding. Un vocabulaire simple est construit pour rester pedagogiquement transparent. Les architectures recurrentes RNN, LSTM et GRU sont comparees selon :

- leur stabilite a l'entrainement;
- leur capacite de memorisation;
- leur cout de calcul;
- la qualite des representations produites.

Un mini systeme Seq2Seq est ensuite entraine pour generer des resumes, avec `teacher forcing` et `gradient clipping`. Deux strategies de decodage sont examinees :

- decodage glouton;
- beam search.

### Interpretation attendue

Le RNN simple souffre rapidement de difficultes de memorisation lorsqu'une dependance s'etend sur plusieurs tokens. Les cellules LSTM et GRU ameliorent ce point grace a leurs mecanismes de portes. Le schema encodeur-decodeur devient pertinent lorsque l'objectif n'est plus seulement de predire le prochain token, mais de transformer une sequence source en sequence cible.

## Discussion transversale

Les trois parties montrent qu'un meme paradigme supervise ne peut pas etre applique de facon uniforme a toutes les structures de donnees. Le MLP convient a des vecteurs tabulaires deja structures. Le CNN exploite explicitement la geometrie locale des images. Les architectures recurrentes et Seq2Seq traitent des dependances temporelles et symboliques dans les sequences textuelles. Ainsi, le deep learning n'est pas uniquement une question de profondeur des modeles, mais d'adequation entre hypothese architecturale et forme des donnees.

## Limites

- Les resultats dependent de la qualite des jeux de donnees reellement utilises.
- Le tokenizer retenu ici est simple et pedagogique, non optimal pour une production avancee.
- Le mini Seq2Seq reste volontairement compact pour rester faisable dans le cadre d'un module.
- Les conclusions experimentales devront etre recalibrees apres execution complete sur les datasets finaux.

## Conclusion

Ce projet montre qu'un scenario e-commerce permet d'unifier de maniere coherente les trois grandes familles de modeles etudiees dans le module : MLP, CNN et architectures recurrentes. Il met en evidence l'importance des choix de representation, de preprocessing et d'architecture. Cette approche presente en outre un interet fort pour un profil data engineer, car elle articule preparation de donnees, experimentation reproductible, modelisation multimodale et interpretation rigoureuse.
