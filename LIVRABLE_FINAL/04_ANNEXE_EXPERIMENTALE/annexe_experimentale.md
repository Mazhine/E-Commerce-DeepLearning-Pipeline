# Annexe experimentale

## 1. Protocoles d'experience

### Partie I - MLP

- Split : 64 pour cent apprentissage, 16 pour cent validation, 20 pour cent test
- Batch size recommande : 64
- Optimiseur : Adam
- Taux d'apprentissage initial : 1e-3
- Initialisations comparees : gaussienne, constante, Xavier
- Architectures comparees : `SequentialMLP`, `CustomMLP`

### Partie II - CNN

- Dataset : Fashion-MNIST
- Batch size recommande : 128
- Baseline : `ImageMLP`
- Modele principal : `LeNetLikeCNN`
- Variantes : padding, stride, pooling, nombre de filtres, convolution 1x1

### Partie III - Sequences

- Colonnes attendues : `text`, `summary`
- Batch size recommande : 32 ou 64 selon la machine
- Tokenisation : decoupage simple par espaces
- Architectures inspectees : `RNN`, `LSTM`, `GRU`
- Systeme de generation : `Seq2SeqModel`

## 2. Tableaux comparatifs a remplir apres execution

### Tableau A - Partie I

| Modele | Initialisation | Accuracy | Precision | Recall | F1-score |
|---|---|---:|---:|---:|---:|
| SequentialMLP | Gaussian | a completer | a completer | a completer | a completer |
| SequentialMLP | Constant | a completer | a completer | a completer | a completer |
| SequentialMLP | Xavier | a completer | a completer | a completer | a completer |
| CustomMLP | Gaussian | a completer | a completer | a completer | a completer |
| CustomMLP | Constant | a completer | a completer | a completer | a completer |
| CustomMLP | Xavier | a completer | a completer | a completer | a completer |

### Tableau B - Partie II

| Modele | Padding | Stride | Pooling | Conv1x1 | Accuracy | F1-score |
|---|---:|---:|---|---|---:|---:|
| ImageMLP | NA | NA | NA | Non | a completer | a completer |
| CNN | 0 | 1 | Max | Non | a completer | a completer |
| CNN | 2 | 1 | Max | Oui | a completer | a completer |
| CNN | 2 | 1 | Average | Oui | a completer | a completer |

### Tableau C - Partie III

| Modele | Role | Observation principale |
|---|---|---|
| RNN | Encodage sequentiel | a completer |
| LSTM | Encodage sequentiel | a completer |
| GRU | Encodage sequentiel | a completer |
| Seq2Seq + Greedy | Generation | a completer |
| Seq2Seq + Beam Search | Generation | a completer |

## 3. Figures attendues

- `artifacts/partie_1_mlp/*_history.png`
- `artifacts/partie_1_mlp/best_mlp_confusion_matrix.png`
- `artifacts/partie_2_cnn/cnn_history.png`
- `artifacts/partie_2_cnn/image_mlp_history.png`
- `artifacts/partie_3_sequences/seq2seq_history.png`

## 4. Points d'analyse a commenter dans le rendu

- effet du preprocessing sur la stabilite du MLP;
- difference pratique entre `nn.Sequential` et classe personnalisee;
- impact de l'initialisation sur la convergence;
- superiorite du CNN sur le MLP image en termes de biais inductif;
- influence du pooling et du padding sur la qualite des representations;
- interet du gradient clipping pour la stabilite des modeles sequentiels;
- difference qualitative entre decodage glouton et beam search.
