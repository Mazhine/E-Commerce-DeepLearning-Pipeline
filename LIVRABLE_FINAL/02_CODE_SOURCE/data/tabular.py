from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class TabularDataBundle:
    """Conteneur pratique pour tout ce que la partie tabulaire doit transporter.

    On y place :
    - les trois DataLoader
    - la dimension d'entree du MLP
    - les noms de classes
    - le preprocesseur sklearn
    """

    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    input_dim: int
    class_names: List[str]
    preprocessor: ColumnTransformer


def load_tabular_dataset(
    csv_path: str | Path,
    target_column: str = "Revenue",
    source: str = "auto",
) -> pd.DataFrame:
    """Charge le dataset tabulaire.

    Le projet est centre sur un cas e-commerce. Par defaut, la cible attendue est
    `Revenue`, une variable binaire souvent presente dans le dataset Online Shoppers.
    """

    csv_path = Path(csv_path)
    if csv_path.exists():
        return pd.read_csv(csv_path)

    if source in ("auto", "uci"):
        try:
            from ucimlrepo import fetch_ucirepo
        except ImportError as exc:
            raise FileNotFoundError(
                f"Dataset tabulaire introuvable localement ({csv_path}) et `ucimlrepo` n'est pas disponible."
            ) from exc

        dataset = fetch_ucirepo(id=468)
        features = dataset.data.features.copy()
        targets = dataset.data.targets.copy()

        # La cible peut arriver en DataFrame booleen ou en serie.
        if hasattr(targets, "columns"):
            target_values = targets.iloc[:, 0]
        else:
            target_values = targets
        dataframe = features.copy()
        dataframe[target_column] = target_values.astype(int)
        return dataframe

    raise FileNotFoundError(f"Dataset tabulaire introuvable : {csv_path}")


def build_tabular_loaders(
    dataframe: pd.DataFrame,
    target_column: str = "Revenue",
    batch_size: int = 64,
    random_state: int = 42,
) -> TabularDataBundle:
    """Prepare les donnees tabulaires pour l'entrainement d'un MLP.

    On separe nettement les etapes de preparation afin de pouvoir les documenter
    proprement dans le rapport : nettoyage, encodage, normalisation et split.
    """

    df = dataframe.copy()
    # On s'assure que la cible est bien sous forme binaire entiere.
    df[target_column] = df[target_column].astype(int)

    x = df.drop(columns=[target_column])
    y = df[target_column].values

    # On separe les colonnes selon leur nature pour appliquer le bon preprocessing.
    categorical_columns = x.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    numerical_columns = [col for col in x.columns if col not in categorical_columns]

    # Pipeline pour les variables numeriques :
    # imputation des manquants puis standardisation.
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    # Pipeline pour les variables categorielles :
    # imputation puis encodage one-hot.
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    # `ColumnTransformer` permet d'appliquer des traitements differents
    # selon les familles de colonnes.
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numerical_columns),
            ("cat", categorical_pipeline, categorical_columns),
        ]
    )

    # Premier split : train complet / test final.
    x_train_full, x_test, y_train_full, y_test = train_test_split(
        x, y, test_size=0.2, random_state=random_state, stratify=y
    )
    # Deuxieme split : train / validation.
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_full, y_train_full, test_size=0.2, random_state=random_state, stratify=y_train_full
    )

    # Le preprocesseur est ajuste uniquement sur train.
    x_train_processed = preprocessor.fit_transform(x_train)
    x_val_processed = preprocessor.transform(x_val)
    x_test_processed = preprocessor.transform(x_test)

    # Conversion en tenseurs PyTorch. Si sklearn produit une matrice sparse,
    # on la densifie d'abord avec `.toarray()`.
    x_train_tensor = torch.tensor(x_train_processed.toarray() if hasattr(x_train_processed, "toarray") else x_train_processed, dtype=torch.float32)
    x_val_tensor = torch.tensor(x_val_processed.toarray() if hasattr(x_val_processed, "toarray") else x_val_processed, dtype=torch.float32)
    x_test_tensor = torch.tensor(x_test_processed.toarray() if hasattr(x_test_processed, "toarray") else x_test_processed, dtype=torch.float32)

    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    y_val_tensor = torch.tensor(y_val, dtype=torch.long)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    # Les DataLoader gerent les mini-lots utilises pendant l'entrainement.
    train_loader = DataLoader(TensorDataset(x_train_tensor, y_train_tensor), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(x_val_tensor, y_val_tensor), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(x_test_tensor, y_test_tensor), batch_size=batch_size, shuffle=False)

    # Nombre total de variables apres encodage et normalisation.
    input_dim = x_train_tensor.shape[1]
    class_names = ["NoPurchase", "Purchase"]

    return TabularDataBundle(train_loader, val_loader, test_loader, input_dim, class_names, preprocessor)
