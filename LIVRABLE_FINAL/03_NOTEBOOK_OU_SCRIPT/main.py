from __future__ import annotations

"""Point d'entree principal du projet.

Ce fichier orchestre les trois grandes parties du devoir :
1. MLP sur donnees tabulaires
2. CNN sur images
3. modeles recursifs et Seq2Seq sur texte

L'idee a retenir est la suivante :
- `src/data/` prepare les donnees
- `src/models/` definit les architectures
- `src/training/` contient les boucles d'entrainement
- `src/utils/` centralise les fonctions communes

Autrement dit, `main.py` ne contient pas la logique detaillee de chaque modele.
Son role est surtout de coordonner les etapes et lancer les bonnes experiences.
"""

import argparse
from pathlib import Path

import torch

from src.data.image import build_fashion_mnist_loaders
from src.data.tabular import build_tabular_loaders, load_tabular_dataset
from src.data.text import build_text_loaders, load_hf_amazon_food_reviews, load_text_dataset
from src.models.cnn import ImageMLP, LeNetLikeCNN, manual_avg_pool2d, manual_cross_correlation2d, manual_max_pool2d
from src.models.mlp import CustomMLP, SequentialMLP, initialize_weights
from src.models.sequence import RecurrentLanguageEncoder, Seq2SeqDecoder, Seq2SeqEncoder, Seq2SeqModel
from src.training.common import evaluate_classifier, train_classifier, train_seq2seq
from src.utils.device import get_device
from src.utils.io import ensure_dir, save_checkpoint, save_json
from src.utils.metrics import confusion_matrix_array
from src.utils.plotting import plot_confusion_matrix, plot_history
from src.utils.seed import set_seed


def inspect_model_parameters(model: torch.nn.Module) -> dict:
    """Retourne un resume des parametres pour la partie MLP."""

    # `named_parameters()` permet d'inspecter les poids et biais appris par le modele.
    # `state_dict()` rassemble tous les tenseurs sauvegardables. Ces deux vues sont
    # utiles pour la partie theorique du devoir sur les parametres PyTorch.
    return {
        "named_parameters": {name: list(param.shape) for name, param in model.named_parameters()},
        "state_dict_keys": list(model.state_dict().keys()),
    }


def run_tabular_experiment(args, device: torch.device) -> None:
    """Execute la partie I du projet."""

    print("\n[Partie I] Debut de l'experience tabulaire MLP")
    # Chargement du dataset tabulaire depuis un CSV local ou une source distante.
    dataframe = load_tabular_dataset(
        args.tabular_csv,
        target_column=args.tabular_target,
        source=args.tabular_source,
    )
    # Preparation complete des donnees :
    # nettoyage, encodage, normalisation, split et creation des DataLoader.
    bundle = build_tabular_loaders(dataframe, target_column=args.tabular_target, batch_size=args.batch_size)

    # Tous les artefacts de la partie I seront stockes dans ce dossier.
    artifacts_dir = ensure_dir(Path("artifacts") / "partie_1_mlp")
    # Le sujet impose la comparaison de plusieurs strategies d'initialisation.
    init_strategies = ["gaussian", "constant", "xavier"]
    # Deux versions du MLP sont comparees :
    # - une version `nn.Sequential`
    # - une version en classe personnalisee
    model_builders = {
        "sequential": SequentialMLP(bundle.input_dim, [128, 64], 2),
        "custom": CustomMLP(bundle.input_dim, [128, 64], 2),
    }

    all_results = {}
    best_model = None
    best_model_name = ""
    best_accuracy = -1.0

    for model_name, model in model_builders.items():
        for init_name in init_strategies:
            # Avant chaque experience, on reset les poids selon la strategie voulue.
            initialize_weights(model, strategy=init_name)
            # Entrainement supervise standard.
            result = train_classifier(
                model=model,
                train_loader=bundle.train_loader,
                val_loader=bundle.val_loader,
                device=device,
                epochs=args.epochs if not args.quick_run else 2,
                learning_rate=args.learning_rate,
            )
            # On recharge les meilleurs poids observes sur la validation avant le test.
            model.load_state_dict(result.best_state_dict)
            test_loss, test_metrics = evaluate_classifier(model, bundle.test_loader, device)
            key = f"{model_name}_{init_name}"
            all_results[key] = {
                "test_loss": test_loss,
                **test_metrics,
                "parameter_summary": inspect_model_parameters(model),
            }
            # Les courbes d'apprentissage sont utiles pour l'annexe experimentale.
            plot_history(result.history, f"MLP - {key}", artifacts_dir / f"{key}_history.png")

            # On memorise le meilleur couple architecture + initialisation.
            if test_metrics["accuracy"] > best_accuracy:
                best_accuracy = test_metrics["accuracy"]
                best_model = model
                best_model_name = key

    if best_model is not None:
        # Sauvegarde du meilleur modele de la partie I.
        save_checkpoint(best_model, artifacts_dir / "best_mlp.pt")
        y_true, y_pred = [], []
        best_model.eval()
        with torch.no_grad():
            for x_batch, y_batch in bundle.test_loader:
                # Le modele et les donnees doivent etre sur le meme device.
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                preds = best_model(x_batch).argmax(dim=1)
                y_true.extend(y_batch.cpu().tolist())
                y_pred.extend(preds.cpu().tolist())
        # La matrice de confusion aide a visualiser les erreurs de classification.
        cm = confusion_matrix_array(y_true, y_pred)
        plot_confusion_matrix(cm, bundle.class_names, "Matrice de confusion MLP", artifacts_dir / "best_mlp_confusion_matrix.png")

    # Sauvegarde d'un resume global pour relire facilement les resultats.
    save_json({"best_model": best_model_name, "results": all_results}, artifacts_dir / "summary.json")
    print(f"[Partie I] Meilleur modele : {best_model_name} | accuracy={best_accuracy:.4f}")


def run_image_experiment(args, device: torch.device) -> None:
    """Execute la partie II du projet."""

    print("\n[Partie II] Debut de l'experience CNN")
    # Chargement de Fashion-MNIST et creation des DataLoader.
    bundle = build_fashion_mnist_loaders(args.image_root, batch_size=args.batch_size)
    artifacts_dir = ensure_dir(Path("artifacts") / "partie_2_cnn")

    # Petit exemple numerique pour illustrer a la main les operations de convolution
    # et de pooling demandees dans la partie theorique du devoir.
    sample = torch.arange(1.0, 17.0).reshape(4, 4)
    kernel = torch.tensor([[1.0, 0.0], [0.0, -1.0]])
    manual_ops = {
        "cross_correlation": manual_cross_correlation2d(sample, kernel).tolist(),
        "max_pooling": manual_max_pool2d(sample, kernel_size=2, stride=2).tolist(),
        "avg_pooling": manual_avg_pool2d(sample, kernel_size=2, stride=2).tolist(),
    }

    # CNN principal de la partie II.
    cnn = LeNetLikeCNN(
        num_classes=len(bundle.class_names),
        num_filters=(16, 32),
        padding=2,
        stride=1,
        pooling="max",
        use_conv1x1=True,
    )
    # Baseline volontairement plus simple pour montrer les limites d'un MLP sur image.
    mlp_baseline = ImageMLP(num_classes=len(bundle.class_names))

    # Entrainement du CNN.
    cnn_result = train_classifier(cnn, bundle.train_loader, bundle.val_loader, device, epochs=args.epochs if not args.quick_run else 2, learning_rate=args.learning_rate)
    cnn.load_state_dict(cnn_result.best_state_dict)
    _, cnn_metrics = evaluate_classifier(cnn, bundle.test_loader, device)
    plot_history(cnn_result.history, "CNN history", artifacts_dir / "cnn_history.png")

    # Entrainement du MLP image sur le meme dataset.
    mlp_result = train_classifier(mlp_baseline, bundle.train_loader, bundle.val_loader, device, epochs=args.epochs if not args.quick_run else 2, learning_rate=args.learning_rate)
    mlp_baseline.load_state_dict(mlp_result.best_state_dict)
    _, mlp_metrics = evaluate_classifier(mlp_baseline, bundle.test_loader, device)
    plot_history(mlp_result.history, "Image MLP history", artifacts_dir / "image_mlp_history.png")

    # Sauvegarde des mesures importantes pour la comparaison finale.
    save_json(
        {
            "manual_operations": manual_ops,
            "cnn_metrics": cnn_metrics,
            "mlp_baseline_metrics": mlp_metrics,
        },
        artifacts_dir / "summary.json",
    )
    save_checkpoint(cnn, artifacts_dir / "best_cnn.pt")
    print(f"[Partie II] Accuracy CNN={cnn_metrics['accuracy']:.4f} | Accuracy MLP image={mlp_metrics['accuracy']:.4f}")


def run_text_experiment(args, device: torch.device) -> None:
    """Execute la partie III du projet."""

    print("\n[Partie III] Debut de l'experience sequences")
    # Selon l'argument choisi, on peut utiliser un dataset Hugging Face
    # ou un CSV local deja exporte.
    if args.text_source == "hf_amazon_food_reviews":
        dataframe = load_hf_amazon_food_reviews(split=args.hf_split, sample_size=args.text_sample_size if not args.quick_run else 500)
    else:
        dataframe = load_text_dataset(args.text_csv, text_col=args.text_column, summary_col=args.summary_column)
        if args.quick_run:
            # Le mode rapide prend seulement un sous-ensemble pour verifier le pipeline.
            dataframe = dataframe.head(500).copy()

    # Construction du vocabulaire, tokenisation, padding et DataLoader.
    bundle = build_text_loaders(
        dataframe,
        text_col=args.text_column,
        summary_col=args.summary_column,
        batch_size=args.batch_size,
    )
    artifacts_dir = ensure_dir(Path("artifacts") / "partie_3_sequences")

    recurrent_summaries = {}
    for rnn_type in ["rnn", "lstm", "gru"]:
        # On construit successivement RNN, LSTM et GRU pour comparer leurs sorties.
        encoder = RecurrentLanguageEncoder(
            vocab_size=len(bundle.vocab),
            embedding_dim=64,
            hidden_dim=128,
            rnn_type=rnn_type,
            pad_idx=bundle.pad_idx,
        ).to(device)
        # Observation rapide d'un lot pour garder une trace des formes de sortie.
        sample_batch = next(iter(bundle.train_loader))[0].to(device)
        outputs, hidden = encoder(sample_batch)
        recurrent_summaries[rnn_type] = {
            "output_shape": list(outputs.shape),
            "hidden_type": type(hidden).__name__,
        }

    # Construction du systeme encodeur-decodeur Seq2Seq.
    seq_encoder = Seq2SeqEncoder(len(bundle.vocab), embedding_dim=128, hidden_dim=256, pad_idx=bundle.pad_idx)
    seq_decoder = Seq2SeqDecoder(len(bundle.vocab), embedding_dim=128, hidden_dim=256, pad_idx=bundle.pad_idx)
    seq2seq = Seq2SeqModel(seq_encoder, seq_decoder, bos_idx=bundle.bos_idx, eos_idx=bundle.eos_idx)

    # Entrainement du modele de generation de resume.
    seq_result = train_seq2seq(
        seq2seq,
        bundle.train_loader,
        bundle.val_loader,
        device=device,
        pad_idx=bundle.pad_idx,
        epochs=args.sequence_epochs if not args.quick_run else 2,
        learning_rate=args.learning_rate,
        gradient_clip=1.0,
    )
    seq2seq.load_state_dict(seq_result.best_state_dict)
    plot_history(seq_result.history, "Seq2Seq history", artifacts_dir / "seq2seq_history.png")
    save_checkpoint(seq2seq, artifacts_dir / "best_seq2seq.pt")

    # Comparaison de deux strategies de decodage :
    # - greedy
    # - beam search
    sample_src, _ = next(iter(bundle.test_loader))
    sample_src = sample_src.to(device)
    greedy_output = seq2seq.greedy_decode(sample_src[:1], max_len=15).cpu().tolist()
    beam_output = seq2seq.beam_search_decode(sample_src[:1], max_len=15, beam_width=3).cpu().tolist()

    save_json(
        {
            "recurrent_model_summaries": recurrent_summaries,
            "decoding_examples": {
                "greedy": greedy_output,
                "beam_search": beam_output,
            },
        },
        artifacts_dir / "summary.json",
    )
    print("[Partie III] Entrainement Seq2Seq termine")


def parse_args():
    """Definit les options de ligne de commande du projet.

    Cette fonction rend le projet flexible :
    on peut lancer une seule partie, changer les chemins des donnees,
    modifier la taille des lots ou activer un mode rapide.
    """

    parser = argparse.ArgumentParser(description="Projet Deep Learning EMSI - pipeline multimodal e-commerce")
    parser.add_argument("--task", choices=["tabular", "image", "text", "all"], default="all")
    parser.add_argument("--quick-run", action="store_true", help="Mode rapide pour verifier le pipeline.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--sequence-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--tabular-csv", type=str, default="data/tabular/online_shoppers_intention.csv")
    parser.add_argument("--tabular-target", type=str, default="Revenue")
    parser.add_argument("--tabular-source", choices=["auto", "local", "uci"], default="local")
    parser.add_argument("--image-root", type=str, default="data/images")
    parser.add_argument("--text-csv", type=str, default="data/text/amazon_food_reviews.csv")
    parser.add_argument("--text-column", type=str, default="text")
    parser.add_argument("--summary-column", type=str, default="summary")
    parser.add_argument("--text-source", choices=["local_csv", "hf_amazon_food_reviews"], default="local_csv")
    parser.add_argument("--hf-split", type=str, default="train")
    parser.add_argument("--text-sample-size", type=int, default=20000)
    return parser.parse_args()


def main():
    """Procedure globale de lancement.

    Etapes :
    1. lecture des arguments
    2. fixation de la seed
    3. choix du device CPU/GPU
    4. execution des parties demandees
    """

    args = parse_args()
    set_seed(42)
    device = get_device()
    print(f"Device detecte : {device}")

    if args.task in ("tabular", "all"):
        run_tabular_experiment(args, device)
    if args.task in ("image", "all"):
        run_image_experiment(args, device)
    if args.task in ("text", "all"):
        run_text_experiment(args, device)


if __name__ == "__main__":
    main()
