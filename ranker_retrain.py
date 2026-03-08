"""
ranker_retrain.py

Retrain the ranker from the choices dataset and save to ranker.pkl.
Supports TF-IDF, embedding (logistic), and neural ranker modes.

Usage:
  python ranker_retrain.py --model all-MiniLM-L6-v2 --out ranker.pkl
  python ranker_retrain.py --tfidf --out ranker.pkl
  python ranker_retrain.py --neural
"""

import argparse

from database import get_choice_dataset
from ranker import train_with_embeddings, train_basic


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="ranker.pkl")
    p.add_argument("--model", default="all-MiniLM-L6-v2")
    p.add_argument("--tfidf", action="store_true")
    p.add_argument("--neural", action="store_true",
                   help="Train the embedding-based neural ranker (Phase 2)")
    p.add_argument("--config", default="config.yaml",
                   help="Path to config.yaml for hyperparameters")
    args = p.parse_args()

    rows = get_choice_dataset()
    if not rows:
        print("No choices available. Generate data in the app first.")
        return

    texts, labels = zip(*rows)

    if args.neural:
        # ---- Neural ranker (Phase 2) ----------------------------------------
        from neural_ranker import train_ranker, load_config
        from utils.hashing import compute_data_sha256
        from seeds import set_deterministic

        config = load_config(args.config)
        set_deterministic(config.get("seed", 42))

        print(f"Training neural ranker with config from {args.config} ...")
        print(f"  Dataset: {len(texts)} samples")
        print(f"  Dataset SHA256: {compute_data_sha256(list(zip(texts, labels)))[:16]}...")

        metrics, exp_dir = train_ranker(list(texts), list(labels), config=config)

        print(f"\nNeural ranker trained.")
        print(f"  Best epoch:    {metrics['best_epoch']}")
        print(f"  Best val loss: {metrics['best_val_loss']:.4f}")
        print(f"  Best val acc:  {metrics['best_val_acc']:.4f}")
        print(f"  Experiment:    {exp_dir}")

    elif args.tfidf:
        acc, rep = train_basic(list(texts), list(labels), save_path=args.out)
        print(f"TF-IDF ranker trained. mean acc={acc:.3f}")
        print(rep)

    else:
        acc, rep = train_with_embeddings(
            list(texts), list(labels),
            embed_model_name=args.model,
            save_path=args.out,
        )
        print(f"Embedding ranker trained with {args.model}. mean acc={acc:.3f}")
        print(rep)


if __name__ == "__main__":
    main()
