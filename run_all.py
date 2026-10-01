import argparse, json, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib

from src.data import (load_and_validate, build_feature_matrix,
                      build_report, sha256_file, FEATURES,
                      TARGET_REG, TARGET_CLF)
from src.regression import (train_test_split_np, standardize,
                            gradient_descent, predict, metrics)
from src.classification import train_classifier
from src.clustering import evaluate_k, final_cluster


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--group", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.output, exist_ok=True)
    os.makedirs("models", exist_ok=True)

    sha = sha256_file(args.data)
    df, missing, dup = load_and_validate(args.data)
    X = build_feature_matrix(df)
    y_reg = df[TARGET_REG].to_numpy(dtype=float)
    y_clf = df[TARGET_CLF].to_numpy(dtype=int)

    # Data report
    with open(f"{args.output}/data_report.json", "w") as f:
        json.dump(build_report(df, missing, dup, args.group, sha), f, indent=2)

    # Regression
    Xtr, Xte, ytr, yte = train_test_split_np(X, y_reg, seed=args.seed)
    Xtr_s, Xte_s, mu, sigma = standardize(Xtr, Xte)
    w, losses = gradient_descent(Xtr_s, ytr, lr=0.05, epochs=2000)
    reg_m = metrics(yte, predict(Xte_s, w))
    reg_m.update({"seed": args.seed, "epochs": 2000,
                  "learning_rate": 0.05, "features": FEATURES})
    with open(f"{args.output}/regression_metrics.json", "w") as f:
        json.dump(reg_m, f, indent=2)

    plt.figure(); plt.plot(losses)
    plt.xlabel("epoch"); plt.ylabel("MSE"); plt.title("Regression loss")
    plt.tight_layout(); plt.savefig(f"{args.output}/regression_loss.png"); plt.close()
    np.savez("models/regression_model.npz", w=w, mu=mu, sigma=sigma)

    # Classification
    clf, scaler_c, clf_m = train_classifier(X, y_clf, seed=args.seed)
    clf_m["cost_note"] = ("False negative is more costly: a consignment "
        "needing attention but predicted 0 may be dispatched unsupervised.")
    with open(f"{args.output}/classification_metrics.json", "w") as f:
        json.dump(clf_m, f, indent=2)

    cm = np.array(clf_m["confusion_matrix"])
    plt.figure(); plt.imshow(cm, cmap="Blues")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center")
    plt.xlabel("Predicted"); plt.ylabel("True"); plt.title("Confusion matrix")
    plt.colorbar(); plt.tight_layout()
    plt.savefig(f"{args.output}/confusion_matrix.png"); plt.close()

    joblib.dump(clf, "models/classifier.joblib")
    joblib.dump(scaler_c, "models/scaler.joblib")

    # Clustering
    scaler_k, k_results = evaluate_k(X, seed=args.seed)
    best_k = max(k_results, key=lambda k: k_results[k]["silhouette"])
    scaler_f, km, labels = final_cluster(X, best_k, seed=args.seed)

    with open(f"{args.output}/clustering_metrics.json", "w") as f:
        json.dump({
            "k_evaluation": {str(k): v for k, v in k_results.items()},
            "selected_k": best_k,
            "selection_reason": "highest silhouette score",
            "seed": args.seed,
            "cautious_note": "Clusters are descriptive groupings, not verified real-world categories.",
        }, f, indent=2)

    out = df[["record_id"] + FEATURES].copy()
    out["cluster"] = labels
    out.to_csv(f"{args.output}/clusters.csv", index=False)

    plt.figure(figsize=(7, 5))
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap="tab10", alpha=0.7)
    plt.xlabel(FEATURES[0]); plt.ylabel(FEATURES[1])
    plt.title(f"Clusters (k={best_k})"); plt.tight_layout()
    plt.savefig(f"{args.output}/cluster_plot.png"); plt.close()

    joblib.dump({"scaler": scaler_f, "km": km, "features": FEATURES},
                "models/cluster_model.joblib")

    print(f"Group: {args.group}")
    print(f"SHA-256: {sha}")
    print("Artifacts written to", args.output)


if __name__ == "__main__":
    main()