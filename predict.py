import argparse, json, sys
import numpy as np, joblib

FEATURES = ["plot_area_ha", "rainfall_mm", "soil_ph",
            "seed_kg", "distance_km", "arrival_hour"]
GROUP = "AI-G15"
MODEL_VERSION = "v1.0"


def validate(rec):
    if not isinstance(rec, dict):
        raise ValueError("Record must be a JSON object")
    for f in FEATURES:
        if f not in rec:
            raise ValueError(f"Missing field: {f}")
        try:
            rec[f] = float(rec[f])
        except (TypeError, ValueError):
            raise ValueError(f"Field {f} must be numeric")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", required=True)
    args = ap.parse_args()

    try:
        if args.record.endswith(".json"):
            with open(args.record, "r", encoding="utf-8-sig") as f:
                rec = json.load(f)
        else:
            rec = json.loads(args.record)
        validate(rec)
    except Exception as e:
        print(json.dumps({"error": str(e), "status": "rejected"}))
        sys.exit(1)

    x = np.array([[rec[f] for f in FEATURES]])

    reg = np.load("models/regression_model.npz")
    xs = (x - reg["mu"]) / reg["sigma"]
    yhat = float((np.hstack([np.ones((1, 1)), xs]) @ reg["w"]).item())

    clf = joblib.load("models/classifier.joblib")
    sc = joblib.load("models/scaler.joblib")
    proba = float(clf.predict_proba(sc.transform(x))[0, 1])
    pred_cls = int(clf.predict(sc.transform(x))[0])

    cm = joblib.load("models/cluster_model.joblib")
    cluster = int(cm["km"].predict(cm["scaler"].transform(x))[0])

    print(json.dumps({
        "regression_prediction": round(yhat, 3),
        "classification_prediction": pred_cls,
        "classification_probability": round(proba, 4),
        "cluster_label": cluster,
        "group_code": GROUP,
        "model_version": MODEL_VERSION,
    }))


if __name__ == "__main__":
    main()