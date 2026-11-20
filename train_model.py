import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

BASE = Path(__file__).resolve().parent
FEATURES = [
    "Area", "Perimeter", "Major_Axis_Length", "Minor_Axis_Length",
    "Eccentricity", "Convex_Area", "Extent",
]
PROFILES = {
    "Cammeo / Osmancik": "Rice_Cammeo_Osmancik.csv",
    "Basmati / Jasmine / Karacadag": "Rice_MSC_Selected.csv",
}


def train():
    models = {}
    metrics = {}
    for index, (profile, file) in enumerate(PROFILES.items()):
        raw = pd.read_csv(BASE / "data" / file)
        data = raw.dropna(subset=[*FEATURES, "Class"]).drop_duplicates(subset=FEATURES)
        X = data[FEATURES]
        y = data["Class"]
        x_train, x_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y,
        )
        model = RandomForestClassifier(
            n_estimators=80, max_depth=16, min_samples_leaf=3,
            random_state=42, n_jobs=2,
        ).fit(x_train, y_train)
        prediction = model.predict(x_test)
        metrics[profile] = {
            "source_rows": len(raw), "training_rows": len(x_train),
            "test_rows": len(x_test),
            "accuracy": accuracy_score(y_test, prediction),
            "macro_f1": f1_score(y_test, prediction, average="macro"),
            "classes": model.classes_.tolist(),
            "confusion": confusion_matrix(y_test, prediction, labels=model.classes_).tolist(),
        }
        models[profile] = {
            "model": model, "min": x_train.min(), "max": x_train.max(),
            "examples": data.groupby("Class")[FEATURES].median(),
        }
        sample = x_test.join(y_test).groupby("Class").head(10)
        sample.to_csv(BASE / "examples" / f"demo_{index + 1}.csv", index=False)
        print(profile, metrics[profile]["accuracy"], flush=True)

    joblib.dump(models, BASE / "models" / "rice_profiles.joblib", compress=3)
    (BASE / "models" / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")


if __name__ == "__main__":
    train()
