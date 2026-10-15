from pathlib import Path
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "placement_predict_50k Dataset1.csv"
if not DATA_PATH.exists():
    DATA_PATH = BASE_DIR / "placement_predict_50k Dataset1.csv"

MODEL_PATH = BASE_DIR / "models" / "placement_model.joblib"

TARGET = "PlacementStatus"

def train_and_save():
    df = pd.read_csv(DATA_PATH)

    if TARGET not in df.columns:
        raise ValueError(
            f"Target '{TARGET}' not found. "
            f"Available columns: {df.columns.tolist()}"
        )

    df = df.dropna(subset=[TARGET])

    # Exclude target and identifier columns
    X = df.drop(columns=[TARGET])

    id_columns = [
        col for col in X.columns
        if col.lower() in ("id", "studentid", "student_id", "name") or "salary" in col.lower() or "anomaly" in col.lower()
    ]
    X = X.drop(columns=id_columns, errors="ignore")

    y = df[TARGET]

    if y.nunique() != 2:
        raise ValueError("This example requires a binary target.")

    numeric_cols = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_cols = X.select_dtypes(
        exclude=["number"]
    ).columns.tolist()

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_cols),
        ("categorical", categorical_pipeline, categorical_cols)
    ])

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000))
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, predictions))

    # Store model and feature information together
    model_package = {
        "model": model,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "feature_names": X.columns.tolist(),
        "categories": {
            col: [str(v) for v in X[col].dropna().unique().tolist()]
            for col in categorical_cols
        }
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_package, MODEL_PATH)
    print("Model saved at:", MODEL_PATH)

if __name__ == "__main__":
    train_and_save()
