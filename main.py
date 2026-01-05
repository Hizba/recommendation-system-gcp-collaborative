import os
from typing import Any, Dict, List

from flask import Flask, request, jsonify
from google.cloud import storage
import joblib
import pandas as pd


# ==============================
# 1) Configuration GCS
# ==============================

GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME", "projet-9-481708-models")
GCS_CF_MODEL_BLOB = os.getenv("CF_MODEL_BLOB", "collab/svd_model.joblib")
GCS_RATINGS_BLOB = os.getenv("CF_RATINGS_BLOB", "collab/ratings_df.pkl")

LOCAL_CF_MODEL_PATH = "/tmp/svd_model.joblib"
LOCAL_RATINGS_PATH = "/tmp/ratings_df.pkl"

_CF_MODEL = None
_CF_RATINGS_DF = None


# ==============================
# 2) Chargement modèle + données
# ==============================

def _load_cf_models_if_needed() -> None:
    global _CF_MODEL, _CF_RATINGS_DF

    if _CF_MODEL is not None and _CF_RATINGS_DF is not None:
        return

    client = storage.Client()
    bucket = client.bucket(GCS_BUCKET_NAME)

    print("Téléchargement du modèle collaboratif et des données depuis GCS...")
    print("Bucket :", GCS_BUCKET_NAME)
    print("CF_MODEL_BLOB :", GCS_CF_MODEL_BLOB)
    print("CF_RATINGS_BLOB :", GCS_RATINGS_BLOB)

    blob_model = bucket.blob(GCS_CF_MODEL_BLOB)
    blob_model.download_to_filename(LOCAL_CF_MODEL_PATH)

    blob_ratings = bucket.blob(GCS_RATINGS_BLOB)
    blob_ratings.download_to_filename(LOCAL_RATINGS_PATH)

    print("Fichiers locaux :")
    print("  modèle :", LOCAL_CF_MODEL_PATH, "taille =", os.path.getsize(LOCAL_CF_MODEL_PATH))
    print("  ratings_df :", LOCAL_RATINGS_PATH, "taille =", os.path.getsize(LOCAL_RATINGS_PATH))

    _CF_MODEL = joblib.load(LOCAL_CF_MODEL_PATH)
    _CF_RATINGS_DF = pd.read_pickle(LOCAL_RATINGS_PATH)

    print("Modèle SVD et ratings_df chargés en mémoire.")


# ==============================
# 3) Logique de reco collaborative
# ==============================

def recommend_top_k_collaborative(user_id: int, k: int = 5) -> List[Dict[str, Any]]:
    from surprise import PredictionImpossible

    _load_cf_models_if_needed()

    svd = _CF_MODEL
    ratings_df = _CF_RATINGS_DF

    user_rows = ratings_df[ratings_df["user_id"] == user_id]
    if user_rows.empty:
        return []

    seen_items = set(user_rows["article_id"].unique())
    all_items = ratings_df["article_id"].unique()
    candidates = [i for i in all_items if i not in seen_items]
    if not candidates:
        return []

    preds: List[Dict[str, Any]] = []
    for iid in candidates:
        try:
            est = svd.predict(str(user_id), str(iid)).est
        except PredictionImpossible:
            continue
        preds.append({"item_id": str(iid), "score": float(est)})

    preds_sorted = sorted(preds, key=lambda x: x["score"], reverse=True)[:k]
    return preds_sorted


# ==============================
# 4) API Flask
# ==============================

app = Flask(__name__)


@app.route("/health", methods=["GET"])
def health() -> Any:
    return jsonify({"status": "ok"}), 200


@app.route("/recommend_collab", methods=["POST"])
def recommend_collab() -> Any:
    """
    Body JSON :
      {
        "user_id": 82705,
        "k": 10
      }
    """
    try:
        payload: Dict[str, Any] = request.get_json(force=True) or {}
    except Exception:
        return jsonify({"error": "Invalid JSON payload"}), 400

    user_id = payload.get("user_id")
    k = payload.get("k", 5)

    if user_id is None:
        return jsonify({"error": "user_id is required"}), 400

    try:
        user_id_int = int(user_id)
        k_int = int(k)
    except ValueError:
        return jsonify({"error": "user_id and k must be integers"}), 400

    try:
        recs = recommend_top_k_collaborative(user_id_int, k_int)
    except Exception as e:
        import traceback
        print(f"Error in recommend_collab(): {e}")
        traceback.print_exc()
        return jsonify({"error": "Internal error"}), 500

    return jsonify(
        {
            "user_id": user_id_int,
            "k": k_int,
            "recommendations": recs,
        }
    ), 200


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8080"))
    app.run(host="0.0.0.0", port=port, debug=True)
