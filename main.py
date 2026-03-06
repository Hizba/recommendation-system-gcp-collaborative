import os
import threading
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

# Globals + thread lock pour lazy loading safe
_CF_MODEL = None
_CF_RATINGS_DF = None
_model_lock = threading.Lock()

# ==============================
# 2) Chargement depuis GCS (inchangé)
# ==============================
def load_cf_models_from_gcs() -> None:
    global _CF_MODEL, _CF_RATINGS_DF
    
    client = storage.Client()
    bucket = client.bucket(GCS_BUCKET_NAME)
    
    print("📥 Téléchargement modèles depuis GCS...")
    print(f"Bucket: {GCS_BUCKET_NAME}")
    print(f"Modèle: {GCS_CF_MODEL_BLOB}")
    print(f"Ratings: {GCS_RATINGS_BLOB}")
    
    # Download
    bucket.blob(GCS_CF_MODEL_BLOB).download_to_filename(LOCAL_CF_MODEL_PATH)
    bucket.blob(GCS_RATINGS_BLOB).download_to_filename(LOCAL_RATINGS_PATH)
    
    print(f"📁 Fichiers locaux: modèle {os.path.getsize(LOCAL_CF_MODEL_PATH)/1e6:.1f}MB, ratings {os.path.getsize(LOCAL_RATINGS_PATH)/1e6:.1f}MB")
    
    # Load en mémoire
    print("🧠 Chargement en RAM...")
    _CF_RATINGS_DF = pd.read_pickle(LOCAL_RATINGS_PATH)
    _CF_MODEL = joblib.load(LOCAL_CF_MODEL_PATH)
    print("✅ Modèles chargés !")

# Lazy loader thread-safe
def ensure_models_loaded():
    global _CF_MODEL, _CF_RATINGS_DF
    if _CF_MODEL is None:
        with _model_lock:
            if _CF_MODEL is None:  # Double-check pattern
                load_cf_models_from_gcs()
    return _CF_MODEL, _CF_RATINGS_DF

# ==============================
# 3) Logique reco (inchangée)
# ==============================
def recommend_top_k_collaborative(user_id: int, k: int = 5) -> List[Dict[str, Any]]:
    from surprise import PredictionImpossible
    
    svd, ratings_df = ensure_models_loaded()
    
    if svd is None or ratings_df is None:
        raise RuntimeError("Collaborative model not loaded")
    
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
    
    return sorted(preds, key=lambda x: x["score"], reverse=True)[:k]

# ==============================
# 4) Flask API
# ==============================
app = Flask(__name__)

@app.route("/health/startup", methods=["GET"])  # Pour startup probe Cloud Run
@app.route("/health", methods=["GET"])
def health():
    try:
        ensure_models_loaded()  # Trigger lazy au premier health
        return jsonify({
            "status": "healthy",
            "models_loaded": True,
            "model_size_mb": os.path.getsize(LOCAL_CF_MODEL_PATH)/1e6 if os.path.exists(LOCAL_CF_MODEL_PATH) else 0,
            "ratings_shape": _CF_RATINGS_DF.shape if _CF_RATINGS_DF is not None else None
        }), 200
    except Exception as e:
        print(f"Health check error: {e}")
        return jsonify({"status": "loading", "error": str(e)}), 503  # Retryable

@app.route("/recommend_collab", methods=["POST"])
def recommend_collab():
    try:
        payload: Dict[str, Any] = request.get_json(force=True) or {}
    except Exception:
        return jsonify({"error": "Invalid JSON"}), 400
    
    user_id = payload.get("user_id")
    k = payload.get("k", 5)
    
    if user_id is None:
        return jsonify({"error": "user_id required"}), 400
    
    try:
        user_id_int = int(user_id)
        k_int = int(k)
    except ValueError:
        return jsonify({"error": "user_id/k must be int"}), 400
    
    try:
        recs = recommend_top_k_collaborative(user_id_int, k_int)
        return jsonify({
            "user_id": user_id_int,
            "k": k_int,
            "recommendations": recs
        }), 200
    except Exception as e:
        import traceback
        print(f"Reco error: {e}\n{traceback.format_exc()}")
        return jsonify({"error": "Internal error"}), 500

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8080"))
    app.run(host="0.0.0.0", port=port, debug=False)
