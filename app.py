import os
import joblib
import pandas as pd
from flask import Flask, request, jsonify, render_template

from preprocessing import engineer_features

app = Flask(__name__)
MODEL = joblib.load(os.path.join(os.path.dirname(__file__), "model.pkl"))

FIELDS = [
    "code_module", "code_presentation", "gender", "region", "highest_education",
    "imd_band", "age_band", "num_of_prev_attempts", "studied_credits", "disability",
    "avg_score", "max_score", "min_score", "total_assessments", "avg_weight",
    "total_clicks", "avg_clicks", "max_clicks", "total_interactions",
    "active_days", "date_registration",
]


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/api/predict", methods=["POST"])
def predict():
    payload = request.get_json(force=True)
    try:
        row = {f: payload[f] for f in FIELDS}
    except KeyError as e:
        return jsonify({"error": f"Missing field: {e}"}), 400

    numeric_fields = [
        "num_of_prev_attempts", "studied_credits", "avg_score", "max_score",
        "min_score", "total_assessments", "avg_weight", "total_clicks",
        "avg_clicks", "max_clicks", "total_interactions", "active_days",
        "date_registration",
    ]
    for f in numeric_fields:
        try:
            row[f] = float(row[f])
        except (TypeError, ValueError):
            return jsonify({"error": f"'{f}' must be a number"}), 400

    df = pd.DataFrame([row])
    df = engineer_features(df)

    pred = MODEL.predict(df)[0]
    probs = MODEL.predict_proba(df)[0]
    classes = list(MODEL.classes_)

    return jsonify({
        "prediction": pred,
        "probabilities": {cls: round(float(p), 4) for cls, p in zip(classes, probs)},
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
