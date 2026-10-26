
from flask import Flask, render_template, request
import pickle
import numpy as np
from pathlib import Path

app = Flask(__name__)

# Locate the saved model files in this project folder
BASE_DIR = Path(__file__).resolve().parent

with open(BASE_DIR / "decision_tree.pkl", "rb") as file:
    model = pickle.load(file)

with open(BASE_DIR / "ohe.pkl", "rb") as file:
    ohe = pickle.load(file)


# Keep this order exactly the same as in the Kaggle notebook
NUMERIC_FIELDS = [
    "user_budget_total_inr",
    "user_daily_budget_inr",
    "stay_days",
    "family_count",
    "adults_count",
    "children_count",
    "seniors_count",
    "rooms_required",
    "requires_central_heating",
    "requires_scenic_view",
    "star_category",
    "google_rating",
    "review_count",
    "base_price_per_night_inr",
    "seasonal_room_rate_inr",
    "estimated_total_stay_cost_inr",
    "budget_fit_ratio",
    "max_guests_per_room",
    "has_central_heating_bukhari",
    "has_power_backup_24x7",
    "has_mountain_or_lake_view",
    "has_wazwan_kitchen",
    "distance_to_sxr_airport_km",
    "distance_to_landmark_km",
    "latitude",
    "longitude",
    "altitude_m",
    "suitability_score"
]

# Keep this order exactly the same as the Kaggle OneHotEncoder
CATEGORICAL_FIELDS = [
    "travel_season",
    "trip_purpose",
    "preferred_property_type",
    "dietary_preference",
    "district",
    "property_type"
]


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Read the 28 numeric values in the trained feature order
        numeric_values = [
            float(request.form[field])
            for field in NUMERIC_FIELDS
        ]

        # Read the 6 categorical values in the encoder's feature order
        categorical_values = [
            request.form[field]
            for field in CATEGORICAL_FIELDS
        ]

        numeric_array = np.array(
            numeric_values, dtype=float
        ).reshape(1, -1)

        encoded = ohe.transform([categorical_values])

        # Support both sparse and dense OneHotEncoder outputs
        if hasattr(encoded, "toarray"):
            encoded = encoded.toarray()

        # Kaggle training order: numeric features first, encoded features second
        final_input = np.hstack((numeric_array, encoded))

        # The trained model expects 65 features
        if final_input.shape[1] != 65:
            raise ValueError(
                f"Expected 65 features, received {final_input.shape[1]}."
            )

        prediction = model.predict(final_input)[0]

        return render_template(
            "index.html",
            prediction=str(prediction)
        )

    except Exception as error:
        app.logger.exception("Prediction failed")
        return render_template(
            "index.html",
            error=f"Prediction failed: {error}"
        ), 400


if __name__ == "__main__":
    app.run(debug=True)
