
from flask import Flask, render_template, request
import pickle
import numpy as np
from pathlib import Path

app = Flask(__name__)

# Find model files in the same folder as app.py
BASE_DIR = Path(__file__).resolve().parent

# Load the trained Decision Tree model
with open(BASE_DIR / "decision_tree.pkl", "rb") as file:
    model = pickle.load(file)

# Load the saved OneHotEncoder
with open(BASE_DIR / "ohe.pkl", "rb") as file:
    ohe = pickle.load(file)

# Accuracy reported from the Kaggle evaluation
MODEL_ACCURACY = 62

# Keep the feature order identical to the Kaggle notebook
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
    return render_template(
        "index.html",
        accuracy=MODEL_ACCURACY
    )


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Read numeric inputs in the required order
        numeric_values = [
            float(request.form[field])
            for field in NUMERIC_FIELDS
        ]

        numeric_array = np.array(
            numeric_values,
            dtype=float
        ).reshape(1, -1)

        # Read categorical inputs in the required order
        categorical_values = [
            request.form[field]
            for field in CATEGORICAL_FIELDS
        ]

        # Apply the saved OneHotEncoder
        encoded = ohe.transform([categorical_values])

        # Convert sparse output to a dense array if necessary
        if hasattr(encoded, "toarray"):
            encoded = encoded.toarray()

        # Combine numeric and encoded categorical features
        final_input = np.hstack((numeric_array, encoded))

        # Verify that the model receives 65 features
        if final_input.shape[1] != 65:
            raise ValueError(
                f"Model expects 65 features, but received "
                f"{final_input.shape[1]}. Check the Kaggle "
                f"feature order and encoder."
            )

        # Predict the recommended hotel/class
        prediction = model.predict(final_input)[0]

        return render_template(
            "index.html",
            prediction=str(prediction),
            accuracy=MODEL_ACCURACY
        )

    except Exception:
        app.logger.exception("Prediction failed")

        return render_template(
            "index.html",
            error=(
                "Prediction failed. Please check your inputs "
                "and confirm that the model and encoder match."
            ),
            accuracy=MODEL_ACCURACY
        ), 400


if __name__ == "__main__":
    app.run(debug=True)
