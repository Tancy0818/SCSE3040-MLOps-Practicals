"""Predict one delivery from the command line."""

from pathlib import Path

import pandas as pd

from delivery import load_model


def main():
    # Find model.joblib inside the work folder
    model_path = Path(__file__).parent / "model.joblib"
    model = load_model(model_path)

    # Create one order with the same features used for training
    order = pd.DataFrame([{
        "distance_km": 7,
        "prep_time_min": 25,
        "traffic_level": 3,
        "rain": 0
    }])

    # Predict the delivery time
    minutes = model.predict(order)[0]

    print(f"PREDICTION: {minutes:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
