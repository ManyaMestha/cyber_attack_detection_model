import joblib

# Load the trained Random Forest model
model = joblib.load("models/random_forest_model.pkl")


def predict_attacks(df):
    """
    Predict attack labels for the uploaded dataframe.
    """

    # Get the exact features used during model training
    expected_features = model.feature_names_in_

    # Check for missing features
    missing_features = [
        feature for feature in expected_features
        if feature not in df.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing required features: {missing_features}"
        )

    # Select features in exactly the same order as training
    X = df[expected_features]

    # Make predictions
    predictions = model.predict(X)

    return predictions