import joblib

# Load the trained Random Forest model
model = joblib.load("models/random_forest_model.pkl")


def predict_attacks(df):
    """
    Predict attack labels for the uploaded dataframe.
    """

    predictions = model.predict(df)

    return predictions