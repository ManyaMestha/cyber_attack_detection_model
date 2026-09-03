import joblib

model = joblib.load("models/random_forest_model.pkl")

print("Number of features:", len(model.feature_names_in_))

print("\nFeatures expected by the model:")
for i, feature in enumerate(model.feature_names_in_, start=1):
    print(i, feature)