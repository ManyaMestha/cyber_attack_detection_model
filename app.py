import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from predict import predict_attacks

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="Cyber Attack Detection System",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ Machine Learning Based Cyber Attack Detection System")

st.write("""
Upload a CSV file containing network traffic features.
The trained Random Forest model will classify each network flow.
""")

# -------------------------------
# Upload CSV
# -------------------------------
uploaded_file = st.file_uploader(
    "Upload Network Traffic CSV",
    type=["csv"]
)

# -------------------------------
# If file uploaded
# -------------------------------
if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Dataset")

    st.dataframe(df.head())

    st.write(f"Rows : {df.shape[0]}")
    st.write(f"Columns : {df.shape[1]}")

    # -------------------------------
    # Predict Button
    # -------------------------------
    if st.button("Predict"):

        predictions = predict_attacks(df)

        df["Prediction"] = predictions

        st.success("Prediction Completed Successfully!")

        st.subheader("Prediction Results")

        st.dataframe(df)

        # -------------------------------
        # Summary
        # -------------------------------
        st.subheader("Attack Summary")

        summary = df["Prediction"].value_counts()

        st.write(summary)

        # -------------------------------
        # Bar Chart
        # -------------------------------
        st.subheader("Attack Distribution")

        fig, ax = plt.subplots(figsize=(7,4))

        summary.plot(kind="bar", ax=ax)

        plt.ylabel("Count")

        plt.tight_layout()

        st.pyplot(fig)

        # -------------------------------
        # Download Results
        # -------------------------------
        csv = df.to_csv(index=False)

        st.download_button(
            label="Download Prediction Results",
            data=csv,
            file_name="prediction_results.csv",
            mime="text/csv"
        )