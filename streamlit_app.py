# Import tools for file paths, model loading, data loading and web app
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


# Configure the Streamlit page.
st.set_page_config(
    page_title="Board Game Rating Predictor",
    page_icon="🎲",
    layout="wide"
)


# Define project paths.
PROJECT_ROOT = Path(__file__).resolve().parent

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "compressed_random_forest_rating_model.joblib"
)

FEATURE_NAMES_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "model_feature_names.csv"
)

DEPLOYMENT_SUMMARY_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "deployment_friendly_model_artifact_summary.csv"
)


@st.cache_resource
def load_model(model_path):
    """Load the trained machine-learning model."""
    return joblib.load(model_path)


@st.cache_data
def load_feature_names(feature_names_path):
    """Load the saved model feature names."""
    feature_names = pd.read_csv(feature_names_path)
    return feature_names["Feature"].tolist()


@st.cache_data
def load_deployment_summary(summary_path):
    """Load the deployment-friendly model artifact summary."""
    return pd.read_csv(summary_path)


# Main page title
st.title("🎲 Board Game Rating Predictor")


# Short project introduction
st.write(
    "This app will use a trained machine-learning model to predict "
    "a board game's average rating based on selected game features."
)


# App status message.
st.info(
    "This version of the app loads the deployment-friendly model artifact. "
    "User input and prediction features will be added in later steps."
)


# Load model assets.
st.header("Model Loading Check")

required_files = {
    "Deployment model": MODEL_PATH,
    "Model feature names": FEATURE_NAMES_PATH,
    "Deployment summary": DEPLOYMENT_SUMMARY_PATH
}

missing_files = [
    file_name
    for file_name, file_path in required_files.items()
    if not file_path.exists()
]

if missing_files:
    st.error(
        "The app could not find all required model files. "
        "Missing files: "
        + ", ".join(missing_files)
    )
    st.stop()


model = load_model(MODEL_PATH)
feature_names = load_feature_names(FEATURE_NAMES_PATH)
deployment_summary = load_deployment_summary(DEPLOYMENT_SUMMARY_PATH)

st.success("Deployment-friendly model loaded successfully.")


# Show basic model information.
st.subheader("Loaded Model Details")

model_details = {
    "Model file": str(MODEL_PATH.relative_to(PROJECT_ROOT)),
    "Model type": type(model).__name__,
    "Number of model input features": model.n_features_in_,
    "Number of saved feature names": len(feature_names)
}

st.json(model_details)


# Show deployment model summary.
st.subheader("Deployment Artifact Summary")

st.dataframe(
    deployment_summary,
    use_container_width=True
)


# Add a simple project overview section.
st.header("Project Overview")

st.write(
    "The project uses BoardGameGeek-style board game data to train and "
    "compare regression models. The current selected deployment artifact is "
    "a compressed Random Forest model."
)


# Add a simple next-steps section.
st.header("Next Steps")

st.write(
    "Next, this app will collect user inputs, convert those inputs into the "
    "same feature format used during training, and display a predicted board "
    "game rating."
)
