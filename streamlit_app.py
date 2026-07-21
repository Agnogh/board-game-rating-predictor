# Import tools for file paths, model loading, data loading, math, and web app
from pathlib import Path

import joblib
import numpy as np
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


def create_model_input(
    feature_names,
    year_published,
    min_players,
    max_players,
    play_time,
    min_age,
    complexity_average,
    selected_mechanics,
    selected_domains
):
    """Convert user-friendly app inputs into the model's feature format."""

    model_input = pd.DataFrame(
        data=0.0,
        index=[0],
        columns=feature_names
    )

    if "Year Published" in model_input.columns:
        model_input.loc[0, "Year Published"] = year_published

    if "Min Players" in model_input.columns:
        model_input.loc[0, "Min Players"] = min_players

    if "Min Age" in model_input.columns:
        model_input.loc[0, "Min Age"] = min_age

    if "Complexity Average" in model_input.columns:
        model_input.loc[0, "Complexity Average"] = complexity_average

    if "Max Players Log" in model_input.columns:
        model_input.loc[0, "Max Players Log"] = np.log1p(max_players)

    if "Play Time Log" in model_input.columns:
        model_input.loc[0, "Play Time Log"] = np.log1p(play_time)

    for mechanic in selected_mechanics:
        mechanic_feature_name = f"Mechanic: {mechanic}"

        if mechanic_feature_name in model_input.columns:
            model_input.loc[0, mechanic_feature_name] = 1.0

    for domain in selected_domains:
        domain_feature_name = f"Domain: {domain}"

        if domain_feature_name in model_input.columns:
            model_input.loc[0, domain_feature_name] = 1.0

    return model_input


# Main page title
st.title("🎲 Board Game Rating Predictor")


# Short project introduction
st.write(
    "This app will use a trained machine-learning model to predict "
    "a board game's average rating based on selected game features."
)


# App status message
st.info(
    "This version of the app converts board game inputs into model features. "
    "Prediction will be added in a later step."
)


# Load model assets
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
    width="stretch"
)


# Build input options from saved feature names.
mechanic_options = sorted(
    [
        feature_name.replace("Mechanic: ", "")
        for feature_name in feature_names
        if feature_name.startswith("Mechanic: ")
    ]
)

domain_options = sorted(
    [
        feature_name.replace("Domain: ", "")
        for feature_name in feature_names
        if feature_name.startswith("Domain: ")
    ]
)


# Add the board game input form.
st.header("Board Game Input Form")

st.write(
    "Enter basic board game information below. "
    "For this step, the app collects inputs and converts them into the "
    "same feature format used during model training."
)

with st.form("board_game_input_form"):
    st.subheader("Basic Game Details")

    col1, col2 = st.columns(2)

    with col1:
        year_published = st.number_input(
            "Year Published",
            min_value=1800,
            max_value=2026,
            value=2020,
            step=1
        )

        min_players = st.number_input(
            "Minimum Players",
            min_value=1,
            max_value=20,
            value=2,
            step=1
        )

        max_players = st.number_input(
            "Maximum Players",
            min_value=1,
            max_value=100,
            value=4,
            step=1
        )

    with col2:
        play_time = st.number_input(
            "Play Time in Minutes",
            min_value=1,
            max_value=600,
            value=60,
            step=5
        )

        min_age = st.number_input(
            "Minimum Age",
            min_value=0,
            max_value=21,
            value=10,
            step=1
        )

        complexity_average = st.number_input(
            "Complexity Average",
            min_value=1.0,
            max_value=5.0,
            value=2.5,
            step=0.1
        )

    st.subheader("Game Categories")

    selected_mechanics = st.multiselect(
        "Select Mechanics",
        options=mechanic_options
    )

    selected_domains = st.multiselect(
        "Select Domains",
        options=domain_options
    )

    submitted = st.form_submit_button("Convert Input to Model Features")


if min_players > max_players:
    st.warning(
        "Minimum players is greater than maximum players. "
        "Please adjust the player counts before prediction is added."
    )


if submitted:
    user_input_preview = pd.DataFrame(
        [
            {
                "Year Published": year_published,
                "Minimum Players": min_players,
                "Maximum Players": max_players,
                "Play Time": play_time,
                "Minimum Age": min_age,
                "Complexity Average": complexity_average,
                "Selected Mechanics": ", ".join(selected_mechanics),
                "Selected Domains": ", ".join(selected_domains)
            }
        ]
    )

    model_input = create_model_input(
        feature_names=feature_names,
        year_published=year_published,
        min_players=min_players,
        max_players=max_players,
        play_time=play_time,
        min_age=min_age,
        complexity_average=complexity_average,
        selected_mechanics=selected_mechanics,
        selected_domains=selected_domains
    )

    feature_order_matches = model_input.columns.tolist() == feature_names
    feature_count_matches = model_input.shape[1] == model.n_features_in_

    st.success("Input converted into model feature format successfully.")

    st.subheader("User Input Preview")

    st.dataframe(
        user_input_preview,
        width="stretch"
    )

    st.subheader("Model Feature Conversion Check")

    conversion_details = {
        "Converted feature row shape": str(model_input.shape),
        "Feature order matches saved feature names": feature_order_matches,
        "Feature count matches model input features": feature_count_matches
    }

    st.json(conversion_details)

    st.subheader("Converted Model Feature Values")

    model_input_preview = (
        model_input
        .T
        .reset_index()
        .rename(
            columns={
                "index": "Feature",
                0: "Value"
            }
        )
    )

    st.dataframe(
        model_input_preview,
        width="stretch"
    )

    st.info(
        "This confirms that the app can convert form inputs into the exact "
        "feature structure expected by the model. The next step will use this "
        "converted feature row to make a prediction."
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
    "Next, this app will use the converted 35-feature model "
    "input row to display a predicted board game rating."
)
