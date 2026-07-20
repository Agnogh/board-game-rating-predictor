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


# App status message
st.info(
    "This version of the app collects board game inputs. "
    "Feature conversion and prediction will be added in later steps."
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
    use_container_width=True
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
    "For this step, the app only collects and previews the inputs. "
    "Prediction will be added in a later step."
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

    submitted = st.form_submit_button("Preview Input")


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

    st.success("Input collected successfully.")

    st.subheader("Input Preview")

    st.dataframe(
        user_input_preview,
        use_container_width=True
    )

    st.info(
        "This preview confirms that the app can collect user input. "
        "The next step will convert these values into the exact model feature "
        "format needed for prediction."
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
    "Next, this app will convert user inputs into the same 35-feature format "
    "used during training, then display a predicted board game rating. "
)
