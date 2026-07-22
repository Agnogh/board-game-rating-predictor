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


def format_selected_items(selected_items):
    """Create clean display text for selected mechanics and domains."""

    if selected_items:
        return ", ".join(selected_items)

    return "None selected"


# Main page title
st.title("🎲 Board Game Rating Predictor")


# Short project introduction
st.write(
    "Estimate a board game's average rating using a trained machine-learning "
    "model based on BoardGameGeek-style game features."
)


st.caption(
    "Portfolio project: predictive analytics, feature engineering, model "
    "selection, and Streamlit deployment."
)


# Load model assets
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


# Sidebar project information
# -> Sidebar + explanation for clarity
with st.sidebar:
    st.header("About this app")

    st.write(
        "This app uses a compressed Random Forest model trained on board game "
        "data. The model estimates an average rating from design features such "
        "as player count, play time, complexity, mechanics, and domain."
    )

    st.subheader("Current model")

    st.write("Random Forest Regressor")

    st.subheader("Model input")

    st.write(f"{len(feature_names)} engineered features")

    st.subheader("Important note")

    st.write(
        "The prediction is an estimate, not a guaranteed real-world rating. "
        "Board game ratings can also depend on theme, art, reviews, marketing, "
        "community attention, and other factors not fully captured here."
    )


# Model status section
# -> model details move from top to inside
with st.expander("Model loading and deployment details", expanded=False):
    st.success("Deployment-friendly model loaded successfully.")

    model_details = {
        "Model file": str(MODEL_PATH.relative_to(PROJECT_ROOT)),
        "Model type": type(model).__name__,
        "Number of model input features": model.n_features_in_,
        "Number of saved feature names": len(feature_names)
    }

    st.subheader("Loaded Model Details")
    st.json(model_details)

    st.subheader("Deployment Artifact Summary")
    st.dataframe(
        deployment_summary,
        width="stretch"
    )


# Add the board game input form.
st.header("Enter Board Game Details")

st.write(
    "Fill in the information below, then click **Predict Rating**. "
    "The app will convert your inputs into the same feature format used during "
    "model training and show the model's estimated rating."
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
            step=1,
            help="The year the game was published."
        )

        min_players = st.number_input(
            "Minimum Players",
            min_value=1,
            max_value=20,
            value=2,
            step=1,
            help="The smallest supported player count."
        )

        max_players = st.number_input(
            "Maximum Players",
            min_value=1,
            max_value=100,
            value=4,
            step=1,
            help="The largest supported player count."
        )

    with col2:
        play_time = st.number_input(
            "Play Time in Minutes",
            min_value=1,
            max_value=600,
            value=60,
            step=5,
            help="Approxamate play time in minutes."
        )

        min_age = st.number_input(
            "Minimum Age",
            min_value=0,
            max_value=21,
            value=10,
            step=1,
            help="Recommended minimum player age."
        )

        complexity_average = st.number_input(
            "Complexity Average",
            min_value=1.0,
            max_value=5.0,
            value=2.5,
            step=0.1,
            help="A rough complexity/weight score from 1.0 to 5.0."
        )

    st.subheader("Game Categories")

    selected_mechanics = st.multiselect(
        "Select Mechanics",
        options=mechanic_options,
        help="Choose one or more mechanics that describe the game."
    )

    selected_domains = st.multiselect(
        "Select Domains",
        options=domain_options,
        help="Choose on or more broad board game categories."
    )

    submitted = st.form_submit_button("Predict Rating")


if min_players > max_players:
    st.warning(
        "Minimum players is greater than maximum players. "
        "Please adjust the player counts before making a prediction."
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
                "Selected Mechanics": format_selected_items(selected_mechanics),
                "Selected Domains": format_selected_items(selected_domains)
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

    if not feature_order_matches or not feature_count_matches:
        st.error(
            "The converted input does not match the model's expected feature "
            "structure, so the app will not make a prediction."
        )
        st.stop()

    prediction = model.predict(model_input)[0]

    st.divider()
    # has its own section now
    st.header("Predicted Board Game Rating")

    prediction_col, explanation_col = st.columns([1, 2])
    # stands out important results
    with prediction_col:
        st.metric(
            label="Predicted Average Rating",
            value=f"{prediction:.2f} / 10"
        )

    with explanation_col:
        st.write(
            "This is the model's estimated average rating for a board game "
            "with the selected features."
        )

        st.write(
            "Use this as a directional prediction rather than a guaranteed "
            "rating. The model is based on patterns in the training data."
        )
    # hiding (unless opened on purpose)
    with st.expander("Show user input summary", expanded=False):
        st.dataframe(
            user_input_preview,
            width="stretch"
        )
    # hiding (to improve clarity / UI)
    with st.expander("Show model feature validation checks", expanded=False):
        conversion_details = {
            "Converted feature row shape": str(model_input.shape),
            "Feature order matches saved feature names": feature_order_matches,
            "Feature count matches model input features": feature_count_matches
        }

        st.json(conversion_details)
    # hidden -> better UI
    with st.expander("Show converted 35-feature model input", expanded=False):
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


st.divider()


# Add a simple project overview section.
st.header("Project Overview")

st.write(
    "This project follows a complete beginner-friendly machine-learning "
    "workflow: data inspection, cleaning, exploratory analysis, feature "
    "engineering, baseline modelling, model improvement, artifact preparation, "
    "and Streamlit app development."
)


# Add a simple model limitation section
st.header("Model Notes")

st.write(
    "The selected model is a compressed Random Forest Regressor. It performed "
    "better than the baseline and linear regression models during testing, but "
    "it still has limitations. The prediction should be interpreted as an "
    "estimate based on available structured features, not as a final judgement "
    "of a game's quality."
)


# Add a simple next-steps section.
st.header("Next Steps")

st.write(
    "Next, the project can be prepared for deployment and documented clearly "
    "in the README so that other people can understand and run the app."
)
