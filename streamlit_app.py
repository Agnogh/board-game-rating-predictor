# Import Streamlit for building web app
import streamlit as st


# Configure the Streamlit page.
st.set_page_config(
    page_title="Board Game Rating Predictor",
    page_icon="🎲",
    layout="wide"
)


# Main page title
st.title("🎲 Board Game Rating Predictor")


# Short project introduction
st.write(
    "This app will use a trained machine-learning model to predict "
    "a board game's average rating based on selected game features."
)


# Current app status
st.info(
    "Initial Streamlit app structure created successfully. "
    "Model loading and prediction features will be added in later steps."
)


# Add a simple project overview section
st.header("Project Overview")

st.write(
    "The project uses BoardGameGeek-style board game data to train and "
    "compare regression models. The current selected model is a compressed "
    "Random Forest model prepared for future deployment."
)


# Add a simple next-steps section
st.header("Next Steps")

st.write(
    "Next, this app will load the deployment-friendly model artifact, "
    "collect user inputs, convert those inputs into model features, "
    "and display a predicted board game rating."
)