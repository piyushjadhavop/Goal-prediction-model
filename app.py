import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
# Load trained model
model = joblib.load('model.pkl')

# Streamlit page configuration
st.set_page_config(
    page_title="Football Goal Prediction",
    page_icon="⚽",
    layout="centered"
)

# Title
st.title("⚽ Football Goal Prediction App")

st.write("Enter player statistics to predict goals.")

# Input Fields
matches_played = st.number_input(
    "Matches Played",
    min_value=0,
    step=1
)

mins = st.number_input(
    "Minutes Played",
    min_value=0,
    step=1
)

xg_avg = st.number_input(
    "xG Per Avg Match",
    min_value=0.0,
    step=0.01
)

shots = st.number_input(
    "Shots",
    min_value=0,
    step=1
)

on_target = st.number_input(
    "On Target",
    min_value=0,
    step=1
)

shots_avg = st.number_input(
    "Shots Per Avg Match",
    min_value=0.0,
    step=0.1
)

on_target_avg = st.number_input(
    "On Target Per Avg Match",
    min_value=0.0,
    step=0.1
)

# Predict Button
if st.button("Predict Goals"):

    # Create dataframe from user input
    input_data = pd.DataFrame({

        'Matches_Played': [matches_played],
        'Mins': [mins],
        'xG Per Avg Match': [xg_avg],
        'Shots': [shots],
        'OnTarget': [on_target],
        'Shots Per Avg Match': [shots_avg],
        'On Target Per Avg Match': [on_target_avg]
    })

    # Prediction
    prediction = model.predict(input_data)

    # Display Result
    st.success(
        f"Predicted Goals: {prediction[0]:.2f}"
    )
    # -------------------------------
# BAR VISUALIZATION
# -------------------------------


feature_df = pd.DataFrame({
    'Features': [
        'Matches Played',
        'Minutes',
        'xG Per Avg Match',
        'Shots',
        'On Target',
        'Shots Per Avg Match',
        'On Target Per Avg Match',
        'Predicted Goals'
    ],
    'Values': [
        matches_played,
        mins,
        xg_avg,
        shots,
        on_target,
        shots_avg,
        on_target_avg,
        prediction[0]
    ]
})

bar_fig = px.bar(
    feature_df,
    x='Features',
    y='Values',
    text='Values',
    color='Features',
    title='Player Statistics & Predicted Goals'
)

# Improve Layout
bar_fig.update_layout(
    xaxis_title="Features",
    yaxis_title="Values",
    showlegend=False
)

# Display Graph
st.plotly_chart(
    bar_fig,
    use_container_width=True
)