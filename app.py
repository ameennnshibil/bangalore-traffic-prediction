import streamlit as st
import pandas as pd
import joblib


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Bangalore Traffic Prediction",
    page_icon="🚦",
    layout="centered"
)


# ============================================================
# 2. TITLE
# ============================================================

st.title("🚦 Bangalore Traffic Prediction System")

st.write(
    "Enter traffic details below to predict the traffic volume."
)


# ============================================================
# 3. LOAD DATASET
# ============================================================

try:
    data = pd.read_csv("Banglore_traffic_Dataset.csv")
except Exception as e:
    st.error("Could not load the dataset.")
    st.error(e)
    st.stop()


# ============================================================
# 4. LOAD TRAINED MODEL
# ============================================================

try:
    model = joblib.load("traffic_model.pkl")
except Exception as e:
    st.error("Could not load traffic_model.pkl")
    st.error(
        "Make sure traffic_model.pkl is present in the same "
        "folder as app.py."
    )
    st.stop()


# ============================================================
# 5. USER INPUTS
# ============================================================

st.subheader("Enter Traffic Details")


average_speed = st.number_input(
    "Average Speed",
    min_value=0.0,
    value=40.0,
    step=1.0
)


travel_time = st.number_input(
    "Travel Time Index",
    min_value=0.0,
    value=1.5,
    step=0.1
)


road_capacity = st.number_input(
    "Road Capacity Utilization",
    min_value=0.0,
    max_value=1.0,
    value=0.75,
    step=0.05
)


# ============================================================
# 6. PREDICTION BUTTON
# ============================================================

if st.button("🚦 Predict Traffic"):

    try:

        # Create empty dictionary for model input
        user_data = {}


        # ----------------------------------------------------
        # Get feature names used by the trained model
        # ----------------------------------------------------

        if hasattr(model, "feature_names_in_"):
            model_features = model.feature_names_in_

        else:
            # Fallback if model does not contain feature names
            model_features = data.columns.tolist()


        # ----------------------------------------------------
        # Create values for every feature required by model
        # ----------------------------------------------------

        for column in model_features:

            # User entered Average Speed
            if column == "Average Speed":

                user_data[column] = average_speed


            # User entered Travel Time Index
            elif column == "Travel Time Index":

                user_data[column] = travel_time


            # User entered Road Capacity Utilization
            elif column == "Road Capacity Utilization":

                user_data[column] = road_capacity


            # If feature exists in dataset,
            # use its average value
            elif column in data.columns:

                # Make sure the column is numeric
                numeric_values = pd.to_numeric(
                    data[column],
                    errors="coerce"
                )

                user_data[column] = numeric_values.mean()


            # If feature doesn't exist anywhere,
            # use 0 as a safe fallback
            else:

                user_data[column] = 0


        # ----------------------------------------------------
        # Convert user data into DataFrame
        # ----------------------------------------------------

        user_input = pd.DataFrame([user_data])


        # ----------------------------------------------------
        # Make prediction
        # ----------------------------------------------------

        prediction = model.predict(user_input)


        # ----------------------------------------------------
        # Display result
        # ----------------------------------------------------

        st.success("Prediction completed successfully! 🎉")


        st.metric(
            label="Predicted Traffic Volume",
            value=f"{prediction[0]:,.2f}"
        )


        # ----------------------------------------------------
        # Additional interpretation
        # ----------------------------------------------------

        st.subheader("Traffic Status")


        if prediction[0] < 15000:

            st.success("🟢 Low Traffic")


        elif prediction[0] < 25000:

            st.warning("🟡 Moderate Traffic")


        else:

            st.error("🔴 High Traffic")


    except Exception as e:

        st.error("Prediction failed.")

        st.error(str(e))