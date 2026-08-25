import streamlit as st
import pandas as pd
import joblib
import requests


st.set_page_config(
    page_title="Bangalore Traffic Prediction",
    page_icon="🚦",
    layout="centered"
)


st.markdown(
    """
    <style>
    .main {
        padding-top: 2rem;
    }

    h1 {
        text-align: center;
        font-size: 42px !important;
        font-weight: 700 !important;
    }

    div.stButton > button {
        width: 100%;
        border-radius: 10px;
        height: 3em;
        font-weight: 600;
    }

    [data-testid="stMetric"] {
        background: rgba(128, 128, 128, 0.10);
        padding: 20px;
        border-radius: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# Load model
try:
    model = joblib.load("traffic_model.pkl")
except Exception as e:
    st.error("Could not load traffic_model.pkl")
    st.error(str(e))
    st.stop()


# Load dataset for traffic-status limits
try:
    dataset = pd.read_csv(
        "Bangalore_traffic_Dataset.csv"
    )

    low_limit = dataset[
        "Traffic Volume"
    ].quantile(0.33)

    high_limit = dataset[
        "Traffic Volume"
    ].quantile(0.66)

except Exception:
    low_limit = 22000
    high_limit = 34000


def get_live_traffic():

    try:

        api_key = st.secrets[
            "MAPPLS_API_KEY"
        ]

        start = "77.5727,12.9767"
        end = "77.6408,12.9784"

        url = (
            "https://route.mappls.com/route/"
            "direction/route_eta/driving/"
            + start
            + ";"
            + end
        )

        params = {
            "region": "ind",
            "rtype": 0,
            "access_token": api_key
        }

        response = requests.get(
            url,
            params=params,
            timeout=10
        )

        if response.status_code != 200:

            st.error(
                f"Mappls API error: "
                f"{response.status_code}"
            )

            return None, None

        data = response.json()

        if not data.get("routes"):

            st.error(
                "No route information received."
            )

            return None, None

        route = data["routes"][0]

        distance = route.get(
            "distance",
            0
        )

        duration = route.get(
            "duration",
            0
        )

        return distance, duration

    except Exception as e:

        st.error(
            f"Live traffic API error: {e}"
        )

        return None, None


# Title
st.title(
    "🚦 Bangalore Traffic Prediction System"
)

st.markdown(
    """
    <div style="text-align:center; font-size:18px;">
    Smart traffic prediction and live traffic
    monitoring for Bangalore roads.
    </div>
    """,
    unsafe_allow_html=True
)


st.divider()


# Traffic input
st.header("📊 Traffic Details")

st.write(
    "Enter the current road conditions "
    "to predict traffic volume."
)


average_speed = st.number_input(
    "🚗 Average Speed (km/h)",
    min_value=0.0,
    max_value=150.0,
    value=40.0,
    step=1.0
)


traffic_delay = st.number_input(
    "⏱️ Traffic Delay (%)",
    min_value=0.0,
    max_value=300.0,
    value=30.0,
    step=5.0
)


road_usage = st.number_input(
    "🛣️ Road Usage (%)",
    min_value=0.0,
    max_value=100.0,
    value=60.0,
    step=5.0
)


st.caption(
    "Traffic Delay = extra travel time caused "
    "by traffic. Road Usage = percentage of "
    "road capacity being used."
)


st.divider()


# Prediction
if st.button("🚦 Predict Traffic"):

    try:

        input_data = pd.DataFrame(
            {
                "Average Speed": [
                    average_speed
                ],
                "Traffic Delay": [
                    traffic_delay
                ],
                "Road Usage": [
                    road_usage
                ]
            }
        )

        predicted_volume = model.predict(
            input_data
        )[0]

        predicted_volume = max(
            0,
            predicted_volume
        )


        st.header(
            "📈 Prediction Result"
        )

        st.success(
            "✅ Prediction completed successfully!"
        )


        st.metric(
            "🚗 Estimated Traffic Volume",
            f"{predicted_volume:,.0f} vehicles"
        )


        st.subheader(
            "🚦 Traffic Status"
        )


        if predicted_volume <= low_limit:

            st.success(
                "🟢 Low Traffic"
            )

        elif predicted_volume <= high_limit:

            st.warning(
                "🟡 Moderate Traffic"
            )

        else:

            st.error(
                "🔴 High Traffic"
            )


        st.caption(
            f"Low: up to {low_limit:,.0f} vehicles | "
            f"Moderate: {low_limit:,.0f}–"
            f"{high_limit:,.0f} vehicles | "
            f"High: above {high_limit:,.0f} vehicles"
        )


        st.subheader(
            "📋 Values Used"
        )


        col1, col2, col3 = st.columns(3)


        with col1:

            st.metric(
                "Speed",
                f"{average_speed:.0f} km/h"
            )


        with col2:

            st.metric(
                "Delay",
                f"{traffic_delay:.0f}%"
            )


        with col3:

            st.metric(
                "Road Usage",
                f"{road_usage:.0f}%"
            )


    except Exception as e:

        st.error(
            "❌ Prediction failed."
        )

        st.error(str(e))


st.divider()


# Live traffic
st.header(
    "🌐 Live Traffic Information"
)

st.write(
    "Get current route distance and travel "
    "time from the live traffic service."
)


if st.button(
    "🌐 Get Live Traffic"
):

    distance, duration = (
        get_live_traffic()
    )


    if distance is not None:

        st.success(
            "✅ Live traffic data received!"
        )


        col1, col2 = st.columns(2)


        with col1:

            st.metric(
                "📍 Route Distance",
                f"{distance / 1000:.2f} km"
            )


        with col2:

            st.metric(
                "⏱️ Current Travel Time",
                f"{duration / 60:.1f} minutes"
            )


        distance_km = (
            distance / 1000
        )

        travel_minutes = (
            duration / 60
        )


        free_flow_speed = 40.0


        free_flow_minutes = (
            distance_km /
            free_flow_speed
        ) * 60


        if free_flow_minutes > 0:

            live_delay = (
                (
                    travel_minutes -
                    free_flow_minutes
                )
                / free_flow_minutes
            ) * 100

            live_delay = max(
                0,
                live_delay
            )

        else:

            live_delay = 0


        st.info(
            f"📊 Live Traffic Delay: "
            f"{live_delay:.1f}%"
        )


        st.caption(
            "Live delay compares the current "
            "travel time with a 40 km/h "
            "free-flow speed."
        )


    else:

        st.error(
            "❌ Unable to get live traffic data."
        )


st.divider()


st.header(
    "ℹ️ What do these values mean?"
)


with st.expander(
    "🚗 Average Speed"
):

    st.write(
        "Average speed of vehicles on "
        "the road in km/h."
    )


with st.expander(
    "⏱️ Traffic Delay"
):

    st.write(
        "Percentage of extra travel time "
        "caused by traffic."
    )


with st.expander(
    "🛣️ Road Usage"
):

    st.write(
        "Percentage of the road's traffic "
        "capacity currently being used."
    )


with st.expander(
    "🚦 Traffic Status"
):

    st.write(
        "Low, Moderate, or High traffic "
        "is determined from the traffic "
        "volume distribution in the dataset."
    )


st.divider()


st.markdown(
    """
    <div style="text-align:center;">
    <small>
    Bangalore Traffic Prediction System |
    Machine Learning + Live Traffic Data
    </small>
    </div>
    """,
    unsafe_allow_html=True
)