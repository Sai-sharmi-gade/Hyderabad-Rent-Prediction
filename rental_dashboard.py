import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pydeck as pdk

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

# Load data
df = pd.read_csv('cleaned_property_rentals.csv')
df['price'] = df['price'] * 83  # Convert USD to INR (Assuming 1 USD = ₹83)

# Set page config
st.set_page_config(
    page_title="House Rent Predictor", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .css-1d391kg {
        background-color: white;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
    .prediction-result {
        background: linear-gradient(90deg, #4CAF50, #45a049);
        color: white;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        margin: 20px 0;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("# 🏠 House Rent Predictor")
st.markdown("### Find the perfect rental price for your property")
st.markdown("---")

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Choose a section:", ["📊 Explore Data", "🎯 Predict Rent", "ℹ️ About"])
# About section
if page == "ℹ️ About":
    st.header("About This App")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("What does this app do?")
        st.write("""
        This app helps you:
        - **Explore** rental property data
        - **Analyze** price trends
        - **Predict** rental prices based on property features
        """)
        st.subheader("How to use:")
        st.write("""
        1. **Explore Data**: View charts and statistics
        2. **Predict Rent**: Enter property details to get price estimate
        """)
    with col2:
        st.subheader("Dataset Overview")
        st.write(f"📋 **Total Properties**: {len(df):,}")
        st.write(f"💰 **Average Rent**: ${df['price'].mean():.2f}")
        st.write(f"🏠 **Property Types**: {df['property_type'].nunique()}")
        st.write(f"🏋️ **Room Types**: {df['room_type'].nunique()}")

# Data exploration
elif page == "📊 Explore Data":
    st.header("Data Exploration")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Properties", f"{len(df):,}")
    with col2:
        st.metric("Average Rent", f"₹{df['price'].mean():,.0f}")
    with col3:
        st.metric("Average Rent", f"₹{df['price'].mean():,.0f}")
    with col4:
        st.metric("Average Rent", f"₹{df['price'].mean():,.0f}")

    st.markdown("---")
    chart_option = st.selectbox("Choose a chart to display:", 
                               ["Price Distribution", "Room Type vs Price", "Bedrooms vs Price"])
    if chart_option == "Price Distribution":
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(df['price'], bins=30, kde=True, ax=ax, color='skyblue')
        ax.set_title('Distribution of Rental Prices')
        st.pyplot(fig)
    elif chart_option == "Room Type vs Price":
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(x='room_type', y='price', data=df, ax=ax, palette='Set2', hue='room_type', legend=False)
        ax.set_title('Price Distribution by Room Type')
        plt.xticks(rotation=45)
        st.pyplot(fig)
    elif chart_option == "Bedrooms vs Price":
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(x='bedrooms', y='price', data=df, ax=ax, palette='viridis', hue='bedrooms', legend=False)
        ax.set_title('Price Distribution by Bedrooms')
        st.pyplot(fig)

# Prediction section
elif page == "🎯 Predict Rent":
    st.header("Predict Your Rental Price")
    col1, col2 = st.columns(2)
    with col1:
        latitude = st.number_input("Latitude", value=17.42, format="%.5f")
        longitude = st.number_input("Longitude", value=78.45, format="%.5f")
        property_type = st.selectbox("Property Type", df['property_type'].unique())
        room_type = st.selectbox("Room Type", df['room_type'].unique())
    with col2:
        bathrooms = st.slider("Number of Bathrooms", 0, 10, 2)
        bedrooms = st.slider("Number of Bedrooms", 0, 10, 2)
        minimum_nights = st.slider("Minimum Nights", 1, 100, 5)

    st.markdown("---")
    if st.button("🔮 Predict Rent", type="primary"):
        with st.spinner("Calculating your rental price..."):
            X = df.drop('price', axis=1)
            y = df['price']
            categorical_features = X.select_dtypes(include='object').columns.tolist()
            preprocessor = ColumnTransformer([
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
            ], remainder='passthrough')
            model = Pipeline([
                ('preprocessor', preprocessor),
                ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
            ])
            model.fit(X, y)
            input_data = pd.DataFrame([{
                'latitude': latitude,
                'longitude': longitude,
                'property_type': property_type,
                'room_type': room_type,
                'bathrooms': bathrooms,
                'bedrooms': bedrooms,
                'minimum_nights': minimum_nights
            }])
            prediction = model.predict(input_data)[0]

            st.markdown(f"""
            <div class="prediction-result">
                💰 Estimated Monthly Rent: ₹{prediction:,.0f}
            </div>
            """, unsafe_allow_html=True)

            st.success("Prediction completed successfully!")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Your Estimate", f"₹{prediction:,.0f}")
            with col2:
                market_avg = df['price'].mean()
                st.metric("Market Average", f"₹{market_avg:.2f}", 
                         f"{((prediction - market_avg) / market_avg * 100):+.1f}%")
            with col3:
                similar_properties = df[(df['property_type'] == property_type) & 
                                        (df['room_type'] == room_type)]['price'].mean()
                st.metric("Similar Properties", f"${similar_properties:.2f}",
                         f"{((prediction - similar_properties) / similar_properties * 100):+.1f}%")

            # Bonus 1: Map
            st.subheader("📍 Property Location on Map")
            map_df = pd.DataFrame({'lat': [latitude], 'lon': [longitude]})
            st.pydeck_chart(pdk.Deck(
                map_style='mapbox://styles/mapbox/light-v9',
                initial_view_state=pdk.ViewState(
                    latitude=latitude,
                    longitude=longitude,
                    zoom=12,
                    pitch=50,
                ),
                layers=[
                    pdk.Layer('ScatterplotLayer',
                              data=map_df,
                              get_position='[lon, lat]',
                              get_radius=100,
                              get_color=[255, 0, 0],
                              pickable=True)
                ]
            ))

            # Bonus 2: Download
            if st.button("📅 Download Estimate as CSV"):
                output_df = input_data.copy()
                output_df["Predicted Rent"] = prediction
                st.download_button("Download Result", data=output_df.to_csv(index=False),
                                   file_name="rent_prediction.csv", mime="text/csv")

# Footer
st.markdown("---")
st.markdown("Made with ❤️ using Streamlit | Data-driven rental predictions By Sai Sharmi Gade")
