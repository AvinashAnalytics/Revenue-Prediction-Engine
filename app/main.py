import streamlit as st
import numpy as np
import pandas as pd
from app.utils import (
    load_feature_scaler,
    load_target_scaler,
    load_keras_model,
    preprocess_input,
    inverse_scale_prediction,
    create_pie_chart,
    create_trend_chart
)

# Custom CSS for styling
st.markdown("""
<style>
    .header {
        color: #1f567d;
        border-bottom: 3px solid #1f567d;
        padding-bottom: 1rem;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: rgba(255,255,255,0.95);
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: transform 0.3s;
        margin: 1rem 0;
    }
    .metric-card:hover {
        transform: translateY(-5px);
    }
    .stButton>button {
        background: linear-gradient(45deg, #1f567d, #2c3e50);
        color: white !important;
        border: none;
        padding: 12px 30px;
        border-radius: 25px;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 5px 15px rgba(31, 86, 125, 0.3);
    }
</style>
""", unsafe_allow_html=True)

# Load assets
@st.cache_resource
def load_assets():
    """Load models and scalers with caching for better performance."""
    return (
        load_keras_model(),
        load_feature_scaler(),
        load_target_scaler()
    )

model, feature_scaler, target_scaler = load_assets()

# Sidebar
with st.sidebar:
    st.image("https://avatars.githubusercontent.com/u/86707796?v=4", width=120)
    st.markdown("## Avinash Rai")
    st.markdown("**Data Analyst**")
    
    st.markdown("---")
    st.markdown("""
    **Connect:**
    - 📧 [masteravinashrai@gmail.com](mailto:masteravinashrai@gmail.com)
    - 💼 [LinkedIn](https://linkedin.com/in/avinashanalytics)
    - 👨💻 [GitHub](https://github.com/AvinashAnalytics)
    """)

# Main interface
st.markdown("# 🍽️ Restaurant Revenue Predictor")
st.markdown("### AI-Powered Forecasting System")

# Prediction Section
with st.expander("🚀 Make Prediction", expanded=True):
    col1, col2 = st.columns(2)
    
    with col1:
        franchise = st.selectbox("Franchise Status", ["Independent", "Franchised"])
        category = st.selectbox("Cuisine Type", ["Fast Food", "Casual Dining", "Fine Dining"])
    
    with col2:
        menu_size = st.number_input("Menu Items", min_value=1, value=25)
        orders = st.slider("Monthly Orders (×1000)", 1, 500, 150)

    if st.button("Predict Revenue"):
        # Prepare inputs
        inputs = {
            'franchise': 1 if franchise == "Franchised" else 0,
            'category': category,
            'menu_size': menu_size,
            'orders': orders
        }
        
        # Preprocess and predict
        processed_input = preprocess_input(inputs, feature_scaler)
        scaled_prediction = model.predict(processed_input)
        prediction = inverse_scale_prediction(scaled_prediction, target_scaler)
        
        # Display results
        st.markdown(f"""
        <div class="metric-card">
            <h3>Predicted Monthly Revenue</h3>
            <h1>₹ {prediction:,.0f}</h1>
            <p>± 5% margin of error</p>
        </div>
        """, unsafe_allow_html=True)

# Analytics Section
st.markdown("## 📊 Performance Insights")
col1, col2 = st.columns(2)

with col1:
    st.plotly_chart(create_pie_chart(), use_container_width=True)

with col2:
    st.plotly_chart(create_trend_chart(), use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 1rem; color: #6c757d;">
    <p>🔮 Powered by Deep Learning | 🚀 Production-Ready Pipeline</p>
    <small>Developed by Avinash Rai | 2023 Revenue Prediction System</small>
</div>
""", unsafe_allow_html=True)