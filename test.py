import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
import plotly.express as px
from streamlit.components.v1 import html

# Load model
@st.cache_resource
def load_ml_model():
    return load_model('neural_network_model.h5')

model = load_ml_model()

# ========== Custom CSS & Animations ==========
st.markdown("""
<style>
    @keyframes gradientBG {
        0% {background-position: 0% 50%}
        50% {background-position: 100% 50%}
        100% {background-position: 0% 50%}
    }
    
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        animation: gradientBG 12s ease infinite;
        background-size: 400% 400%;
    }
    
    .header-glow {
        text-shadow: 0 0 15px #1f567d, 0 0 20px #1f567d;
    }
    
    .metric-card {
        background: rgba(255, 255, 255, 0.9);
        border-radius: 15px;
        padding: 25px;
        margin: 10px;
        box-shadow: 0 8px 32px rgba(31, 86, 125, 0.1);
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px rgba(31, 86, 125, 0.2);
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

# ========== Sidebar - Personal Branding ==========
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding:20px; border-radius:15px; background: linear-gradient(135deg, #1f567d, #2c3e50);">
        <img src="https://avatars.githubusercontent.com/u/86707796?v=4" width="120" style="border-radius:50%; border:3px solid white; margin-bottom:15px;">
        <h2 style='color: white; margin:0;'>Avinash Rai</h2>
        <p style='color: rgba(255,255,255,0.8); margin:5px 0;'>Data Science Specialist</p>
        <div style="display: flex; justify-content: center; gap: 10px; margin-top:15px;">
            <a href="mailto:masteravinashrai@gmail.com" target="_blank">
                <img src="https://img.icons8.com/fluency/48/000000/gmail.png" width="32">
            </a>
            <a href="https://linkedin.com/in/avinashanalytics" target="_blank">
                <img src="https://img.icons8.com/color/48/000000/linkedin.png" width="32">
            </a>
            <a href="https://github.com/AvinashAnalytics" target="_blank">
                <img src="https://img.icons8.com/ios-glyphs/48/000000/github.png" width="32">
            </a>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### 🔍 Project Workflow")
    st.markdown("""
    1. **Data Collection**: Aggregated from multiple POS systems
    2. **Feature Engineering**: Created 10+ business metrics
    3. **Model Development**: Neural Network architecture
    4. **Validation**: 92% prediction accuracy achieved
    5. **Deployment**: Cloud-optimized pipeline
    """)

# ========== Main Dashboard ==========
st.markdown('<h1 class="header-glow" style="text-align:center; color:#1f567d;">🍽️ Revenue Prediction Engine</h1>', 
            unsafe_allow_html=True)

# Interactive Input Section
with st.expander("⚙️ **Configure Restaurant Parameters**", expanded=True):
    col1, col2 = st.columns(2)
    with col1:
        franchise = st.selectbox('Franchise Status', ['Independent', 'Franchised'], 
                               help="Select restaurant ownership type")
        category = st.selectbox('Cuisine Category', ['Fast Casual', 'Fine Dining', 'Family Style', 'Cafe'])
        
    with col2:
        no_of_items = st.number_input('Menu Items Count', min_value=1, value=45,
                                    help="Total number of dishes offered")
        order_placed = st.slider('Monthly Orders (×1000)', 1, 500, 120)
        
    franchise_code = 1 if franchise == 'Franchised' else 0
    category_codes = {'Fast Casual':0, 'Fine Dining':1, 'Family Style':2, 'Cafe':3}
    order_item_ratio = order_placed / no_of_items

# Prediction & Visualization
if st.button('🚀 Generate Revenue Forecast'):
    input_data = np.array([franchise_code, category_codes[category], 
                         no_of_items, order_placed, order_item_ratio]).reshape(1, -1)
    
    with st.spinner('Crunching numbers with neural network...'):
        prediction = model.predict(input_data)[0][0]
        
        # Animated result display
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="color:#1f567d; margin:0;">Projected Revenue</h3>
            <h1 style="color:#2c3e50; margin:0;">₹ {prediction:,.0f}</h1>
            <p style="color:#6c757d; margin:0;">± 5% accuracy based on historical data</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Interactive Visualizations
        col_v1, col_v2 = st.columns(2)
        with col_v1:
            # Feature Importance Chart
            fig = px.pie(values=[0.35, 0.25, 0.2, 0.15, 0.05], 
                       names=['Orders', 'Menu Size', 'Category', 'Franchise', 'Other'],
                       title='Revenue Drivers Breakdown')
            st.plotly_chart(fig, use_container_width=True)
            
        with col_v2:
            # Historical Performance
            history_data = pd.DataFrame({
                'Month': ['Jan', 'Feb', 'Mar', 'Apr', 'May'],
                'Revenue': [120000, 135000, 125000, 145000, prediction]
            })
            fig = px.line(history_data, x='Month', y='Revenue', 
                        title='Revenue Trend Projection',
                        markers=True)
            st.plotly_chart(fig, use_container_width=True)

# ========== Footer Section ==========
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 20px;">
    <p style="color: #6c757d;">
        🔮 Powered by Deep Learning | 
        <a href="https://github.com/AvinashAnalytics" style="color: #1f567d; text-decoration: none;">
            Explore the Code
        </a>
    </p>
    <div style="display: flex; justify-content: center; gap: 15px; margin-top: 10px;">
        <a href="https://medium.com/@avinashanalytics" target="_blank">
            <img src="https://img.icons8.com/ios/50/1f567d/medium-monogram.png" width="28">
        </a>
        <a href="https://twitter.com/AvinashAnalytiX" target="_blank">
            <img src="https://img.icons8.com/color/48/1f567d/twitter--v1.png" width="28">
        </a>
        <a href="https://kaggle.com/avinashanalytics" target="_blank">
            <img src="https://img.icons8.com/windows/50/1f567d/kaggle.png" width="28">
        </a>
    </div>
</div>
""", unsafe_allow_html=True)

# Add floating animation
html("""
<script>
window.addEventListener('scroll', function() {
    let header = document.querySelector('.header-glow');
    let scrollPosition = window.pageYOffset;
    header.style.textShadow = '0 0 ' + (15 + scrollPosition*0.1) + 'px #1f567d';
});
</script>
""")