import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the trained Keras model
model = load_model('neural_network_model.h5')

# Configure visual settings
sns.set_theme(style="whitegrid", palette="pastel")
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelcolor'] = '#2c3e50'
plt.rcParams['figure.titlesize'] = 16

# ========== Custom CSS ==========

st.markdown("""
<style>
    .header-style { 
        font-size: 36px !important; 
        color: #1f567d !important;
        border-bottom: 3px solid #1f567d;
        padding-bottom: 12px;
        margin-bottom: 1.5rem;
        font-family: 'Helvetica Neue', sans-serif;
    }
    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        transition: transform 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-3px);
    }
</style>
""", unsafe_allow_html=True)

# ========== Sidebar ==========

with st.sidebar:
    st.image("https://avatars.githubusercontent.com/u/86707796?v=4", width=100)
    st.markdown("<h2 style='color: #1f567d;'>Your Name</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color: #6c757d;'>Data Analyst</p>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("### 🛠️ Technical Showcase")
    st.markdown("""- **Data Processing**: Pandas, NumPy\n- **Machine Learning**: Keras (Neural Network)\n- **Visualization**: Matplotlib, Seaborn\n- **Web App**: Streamlit""")
    st.markdown("---")
    st.markdown("**Let's Connect:**  \n 📧 [Email](mailto:your.email@example.com)  \n 💼 [LinkedIn](https://linkedin.com/in/yourprofile)  \n 👨💻 [GitHub](https://github.com/yourgithub)")

# ========== Main Dashboard ==========

st.markdown('<h1 class="header-style">🍽️ Restaurant Revenue Prediction</h1>', unsafe_allow_html=True)

# Project Overview Section
with st.expander("📌 **Project Overview**", expanded=True):
    st.markdown("""
    **Objective:** Predict restaurant revenue using machine learning models based on various input features.
    
    **Features Considered:**
    - **Franchise**: Whether the restaurant is part of a franchise or not.
    - **Category**: Type of restaurant (Fast Food, Casual Dining, Fine Dining).
    - **Menu Size**: Number of items on the menu.
    - **Orders Placed**: The total number of orders placed (in lacs).
    """)

# ========== Input Features ==========

st.sidebar.header('Input Features')

# Sidebar Inputs
franchise = st.sidebar.selectbox('Franchise (0 = No, 1 = Yes)', [0, 1])
category = st.sidebar.selectbox('Category', ['Fast Food', 'Casual Dining', 'Fine Dining'])
no_of_items = st.sidebar.number_input('Number of Items Offered', min_value=1, value=10)
order_placed = st.sidebar.number_input('Orders Placed (in lacs)', min_value=1, value=100)
order_item_ratio = order_placed / no_of_items

# Category encoding
category_mapping = {'Fast Food': 0, 'Casual Dining': 1, 'Fine Dining': 2}
category_encoded = category_mapping[category]

# ========== Prediction Button and Result ==========

if st.sidebar.button('Predict Revenue'):
    input_data = np.array([franchise, category_encoded, no_of_items, order_placed, order_item_ratio]).reshape(1, -1)
    prediction = model.predict(input_data)
    
    st.subheader('📊 Predicted Revenue:')
    st.markdown(f"### ₹ {prediction[0][0]:,.2f}")  # Format the prediction in INR

    st.markdown("""**Note**: The model was trained on restaurant features like franchise type, category, menu size, and orders placed.""")

# ========== Visualization Section ==========

st.markdown("### 📈 Visual Insights")

# Example Data for Visualization
data = {
    "Category": ['Fast Food', 'Casual Dining', 'Fine Dining'],
    "Avg Revenue": [150000, 250000, 350000]  # Example revenues
}

df_vis = pd.DataFrame(data)

# Plot bar chart for visual insights
fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(x='Category', y='Avg Revenue', data=df_vis, ax=ax, palette='Blues')
ax.set_title('Average Revenue per Category')
ax.set_ylabel('Average Revenue (₹)')
ax.set_xlabel('Restaurant Category')
sns.despine()
st.pyplot(fig)

# ========== Footer ==========

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #6c757d; padding: 20px">
    <p style="font-size: 0.9rem;">
        Portfolio Project • Built with Python & Streamlit • 
        <a href="https://github.com/yourgithub" style="color: #1f567d; text-decoration: none;">View Source Code</a>
    </p>
</div>
""", unsafe_allow_html=True)
