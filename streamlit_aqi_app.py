# streamlit_aqi_app.py
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import random

# Page configuration
st.set_page_config(
    page_title="Air Quality Predictor",
    page_icon="🌍",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .main-header {
        color: white;
        text-align: center;
        padding: 2rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .result-card {
        background: white;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    .aqi-good { color: #28a745; font-size: 3rem; font-weight: bold; }
    .aqi-moderate { color: #ffc107; font-size: 3rem; font-weight: bold; }
    .aqi-poor { color: #dc3545; font-size: 3rem; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>🌬️ Air Quality Prediction System</h1>
    <p>Enter pollutant values to predict AQI and get health recommendations</p>
</div>
""", unsafe_allow_html=True)

# Initialize session state
if 'predicted' not in st.session_state:
    st.session_state.predicted = False

# Create two columns
col1, col2 = st.columns([1, 1.2])

with col1:
    st.markdown("### 📊 Enter Pollutant Values")
    
    with st.container():
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        
        # Input fields with icons
        pm25 = st.number_input("🌫️ PM2.5 (μg/m³)", min_value=0.0, value=35.0, step=0.1)
        pm10 = st.number_input("🌪️ PM10 (μg/m³)", min_value=0.0, value=70.0, step=0.1)
        no2 = st.number_input("🏭 NO2 (ppb)", min_value=0.0, value=40.0, step=0.1)
        so2 = st.number_input("🔥 SO2 (ppb)", min_value=0.0, value=20.0, step=0.1)
        co = st.number_input("🚗 CO (ppm)", min_value=0.0, value=2.0, step=0.1)
        o3 = st.number_input("☀️ O3 (ppb)", min_value=0.0, value=60.0, step=0.1)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Predict button
    if st.button("🔮 Predict AQI", type="primary", use_container_width=True):
        if pm25 == 0 and pm10 == 0 and no2 == 0 and so2 == 0 and co == 0 and o3 == 0:
            st.warning("⚠️ Please enter at least one pollutant value")
        else:
            # Calculate AQI using your Flask logic
            aqi = calculate_aqi(pm25, pm10, no2, so2, co, o3)
            category_info = get_aqi_category(aqi)
            health_tips = get_health_tips(aqi)
            nearest_city = get_nearest_city()
            
            # Store in session state
            st.session_state.predicted = True
            st.session_state.aqi = aqi
            st.session_state.category_info = category_info
            st.session_state.health_tips = health_tips
            st.session_state.nearest_city = nearest_city
            st.session_state.values = {
                'PM2.5': pm25, 'PM10': pm10, 'NO2': no2,
                'SO2': so2, 'CO': co, 'O3': o3
            }

with col2:
    st.markdown("### 📈 Prediction Results")
    
    if st.session_state.predicted:
        aqi = st.session_state.aqi
        category_info = st.session_state.category_info
        health_tips = st.session_state.health_tips
        nearest_city = st.session_state.nearest_city
        
        # AQI Display
        st.markdown(f"""
        <div class="result-card" style="text-align: center;">
            <h2>Air Quality Index</h2>
            <div class="aqi-{category_info['class']}">{aqi}</div>
            <h3 style="color: {category_info['color']};">{category_info['category']} {category_info['emoji']}</h3>
            <p>📍 Nearest City: {nearest_city}</p>
            <p>🕐 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Metrics in columns
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("PM2.5", f"{st.session_state.values['PM2.5']} μg/m³")
        with col_b:
            st.metric("PM10", f"{st.session_state.values['PM10']} μg/m³")
        with col_c:
            st.metric("NO2", f"{st.session_state.values['NO2']} ppb")
        
        # Pollutant contribution chart
        st.markdown("### 📊 Pollutant Contribution")
        plot_pollutant_chart(st.session_state.values)
        
        # Health recommendations
        st.markdown("### 🏥 Health Recommendations")
        
        if health_tips['level'] == 'good':
            st.success("\n".join([f"✅ {tip}" for tip in health_tips['tips']]))
        elif health_tips['level'] == 'moderate':
            st.warning("\n".join([f"⚠️ {tip}" for tip in health_tips['tips']]))
        else:
            st.error("\n".join([f"🚨 {tip}" for tip in health_tips['tips']]))
        
        # Alert if AQI is high
        if aqi > 150:
            st.markdown("""
            <div style="background: #f8d7da; color: #721c24; padding: 1rem; border-radius: 5px; border-left: 5px solid #dc3545;">
                <strong>🚨 HIGH AQI ALERT!</strong> Take necessary precautions.
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("👈 Enter values and click 'Predict AQI' to see results")

# Footer
st.markdown("---")
st.markdown("© 2024 Air Quality Prediction System | Data from environmental monitoring stations")

# Function definitions (your Flask logic converted)
def calculate_aqi(pm25, pm10, no2, so2, co, o3):
    """Calculate AQI based on pollutant values"""
    aqi = pm25 * 2.5 + pm10 * 1.2 + no2 * 0.5 + so2 * 0.3 + co * 10 + o3 * 0.4
    return round(aqi / 10, 1)

def get_aqi_category(aqi):
    """Determine AQI category"""
    if aqi <= 50:
        return {'category': 'Good', 'class': 'good', 'emoji': '😊', 'color': '#28a745'}
    elif aqi <= 100:
        return {'category': 'Moderate', 'class': 'moderate', 'emoji': '😐', 'color': '#ffc107'}
    elif aqi <= 150:
        return {'category': 'Unhealthy for Sensitive Groups', 'class': 'moderate', 'emoji': '😷', 'color': '#ffc107'}
    elif aqi <= 200:
        return {'category': 'Unhealthy', 'class': 'poor', 'emoji': '😷', 'color': '#dc3545'}
    elif aqi <= 300:
        return {'category': 'Very Unhealthy', 'class': 'poor', 'emoji': '😨', 'color': '#dc3545'}
    else:
        return {'category': 'Hazardous', 'class': 'poor', 'emoji': '☠️', 'color': '#dc3545'}

def get_nearest_city():
    """Simulate nearest city detection"""
    cities = ["Chennai", "Mumbai", "Delhi", "Bangalore", "Hyderabad", 
              "Kolkata", "Pune", "Ahmedabad", "Coimbatore", "Madurai"]
    return random.choice(cities)

def get_health_tips(aqi):
    """Get health recommendations based on AQI"""
    if aqi <= 50:
        return {
            'tips': ['Air quality is satisfactory', 'No health risks', 'Outdoor activities are safe'],
            'level': 'good'
        }
    elif aqi <= 100:
        return {
            'tips': ['Acceptable air quality', 'Sensitive individuals may experience minor effects', 
                    'Consider limiting prolonged outdoor exertion'],
            'level': 'moderate'
        }
    elif aqi <= 150:
        return {
            'tips': ['Sensitive groups should reduce outdoor activities', 
                    'People with lung/heart disease, children & elderly should be cautious',
                    'Consider wearing masks outdoors'],
            'level': 'moderate'
        }
    elif aqi <= 200:
        return {
            'tips': ['Everyone may begin to experience health effects', 
                    'Sensitive groups should avoid outdoor activities',
                    'Wear N95 masks when going outside', 'Use air purifiers indoors'],
            'level': 'poor'
        }
    elif aqi <= 300:
        return {
            'tips': ['HEALTH WARNING: Very unhealthy conditions', 'Avoid outdoor activities',
                    'Keep windows and doors closed', 'Run air purifiers continuously',
                    'Sensitive groups should stay indoors'],
            'level': 'poor'
        }
    else:
        return {
            'tips': ['EMERGENCY CONDITIONS: HAZARDOUS', 'Avoid all outdoor activities',
                    'Remain indoors with windows closed', 'Use air purifiers with HEPA filters',
                    'Consider relocating if possible', 'Seek medical attention if experiencing breathing difficulties'],
            'level': 'poor'
        }

def plot_pollutant_chart(values):
    """Create pollutant contribution chart"""
    pollutants = list(values.keys())
    values_list = list(values.values())
    
    # Filter out zero values
    data = [(p, v) for p, v in zip(pollutants, values_list) if v > 0]
    
    if data:
        df = pd.DataFrame(data, columns=['Pollutant', 'Value'])
        
        fig = go.Figure(data=[
            go.Bar(
                x=df['Pollutant'],
                y=df['Value'],
                marker_color=['#667eea', '#764ba2', '#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4']
            )
        ])
        
        fig.update_layout(
            title="Pollutant Levels",
            xaxis_title="Pollutant",
            yaxis_title="Concentration",
            height=300,
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        st.plotly_chart(fig, use_container_width=True)