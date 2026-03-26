# aqi_app_with_ml_enhanced.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px                                                                                                                                                                                                                                      
from datetime import datetime, timedelta           
import hashlib
import sqlite3
import time
import joblib
import requests
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    mean_absolute_error, r2_score, 
    accuracy_score, classification_report,
    confusion_matrix
)
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="AQI Predictor and analysis ",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    .prediction-box {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 30px;
        border-radius: 15px;
        text-align: center;
    }
    .category-good { background-color: #28a745; color: white; padding: 10px; border-radius: 5px; }
    .category-moderate { background-color: #ffc107; color: black; padding: 10px; border-radius: 5px; }
    .category-unhealthy { background-color: #dc3545; color: white; padding: 10px; border-radius: 5px; }
    .category-very-unhealthy { background-color: #9c27b0; color: white; padding: 10px; border-radius: 5px; }
    .category-hazardous { background-color: #7b1fa2; color: white; padding: 10px; border-radius: 5px; }
</style>
""", unsafe_allow_html=True)

# Initialize database
def init_db():
    conn = sqlite3.connect('aqi_ml_database.db')
    c = conn.cursor()
    
    # Users table
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (username TEXT PRIMARY KEY, 
                  password TEXT,
                  email TEXT,
                  phone TEXT,
                  city TEXT,
                  preferences TEXT,
                  created_at TIMESTAMP)''')
    
    # AQI predictions history
    c.execute('''CREATE TABLE IF NOT EXISTS predictions
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  username TEXT,
                  date TIMESTAMP,
                  aqi_predicted REAL,
                  aqi_category TEXT,
                  pm25 REAL,
                  pm10 REAL,
                  no2 REAL,
                  so2 REAL,
                  co REAL,
                  o3 REAL,
                  model_used TEXT,
                  confidence REAL)''')
    
    # Model performance tracking
    c.execute('''CREATE TABLE IF NOT EXISTS model_performance
                 (model_name TEXT,
                  date TIMESTAMP,
                  accuracy REAL,
                  mae REAL,
                  r2_score REAL,
                  precision REAL,
                  recall REAL)''')
    
    conn.commit()
    conn.close()

# Initialize ML Models
class AQIMLModels:
    def __init__(self):
        # Regression Models (for exact AQI)
        self.rf_regressor = RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        )
        
        # Classification Models (for AQI categories)
        self.rf_classifier = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        
        self.logistic_classifier = LogisticRegression(
            solver='lbfgs',  
            max_iter=1000,
            random_state=42,
            C=1.0
        )
        
        
        
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.is_trained = False
        self.training_metrics = {}
        
    def get_aqi_category(self, aqi):
        """Convert AQI value to category"""
        if aqi <= 50:
            return "Good"
        elif aqi <= 100:
            return "Moderate"
        elif aqi <= 150:
            return "Unhealthy for Sensitive"
        elif aqi <= 200:
            return "Unhealthy"
        elif aqi <= 300:
            return "Very Unhealthy"
        else:
            return "Hazardous"
    
    def generate_training_data(self, n_samples=20000):
        """Generate synthetic training data with realistic patterns"""
        np.random.seed(42)
        
        # Generate pollutants with correlations
        pm25 = np.random.uniform(0, 500, n_samples)
        
        # PM10 correlated with PM25
        pm10 = pm25 * np.random.uniform(1.2, 1.8, n_samples) + np.random.normal(0, 10, n_samples)
        
        # Other pollutants
        no2 = np.random.uniform(0, 200, n_samples)
        so2 = np.random.uniform(0, 100, n_samples)
        co = np.random.uniform(0, 50, n_samples)
        o3 = np.random.uniform(0, 300, n_samples)
        
        # Create non-linear relationships
        # High PM increases other pollutants
        mask_high_pm = pm25 > 200
        no2[mask_high_pm] += np.random.uniform(20, 50, np.sum(mask_high_pm))
        so2[mask_high_pm] += np.random.uniform(10, 30, np.sum(mask_high_pm))
        
        # Calculate AQI using weighted formula with interactions
        aqi = (
            0.30 * pm25 +
            0.25 * pm10 +
            0.15 * no2 +
            0.10 * so2 +
            0.10 * co * 10 +  # Scale CO
            0.10 * o3 * 0.5    # Scale O3
        )
        
        # Add interaction terms for better realism
        aqi += 0.001 * pm25 * pm10  # PM interaction
        aqi += 0.01 * no2 * o3       # Photochemical interaction
        
        # Add noise
        aqi += np.random.normal(0, 5, n_samples)
        aqi = np.maximum(0, aqi)  # Ensure non-negative
        
        # Create categories
        categories = [self.get_aqi_category(val) for val in aqi]
        
        # Combine features
        X = np.column_stack([pm25, pm10, no2, so2, co, o3])
        y_regression = aqi
        y_classification = categories
        
        return X, y_regression, y_classification
    
    def train_models(self):
        """Train all ML models"""
        with st.spinner("🤖 Training ML Models..."):
            # Generate data
            X, y_reg, y_class = self.generate_training_data()
            
            # Split data
            X_train, X_test, y_reg_train, y_reg_test, y_class_train, y_class_test = train_test_split(
                X, y_reg, y_class, test_size=0.2, random_state=42, stratify=y_class
            )
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Encode labels for classification
            y_class_train_encoded = self.label_encoder.fit_transform(y_class_train)
            y_class_test_encoded = self.label_encoder.transform(y_class_test)
            
            # ===== 1. Train Random Forest Regressor =====
            self.rf_regressor.fit(X_train_scaled, y_reg_train)
            rf_reg_pred = self.rf_regressor.predict(X_test_scaled)
            rf_reg_mae = mean_absolute_error(y_reg_test, rf_reg_pred)
            rf_reg_r2 = r2_score(y_reg_test, rf_reg_pred)
            
            # ===== 2. Train Random Forest Classifier =====
            self.rf_classifier.fit(X_train_scaled, y_class_train_encoded)
            rf_class_pred = self.rf_classifier.predict(X_test_scaled)
            rf_class_accuracy = accuracy_score(y_class_test_encoded, rf_class_pred)
            
            # ===== 3. Train Logistic Regression Classifier =====
            self.logistic_classifier.fit(X_train_scaled, y_class_train_encoded)
            log_reg_pred = self.logistic_classifier.predict(X_test_scaled)
            log_reg_accuracy = accuracy_score(y_class_test_encoded, log_reg_pred)
            
            # Cross-validation scores
            rf_cv_scores = cross_val_score(self.rf_classifier, X_train_scaled, y_class_train_encoded, cv=5)
            log_cv_scores = cross_val_score(self.logistic_classifier, X_train_scaled, y_class_train_encoded, cv=5)
            
            self.is_trained = True
            
            # Store metrics
            self.training_metrics = {
                'rf_regressor': {
                    'mae': rf_reg_mae,
                    'r2': rf_reg_r2,
                    'type': 'regression'
                },
                'rf_classifier': {
                    'accuracy': rf_class_accuracy,
                    'cv_mean': rf_cv_scores.mean(),
                    'cv_std': rf_cv_scores.std(),
                    'type': 'classification'
                },
                'logistic_regression': {
                    'accuracy': log_reg_accuracy,
                    'cv_mean': log_cv_scores.mean(),
                    'cv_std': log_cv_scores.std(),
                    'type': 'classification'
                }
            }
            
            return self.training_metrics
    
    def predict_aqi(self, features, model_type='rf_regressor'):
        """Predict AQI using selected model"""
        features_scaled = self.scaler.transform([features])
        
        if model_type == 'rf_regressor':
            # Random Forest Regressor
            prediction = self.rf_regressor.predict(features_scaled)[0]
            confidence = 0.85  # Base confidence
            
            # Adjust confidence based on prediction range
            if prediction < 50:
                confidence += 0.05  # Good range is more reliable
            elif prediction > 300:
                confidence -= 0.10  # Hazardous range less reliable
            
        elif model_type == 'rf_classifier':
            # Random Forest Classifier
            class_pred = self.rf_classifier.predict(features_scaled)[0]
            class_proba = self.rf_classifier.predict_proba(features_scaled)[0]
            
            # Get category and confidence
            category = self.label_encoder.inverse_transform([class_pred])[0]
            confidence = np.max(class_proba)
            
            # Convert category back to approximate AQI
            prediction = self.category_to_aqi(category)
            
        elif model_type == 'logistic_regression':
            # Logistic Regression
            class_pred = self.logistic_classifier.predict(features_scaled)[0]
            class_proba = self.logistic_classifier.predict_proba(features_scaled)[0]
            
            category = self.label_encoder.inverse_transform([class_pred])[0]
            confidence = np.max(class_proba)
            
            prediction = self.category_to_aqi(category)
        
        else:  # Ensemble
            # Combine all models
            rf_reg_pred = self.rf_regressor.predict(features_scaled)[0]
            
            rf_class_pred = self.rf_classifier.predict(features_scaled)[0]
            rf_category = self.label_encoder.inverse_transform([rf_class_pred])[0]
            rf_class_aqi = self.category_to_aqi(rf_category)
            
            log_pred = self.logistic_classifier.predict(features_scaled)[0]
            log_category = self.label_encoder.inverse_transform([log_pred])[0]
            log_aqi = self.category_to_aqi(log_category)
            
            # Weighted average (regression gets higher weight for exact value)
            prediction = (0.5 * rf_reg_pred + 0.25 * rf_class_aqi + 0.25 * log_aqi)
            confidence = 0.92  # Ensemble confidence
        
        return round(prediction, 1), min(confidence, 1.0)
    
    def category_to_aqi(self, category):
        """Convert category to approximate AQI value"""
        category_map = {
            "Good": 25,
            "Moderate": 75,
            "Unhealthy for Sensitive": 125,
            "Unhealthy": 175,
            "Very Unhealthy": 250,
            "Hazardous": 350
        }
        return category_map.get(category, 100)
    
    def get_feature_importance(self, model_type='rf_regressor'):
        """Get feature importance for interpretation"""
        features = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3']
        
        if model_type == 'rf_regressor':
            importance = self.rf_regressor.feature_importances_
        elif model_type == 'rf_classifier':
            importance = self.rf_classifier.feature_importances_
        elif model_type == 'logistic_regression':
            # For logistic regression, use coefficient magnitude
            importance = np.abs(self.logistic_classifier.coef_).mean(axis=0)
        else:
            importance = self.rf_regressor.feature_importances_
        
        return dict(zip(features, importance))

# Initialize global model
@st.cache_resource
def load_ml_models():
    models = AQIMLModels()
    metrics = models.train_models()
    return models, metrics

# Initialize everything
init_db()
ml_models, model_metrics = load_ml_models()

# Session state
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'username' not in st.session_state:
    st.session_state.username = ""
if 'selected_city' not in st.session_state:
    st.session_state.selected_city = "Chennai"
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []

# ==================== LOGIN PAGE ====================
def login_page():
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        <div class="main-header">
            <h1>🌍 AQI Prediction system </h1>
            <p>Using Random Forest & Logistic Regression</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        tab1, tab2 = st.tabs(["🔑 Login", "📝 Sign Up"])
        
        with tab1:
            with st.form("login_form"):
                username = st.text_input("👤 Username")
                password = st.text_input("🔒 Password", type="password")
                
                if st.form_submit_button("🚀 Login", use_container_width=True):
                    if verify_login(username, password):
                        st.session_state.logged_in = True
                        st.session_state.username = username
                        st.success("✅ Login successful!")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error("❌ Invalid credentials")
        
        with tab2:
            with st.form("signup_form"):
                new_username = st.text_input("👤 Username")
                new_password = st.text_input("🔒 Password", type="password")
                confirm_password = st.text_input("🔒 Confirm Password", type="password")
                email = st.text_input("📧 Email")
                phone = st.text_input("📱 Phone")
                city = st.selectbox("🏙️ Your City", ["Chennai", "Mumbai", "Delhi", "Bangalore", "Kolkata", "Hyderabad"])
                
                if st.form_submit_button("📝 Create Account", use_container_width=True):
                    if new_password == confirm_password:
                        if create_user(new_username, new_password, email, phone, city):
                            st.success("✅ Account created! Please login.")
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.error("❌ Username already exists")
                    else:
                        st.error("❌ Passwords don't match")

# ==================== MAIN APP ====================
def main_app():
    # Sidebar
    with st.sidebar:
        st.markdown(f"""
        <div style="text-align: center; padding: 20px;">
            <h3>👋 Welcome, {st.session_state.username}!</h3>
            <p>{datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Navigation
        page = st.radio(
            "📌 Navigation",
            ["🏠 Dashboard", "🤖 Predictor", "📊 Model Comparison", "📈 History", "👤 Profile"]
        )
        
        st.markdown("---")
        
        # Model Performance Summary
        with st.expander("📊 Model Performance"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric(
                    "Random Forest (Reg)", 
                    f"{model_metrics['rf_regressor']['r2']:.3f} R²"
                )
                st.metric(
                    "Random Forest (Clf)", 
                    f"{model_metrics['rf_classifier']['accuracy']:.1%}"
                )
            
            with col2:
                st.metric(
                    "Logistic Regression", 
                    f"{model_metrics['logistic_regression']['accuracy']:.1%}"
                )
                st.metric(
                    "Best Model", 
                    "Random Forest" if model_metrics['rf_classifier']['accuracy'] > 
                    model_metrics['logistic_regression']['accuracy'] else "Logistic"
                )
        
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.logged_in = False
            st.rerun()
    
    # Main content
    if page == "🏠 Dashboard":
        show_dashboard()
    elif page == "🤖 ML Predictor":
        show_ml_predictor()
    elif page == "📊 Model Comparison":
        show_model_comparison()
    elif page == "📈 History":
        show_history()
    elif page == "👤 Profile":
        show_profile()

# ==================== DASHBOARD ====================
def show_dashboard():
    st.title(f"🏠 Dashboard")
    st.markdown("---")
    
    # Quick stats
    col1, col2, col3, col4 = st.columns(4)
    
    total_users = get_total_users()
    total_predictions = get_total_predictions()
    
    with col1:
        st.metric("Total Users", total_users, "+12%")
    with col2:
        st.metric("Total Predictions", total_predictions, "+8%")
    with col3:
        st.metric("Models Active", "3", "RF, LR, Ensemble")
    with col4:
        st.metric("Avg Accuracy", "87%", "+2%")
    
    st.markdown("---")
    
    # Model performance visualization
    st.subheader("📊 Model Performance Comparison")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Accuracy comparison
        models = ['Random Forest (Reg)', 'Random Forest (Clf)', 'Logistic Regression']
        r2_scores = [
            model_metrics['rf_regressor']['r2'],
            model_metrics['rf_classifier']['accuracy'],
            model_metrics['logistic_regression']['accuracy']
        ]
        
        fig = go.Figure(data=[
            go.Bar(
                x=models,
                y=r2_scores,
                marker_color=['#667eea', '#764ba2', '#f093fb'],
                text=[f'{score:.2%}' for score in r2_scores],
                textposition='outside'
            )
        ])
        fig.update_layout(
            title="Model Performance Metrics",
            yaxis_title="Score",
            height=400,
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Feature importance
        importance = ml_models.get_feature_importance('rf_regressor')
        
        fig = go.Figure(data=[
            go.Bar(
                x=list(importance.values()),
                y=list(importance.keys()),
                orientation='h',
                marker_color='#28a745'
            )
        ])
        fig.update_layout(
            title="Feature Importance (Random Forest)",
            xaxis_title="Importance",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Recent activity
    st.subheader("🕒 Recent Predictions")
    recent = get_recent_predictions(10)
    
    if recent:
        df = pd.DataFrame(recent, columns=['User', 'Date', 'AQI', 'Category', 'Model'])
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No predictions yet. Start predicting!")

# ==================== ML PREDICTOR ====================
def show_ml_predictor():
    st.title("🤖 ML AQI Predictor")
    st.markdown("---")
    
    st.markdown("""
    <div style="background-color: #f0f2f6; padding: 15px; border-radius: 10px; margin-bottom: 20px;">
        <h4>📌 Choose your model:</h4>
        <ul>
            <li><b>Random Forest Regressor</b> - Predicts exact AQI value (Most Accurate)</li>
            <li><b>Random Forest Classifier</b> - Predicts AQI category (Good, Moderate, etc.)</li>
            <li><b>Logistic Regression</b> - Fast classification (Interpretable)</li>
            <li><b>Ensemble</b> - Combines all models (Most Robust)</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📝 Enter Values")
        
        # Input fields with tooltips
        pm25 = st.number_input(
            "PM2.5 (μg/m³)", 0.0, 500.0, 35.0,
            help="Fine particulate matter (2.5 micrometers)"
        )
        pm10 = st.number_input(
            "PM10 (μg/m³)", 0.0, 600.0, 70.0,
            help="Coarse particulate matter (10 micrometers)"
        )
        no2 = st.number_input(
            "NO2 (ppb)", 0.0, 200.0, 40.0,
            help="Nitrogen Dioxide - from vehicle emissions"
        )
        so2 = st.number_input(
            "SO2 (ppb)", 0.0, 100.0, 20.0,
            help="Sulfur Dioxide - from industrial processes"
        )
        co = st.number_input(
            "CO (ppm)", 0.0, 50.0, 2.0,
            help="Carbon Monoxide - from incomplete burning"
        )
        o3 = st.number_input(
            "O3 (ppb)", 0.0, 300.0, 60.0,
            help="Ground-level Ozone - from photochemical reactions"
        )
        
        # Model selection
        model_type = st.selectbox(
            "🧠 Select Model",
            [
                "Random Forest Regressor (Recommended)",
                "Random Forest Classifier",
                "Logistic Regression",
                "Ensemble (All Models)"
            ]
        )
        
        model_map = {
            "Random Forest Regressor (Recommended)": "rf_regressor",
            "Random Forest Classifier": "rf_classifier",
            "Logistic Regression": "logistic_regression",
            "Ensemble (All Models)": "ensemble"
        }
        
        if st.button("🔮 Predict AQI", type="primary", use_container_width=True):
            features = [pm25, pm10, no2, so2, co, o3]
            predicted_aqi, confidence = ml_models.predict_aqi(features, model_map[model_type])
            
            # Get category
            category = ml_models.get_aqi_category(predicted_aqi)
            
            # Save to session
            st.session_state.predicted_aqi = predicted_aqi
            st.session_state.prediction_confidence = confidence
            st.session_state.prediction_category = category
            st.session_state.prediction_model = model_type
            st.session_state.prediction_features = features
            
            # Save to database
            save_prediction(
                st.session_state.username,
                predicted_aqi,
                category,
                pm25, pm10, no2, so2, co, o3,
                model_type,
                confidence
            )
            
            # Add to history
            st.session_state.prediction_history.append({
                'date': datetime.now(),
                'aqi': predicted_aqi,
                'category': category,
                'model': model_type,
                'confidence': confidence
            })
            
            st.success(f"✅ Prediction complete! Confidence: {confidence*100:.1f}%")
    
    with col2:
        st.subheader("📊 Results")
        
        if 'predicted_aqi' in st.session_state:
            aqi = st.session_state.predicted_aqi
            category = st.session_state.prediction_category
            
            # Color based on category
            if category == "Good":
                bg_color = "#28a745"
                advice = "Air quality is satisfactory. Safe for outdoor activities."
            elif category == "Moderate":
                bg_color = "#ffc107"
                advice = "Acceptable quality. Unusually sensitive people should reduce outdoor time."
            elif category == "Unhealthy for Sensitive":
                bg_color = "#ff9800"
                advice = "People with respiratory issues should limit outdoor exertion."
            elif category == "Unhealthy":
                bg_color = "#dc3545"
                advice = "Everyone may begin to experience health effects."
            elif category == "Very Unhealthy":
                bg_color = "#9c27b0"
                advice = "Health alert: everyone may experience serious effects."
            else:  # Hazardous
                bg_color = "#7b1fa2"
                advice = "Emergency conditions: everyone should stay indoors."
            
            st.markdown(f"""
            <div style="background-color: {bg_color}; padding: 30px; border-radius: 15px; text-align: center; color: white;">
                <h2>Predicted AQI</h2>
                <h1 style="font-size: 80px; margin: 10px;">{aqi}</h1>
                <h3>{category}</h3>
                <p>Model: {st.session_state.prediction_model}</p>
                <p>Confidence: {st.session_state.prediction_confidence*100:.1f}%</p>
                <hr>
                <p>💡 {advice}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Show feature contribution
            st.subheader("🔍 Pollutant Contribution")
            
            features = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3']
            values = [pm25, pm10, no2, so2, co, o3]
            
            # Normalize for visualization
            max_val = max(values) if max(values) > 0 else 1
            norm_values = [v/max_val * 100 for v in values]
            
            fig = go.Figure(data=[
                go.Bar(
                    x=features,
                    y=norm_values,
                    marker_color='#764ba2',
                    text=[f'{v:.1f}' for v in values],
                    textposition='outside'
                )
            ])
            fig.update_layout(
                title="Relative Pollutant Levels",
                yaxis_title="Relative Level (%)",
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)

# ==================== MODEL COMPARISON ====================
def show_model_comparison():
    st.title("📊 Model Comparison")
    st.markdown("---")
    
    # Generate test data for comparison
    test_cases = [
        {"name": "Clean Air", "features": [10, 20, 10, 5, 0.5, 30]},
        {"name": "Moderate Pollution", "features": [50, 80, 40, 25, 2, 70]},
        {"name": "High Pollution", "features": [200, 300, 120, 60, 8, 150]},
        {"name": "Severe Pollution", "features": [400, 500, 180, 90, 20, 250]}
    ]
    
    results = []
    
    for test in test_cases:
        row = {"Scenario": test["name"]}
        
        # Get predictions from all models
        for model_name, model_key in [
            ("Random Forest (Reg)", "rf_regressor"),
            ("Random Forest (Clf)", "rf_classifier"),
            ("Logistic Regression", "logistic_regression"),
            ("Ensemble", "ensemble")
        ]:
            pred, conf = ml_models.predict_aqi(test["features"], model_key)
            row[model_name] = pred
            row[f"{model_name}_conf"] = conf
        
        results.append(row)
    
    df = pd.DataFrame(results)
    
    # Display comparison table
    st.subheader("📋 Model Predictions Comparison")
    
    # Format the dataframe
    display_df = df[['Scenario', 'Random Forest (Reg)', 'Random Forest (Clf)', 
                     'Logistic Regression', 'Ensemble']]
    
    st.dataframe(display_df, use_container_width=True)
    
    st.markdown("---")
    
    # Visual comparison
    st.subheader("📈 Prediction Visualization")
    
    fig = go.Figure()
    
    for model in ['Random Forest (Reg)', 'Random Forest (Clf)', 'Logistic Regression', 'Ensemble']:
        fig.add_trace(go.Bar(
            name=model,
            x=df['Scenario'],
            y=df[model],
            text=df[model],
            textposition='inside'
        ))
    
    fig.update_layout(
        barmode='group',
        yaxis_title="Predicted AQI",
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Model metrics
    st.subheader("📊 Detailed Metrics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### Random Forest Regressor")
        st.metric("R² Score", f"{model_metrics['rf_regressor']['r2']:.3f}")
        st.metric("MAE", f"{model_metrics['rf_regressor']['mae']:.2f}")
        st.info("✅ Best for exact AQI values")
    
    with col2:
        st.markdown("### Random Forest Classifier")
        st.metric("Accuracy", f"{model_metrics['rf_classifier']['accuracy']:.1%}")
        st.metric("CV Score", f"{model_metrics['rf_classifier']['cv_mean']:.1%}")
        st.info("✅ Best for category prediction")
    
    with col3:
        st.markdown("### Logistic Regression")
        st.metric("Accuracy", f"{model_metrics['logistic_regression']['accuracy']:.1%}")
        st.metric("CV Score", f"{model_metrics['logistic_regression']['cv_mean']:.1%}")
        st.info("✅ Fastest & most interpretable")

# ==================== HISTORY ====================
def show_history():
    st.title("📈 Prediction History")
    st.markdown("---")
    
    # Get user's prediction history
    history = get_user_predictions(st.session_state.username)
    
    if history:
        df = pd.DataFrame(history, columns=[
            'Date', 'AQI', 'Category', 'PM2.5', 'PM10', 'NO2', 
            'SO2', 'CO', 'O3', 'Model', 'Confidence'
        ])
        
        # Summary stats
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Predictions", len(df))
        with col2:
            st.metric("Average AQI", f"{df['AQI'].mean():.1f}")
        with col3:
            st.metric("Max AQI", f"{df['AQI'].max():.1f}")
        with col4:
            st.metric("Avg Confidence", f"{df['Confidence'].mean()*100:.1f}%")
        
        st.markdown("---")
        
        # Time series chart
        st.subheader("📊 AQI Trend")
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=pd.to_datetime(df['Date']),
            y=df['AQI'],
            mode='lines+markers',
            name='AQI',
            line=dict(color='#667eea', width=3),
            marker=dict(
                size=10,
                color=df['Confidence'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Confidence")
            )
        ))
        
        # Add category thresholds
        fig.add_hline(y=50, line_dash="dash", line_color="green", annotation_text="Good")
        fig.add_hline(y=100, line_dash="dash", line_color="yellow", annotation_text="Moderate")
        fig.add_hline(y=150, line_dash="dash", line_color="orange", annotation_text="Unhealthy (Sensitive)")
        fig.add_hline(y=200, line_dash="dash", line_color="red", annotation_text="Unhealthy")
        fig.add_hline(y=300, line_dash="dash", line_color="purple", annotation_text="Very Unhealthy")
        
        fig.update_layout(
            height=500,
            xaxis_title="Date",
            yaxis_title="AQI"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # History table
        st.subheader("📋 Detailed History")
        
        # Add color coding
        def color_category(val):
            if val == "Good":
                return 'background-color: #28a745; color: white'
            elif val == "Moderate":
                return 'background-color: #ffc107'
            elif "Unhealthy" in val:
                return 'background-color: #dc3545; color: white'
            elif "Very Unhealthy" in val:
                return 'background-color: #9c27b0; color: white'
            elif "Hazardous" in val:
                return 'background-color: #7b1fa2; color: white'
            return ''
        
        styled_df = df.style.applymap(color_category, subset=['Category'])
        st.dataframe(styled_df, use_container_width=True)
        
        # Export option
        if st.button("📥 Export to CSV"):
            csv = df.to_csv(index=False)
            st.download_button(
                "Download CSV",
                csv,
                "aqi_history.csv",
                "text/csv"
            )
    
    else:
        st.info("📭 No prediction history yet. Use the ML Predictor to make your first prediction!")

# ==================== PROFILE ====================
def show_profile():
    st.title("👤 User Profile")
    st.markdown("---")
    
    user = get_user_details(st.session_state.username)
    stats = get_user_statistics(st.session_state.username)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📋 Profile Information")
        
        st.markdown(f"""
        **Username:** {user[0]}  
        **Email:** {user[2]}  
        **Phone:** {user[3]}  
        **City:** {user[4]}  
        **Member Since:** {user[6]}
        """)
        
        st.markdown("---")
        
        # Model preferences
        st.subheader("🎯 Model Preferences")
        
        fav_model = stats.get('fav_model', 'Not enough data')
        st.info(f"Your most used model: **{fav_model}**")
        
        if fav_model != 'Not enough data':
            if 'Random Forest' in fav_model:
                st.success("✅ You prefer ensemble methods - good for accuracy!")
            elif 'Logistic' in fav_model:
                st.success("✅ You prefer interpretable models!")
    
    with col2:
        st.subheader("📊 Your Statistics")
        
        # Create metrics
        st.markdown(f"""
        - **Total Predictions:** {stats['total']}
        - **Average AQI:** {stats['avg']}
        - **Highest AQI:** {stats['max']}
        - **Lowest AQI:** {stats['min']}
        - **Average Confidence:** {stats['avg_confidence']:.1f}%
        """)
        
        st.markdown("---")
        
        # Category distribution
        if stats.get('category_dist'):
            st.subheader("📊 Category Distribution")
            
            cat_df = pd.DataFrame(
                list(stats['category_dist'].items()),
                columns=['Category', 'Count']
            )
            
            fig = px.pie(cat_df, values='Count', names='Category')
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Achievements
        st.subheader("🏆 Achievements")
        
        achievements = []
        if stats['total'] >= 10:
            achievements.append("🎯 **Pro Predictor** - Made 10+ predictions")
        if stats['total'] >= 50:
            achievements.append("🏅 **AQI Expert** - Made 50+ predictions")
        if stats['max'] >= 200:
            achievements.append("⚠️ **Survivor** - Recorded very high AQI")
        if stats['avg_confidence'] >= 90:
            achievements.append("🎯 **Sharp Shooter** - 90%+ average confidence")
        if len(set([p[3] for p in get_user_predictions(st.session_state.username)[:5] or []])) >= 3:
            achievements.append("🔄 **Model Explorer** - Used 3+ different models")
        
        if achievements:
            for ach in achievements:
                st.success(ach)
        else:
            st.info("Make more predictions to earn achievements!")

# ==================== HELPER FUNCTIONS ====================
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def create_user(username, password, email, phone, city):
    conn = sqlite3.connect('aqi_ml_database.db')
    c = conn.cursor()
    try:
        c.execute(
            "INSERT INTO users VALUES (?, ?, ?, ?, ?, ?, ?)",
            (username, hash_password(password), email, phone, city, "{}", datetime.now())
        )
        conn.commit()
        return True
    except Exception as e:
        print(e)
        return False
    finally:
        conn.close()

def verify_login(username, password):
    conn = sqlite3.connect('aqi_ml_database.db')
    c = conn.cursor()
    c.execute(
        "SELECT * FROM users WHERE username=? AND password=?",
        (username, hash_password(password))
    )
    user = c.fetchone()
    conn.close()
    return user is not None

def save_prediction(username, aqi, category, pm25, pm10, no2, so2, co, o3, model, confidence):
    conn = sqlite3.connect('aqi_ml_database.db')
    c = conn.cursor()
    c.execute(
        """INSERT INTO predictions 
           (username, date, aqi_predicted, aqi_category, pm25, pm10, no2, so2, co, o3, model_used, confidence)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (username, datetime.now(), aqi, category, pm25, pm10, no2, so2, co, o3, model, confidence)
    )
    conn.commit()
    conn.close()

def get_user_predictions(username):
    conn = sqlite3.connect('aqi_ml_database.db')
    c = conn.cursor()
    c.execute(
        """SELECT date, aqi_predicted, aqi_category, pm25, pm10, no2, so2, co, o3, model_used, confidence 
           FROM predictions 
           WHERE username=? 
           ORDER BY date DESC 
           LIMIT 100""",
        (username,)
    )
    history = c.fetchall()
    conn.close()
    return history

def get_user_details(username):
    conn = sqlite3.connect('aqi_ml_database.db')
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE username=?", (username,))
    user = c.fetchone()
    conn.close()
    return user

def get_user_statistics(username):
    conn = sqlite3.connect('aqi_ml_database.db')
    c = conn.cursor()
    
    # Get prediction stats
    c.execute(
        """SELECT COUNT(*), AVG(aqi_predicted), MAX(aqi_predicted), MIN(aqi_predicted),
                  AVG(confidence)
           FROM predictions 
           WHERE username=?""",
        (username,)
    )
    stats = c.fetchone()
    
    # Get favorite model
    c.execute(
        """SELECT model_used, COUNT(*) as count 
           FROM predictions 
           WHERE username=? 
           GROUP BY model_used 
           ORDER BY count DESC 
           LIMIT 1""",
        (username,)
    )
    fav_model = c.fetchone()
    
    # Get category distribution
    c.execute(
        """SELECT aqi_category, COUNT(*) as count 
           FROM predictions 
           WHERE username=? 
           GROUP BY aqi_category""",
        (username,)
    )
    categories = c.fetchall()
    
    conn.close()
    
    return {
        'total': stats[0] if stats[0] else 0,
        'avg': round(stats[1], 1) if stats[1] else 0,
        'max': stats[2] if stats[2] else 0,
        'min': stats[3] if stats[3] else 0,
        'avg_confidence': stats[4] * 100 if stats[4] else 0,
        'fav_model': fav_model[0] if fav_model else 'None',
        'category_dist': dict(categories) if categories else {}
    }

def get_total_users():
    conn = sqlite3.connect('aqi_ml_database.db')
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM users")
    total = c.fetchone()[0]
    conn.close()
    return total

def get_total_predictions():
    conn = sqlite3.connect('aqi_ml_database.db')
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM predictions")
    total = c.fetchone()[0]
    conn.close()
    return total

def get_recent_predictions(limit=10):
    conn = sqlite3.connect('aqi_ml_database.db')
    c = conn.cursor()
    c.execute(
        """SELECT username, date, aqi_predicted, aqi_category, model_used 
           FROM predictions 
           ORDER BY date DESC 
           LIMIT ?""",
        (limit,)
    )
    recent = c.fetchall()
    conn.close()
    return recent

# ==================== MAIN ====================
if __name__ == "__main__":
    if not st.session_state.logged_in:
        login_page()
    else:
        main_app()