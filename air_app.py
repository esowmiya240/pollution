import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import hashlib
import sqlite3
import time
import smtplib
from email.mime.text import MIMEText
import requests
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Page config
st.set_page_config(
    page_title="AQI Predictor - Smart Air Quality Monitor",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    /* Main container styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 20px;
    }
    
    /* Card styling */
    .custom-card {
        background: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    
    /* Metric styling */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
    }
    
    /* Alert styling */
    .alert-good {
        background: #28a745;
        color: white;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
    }
    
    .alert-moderate {
        background: #ffc107;
        color: black;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
    }
    
    .alert-unhealthy {
        background: #dc3545;
        color: white;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
    }
    
    /* Icon styling */
    .icon-large {
        font-size: 48px;
        margin-bottom: 10px;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 10px 24px;
        border-radius: 25px;
        font-weight: 600;
        transition: all 0.3s;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }
    
    /* Sidebar styling */
    .sidebar-user-info {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 20px;
    }
    
    /* Success message styling */
    .success-message {
        background: #d4edda;
        color: #155724;
        padding: 10px;
        border-radius: 5px;
        border-left: 5px solid #28a745;
        margin: 10px 0;
    }
    
    /* Error message styling */
    .error-message {
        background: #f8d7da;
        color: #721c24;
        padding: 10px;
        border-radius: 5px;
        border-left: 5px solid #dc3545;
        margin: 10px 0;
    }
    
    /* Info message styling */
    .info-message {
        background: #d1ecf1;
        color: #0c5460;
        padding: 10px;
        border-radius: 5px;
        border-left: 5px solid #17a2b8;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize database with correct schema
def init_db():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    
    # Users table with correct schema
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (username TEXT PRIMARY KEY, 
                  password TEXT,
                  email TEXT,
                  phone TEXT,
                  created_at TIMESTAMP)''')
    
    # History table
    c.execute('''CREATE TABLE IF NOT EXISTS history
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  username TEXT, 
                  date TIMESTAMP, 
                  aqi REAL,
                  pm25 REAL, 
                  pm10 REAL, 
                  no2 REAL, 
                  so2 REAL, 
                  co REAL, 
                  o3 REAL,
                  status TEXT)''')
    
    # Settings table
    c.execute('''CREATE TABLE IF NOT EXISTS settings
                 (username TEXT PRIMARY KEY,
                  email_notify INTEGER DEFAULT 1,
                  sms_notify INTEGER DEFAULT 0,
                  alert_threshold INTEGER DEFAULT 150,
                  theme TEXT DEFAULT 'Light',
                  language TEXT DEFAULT 'English',
                  chart_type TEXT DEFAULT 'Line',
                  show_grid INTEGER DEFAULT 1,
                  FOREIGN KEY (username) REFERENCES users(username))''')
    
    conn.commit()
    conn.close()

# Hash password
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# Create new user (fixed parameter order)
def create_user(username, password, email, phone):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    try:
        c.execute("INSERT INTO users (username, password, email, phone, created_at) VALUES (?, ?, ?, ?, ?)",
                 (username, hash_password(password), email, phone, datetime.now()))
        # Create default settings for user
        c.execute("INSERT INTO settings (username) VALUES (?)", (username,))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False  # Username already exists
    except Exception as e:
        print(f"Error creating user: {e}")
        return False
    finally:
        conn.close()

# Verify login
def verify_login(username, password):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE username=? AND password=?",
             (username, hash_password(password)))
    user = c.fetchone()
    conn.close()
    return user is not None

# Get user settings
def get_user_settings(username):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute("SELECT * FROM settings WHERE username=?", (username,))
    settings = c.fetchone()
    conn.close()
    
    if settings:
        return {
            'email_notify': bool(settings[1]),
            'sms_notify': bool(settings[2]),
            'alert_threshold': settings[3],
            'theme': settings[4],
            'language': settings[5],
            'chart_type': settings[6],
            'show_grid': bool(settings[7])
        }
    return {
        'email_notify': True,
        'sms_notify': False,
        'alert_threshold': 150,
        'theme': 'Light',
        'language': 'English',
        'chart_type': 'Line',
        'show_grid': True
    }

# Save user settings
def save_user_settings(username, settings):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    try:
        c.execute('''UPDATE settings SET 
                     email_notify=?, sms_notify=?, alert_threshold=?,
                     theme=?, language=?, chart_type=?, show_grid=?
                     WHERE username=?''',
                 (int(settings['email_notify']), int(settings['sms_notify']),
                  settings['alert_threshold'], settings['theme'],
                  settings['language'], settings['chart_type'],
                  int(settings['show_grid']), username))
        conn.commit()
        return True
    except Exception as e:
        print(f"Error saving settings: {e}")
        return False
    finally:
        conn.close()

# Send email alert
def send_email_alert(recipient, subject, message):
    try:
        sender = os.getenv('EMAIL_SENDER', 'yourmail@gmail.com')
        password = os.getenv('EMAIL_PASSWORD', 'your-app-password')
        
        # Skip if credentials are not set
        if sender == 'yourmail@gmail.com' or password == 'your-app-password':
            return False, "Email credentials not configured"
        
        msg = MIMEText(message)
        msg['Subject'] = subject
        msg['From'] = sender
        msg['To'] = recipient
        
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(sender, password)
        server.send_message(msg)
        server.quit()
        return True, "Email sent successfully"
    except Exception as e:
        return False, str(e)

# Send SMS alert (using Fast2SMS)
def send_sms_alert(phone, message):
    try:
        api_key = os.getenv('FAST2SMS_API_KEY', 'your-api-key')
        
        # Skip if API key is not set
        if api_key == 'your-api-key':
            return False, "SMS API key not configured"
        
        url = "https://www.fast2sms.com/dev/bulkV2"
        
        # Clean phone number
        phone = str(phone).replace('+91', '').strip()
        
        payload = {
            'sender_id': 'TXTIND',
            'message': message[:160],
            'language': 'english',
            'route': 'v3',
            'numbers': phone,
        }
        
        headers = {'authorization': api_key}
        response = requests.post(url, data=payload, headers=headers)
        result = response.json()
        
        if result.get('return'):
            return True, "SMS sent successfully"
        else:
            return False, result.get('message', 'Unknown error')
    except Exception as e:
        return False, str(e)

# Function to get AQI status, color, icon, and message
def get_aqi_status(aqi):
    if aqi <= 50:
        return {
            'status': 'Good',
            'color': '#28a745',
            'icon': '✅',
            'icon_large': '😊',
            'message': 'Air quality is good. Safe for outdoor activities.',
            'recommendations': '• Great day for outdoor activities!\n• Keep windows open for ventilation\n• Perfect for exercise and walks'
        }
    elif aqi <= 100:
        return {
            'status': 'Moderate',
            'color': '#ffc107',
            'icon': '⚠️',
            'icon_large': '😐',
            'message': 'Air quality is moderate. Sensitive individuals should be cautious.',
            'recommendations': '• Sensitive individuals should limit outdoor exertion\n• Keep windows closed if sensitive\n• Reduce prolonged outdoor activities'
        }
    elif aqi <= 150:
        return {
            'status': 'Unhealthy for Sensitive Groups',
            'color': '#ff9800',
            'icon': '😷',
            'icon_large': '😷',
            'message': 'Unhealthy for sensitive groups. Wear masks if going outside.',
            'recommendations': '• Wear masks when going outside\n• Reduce prolonged outdoor activities\n• Keep windows closed\n• Sensitive groups should stay indoors'
        }
    elif aqi <= 200:
        return {
            'status': 'Unhealthy',
            'color': '#dc3545',
            'icon': '🚨',
            'icon_large': '🤢',
            'message': 'Unhealthy air quality. Avoid outdoor activities.',
            'recommendations': '• Avoid all outdoor activities\n• Wear N95 masks if necessary\n• Use air purifiers indoors\n• Keep windows and doors closed'
        }
    elif aqi <= 300:
        return {
            'status': 'Very Unhealthy',
            'color': '#9c27b0',
            'icon': '🚫',
            'icon_large': '😫',
            'message': 'Very unhealthy. Stay indoors, wear masks if outside.',
            'recommendations': '• Stay indoors at all times\n• Wear N95 masks if absolutely necessary\n• Use air purifiers\n• Avoid physical exertion'
        }
    else:
        return {
            'status': 'Hazardous',
            'color': '#6c757d',
            'icon': '☣️',
            'icon_large': '💀',
            'message': 'Hazardous! Emergency conditions. Stay inside!',
            'recommendations': '• DO NOT go outside\n• Seal windows and doors\n• Use air purifiers with HEPA filters\n• Emergency alert: Seek shelter immediately'
        }

# Calculate AQI with proper formula
def calculate_aqi(pm25, pm10, no2, so2, co, o3):
    # Breakpoint concentrations for AQI calculation
    # This is a simplified but more accurate calculation
    
    # PM2.5 sub-index (based on EPA standards)
    if pm25 <= 12: 
        pm25_index = (50/12) * pm25
    elif pm25 <= 35.4: 
        pm25_index = ((100-51)/(35.4-12.1)) * (pm25-12.1) + 51
    elif pm25 <= 55.4: 
        pm25_index = ((150-101)/(55.4-35.5)) * (pm25-35.5) + 101
    elif pm25 <= 150.4: 
        pm25_index = ((200-151)/(150.4-55.5)) * (pm25-55.5) + 151
    elif pm25 <= 250.4: 
        pm25_index = ((300-201)/(250.4-150.5)) * (pm25-150.5) + 201
    else: 
        pm25_index = ((500-301)/(500-250.5)) * (pm25-250.5) + 301
    
    # PM10 sub-index
    if pm10 <= 54:
        pm10_index = (50/54) * pm10
    elif pm10 <= 154:
        pm10_index = ((100-51)/(154-55)) * (pm10-55) + 51
    elif pm10 <= 254:
        pm10_index = ((150-101)/(254-155)) * (pm10-155) + 101
    elif pm10 <= 354:
        pm10_index = ((200-151)/(354-255)) * (pm10-255) + 151
    elif pm10 <= 424:
        pm10_index = ((300-201)/(424-355)) * (pm10-355) + 201
    else:
        pm10_index = ((500-301)/(500-425)) * (pm10-425) + 301
    
    # Take maximum of all pollutants
    aqi = max(pm25_index, pm10_index)
    
    return round(aqi, 1)

# Save prediction to history
def save_to_history(username, aqi, pm25, pm10, no2, so2, co, o3):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    status = get_aqi_status(aqi)['status']
    try:
        c.execute('''INSERT INTO history 
                     (username, date, aqi, pm25, pm10, no2, so2, co, o3, status)
                     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                 (username, datetime.now(), aqi, pm25, pm10, no2, so2, co, o3, status))
        conn.commit()
    except Exception as e:
        print(f"Error saving to history: {e}")
    finally:
        conn.close()

# Get user history
def get_user_history(username, limit=20):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    try:
        c.execute('''SELECT date, aqi, pm25, pm10, no2, so2, co, o3, status 
                     FROM history 
                     WHERE username=? 
                     ORDER BY date DESC 
                     LIMIT ?''', (username, limit))
        history = c.fetchall()
    except Exception as e:
        print(f"Error getting history: {e}")
        history = []
    finally:
        conn.close()
    return history

# Get user details
def get_user_details(username):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute("SELECT username, password, email, phone, created_at FROM users WHERE username=?", (username,))
    user = c.fetchone()
    conn.close()
    return user

# Get user statistics
def get_user_stats(username):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    try:
        c.execute('''SELECT COUNT(*), AVG(aqi), MAX(aqi), MIN(aqi) 
                     FROM history 
                     WHERE username=?''', (username,))
        stats = c.fetchone()
        stats = stats if stats else (0, 0, 0, 0)
    except:
        stats = (0, 0, 0, 0)
    finally:
        conn.close()
    
    return {
        'total': stats[0] if stats[0] else 0,
        'avg': round(stats[1], 1) if stats[1] else 0,
        'max': stats[2] if stats[2] else 0,
        'min': stats[3] if stats[3] else 0
    }

# Initialize DB
init_db()

# Session state initialization
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'username' not in st.session_state:
    st.session_state.username = ""
if 'page' not in st.session_state:
    st.session_state.page = "login"
if 'last_prediction' not in st.session_state:
    st.session_state.last_prediction = None
if 'notification_shown' not in st.session_state:
    st.session_state.notification_shown = False
if 'settings' not in st.session_state:
    st.session_state.settings = None

# Login/Signup Page
def login_page():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🌍 Smart Air Quality Monitor</h1>
        <p>Login to monitor and predict air quality in real-time</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Center the login box
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        # Login/Signup tabs
        tab1, tab2 = st.tabs(["🔑 Login", "📝 Sign Up"])
        
        with tab1:
            with st.form("login_form"):
                st.markdown("### Welcome Back! 👋")
                username = st.text_input("👤 Username", placeholder="Enter your username")
                password = st.text_input("🔒 Password", type="password", placeholder="Enter your password")
                
                if st.form_submit_button("🚀 Login", use_container_width=True):
                    if username and password:
                        if verify_login(username, password):
                            st.session_state.logged_in = True
                            st.session_state.username = username
                            st.session_state.settings = get_user_settings(username)
                            st.success("✅ Login successful!")
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.error("❌ Invalid username or password")
                    else:
                        st.warning("⚠️ Please enter both username and password")
        
        with tab2:
            with st.form("signup_form"):
                st.markdown("### Create Account 🆕")
                col_a, col_b = st.columns(2)
                
                with col_a:
                    new_username = st.text_input("👤 Username", placeholder="Choose username")
                    new_password = st.text_input("🔒 Password", type="password", placeholder="Choose password")
                
                with col_b:
                    confirm_password = st.text_input("🔒 Confirm Password", type="password", placeholder="Re-enter password")
                    email = st.text_input("📧 Email", placeholder="Enter email")
                
                phone = st.text_input("📱 Phone", placeholder="Enter phone number (10 digits)")
                
                if st.form_submit_button("📝 Create Account", use_container_width=True):
                    if all([new_username, new_password, confirm_password, email, phone]):
                        if new_password == confirm_password:
                            if create_user(new_username, new_password, email, phone):
                                st.success("✅ Account created! Please login.")
                                time.sleep(1)
                                st.rerun()
                            else:
                                st.error("❌ Username already exists")
                        else:
                            st.error("❌ Passwords do not match")
                    else:
                        st.warning("⚠️ Please fill all fields")

# Main AQI App
def main_app():
    # Load user settings
    if st.session_state.settings is None:
        st.session_state.settings = get_user_settings(st.session_state.username)
    
    # Sidebar with user info
    with st.sidebar:
        st.markdown(f"""
        <div class="sidebar-user-info">
            <h3>👋 Welcome!</h3>
            <h2>{st.session_state.username}</h2>
            <p>{datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Navigation with icons
        if 'nav_page' not in st.session_state:
            st.session_state.nav_page = "🏠 Dashboard"
            
        pages = {
            "🏠 Dashboard": "📊 Overview",
            "📊 Predict AQI": "🔮 Prediction",
            "📈 History": "📜 Records",
            "👤 Profile": "👤 Profile",
            "⚙️ Settings": "🔧 Settings"
        }
        
        for page, desc in pages.items():
            if st.sidebar.button(f"{page}", use_container_width=True):
                st.session_state.nav_page = page
        
        st.markdown("---")
        
        # Quick stats
        stats = get_user_stats(st.session_state.username)
        st.markdown("### 📊 Quick Stats")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total", stats['total'])
        with col2:
            st.metric("Avg AQI", stats['avg'])
        
        st.markdown("---")
        
        # Logout button
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.logged_in = False
            st.session_state.username = ""
            st.session_state.settings = None
            st.rerun()
    
    # Main content based on navigation
    if st.session_state.nav_page == "🏠 Dashboard":
        show_dashboard()
    elif st.session_state.nav_page == "📊 Predict AQI":
        show_predictor()
    elif st.session_state.nav_page == "📈 History":
        show_history()
    elif st.session_state.nav_page == "👤 Profile":
        show_profile()
    elif st.session_state.nav_page == "⚙️ Settings":
        show_settings()

def show_dashboard():
    st.markdown("""
    <div class="main-header">
        <h1>🏠 Dashboard</h1>
        <p>Welcome to your personal air quality monitoring dashboard</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Get user history
    history = get_user_history(st.session_state.username, limit=50)
    
    if history:
        df = pd.DataFrame(history, columns=['Date', 'AQI', 'PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3', 'Status'])
        df['Date'] = pd.to_datetime(df['Date'])
        
        latest_aqi = df['AQI'].iloc[0] if not df.empty else 0
        latest_info = get_aqi_status(latest_aqi)
        
        # Stats cards
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="icon-large">{latest_info['icon_large']}</div>
                <h3>Current AQI</h3>
                <h1>{latest_aqi}</h1>
                <p>{latest_info['status']}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.metric("Total Predictions", len(df), 
                     f"+{len([x for x in df['AQI'] if x > latest_aqi])} this week")
        with col3:
            st.metric("Average AQI", f"{df['AQI'].mean():.1f}", 
                     f"{'↑' if df['AQI'].mean() > 100 else '↓'} compared to safe")
        with col4:
            st.metric("Highest AQI", f"{df['AQI'].max():.1f}", 
                     f"on {df.loc[df['AQI'].idxmax(), 'Date'].strftime('%m/%d')}")
        
        st.markdown("---")
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 AQI Trend (Last 10 readings)")
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df['Date'].head(10),
                y=df['AQI'].head(10),
                mode='lines+markers',
                name='AQI',
                line=dict(color='#667eea', width=3),
                marker=dict(size=10, color=df['AQI'].head(10), colorscale='RdYlGn_r')
            ))
            
            # Add threshold lines
            fig.add_hline(y=50, line_dash="dash", line_color="green", annotation_text="Good")
            fig.add_hline(y=100, line_dash="dash", line_color="orange", annotation_text="Moderate")
            fig.add_hline(y=150, line_dash="dash", line_color="red", annotation_text="Unhealthy")
            
            fig.update_layout(
                height=400,
                xaxis_title="Date",
                yaxis_title="AQI Value",
                hovermode='x',
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("📊 Status Distribution")
            
            status_counts = df['Status'].value_counts()
            
            fig = go.Figure(data=[
                go.Pie(
                    labels=status_counts.index,
                    values=status_counts.values,
                    hole=0.4,
                    marker=dict(colors=['#28a745', '#ffc107', '#ff9800', '#dc3545', '#9c27b0'])
                )
            ])
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Recent predictions table
        st.subheader("📋 Recent Predictions")
        
        # Format for display
        display_df = df[['Date', 'AQI', 'Status', 'PM2.5', 'PM10']].head(10).copy()
        display_df['Date'] = display_df['Date'].dt.strftime('%Y-%m-%d %H:%M')
        
        st.dataframe(display_df, use_container_width=True, height=400)
        
    else:
        st.info("📭 No predictions yet. Go to Predict AQI page to make your first prediction!")
        
        # Welcome guide
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="custom-card">
                <h3>🚀 Get Started</h3>
                <ol>
                    <li>Click on <b>Predict AQI</b> in sidebar</li>
                    <li>Enter pollutant values</li>
                    <li>Click Predict to see results</li>
                    <li>View history in <b>History</b> tab</li>
                </ol>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="custom-card">
                <h3>📊 Sample Data</h3>
                <p>Here's what your dashboard will look like:</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Sample chart
            sample_data = pd.DataFrame({
                'Date': pd.date_range(start='2024-01-01', periods=10, freq='D'),
                'AQI': [85, 110, 95, 145, 168, 152, 130, 118, 142, 135]
            })
            
            fig = go.Figure(data=[
                go.Scatter(x=sample_data['Date'], y=sample_data['AQI'], mode='lines+markers')
            ])
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)

def show_predictor():
    st.markdown("""
    <div class="main-header">
        <h1>📊 AQI Predictor</h1>
        <p>Enter pollutant values to predict air quality</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### Enter Pollutant Values")
        
        with st.form("prediction_form"):
            # Input fields with icons and tooltips
            col_a, col_b = st.columns(2)
            
            with col_a:
                pm25 = st.number_input(
                    "🌫️ PM2.5 (μg/m³)", 
                    0.0, 500.0, 35.0, step=1.0,
                    help="Fine particulate matter (2.5 micrometers)"
                )
                pm10 = st.number_input(
                    "🏭 PM10 (μg/m³)", 
                    0.0, 600.0, 70.0, step=1.0,
                    help="Coarse particulate matter (10 micrometers)"
                )
                no2 = st.number_input(
                    "🚗 NO₂ (ppb)", 
                    0.0, 200.0, 40.0, step=1.0,
                    help="Nitrogen Dioxide - from vehicle emissions"
                )
            
            with col_b:
                so2 = st.number_input(
                    "🏭 SO₂ (ppb)", 
                    0.0, 100.0, 20.0, step=1.0,
                    help="Sulfur Dioxide - from industrial processes"
                )
                co = st.number_input(
                    "🔥 CO (ppm)", 
                    0.0, 50.0, 2.0, step=0.1,
                    help="Carbon Monoxide - from incomplete burning"
                )
                o3 = st.number_input(
                    "☀️ O₃ (ppb)", 
                    0.0, 300.0, 60.0, step=1.0,
                    help="Ground-level Ozone"
                )
            
            st.markdown("---")
            
            if st.form_submit_button("🔮 Predict AQI", type="primary", use_container_width=True):
                # Calculate AQI
                aqi = calculate_aqi(pm25, pm10, no2, so2, co, o3)
                
                # Store in session
                st.session_state.last_prediction = {
                    'aqi': aqi,
                    'pm25': pm25,
                    'pm10': pm10,
                    'no2': no2,
                    'so2': so2,
                    'co': co,
                    'o3': o3,
                    'timestamp': datetime.now()
                }
                
                # Save to history
                save_to_history(st.session_state.username, aqi, pm25, pm10, no2, so2, co, o3)
                
                # Reset notification flag for new prediction
                st.session_state.notification_shown = False
                
                st.success("✅ Prediction complete!")
                st.rerun()
    
    with col2:
        st.markdown("### Prediction Results")
        
        if st.session_state.last_prediction:
            aqi = st.session_state.last_prediction['aqi']
            info = get_aqi_status(aqi)
            
            # Show AQI value with styling
            st.markdown(f"""
            <div style="text-align: center; padding: 30px; background: #f8f9fa; border-radius: 15px; margin-bottom: 20px;">
                <div style="font-size: 72px;">{info['icon_large']}</div>
                <h1 style="font-size: 72px; color: {info['color']}; margin: 0;">{aqi}</h1>
                <h2 style="color: {info['color']};">{info['status']}</h2>
                <p style="color: #666; font-size: 16px;">{info['message']}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Show alert if threshold exceeded
            if aqi > st.session_state.settings['alert_threshold'] and not st.session_state.notification_shown:
                st.error(f"🚨 **ALERT:** AQI exceeds your threshold of {st.session_state.settings['alert_threshold']}!")
                
                # Send notifications based on settings
                user = get_user_details(st.session_state.username)
                
                if st.session_state.settings['email_notify'] and user and user[2]:
                    email_msg = f"""AQI Alert - {info['status']}
                    
AQI Value: {aqi}
Status: {info['status']}
Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

{info['recommendations']}

Stay safe!
"""
                    success, msg = send_email_alert(user[2], f"🚨 AQI Alert: {info['status']}", email_msg)
                    if success:
                        st.success("📧 Email alert sent!")
                    else:
                        st.warning(f"⚠️ Email not sent: {msg}")
                
                if st.session_state.settings['sms_notify'] and user and user[3]:
                    sms_msg = f"AQI Alert: {aqi} - {info['status']}. {info['message'][:50]}"
                    success, msg = send_sms_alert(user[3], sms_msg)
                    if success:
                        st.success("📱 SMS alert sent!")
                    else:
                        st.warning(f"⚠️ SMS not sent: {msg}")
                
                st.balloons()
                st.session_state.notification_shown = True
            
            # Show recommendations
            st.markdown(f"""
            <div class="custom-card">
                <h3>💪 Health Recommendations</h3>
                <pre style="background: none; border: none; white-space: pre-wrap;">{info['recommendations']}</pre>
            </div>
            """, unsafe_allow_html=True)
            
            # Show pollutant details
            with st.expander("📊 View Pollutant Details"):
                details_df = pd.DataFrame({
                    'Pollutant': ['PM2.5', 'PM10', 'NO₂', 'SO₂', 'CO', 'O₃'],
                    'Icon': ['🌫️', '🏭', '🚗', '🏭', '🔥', '☀️'],
                    'Value': [
                        st.session_state.last_prediction['pm25'],
                        st.session_state.last_prediction['pm10'],
                        st.session_state.last_prediction['no2'],
                        st.session_state.last_prediction['so2'],
                        st.session_state.last_prediction['co'],
                        st.session_state.last_prediction['o3']
                    ],
                    'Unit': ['μg/m³', 'μg/m³', 'ppb', 'ppb', 'ppm', 'ppb']
                })