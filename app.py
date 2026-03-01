import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load the trained pipeline
model = joblib.load('mental_health_model.pkl')

# --- Custom colorful CSS with mental health theme ---
st.set_page_config(page_title="Mental Wellness Check", layout="wide")
st.markdown("""
<style>
    /* Main header with soothing gradient */
    .main-header {
        background: linear-gradient(135deg, #6B8E9B 0%, #A7C5D3 100%);
        padding: 2rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 20px rgba(0,0,0,0.1);
    }
    /* Section headers with lighter gradient */
    .section-header {
        background: linear-gradient(90deg, #D4E6F1 0%, #F0F8FF 100%);
        padding: 1rem 2rem;
        border-radius: 15px;
        margin: 1.5rem 0 1rem 0;
        color: #2C3E50;
        font-weight: 500;
        border-left: 5px solid #5D9B9B;
    }
    /* Style the form */
    .stForm {
        background: rgba(255,255,255,0.9);
        backdrop-filter: blur(10px);
        border-radius: 30px;
        padding: 2rem;
        border: 1px solid rgba(255,255,255,0.3);
        box-shadow: 0 20px 40px rgba(0,0,0,0.1);
    }
    /* Style buttons */
    .stButton > button {
        background: linear-gradient(45deg, #5D9B9B, #417D7D);
        color: white;
        font-weight: bold;
        border: none;
        padding: 0.75rem 3rem;
        border-radius: 50px;
        font-size: 1.2rem;
        box-shadow: 0 5px 15px rgba(93,155,155,0.4);
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(93,155,155,0.6);
    }
    /* Metric cards */
    .stMetric {
        background: white;
        border-radius: 15px;
        padding: 1rem;
        box-shadow: 0 5px 15px rgba(0,0,0,0.05);
        border: 1px solid #E8F0F5;
    }
    /* Progress bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #5D9B9B, #A7C5D3) !important;
    }
    /* Expander */
    .streamlit-expanderHeader {
        background: #F0F8FF;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header"><h1>🌿 Mindful Health Check</h1><p>Answer 10 quick questions to get a personal wellness insight</p></div>', unsafe_allow_html=True)

# --- Exactly 10 questions in a clean form ---
with st.form("mental_health_form"):
    st.markdown('<div class="section-header">📋 About You</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=30, help="Your current age")
        gender = st.selectbox("Gender", ["Female", "Male", "Non‑binary", "Prefer not to say"])

    with col2:
        work_hours = st.number_input("Work hours per week", min_value=0, max_value=168, value=40,
                                     help="Average hours you work weekly")
        sleep_hours = st.number_input("Sleep hours per night", min_value=0.0, max_value=24.0, value=7.0, step=0.5,
                                      help="How many hours do you typically sleep?")

    with col3:
        exercise = st.selectbox("Exercise frequency", ["Never", "1-2 times/week", "3-4 times/week", "5+ times/week"])
        diet_quality = st.selectbox("Diet quality", ["Poor", "Average", "Good", "Excellent"])

    st.markdown('<div class="section-header">⚖️ Stress & Lifestyle</div>', unsafe_allow_html=True)
    col4, col5, col6 = st.columns(3)

    with col4:
        work_stress = st.slider("Work stress (1-10)", 1, 10, 5,
                                help="How stressed do you feel at work?")
        financial_stress = st.slider("Financial stress (1-10)", 1, 10, 5,
                                     help="Stress related to finances")

    with col5:
        social_media = st.number_input("Social media (hrs/day)", min_value=0.0, max_value=24.0, value=2.0, step=0.5,
                                       help="Daily time on social platforms")
        caffeine = st.number_input("Caffeine drinks/day", min_value=0, max_value=20, value=2,
                                   help="Cups of coffee/tea/energy drinks")

    with col6:
        smoking = st.selectbox("Smoking", ["Never", "Former", "Current"])
        alcohol = st.selectbox("Alcohol", ["Never", "Rarely", "Weekly", "Daily"])

    submitted = st.form_submit_button("See My Results")

# --- When form is submitted ---
if submitted:
    # Default values for all other features (set to neutral)
    defaults = {
        'country': "Other",
        'education': "Bachelor",
        'marital_status': "Single",
        'income_level': "Middle",
        'employment_status': "Full-time",
        'remote_work': "No",
        'job_satisfaction': 5,
        'work_life_balance': 5,
        'ever_bullied': "0",
        'company_support': "No",
        'hobby_time': 5,
        'screen_time': 5,
        'symptom': 1,           # all 15 symptoms set to 1 (minimal)
        'family_history': 0,
        'prev_diagnosed': 0,
        'ever_sought_treatment': 0,
        'on_therapy': 0,
        'on_medication': 0,
        'trauma_history': 0,
        'social_support': 5,
        'close_friends': 3,
        'feel_understood': 5,
        'loneliness': 5,
        'discuss_mental': "Never"
    }

    # Build full 50-feature dataframe (exact order as training)
    input_data = pd.DataFrame([[
        age, gender, defaults['country'], defaults['education'], defaults['marital_status'], defaults['income_level'],
        defaults['employment_status'], work_hours, defaults['remote_work'], defaults['job_satisfaction'],
        work_stress, defaults['work_life_balance'],
        defaults['ever_bullied'], defaults['company_support'],
        exercise, sleep_hours, caffeine, alcohol, smoking, defaults['screen_time'],
        social_media, defaults['hobby_time'], diet_quality, financial_stress,
        defaults['symptom'],  # Feeling_Sad_Down
        defaults['symptom'],  # Loss_Of_Interest
        defaults['symptom'],  # Sleep_Trouble
        defaults['symptom'],  # Fatigue
        defaults['symptom'],  # Poor_Appetite_Or_Overeating
        defaults['symptom'],  # Feeling_Worthless
        defaults['symptom'],  # Concentration_Difficulty
        defaults['symptom'],  # Anxious_Nervous
        defaults['symptom'],  # Panic_Attacks
        defaults['symptom'],  # Mood_Swings
        defaults['symptom'],  # Irritability
        defaults['symptom'],  # Obsessive_Thoughts
        defaults['symptom'],  # Compulsive_Behavior
        defaults['symptom'],  # Self_Harm_Thoughts
        defaults['symptom'],  # Suicidal_Thoughts
        defaults['family_history'],
        defaults['prev_diagnosed'],
        defaults['ever_sought_treatment'],
        defaults['on_therapy'],
        defaults['on_medication'],
        defaults['trauma_history'],
        defaults['social_support'],
        defaults['close_friends'],
        defaults['feel_understood'],
        defaults['loneliness'],
        defaults['discuss_mental']
    ]], columns=[
        'Age', 'Gender', 'Country', 'Education', 'Marital_Status', 'Income_Level',
        'Employment_Status', 'Work_Hours_Per_Week', 'Remote_Work', 'Job_Satisfaction',
        'Work_Stress_Level', 'Work_Life_Balance', 'Ever_Bullied_At_Work', 'Company_Mental_Health_Support',
        'Exercise_Per_Week', 'Sleep_Hours_Night', 'Caffeine_Drinks_Day', 'Alcohol_Frequency', 'Smoking',
        'Screen_Time_Hours_Day', 'Social_Media_Hours_Day', 'Hobby_Time_Hours_Week', 'Diet_Quality',
        'Financial_Stress', 'Feeling_Sad_Down', 'Loss_Of_Interest', 'Sleep_Trouble', 'Fatigue',
        'Poor_Appetite_Or_Overeating', 'Feeling_Worthless', 'Concentration_Difficulty', 'Anxious_Nervous',
        'Panic_Attacks', 'Mood_Swings', 'Irritability', 'Obsessive_Thoughts', 'Compulsive_Behavior',
        'Self_Harm_Thoughts', 'Suicidal_Thoughts', 'Family_History_Mental_Illness', 'Previously_Diagnosed',
        'Ever_Sought_Treatment', 'On_Therapy_Now', 'On_Medication', 'Trauma_History', 'Social_Support',
        'Close_Friends_Count', 'Feel_Understood', 'Loneliness', 'Discuss_Mental_Health'
    ])

    # Predict probability
    probability = model.predict_proba(input_data)[0][1]

    # Risk level
    if probability < 0.33:
        risk = "Low"
        color = "#5D9B9B"
    elif probability < 0.66:
        risk = "Medium"
        color = "#E6B800"
    else:
        risk = "High"
        color = "#C44545"

    # --- Display result with gradient card ---
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, {color}20, {color}40); padding: 2rem; border-radius: 30px; margin: 2rem 0; border: 2px solid {color};">
        <h2 style="color: {color}; text-align: center;">Your Wellness Insight</h2>
        <div style="display: flex; justify-content: center; align-items: center; gap: 2rem; flex-wrap: wrap;">
            <div style="font-size: 3rem;">{'🟢' if risk=='Low' else '🟡' if risk=='Medium' else '🔴'}</div>
            <div style="font-size: 2.5rem; color: {color};">{risk} Risk</div>
            <div style="font-size: 1.5rem; background: white; padding: 0.5rem 1.5rem; border-radius: 50px;">
                Probability: {probability:.1%}
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # --- Create two colorful charts ---
    st.markdown('<div class="section-header">📊 Your Personal Dashboard</div>', unsafe_allow_html=True)

    col_left, col_right = st.columns(2)

    with col_left:
        # Bar chart: stress & satisfaction scores
        fig, ax = plt.subplots(figsize=(6, 4))
        categories = ['Work Stress', 'Financial Stress', 'Sleep (scaled)']
        values = [work_stress, financial_stress, sleep_hours/2.4]  # scale sleep to 0-10 roughly
        bars = ax.bar(categories, values, color=['#FF6B6B', '#FFA07A', '#5D9B9B'])
        ax.set_ylim(0, 10)
        ax.set_ylabel('Score (1-10 scale)')
        ax.set_title('Key Stress Indicators', fontsize=14, fontweight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.2, f'{height:.1f}', ha='center', fontsize=12)
        st.pyplot(fig)
        plt.close(fig)

    with col_right:
        # Pie chart: lifestyle balance (exercise, diet, social media)
        # We'll create a simple donut chart of categorical inputs
        labels = ['Exercise', 'Diet', 'Social Media']
        # Convert exercise to numeric score
        ex_map = {"Never": 2, "1-2 times/week": 4, "3-4 times/week": 7, "5+ times/week": 9}
        diet_map = {"Poor": 3, "Average": 5, "Good": 7, "Excellent": 9}
        # Social media: higher hours = lower score (inverse)
        sm_score = max(1, 10 - social_media*1.5)  # rough mapping

        sizes = [ex_map[exercise], diet_map[diet_quality], sm_score]
        colors = ['#98D8C8', '#F7DC6F', '#BB8FCE']
        fig, ax = plt.subplots(figsize=(6, 4))
        wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors, autopct='%1.0f',
                                           startangle=90, pctdistance=0.85,
                                           wedgeprops=dict(width=0.4, edgecolor='white'))
        # Draw center circle for donut
        centre_circle = plt.Circle((0,0), 0.70, fc='white')
        fig.gca().add_artist(centre_circle)
        ax.set_title('Lifestyle Balance', fontsize=14, fontweight='bold')
        st.pyplot(fig)
        plt.close(fig)

    # --- Lifestyle summary metrics ---
    st.markdown('<div class="section-header">✨ Quick Snapshot</div>', unsafe_allow_html=True)
    mcol1, mcol2, mcol3, mcol4 = st.columns(4)
    mcol1.metric("Sleep", f"{sleep_hours} hrs")
    mcol2.metric("Exercise", exercise)
    mcol3.metric("Caffeine", f"{caffeine}/day")
    mcol4.metric("Social Media", f"{social_media} hrs")

    # Optional raw data expander
    with st.expander("🔍 Show all data used (including defaults)"):
        st.dataframe(input_data)
