import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time
import plotly.graph_objects as go

# -----------------------------------------------------------------------------
# 1. APP CONFIGURATION & ASSETS
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="HeartGuard Pro",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load Models with Caching
@st.cache_resource
def load_assets():
    try:
        model = joblib.load('SVM_heart.pkl')
        scaler = joblib.load('scaler_heart.pkl')
        columns = joblib.load('columns_heart.pkl')
        return model, scaler, columns
    except FileNotFoundError:
        return None, None, None

model, scaler, model_columns = load_assets()

# -----------------------------------------------------------------------------
# 2. ADVANCED STYLING (Cyber-Medical Dark Theme)
# -----------------------------------------------------------------------------
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&family=JetBrains+Mono:wght@400&display=swap');

    :root {
        --primary: #FF4B4B;
        --secondary: #00D4FF;
        --bg-dark: #0E1117;
        --card-bg: #1A1C24;
    }

    html, body, .stApp {
        font-family: 'Inter', sans-serif;
        background-color: var(--bg-dark);
        color: #E0E0E0;
    }

    /* Glassmorphism Header */
    .header-container {
        background: linear-gradient(90deg, rgba(255, 75, 75, 0.1), rgba(0, 212, 255, 0.1));
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        padding: 20px;
        text-align: center;
        margin-bottom: 30px;
        backdrop-filter: blur(10px);
    }

    .header-title {
        font-size: 2.5rem;
        font-weight: 700;
        background: -webkit-linear-gradient(left, #FF4B4B, #FF9068);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    /* Input Cards */
    .styled-card {
        background-color: var(--card-bg);
        border: 1px solid #333;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.4);
        transition: transform 0.2s;
    }
    .styled-card:hover {
        border-color: #555;
    }
    .card-header {
        color: var(--secondary);
        font-weight: 600;
        margin-bottom: 15px;
        display: flex;
        align-items: center;
        gap: 10px;
    }

    /* Metric Containers */
    .metric-box {
        background: #262730;
        padding: 15px;
        border-radius: 8px;
        text-align: center;
        border: 1px solid #333;
    }
    .metric-value {
        font-size: 1.5rem;
        font-weight: bold;
        color: white;
    }
    .metric-label {
        font-size: 0.8rem;
        color: #888;
        text-transform: uppercase;
    }

    /* Custom Button */
    .stButton > button {
        background: linear-gradient(135deg, #FF4B4B 0%, #C20000 100%);
        color: white;
        border: none;
        padding: 15px;
        font-weight: bold;
        text-transform: uppercase;
        letter-spacing: 1px;
        border-radius: 8px;
        transition: all 0.3s;
    }
    .stButton > button:hover {
        box-shadow: 0 0 20px rgba(255, 75, 75, 0.6);
        transform: scale(1.02);
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #1A1C24;
        border-radius: 4px;
        color: white;
    }
    .stTabs [aria-selected="true"] {
        background-color: #FF4B4B;
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 3. UTILITY FUNCTIONS
# -----------------------------------------------------------------------------
def preprocess_input(data, columns):
    df = pd.DataFrame([data])
    df['Sex'] = df['Sex'].map({'Male': 1, 'Female': 0})
    df_processed = pd.DataFrame(np.zeros((1, len(columns))), columns=columns)

    numeric_cols = ['Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak']
    for col in numeric_cols:
        if col in df.columns:
            df_processed[col] = df[col]

    if 'Sex_1' in columns: df_processed['Sex_1'] = 1 if df['Sex'].iloc[0] == 1 else 0
    elif 'isMale' in columns: df_processed['isMale'] = 1 if df['Sex'].iloc[0] == 1 else 0

    cat_mappings = {
        'ChestPainType': df['ChestPainType'].iloc[0],
        'RestingECG': df['RestingECG'].iloc[0],
        'ExerciseAngina': df['ExerciseAngina'].iloc[0],
        'ST_Slope': df['ST_Slope'].iloc[0]
    }

    for col_base, value in cat_mappings.items():
        dummy_col = f"{col_base}_{value}"
        if dummy_col in df_processed.columns: df_processed[dummy_col] = 1

    return df_processed

def get_health_tips(data):
    tips = []
    if data['RestingBP'] > 140: tips.append("⚠️ **Hypertension Stage 2:** BP is significantly high. Monitor daily.")
    elif data['RestingBP'] > 130: tips.append("⚠️ **Hypertension Stage 1:** BP is elevated.")

    if data['Cholesterol'] > 240: tips.append("⚠️ **High Cholesterol:** Limit saturated fats and consider checking LDL levels.")

    if data['FastingBS'] == 1: tips.append("⚠️ **High Blood Sugar:** Diabetic risk factor detected.")

    if data['MaxHR'] < 100 and data['Age'] < 60: tips.append("ℹ️ **Low Max HR:** Exercise tolerance seems low for your age.")

    if not tips: tips.append("✅ **Vitals Check:** Key individual vitals appear within normal ranges.")
    return tips

# -----------------------------------------------------------------------------
# 4. PAGE LAYOUT: TABS
# -----------------------------------------------------------------------------
st.markdown('<div class="header-container"><div class="header-title">🫀 HeartGuard Pro</div><p>AI-Powered Cardiac Risk Analytics & Visualization Engine</p></div>', unsafe_allow_html=True)

tab_pred, tab_batch, tab_about = st.tabs(["🔍 Individual Analysis", "📂 Batch Processing", "ℹ️ Model Specs"])

# =============================================================================
# TAB 1: INDIVIDUAL PREDICTION
# =============================================================================
with tab_pred:
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('<div class="styled-card"><div class="card-header">👤 Patient Profile</div>', unsafe_allow_html=True)
        age = st.number_input("Age", 20, 100, 50)
        sex = st.selectbox("Sex", ["Male", "Female"])
        resting_bp = st.slider("Resting BP (mm Hg)", 80, 200, 120)
        cholesterol = st.slider("Cholesterol (mg/dl)", 100, 600, 200)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="styled-card"><div class="card-header">📝 Clinical History</div>', unsafe_allow_html=True)
        chest_pain = st.selectbox("Chest Pain", ["ASY", "ATA", "NAP", "TA"], help="ASY: Asymptomatic\nATA: Atypical\nNAP: Non-Anginal\nTA: Typical")
        fasting_bs = st.radio("Fasting BS > 120?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No", horizontal=True)
        resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="styled-card"><div class="card-header">⚡ Stress Test</div>', unsafe_allow_html=True)
        max_hr = st.slider("Max Heart Rate", 60, 220, 150)
        ex_angina = st.selectbox("Exercise Angina?", ["N", "Y"], format_func=lambda x: "Yes" if x == "Y" else "No")
        oldpeak = st.number_input("Oldpeak (ST)", 0.0, 6.0, 0.0, 0.1)
        st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("RUN ADVANCED DIAGNOSTICS", type="primary", use_container_width=True):
        if model:
            with st.spinner("Calculating probability vectors..."):
                time.sleep(1)

                # Prepare Data
                input_dict = {
                    'Age': age, 'Sex': sex, 'ChestPainType': chest_pain,
                    'RestingBP': resting_bp, 'Cholesterol': cholesterol,
                    'FastingBS': fasting_bs, 'RestingECG': resting_ecg,
                    'MaxHR': max_hr, 'ExerciseAngina': ex_angina,
                    'Oldpeak': oldpeak, 'ST_Slope': st_slope
                }

                processed = preprocess_input(input_dict, model_columns)
                scaled = scaler.transform(processed)

                # Prediction & Confidence Calculation
                pred_class = model.predict(scaled)[0]

                # Advanced: Use decision_function to estimate probability/confidence for SVM
                # Distance from hyperplane
                dist = model.decision_function(scaled)[0]
                # Sigmoid conversion to 0-1 probability space
                probability = 1 / (1 + np.exp(-dist))

                # --- RESULTS SECTION ---
                st.markdown("---")
                r_col1, r_col2 = st.columns([1, 1.5])

                with r_col1:
                    # 1. Gauge Chart
                    fig_gauge = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = probability * 100,
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        title = {'text': "Risk Probability", 'font': {'size': 24, 'color': 'white'}},
                        gauge = {
                            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "white"},
                            'bar': {'color': "#FF4B4B" if pred_class == 1 else "#00D4FF"},
                            'bgcolor': "rgba(0,0,0,0)",
                            'borderwidth': 2,
                            'bordercolor': "#333",
                            'steps': [
                                {'range': [0, 50], 'color': 'rgba(0, 212, 255, 0.1)'},
                                {'range': [50, 100], 'color': 'rgba(255, 75, 75, 0.1)'}],
                        }))
                    fig_gauge.update_layout(paper_bgcolor="rgba(0,0,0,0)", font={'color': "white", 'family': "Inter"})
                    st.plotly_chart(fig_gauge, use_container_width=True)

                    if pred_class == 1:
                        st.error("⚠️ **RESULT: HIGH RISK DETECTED**")
                    else:
                        st.success("✅ **RESULT: LOW RISK PROFILE**")

                with r_col2:
                    # 2. Radar Chart Comparison
                    st.subheader("📊 Vitals Comparison")

                    # Normalize data for visual comparison (Simple normalization against typical max values)
                    # categories: BP (200), Chol (600), MaxHR (220), Age (100)
                    categories = ['Resting BP', 'Cholesterol', 'Max HR', 'Age']
                    # Patient values normalized
                    patient_vals = [resting_bp/200, cholesterol/600, max_hr/220, age/100]
                    # "Healthy Average" values (Approximation)
                    avg_vals = [120/200, 200/600, 150/220, 50/100]

                    fig_radar = go.Figure()
                    fig_radar.add_trace(go.Scatterpolar(
                        r=patient_vals, theta=categories, fill='toself', name='Patient',
                        line_color='#FF4B4B'
                    ))
                    fig_radar.add_trace(go.Scatterpolar(
                        r=avg_vals, theta=categories, fill='toself', name='Healthy Avg',
                        line_color='#00D4FF', opacity=0.5
                    ))
                    fig_radar.update_layout(
                        polar=dict(radialaxis=dict(visible=True, range=[0, 1], showticklabels=False)),
                        showlegend=True,
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font={'color': "white"},
                        margin=dict(l=20, r=20, t=20, b=20)
                    )
                    st.plotly_chart(fig_radar, use_container_width=True)

                # 3. AI Doctor Notes
                st.subheader("🩺 AI Health Insights")
                tips = get_health_tips(input_dict)
                for tip in tips:
                    st.markdown(f"""
                        <div style="background: rgba(255,255,255,0.05); padding: 10px; border-radius: 5px; margin-bottom: 5px; border-left: 3px solid #FF4B4B;">
                        {tip}
                        </div>
                    """, unsafe_allow_html=True)
        else:
            st.error("Model files not loaded. Check directory.")

# =============================================================================
# TAB 2: BATCH PROCESSING
# =============================================================================
with tab_batch:
    st.markdown("### 📂 Hospital Batch Mode")
    st.write("Upload a CSV file containing patient data for bulk prediction.")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file is not None:
        try:
            input_df = pd.read_csv(uploaded_file)
            st.write("Preview:", input_df.head())

            if st.button("Process Batch"):
                with st.spinner("Processing batch..."):
                    results = []
                    # Simple loop for demonstration (vectorization preferred for large datasets)
                    for index, row in input_df.iterrows():
                        # Map input row to processor
                        try:
                            # Assuming CSV headers match inputs
                            row_dict = row.to_dict()
                            # Handle Sex if string
                            p_data = preprocess_input(row_dict, model_columns)
                            s_data = scaler.transform(p_data)
                            pred = model.predict(s_data)[0]
                            results.append("High Risk" if pred == 1 else "Low Risk")
                        except Exception as e:
                            results.append("Error")

                    input_df['AI Prediction'] = results
                    st.success("Batch processing complete!")
                    st.dataframe(input_df.style.map(lambda x: 'color: #FF4B4B' if x == 'High Risk' else 'color: #00D4FF', subset=['AI Prediction']))

                    # Download
                    csv = input_df.to_csv(index=False).encode('utf-8')
                    st.download_button("Download Report", csv, "heart_guard_report.csv", "text/csv")
        except Exception as e:
            st.error(f"Format Error: Ensure CSV columns match training data. ({e})")
    else:
        st.info("CSV should contain columns: Age, Sex, ChestPainType, RestingBP, Cholesterol, FastingBS, RestingECG, MaxHR, ExerciseAngina, Oldpeak, ST_Slope")

# =============================================================================
# TAB 3: ABOUT
# =============================================================================
with tab_about:
    st.markdown("### 🧬 About the Architecture")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Model Specs**")
        st.code("""
Algorithm: Support Vector Classifier (SVC)
Kernel:    Radial Basis Function (RBF)
Scaler:    StandardScaler (Z-Score Normalization)
Encoder:   One-Hot Encoding (Categorical)
        """)
    with c2:
        st.markdown("**Metric Definitions**")
        st.write("- **Oldpeak:** ST depression induced by exercise relative to rest.")
        st.write("- **ST Slope:** The slope of the peak exercise ST segment.")
        st.write("- **FastingBS:** Fasting Blood Sugar > 120 mg/dl.")
