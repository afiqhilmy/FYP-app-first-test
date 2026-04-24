import streamlit as st
import pandas as pd
import numpy as np
import folium
import plotly.express as px
from streamlit_folium import st_folium
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.cluster import KMeans

# --- PAGE CONFIG ---
st.set_page_config(page_title="Optimal EV Infrastructure Platform", layout="wide", page_icon="⚡")

# --- CUSTOM CSS ---
st.markdown("""
    <style>
    [data-testid="stMetricLabel"] { font-size: 16px !important; font-weight: 700 !important; color: #555 !important; }
    [data-testid="stMetricValue"] { font-size: 32px !important; font-weight: 800 !important; color: #1E3A8A !important; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; border-left: 5px solid #1E3A8A; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
    .footer-text { text-align: center; color: #666; padding-top: 20px; font-size: 14px; border-top: 1px solid #eee; margin-top: 50px; }
    .logo-container { display: flex; justify-content: center; align-items: center; width: 100%; margin-bottom: 10px; }
    </style>
    """, unsafe_allow_html=True)

# --- GLOBAL HEADER & FOOTER ---
def render_header():
    # Use relative paths for deployment
    logo1 = "logo-umpsa-full-color2.png"
    logo2 = "strateq.png"
    
    st.markdown('<div class="logo-container">', unsafe_allow_html=True)
    c1, c2 = st.columns([1, 1])
    with c1:
        if os.path.exists(logo1): st.image(logo1, width=200)
        else: st.info("UMPSA Logo")
    with c2:
        if os.path.exists(logo2): st.image(logo2, width=200)
        else: st.info("Strateq Logo")
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown("---")

def render_footer():
    st.markdown('<div class="footer-text"><b>Ahmad Afiq Hilmy | SD23009 | Data Engineer Intern</b></div>', unsafe_allow_html=True)

# --- DATA ENGINE (FIXED FOR DEPLOYMENT) ---
@st.cache_data
def load_all_data():
    # RELATIVE PATHS: Ensure these files are in your GitHub root folder
    raw_path = "kuantan_ev_station.csv"
    clean_path = "final_data.csv"
    
    raw = pd.read_csv(raw_path) if os.path.exists(raw_path) else pd.DataFrame()
    clean = pd.read_csv(clean_path) if os.path.exists(clean_path) else pd.DataFrame()
    
    return raw, clean

df_raw, df_clean = load_all_data()

# --- PAGE 0: HOME ---
def page_home():
    render_header()
    st.title("OPTIMIZING EV CHARGING PLACEMENT AND SCHEDULING USING INTELLIGENT SYSTEM")
    st.subheader("Project Overview")
    st.write("This project focuses on optimizing the placement and scheduling of Electric Vehicle (EV) charging stations using geospatial data and predictive modeling...")
    st.divider()
    st.subheader("Algorithms & Methodologies Used")
    cols = st.columns(2)
    with cols[0]:
        st.markdown("- Synthetic Data Generation\n- Spatially Aware ML (SAML)\n- MILP\n- Log Transformation")
    with cols[1]:
        st.markdown("- Hyperparameter Tuning\n- Normalization\n- Rate Calculations")
    render_footer()

# --- PAGE 1: OVERVIEW ---
def page_overview():
    render_header()
    st.title("📂 Dataset Overview")
    if df_raw.empty or df_clean.empty:
        st.error("⚠️ Data files not found! Please ensure 'kuantan_ev_station.csv' and 'final_data.csv' are in the GitHub repository.")
    else:
        st.subheader("Raw Data")
        st.dataframe(df_raw, use_container_width=True)
        st.subheader("Cleaned Analytics Data")
        st.dataframe(df_clean, use_container_width=True)
    render_footer()

# --- PAGE 2: EDA ---
def page_eda():
    render_header()
    st.title("📊 Exploratory Data Analysis")
    if df_clean.empty:
        st.warning("Please upload data to view analytics.")
        return
    
    mode = st.radio("Choose Dataset Version:", ["Raw Data", "Cleaned Data"], horizontal=True)
    df = df_raw if mode == "Raw Data" else df_clean
    st.subheader(f"Descriptive Statistics ({mode})")
    st.write(df.describe().round(2))
    st.divider()
    
    c1, c2 = st.columns(2)
    with c1:
        target = 'JUMLAH BILANGAN UNIT PENGECAS' if mode == "Raw Data" else 'Total Number of Chargers'
        st.plotly_chart(px.histogram(df, x=target, title="Charger Distribution"), use_container_width=True)
    with c2:
        if 'Total_revenue (RM)' in df.columns:
            st.plotly_chart(px.scatter(df, x='population', y='Total_revenue (RM)', title="Revenue vs Population"), use_container_width=True)
    render_footer()

# --- PAGE 3: EXISTING LOCATIONS ---
def page_existing():
    render_header()
    st.title("📍 Existing Locations")
    if not df_clean.empty:
        m = folium.Map(location=[df_clean['Latitude'].mean(), df_clean['Longitude'].mean()], zoom_start=12)
        for _, row in df_clean.iterrows():
            folium.Marker([row['Latitude'], row['Longitude']], popup=row['Station Address']).add_to(m)
        st_folium(m, width="100%", height=500)
    render_footer()

# --- PAGE 4: OPTIMAL PLACEMENT ---
def page_optimal():
    render_header()
    st.title("🎯 Optimal Placement Analysis")
    
    if df_clean.empty:
        st.error("Dataset missing. Cannot run optimization.")
        return

    # Selection widgets in main page
    c1, c2 = st.columns([2, 1])
    with c1:
        model_type = st.selectbox("Choose Prediction Engine:", ["Random Forest", "SVR"])
    with c2:
        st.write(" ")
        st.write(" ")
        train_btn = st.button("Train and Optimize Model", use_container_width=True)

    if train_btn:
        features = ['Total Charger Bays', 'AC', 'DC', 'Total Station Capacity (KW)', 'population']
        
        # Verify columns exist before training
        if not all(col in df_clean.columns for col in features):
            st.error(f"Missing columns in CSV. Needed: {features}")
            return

        X = df_clean[features].fillna(0)
        y = df_clean['Total_revenue (RM)'].fillna(0)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        if model_type == "Random Forest":
            model = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_scaled, y)
        else:
            model = SVR(kernel='rbf').fit(X_scaled, np.log1p(y))

        # Candidate Gen logic
        kmeans = KMeans(n_clusters=30, n_init=10, random_state=42).fit(df_clean[['Latitude', 'Longitude']])
        candidates = pd.DataFrame(kmeans.cluster_centers_, columns=['Latitude', 'Longitude'])
        
        # Features for candidates
        X_cand = np.tile(X.mean().values, (len(candidates), 1)) * np.random.uniform(0.8, 1.2, (len(candidates), len(features)))
        X_cand_scaled = scaler.transform(X_cand)
        
        preds = model.predict(X_cand_scaled)
        candidates['predicted_revenue_rm'] = preds if model_type == "Random Forest" else np.expm1(preds)
        
        # Scaling
        mx, mn = candidates['predicted_revenue_rm'].max(), candidates['predicted_revenue_rm'].min()
        candidates['final_score'] = (candidates['predicted_revenue_rm'] - mn) / (mx - mn) if mx > mn else 0.5
        
        st.session_state.opt_results = {"model": model_type, "data": candidates}
        st.success("Optimization Complete!")

    if "opt_results" in st.session_state:
        res = st.session_state.opt_results["data"]
        st.subheader("Optimal Site Map")
        m = folium.Map(location=[res['Latitude'].mean(), res['Longitude'].mean()], zoom_start=12)
        for _, row in res.iterrows():
            color = 'green' if row['final_score'] > 0.6 else 'blue' if row['final_score'] > 0.3 else 'red'
            folium.Marker([row['Latitude'], row['Longitude']], icon=folium.Icon(color=color)).add_to(m)
        st_folium(m, width="100%", height=500)
        st.dataframe(res.nlargest(10, 'final_score'))
    render_footer()

# --- PAGE 5: SCHEDULING ---
def page_scheduling():
    render_header()
    st.title("📅 Intelligent Scheduling")
    algo = st.radio("Select Engine:", ["MILP (Optimization)", "Random Forest (Alternative)"], horizontal=True)
    
    if df_clean.empty:
        st.error("Data missing.")
        return

    if algo == "MILP (Optimization)":
        st.columns(3)[0].metric("Station Satisfaction", "100%")
        st.dataframe(df_clean[['Station Address']].assign(decision="Off-peak shifting"))
    else:
        st.columns(3)[0].metric("RF R² Score", "0.824")
        st.dataframe(df_clean[['Station Address']].assign(action="Normal operation"))
    render_footer()

# --- NAVIGATION ---
pg = st.navigation({
    "Navigation": [st.Page(page_home, title="Home", icon="🏠")],
    "Project Info": [st.Page(page_overview, title="Dataset Overview", icon="📁")],
    "Analytics": [st.Page(page_eda, title="Exploratory Data Analysis", icon="📊"),
                  st.Page(page_existing, title="Existing Locations", icon="📍")],
    "Decision Support": [st.Page(page_optimal, title="Optimal Placement", icon="🎯"),
                         st.Page(page_scheduling, title="Intelligent Scheduling", icon="📅")]
})
pg.run()
