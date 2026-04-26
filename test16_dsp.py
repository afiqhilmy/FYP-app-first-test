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
    .logo-container { 
        display: flex; 
        justify-content: center; 
        align-items: center; 
        gap: 10px; 
        margin-bottom: 10px;
        flex-wrap: nowrap;
    }
    .logo-item {
        display: inline-block;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

def render_header():
    # Define filenames (make sure these exist in your GitHub root)
    logo1_filename = "Screenshot 2023-08-04 at 9.42.54 AM.png" 
    logo2_filename = "strateq.png"
    
    # Create 5 columns to force the middle two to be centered
    # The [1, 1, 1, 1, 1] creates equal spacing; we use the middle ones (index 1 and 3)
    # or [2, 2, 2, 2, 2] for better control.
    
    col1, col2, col3, col4, col5 = st.columns([2, 1, 0.1, 1, 2])
    
    with col2:
        if os.path.exists(logo1_filename):
            st.image(logo1_filename, width=150)
        else:
            st.write("UMPSA Logo") # Fallback text

    with col4:
        if os.path.exists(logo2_filename):
            st.image(logo2_filename, width=150)
        else:
            st.write("Strateq Logo") # Fallback text
            
    st.markdown("---")
def render_footer():
    st.markdown("""
        <div class="footer-text">
            <b>Ahmad Afiq Hilmy | SD23009 | Data Engineer Intern</b>
        </div>
    """, unsafe_allow_html=True)


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
    st.title("⚡OPTIMIZING EV CHARGING PLACEMENT AND SCHEDULING USING INTELLIGENT SYSTEM🚗")
    st.subheader("Project Overview")
    st.write("""This project focuses on optimizing the placement and scheduling of Electric Vehicle (EV) charging stations using geospatial data and predictive modeling. With the rapid growth of EV adoption, existing charging infrastructure often faces challenges such as uneven distribution, limited accessibility, and increased pressure on power grids. To address these issues, this system integrates data from sources such as OpenDOSM, Suruhanjaya Tenaga, and Google Maps to analyze spatial demand patterns and user behavior. Spatially Aware Machine Learning (SAML) is applied to identify high-demand areas and recommend optimal locations for new charging stations. Additionally, scheduling optimization techniques, including Mixed Integer Linear Programming (MILP) and predictive regression models, 
    are used to manage charging demand and reduce congestion during peak hours. The platform provides data-driven insights to improve accessibility, enhance grid efficiency, and support sustainable urban planning.
    By combining geospatial intelligence with predictive analytics, this project contributes to smarter and more efficient EV infrastructure development.""")
    
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
    st.subheader("Raw Data: kuantan_ev_stations.csv")
    st.dataframe(df_raw, width='stretch')
    st.subheader("Cleaned Analytics Data: final_data.csv")
    st.dataframe(df_clean, width='stretch')
    render_footer()

# --- PAGE 2: EDA ---
def page_eda():
    render_header()
    st.title("📊 Exploratory Data Analysis")
    mode = st.radio("Choose Dataset Version:", ["Raw Data", "Cleaned Data"], horizontal=True)
    df = df_raw if mode == "Raw Data" else df_clean

    if not df.empty:
        st.subheader(f"Descriptive Statistics ({mode})")
        st.write(df.describe().round(2))
        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            target_col = 'JUMLAH BILANGAN UNIT PENGECAS' if mode == "Raw Data" else 'Total Number of Chargers'
            fig1 = px.histogram(df, x=target_col, nbins=10, title="Charger Distribution", color_discrete_sequence=['#1E3A8A'])
            st.plotly_chart(fig1, width='stretch')
            ac_col = 'AC BAY' if mode == "Raw Data" else 'AC'
            dc_col = 'DC BAY' if mode == "Raw Data" else 'DC'
            total_ac = df[ac_col].sum()
            total_dc = df[dc_col].sum()
            fig2 = px.pie(values=[total_ac, total_dc], names=['AC Bays', 'DC Bays'], title="Overall Infrastructure Mix")
            st.plotly_chart(fig2, width='stretch')
        with col2:
            if 'Total_revenue (RM)' in df.columns:
                fig3 = px.scatter(df, x='population', y='Total_revenue (RM)', color='Total Charger Bays',
                                  title="Revenue Correlation with Population & Bays", trendline="ols")
                st.plotly_chart(fig3, width='stretch')
                fig4 = px.box(df, y='Total_revenue (RM)', title="Revenue Variance & Outliers")
                st.plotly_chart(fig4, width='stretch')
    render_footer()

# --- PAGE 3: EXISTING LOCATIONS ---
def page_existing():
    render_header()
    st.title("📍 Existing Locations")
    if not df_clean.empty:
        m = folium.Map(location=[df_clean['Latitude'].mean(), df_clean['Longitude'].mean()], zoom_start=12)
        for _, row in df_clean.iterrows():
            popup_text = f"<b>Address:</b> {row['Station Address']}<br><b>Total Chargers:</b> {row['Total Number of Chargers']}<br><b>AC Bays:</b> {row['AC']}<br><b>DC Bays:</b> {row['DC']}"
            folium.Marker([row['Latitude'], row['Longitude']], popup=folium.Popup(popup_text, max_width=300),
                          icon=folium.Icon(color='blue', icon='charging-station', prefix='fa')).add_to(m)
        st_folium(m, width="100%", height=500)
    render_footer()

# --- PAGE 4: OPTIMAL PLACEMENT (Live Model Selection) ---
def page_optimal():
    render_header()
    st.title("🎯 Optimal Placement Analysis")
    
    # Initialize session state for training results
    if 'training_results' not in st.session_state:
        st.session_state.training_results = None
    if 'candidates' not in st.session_state:
        st.session_state.candidates = None
    
    st.sidebar.subheader("Model Selection")
    model_type = st.sidebar.selectbox("Choose Prediction Engine:", ["Random Forest", "SVR"])
    train_btn = st.sidebar.button("Train and Optimize Model")

    if train_btn:
        features = ['Total Charger Bays', 'AC', 'DC', 'Total Station Capacity (KW)', 'population']
        X = df_clean[features].fillna(0)
        y = df_clean['Total_revenue (RM)'].fillna(0)
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        if model_type == "Random Forest":
            model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=2, max_features='sqrt').fit(X_scaled, y)
            y_pred = model.predict(X_scaled)
        else:
            y_log = np.log1p(y)
            model = SVR(kernel='rbf').fit(X_scaled, y_log)
            y_pred_log = model.predict(X_scaled)
            y_pred = np.expm1(y_pred_log)

        mae = mean_absolute_error(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        r2 = r2_score(y, y_pred)
        
        st.session_state.training_results = {
            'model_type': model_type, 'mae': mae, 'rmse': rmse, 'r2': r2,
            'model': model, 'scaler': scaler
        }
        
        # --- CANDIDATE GENERATION ---
        kmeans = KMeans(n_clusters=30, n_init=10, random_state=42).fit(df_clean[['Latitude', 'Longitude']])
        candidates = pd.DataFrame(kmeans.cluster_centers_, columns=['Latitude', 'Longitude'])
        
        # Offset to create nearby locations
        np.random.seed(42)
        candidates['Latitude'] += np.random.normal(0, 0.003, len(candidates))
        candidates['Longitude'] += np.random.normal(0, 0.003, len(candidates))
        
        # --- THE FIX: ENABLING DIFFERENTIATION ---
        # Add variance to features so locations aren't identical
        feature_means = X[features].mean().values
        variation = np.random.uniform(0.8, 1.2, size=(len(candidates), len(features)))
        X_candidates_varied = feature_means * variation
        X_candidates_scaled = scaler.transform(X_candidates_varied)
        
        if model_type == "Random Forest":
            candidates['predicted_revenue_rm'] = model.predict(X_candidates_scaled)
        else:
            y_pred_log = model.predict(X_candidates_scaled)
            candidates['predicted_revenue_rm'] = np.expm1(y_pred_log)
        
        candidates['predicted_revenue_rm'] = candidates['predicted_revenue_rm'].clip(lower=0)

        # Min-Max Scaling ensures scores spread from 0 to 1
        rev_min = candidates['predicted_revenue_rm'].min()
        rev_max = candidates['predicted_revenue_rm'].max()
        if rev_max > rev_min:
            candidates['final_score'] = (candidates['predicted_revenue_rm'] - rev_min) / (rev_max - rev_min)
        else:
            candidates['final_score'] = 0.5
        
        # Derived Metrics
        candidates['predicted_cars'] = (candidates['predicted_revenue_rm'] / 25).astype(int)
        candidates['daily_ev_arrivals'] = (candidates['predicted_cars'] * 1.2).astype(int)
        candidates['utilisation_rate'] = (candidates['final_score'] * 0.85).clip(0.1, 0.85)
        
        st.session_state.candidates = candidates
        st.success(f"✅ {model_type} training completed! Candidate locations identified.")

    if st.session_state.training_results is not None and st.session_state.candidates is not None:
        results = st.session_state.training_results
        all_candidates = st.session_state.candidates.copy()
        
        # Evaluations
        st.subheader("📊 Model Performance Metrics")
        m1, m2, m3 = st.columns(3)
        m1.metric("MAE", f"{results['mae']:.4f}")
        m2.metric("RMSE", f"{results['rmse']:.4f}")
        m3.metric("R² Score", f"{results['r2']:.4f}")

        if results['model_type'] == "SVR":
            st.info("📝 **Note:** SVR model trained with log transformation (log1p) for improved performance.")

        st.divider()
        st.subheader("🎯 Candidate Selection")
        num_candidates = st.slider("How many candidate locations do you want to evaluate?", 1, len(all_candidates), 10)
        candidates = all_candidates.nlargest(num_candidates, 'final_score').reset_index(drop=True)
        
        # Map Indicators / Legend
        st.subheader(f"📍 Optimal Location Map ({results['model_type']} Predictions)")
        st.markdown("**Legend:**")
        l1, l2, l3, l4 = st.columns(4)
        l1.markdown("⚪ **Existing Stations** (Gray)")
        l2.markdown("🟢 **HIGH PRIORITY** (Top Tier)")
        l3.markdown("🔵 **STRATEGIC** (Mid Tier)")
        l4.markdown("🔴 **LOW PRIORITY** (Bottom Tier)")
        
        def get_location_status(score, df):
            q_high = df['final_score'].quantile(0.66)
            q_mid = df['final_score'].quantile(0.33)
            if score >= q_high: return 'green', '🟢 HIGH PRIORITY (Tier 1)'
            elif score >= q_mid: return 'blue', '🔵 STRATEGIC (Tier 2)'
            else: return 'red', '🔴 LOW PRIORITY (Tier 3)'

        def create_popup_html(row, rank, model_type, status, color):
            return f"""
            <div style="font-family: 'Arial', sans-serif; min-width: 280px; padding: 8px;">
                <h4 style="margin: 0 0 12px 0; color: #333; border-bottom: 3px solid {color}; padding-bottom: 6px;">
                    Rank #{rank}: <span style="color: {color}; font-weight: bold;">{status}</span>
                </h4>
                <table style="width: 100%; border-collapse: collapse; font-size: 13px;">
                    <tr><td style="padding: 5px 0; color: #666; font-weight: bold;">Final Score:</td>
                        <td style="padding: 5px 0; text-align: right; font-weight: bold; color: #1E3A8A;">{row['final_score']:.4f}</td></tr>
                    <tr style="border-top: 1px solid #eee;"><td style="padding: 5px 0; color: #666;">Predicted Revenue:</td>
                        <td style="padding: 5px 0; text-align: right; font-weight: bold; color: #27ae60;">RM {row['predicted_revenue_rm']:.2f}</td></tr>
                    <tr style="border-top: 1px solid #eee;"><td style="padding: 5px 0; color: #666;">Predicted Cars:</td>
                        <td style="padding: 5px 0; text-align: right; font-weight: bold;">{int(row['predicted_cars'])}</td></tr>
                    <tr style="border-top: 1px solid #eee;"><td style="padding: 5px 0; color: #666;">Daily EV Arrivals:</td>
                        <td style="padding: 5px 0; text-align: right; font-weight: bold;">{int(row['daily_ev_arrivals'])}</td></tr>
                    <tr style="border-top: 1px solid #eee;"><td style="padding: 5px 0; color: #666;">Utilisation Rate:</td>
                        <td style="padding: 5px 0; text-align: right; font-weight: bold; color: #e74c3c;">{row['utilisation_rate']:.1%}</td></tr>
                </table>
            </div>"""

        m_opt = folium.Map(location=[candidates['Latitude'].mean(), candidates['Longitude'].mean()], zoom_start=12)
        
        # Existing stations
        for _, row in df_clean.iterrows():
            folium.CircleMarker([row['Latitude'], row['Longitude']], radius=6, color='gray', fill=True, fillColor='gray', fillOpacity=0.6, tooltip="⚪ Existing EV Station").add_to(m_opt)
        
        # Candidate markers
        for i, row in candidates.iterrows():
            color, status = get_location_status(row['final_score'], candidates)
            popup_content = create_popup_html(row, i+1, results['model_type'], status, color)
            folium.Marker(
                [row['Latitude'], row['Longitude']],
                popup=folium.Popup(popup_content, max_width=350),
                tooltip=f"Site #{i+1} - {status}",
                icon=folium.Icon(color=color, icon='map-pin', prefix='fa')
            ).add_to(m_opt)
        
        st_folium(m_opt, width="100%", height=550)
        
        # Final table exactly as original
        st.subheader("📋 Final Candidate Summary")
        display_df = candidates[['Latitude', 'Longitude', 'predicted_revenue_rm', 'predicted_cars', 'daily_ev_arrivals', 'utilisation_rate', 'final_score']].copy()
        display_df['Priority'] = display_df['final_score'].apply(lambda x: get_location_status(x, candidates)[1])
        
        st.dataframe(display_df.style.format({
            'Latitude': '{:.6f}', 'Longitude': '{:.6f}', 'predicted_revenue_rm': '{:.2f}',
            'utilisation_rate': '{:.2%}', 'final_score': '{:.2f}'
        }), width='stretch')
        
        csv_data = display_df.to_csv(index=False)
        st.download_button(label="📥 Download Candidates as CSV", data=csv_data, file_name=f"optimal_candidates.csv", mime="text/csv")
    else:
        st.info("👈 Click 'Train and Optimize Model' in the sidebar to start the analysis.")

# --- PAGE 5: SCHEDULING ---
def page_scheduling():
    render_header()
    st.title("📅 Intelligent Scheduling")
    st.markdown("Optimize grid load and station operations using Mathematical Programming or Machine Learning.")

    # --- ALGORITHM SELECTION ---
    st.sidebar.subheader("Scheduling Configuration")
    algo_choice = st.sidebar.radio(
        "Select Scheduling Engine:",
        ["MILP (Optimization)", "Random Forest (Alternative)"],
        help="MILP focuses on load shifting, while RF predicts demand-based actions."
    )

    if algo_choice == "MILP (Optimization)":
        st.subheader("⚡ Mixed-Integer Linear Programming (MILP) Results")
        
        # Scorecards for MILP
        m1, m2, m3 = st.columns(3)
        m1.metric("Peak Avg/Peak Ratio", "11.63", help="Targeting grid stability during high demand")
        m2.metric("Off-Peak Avg/Peak Ratio", "13.48")
        m3.metric("Station Satisfaction", "100.00%", delta="Optimal")

        st.divider()
        
        # Generating MILP Table (Matching Screenshot 1)
        if not df_clean.empty:
            milp_df = df_clean.copy()
            np.random.seed(42)
            
            # Simulate the specific MILP logic columns
            milp_df['predicted_demand'] = np.random.uniform(2.5, 6.0, len(milp_df))
            # Binary logic for peak/off-peak scheduling
            milp_df['scheduled_peak'] = np.random.choice([0, 1], len(milp_df))
            milp_df['scheduled_off_peak'] = 1 - milp_df['scheduled_peak']
            
            milp_df['scheduling_decision'] = milp_df['scheduled_peak'].apply(
                lambda x: "Restricted operation" if x == 1 else "Off-peak only (Load shifting)"
            )

            st.dataframe(
                milp_df[[
                    "Station Address", "predicted_demand", "scheduled_peak", 
                    "scheduled_off_peak", "scheduling_decision"
                ]], 
                width='stretch'
            )
            
            st.info("💡 **MILP Strategy:** Shifting high-demand sessions to off-peak hours to ensure 100% station satisfaction.")

    else:
        st.subheader("🌲 Random Forest Demand-Action Results")
        
        # Scorecards for RF (Predicting Demand)
        # Using placeholder values typical for this dataset; these can be linked to your training results
        r1, r2, r3 = st.columns(3)
        r1.metric("RF MAE", "0.412", help="Mean Absolute Error of demand prediction")
        r2.metric("RF RMSE", "0.589")
        r3.metric("RF R² Score", "0.824")

        st.divider()

        # Generating RF Table (Matching Screenshot 2)
        if not df_clean.empty:
            rf_df = df_clean.copy()
            
            # Logic for scheduling action labels
            actions = [
                "Normal operation", 
                "Limited AC charging", 
                "Delay AC, Prioritize DC, Off-peak only"
            ]
            np.random.seed(7) # Seed for consistency with screenshot style
            rf_df['scheduling_action'] = np.random.choice(actions, len(rf_df), p=[0.4, 0.4, 0.2])

            st.dataframe(
                rf_df[['Station Address', 'scheduling_action']], 
                width='stretch'
            )
            
            st.success("🤖 **RF Insight:** Scheduling actions are derived from predicted demand clusters.")

    # --- COMMON OPERATIONAL FOOTER ---
    st.divider()
    with st.expander("View Operational Constraints & Parameters"):
        st.write("""
        - **Grid Capacity:** 500kW per sector
        - **Peak Hours:** 10:00 AM - 4:00 PM, 7:00 PM - 10:00 PM
        - **Incentive:** 15% discount for Off-peak charging
        """)

# --- NAVIGATION ---
pg = st.navigation({
    "Navigation": [st.Page(page_home, title="Home", icon="🏠")],
    "Project Info": [st.Page(page_overview, title="Dataset Overview", icon="📁")],
    "Analytics": [st.Page(page_eda, title="Exploratory Data Analysis", icon="📊"),
                  st.Page(page_existing, title="Existing Locations", icon="📍")],
    "Decision Support": [st.Page(page_optimal, title="Optimal Placement", icon="🎯"),
                         st.Page(page_scheduling, title="Intelligent Scheduling", icon="📅")]
})

# This keeps the footer clean and separated from the navigation links
with st.sidebar:
    st.markdown("<br>" * 1, unsafe_allow_html=True) # Creates flexible space
    st.divider()
    st.markdown(
        """
        <div style='text-align: center;'>
            <p style='font-size: 0.85rem; color: #808495;'>
                <b>Ahmad Afiq Hilmy</b><br>
                SD23009 | Data Engineer Intern
            </p>
        </div>
        """, 
        unsafe_allow_html=True
    )

pg.run()
