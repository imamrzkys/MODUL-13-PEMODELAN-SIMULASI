"""
Streamlit App: Predator-Prey Analysis Lab
Multi-page interactive application for Lotka-Volterra analysis
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import io
import zipfile
import logging

# Import our modules
from src import data, models, plots, utils
from src.utils import Config

# Setup logging
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Predator-Prey Lab",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items=None
)

# Custom CSS for animations, mobile responsiveness, and styling
st.markdown("""
<style>
    /* Main styling */
    .main {
        padding: 1rem 2rem;
    }
    
    /* Smooth animations */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes slideIn {
        from { opacity: 0; transform: translateX(-20px); }
        to { opacity: 1; transform: translateX(0); }
    }
    
    .fade-in {
        animation: fadeIn 0.5s ease-in;
    }
    
    .slide-in {
        animation: slideIn 0.4s ease-out;
    }
    
    /* Metric cards with hover effect */
    [data-testid="stMetricValue"] {
        font-size: 1.5rem;
        font-weight: 600;
        color: #1f77b4;
        transition: all 0.3s ease;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 0.9rem;
        color: #666;
        font-weight: 500;
    }
    
    /* Button styling */
    .stButton > button {
        border-radius: 8px;
        transition: all 0.3s ease;
        font-weight: 500;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
    }
    
    /* Sidebar styling - Make it very visible with dark background */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #2c3e50 0%, #34495e 100%) !important;
        min-width: 280px !important;
        padding: 1.5rem !important;
        border-right: 3px solid #1f77b4 !important;
        box-shadow: 2px 0 12px rgba(0,0,0,0.3) !important;
    }
    
    /* Sidebar text color - make it white/light */
    [data-testid="stSidebar"] * {
        color: #ffffff !important;
    }
    
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #ffffff !important;
    }
    
    [data-testid="stSidebar"] > div:first-child {
        padding-top: 1rem !important;
    }
    
    /* Sidebar content wrapper */
    [data-testid="stSidebar"] > div {
        background: transparent !important;
    }
    
    /* Sidebar title */
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #1f77b4 !important;
        font-weight: 700 !important;
    }
    
    /* Sidebar radio buttons - make them more visible with better contrast */
    [data-testid="stSidebar"] [data-baseweb="radio"] {
        margin: 0.5rem 0 !important;
        padding: 0 !important;
        background-color: transparent !important;
    }
    
    [data-testid="stSidebar"] [data-baseweb="radio"] > div {
        display: flex !important;
        flex-direction: column !important;
        gap: 0.5rem !important;
    }
    
    [data-testid="stSidebar"] label {
        font-size: 1rem !important;
        font-weight: 500 !important;
        padding: 0.75rem 1rem !important;
        border-radius: 8px !important;
        transition: all 0.3s ease !important;
        color: #ffffff !important;
        background-color: rgba(255,255,255,0.1) !important;
        border: 2px solid rgba(255,255,255,0.2) !important;
        margin: 0.25rem 0 !important;
        display: block !important;
        cursor: pointer !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2) !important;
    }
    
    [data-testid="stSidebar"] label:hover {
        background-color: rgba(255,255,255,0.2) !important;
        border-color: #1f77b4 !important;
        transform: translateX(5px) !important;
        box-shadow: 0 4px 8px rgba(31,119,180,0.4) !important;
        color: #ffffff !important;
        font-weight: 600 !important;
    }
    
    /* Selected/Active state - very clear with light background and dark text */
    [data-testid="stSidebar"] [data-baseweb="radio"] [checked="true"] + label,
    [data-testid="stSidebar"] label:has([checked="true"]),
    [data-testid="stSidebar"] [data-baseweb="radio"] label[aria-checked="true"] {
        background-color: #ffffff !important;
        color: #2c3e50 !important;
        font-weight: 700 !important;
        border-color: #1f77b4 !important;
        border-width: 3px !important;
        box-shadow: 0 4px 12px rgba(255,255,255,0.3) !important;
        transform: translateX(5px) !important;
    }
    
    /* Radio button circle styling */
    [data-testid="stSidebar"] [data-baseweb="radio"] input[type="radio"] {
        margin-right: 0.75rem !important;
        width: 18px !important;
        height: 18px !important;
    }
    
    [data-testid="stSidebar"] [data-baseweb="radio"] input[type="radio"]:checked {
        background-color: white !important;
        border-color: white !important;
    }
    
    /* Sidebar markdown */
    [data-testid="stSidebar"] .stMarkdown {
        color: #ffffff !important;
    }
    
    /* Sidebar info box */
    [data-testid="stSidebar"] .stInfo {
        margin-top: 1rem !important;
        padding: 1rem !important;
        background-color: rgba(31,119,180,0.3) !important;
        border-left: 4px solid #1f77b4 !important;
        border-radius: 4px !important;
        color: #ffffff !important;
    }
    
    [data-testid="stSidebar"] .stInfo * {
        color: #ffffff !important;
    }
    
    /* Sidebar hr/divider */
    [data-testid="stSidebar"] hr {
        border-color: rgba(255,255,255,0.3) !important;
        margin: 1rem 0 !important;
    }
    
    /* Mobile responsiveness */
    @media (max-width: 768px) {
        .main {
            padding: 0.5rem 1rem;
        }
        
        [data-testid="stMetricValue"] {
            font-size: 1.2rem;
        }
        
        h1 {
            font-size: 1.8rem !important;
        }
        
        h2 {
            font-size: 1.4rem !important;
        }
        
        h3 {
            font-size: 1.2rem !important;
        }
        
        .stButton > button {
            width: 100%;
            margin: 0.25rem 0;
        }
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 6px 6px 0 0;
        padding: 0.5rem 1rem;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #1f77b4;
        color: white;
    }
    
    /* Success/Info boxes */
    .stSuccess {
        border-left: 4px solid #28a745;
        border-radius: 4px;
    }
    
    .stInfo {
        border-left: 4px solid #17a2b8;
        border-radius: 4px;
    }
    
    .stWarning {
        border-left: 4px solid #ffc107;
        border-radius: 4px;
    }
    
    /* Progress bar */
    .stProgress > div > div > div {
        background-color: #1f77b4;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        font-weight: 600;
        color: #1f77b4;
    }
    
    /* Hide Streamlit menu and footer for cleaner look */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Sidebar toggle button - make it more visible */
    [data-testid="stSidebar"] [data-testid="baseButton-header"] {
        background-color: #1f77b4;
        color: white;
        border-radius: 4px;
        padding: 0.5rem;
    }
    
    /* Ensure sidebar is always visible on desktop */
    @media (min-width: 769px) {
        [data-testid="stSidebar"] {
            display: block !important;
            visibility: visible !important;
        }
    }
    
    /* Custom card effect */
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
        transform: translateY(-2px);
    }
    
    /* Loading spinner animation */
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    /* Responsive columns */
    @media (max-width: 768px) {
        [data-testid="column"] {
            width: 100% !important;
            margin-bottom: 1rem;
        }
        
        /* Stack columns on mobile */
        .stColumn {
            flex-direction: column !important;
        }
        
        /* Adjust metric cards */
        [data-testid="stMetricContainer"] {
            margin-bottom: 0.5rem;
        }
        
        /* Smaller text on mobile */
        p, li {
            font-size: 0.9rem;
        }
        
        /* Sidebar on mobile - make it collapsible but visible */
        [data-testid="stSidebar"] {
            min-width: 200px !important;
            max-width: 250px !important;
        }
        
        /* Sidebar content on mobile */
        [data-testid="stSidebar"] h1 {
            font-size: 1.2rem;
        }
        
        [data-testid="stSidebar"] label {
            font-size: 0.9rem;
            padding: 0.4rem;
        }
        
        /* Table responsiveness */
        .dataframe {
            font-size: 0.75rem;
            overflow-x: auto;
        }
    }
    
    /* Tablet responsiveness */
    @media (max-width: 1024px) and (min-width: 769px) {
        .main {
            padding: 1rem 1.5rem;
        }
        
        [data-testid="stMetricValue"] {
            font-size: 1.3rem;
        }
    }
    
    /* Smooth page transitions */
    .element-container {
        animation: fadeIn 0.6s ease-in;
    }
    
    /* Better spacing */
    .stMarkdown {
        margin-bottom: 1rem;
    }
    
    /* Image responsiveness */
    img {
        max-width: 100%;
        height: auto;
    }
    
    /* Table responsiveness */
    .dataframe {
        font-size: 0.85rem;
        overflow-x: auto;
        display: block;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'best_params' not in st.session_state:
    st.session_state.best_params = None
if 'fitting_complete' not in st.session_state:
    st.session_state.fitting_complete = False
if 'sim_results' not in st.session_state:
    st.session_state.sim_results = None
if 'scale_factor' not in st.session_state:
    st.session_state.scale_factor = None


# --- Cached Data Loading ---
@st.cache_data
def load_and_prepare_data():
    """Load and prepare all data."""
    df_raw = data.load_data("wolf_moose_nps.csv")
    df_scaled, scale_factor = data.add_scaling(df_raw, method="max")
    
    prey_s = df_scaled['prey_scaled'].values
    pred_s = df_scaled['predator_scaled'].values
    t_eval = np.arange(len(df_scaled))
    initial_conditions = (prey_s[0], pred_s[0])
    
    return df_raw, df_scaled, prey_s, pred_s, t_eval, initial_conditions, scale_factor


# Load data once
try:
    df_raw, df_scaled, prey_s, pred_s, t_eval, initial_conditions, scale_factor = load_and_prepare_data()
    st.session_state.scale_factor = scale_factor
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()


# --- Sidebar Navigation ---
st.sidebar.markdown("""
<div style='text-align: center; padding: 1rem 0;'>
    <h2 style='color: #1f77b4; margin: 0; font-size: 1.5rem;'>Predator-Prey Lab</h2>
    <p style='color: #666; font-size: 0.9rem; margin-top: 0.5rem;'>Lotka-Volterra Analysis</p>
</div>
""", unsafe_allow_html=True)
st.sidebar.markdown("---")

st.sidebar.markdown("""
<div style='margin: 1rem 0;'>
    <h3 style='color: #1f77b4; font-weight: 700; margin-bottom: 0.75rem; font-size: 1.1rem;'>Navigasi Halaman</h3>
</div>
""", unsafe_allow_html=True)

page = st.sidebar.radio(
    "Pilih Halaman",
    [
        "Dashboard",
        "Data Exploration (EDA)",
        "Lotka-Volterra Simulator",
        "Parameter Fitting",
        "Phase Portrait & Dynamics",
        "Export & Gallery"
    ],
    label_visibility="collapsed"
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Informasi")
st.sidebar.info(
    "Aplikasi analisis Predator-Prey menggunakan model Lotka-Volterra.\n\n"
    "**Data:** Isle Royale (1980-2019)\n\n"
    "**Spesies:**\n"
    "- Prey: Moose\n"
    "- Predator: Wolves"
)


# ============================================================================
# PAGE 1: DASHBOARD
# ============================================================================
if page == "Dashboard":
    st.title("Dashboard")
    st.markdown("### Ringkasan Data & Metrik Utama")
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Year Range", f"{int(df_raw.index.min())} - {int(df_raw.index.max())}")
    with col2:
        st.metric("N Data Points", len(df_raw))
    with col3:
        st.metric("Min Prey", f"{df_raw['prey'].min():.0f}")
    with col4:
        st.metric("Max Prey", f"{df_raw['prey'].max():.0f}")
    
    col5, col6, col7, col8 = st.columns(4)
    with col5:
        st.metric("Min Predator", f"{df_raw['predator'].min():.0f}")
    with col6:
        st.metric("Max Predator", f"{df_raw['predator'].max():.0f}")
    with col7:
        if st.session_state.best_params:
            best_loss = "N/A"
            if st.session_state.sim_results is not None:
                from sklearn.metrics import mean_squared_error
                mse_prey = mean_squared_error(prey_s, st.session_state.sim_results[:, 0])
                mse_pred = mean_squared_error(pred_s, st.session_state.sim_results[:, 1])
                best_loss = f"{(mse_prey + 3*mse_pred):.4f}"
            st.metric("Best-Fit Loss", best_loss)
        else:
            st.metric("Best-Fit Loss", "Not fitted")
    with col8:
        st.metric("Scale Factor", f"{scale_factor:.2f}")
    
    st.markdown("---")
    
    # Tabs for plot and interpretation
    tab1, tab2 = st.tabs(["Plot", "Interpretasi"])
    
    with tab1:
        st.subheader("Raw Time Series Data")
        fig_raw = plots.plot_raw_data(df_raw, save=False)
        st.pyplot(fig_raw)
        st.caption("**Gambar 1.** Data Asli: Populasi Moose (Prey) dan Wolves (Predator) di Isle Royale dari tahun 1980 hingga 2019.")
        
        # Save plot
        if st.button("Simpan Plot", key="save_dashboard"):
            plots.plot_raw_data(df_raw, save=True)
            st.success("Plot disimpan sebagai 01_data_asli.png")
    
    with tab2:
        st.markdown("""
        ### Interpretasi Dashboard
        
        Dashboard ini menampilkan ringkasan data populasi predator-prey dari Isle Royale National Park.
        
        **Karakteristik Data:**
        - **Rentang Tahun**: Data mencakup periode 40 tahun (1980-2019)
        - **Prey (Moose)**: Populasi berkisar antara 385 hingga 2400 individu
        - **Predator (Wolves)**: Populasi berkisar antara 2 hingga 50 individu
        
        **Pola yang Teramati:**
        - Terdapat osilasi dalam populasi kedua spesies
        - Ketika populasi moose tinggi, populasi serigala cenderung meningkat (lag)
        - Ketika populasi serigala tinggi, populasi moose menurun
        - Pola ini menunjukkan dinamika predator-prey klasik
        """)
    
    st.markdown("---")
    
    # Data preview
    st.subheader("Data Preview")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Top 5 Rows**")
        st.dataframe(df_raw.head(), use_container_width=True)
    
    with col2:
        st.markdown("**Bottom 5 Rows**")
        st.dataframe(df_raw.tail(), use_container_width=True)
    
    # Download filtered dataset
    st.markdown("---")
    st.subheader("Download Dataset")
    csv = df_raw.to_csv()
    st.download_button(
        label="Download Filtered Dataset (CSV)",
        data=csv,
        file_name="wolf_moose_filtered.csv",
        mime="text/csv"
    )


# ============================================================================
# PAGE 2: DATA EXPLORATION (EDA)
# ============================================================================
elif page == "Data Exploration (EDA)":
    st.title("Data Exploration (EDA)")
    
    with st.expander("Tentang EDA", expanded=False):
        st.markdown("""
        **Exploratory Data Analysis (EDA)** adalah langkah penting dalam memahami data sebelum membangun model.
        
        Pada halaman ini, kita akan:
        1. Memvisualisasikan data asli dalam berbagai skala
        2. Menerapkan smoothing untuk melihat tren
        3. Menganalisis pola osilasi dalam data
        """)
    
    # Tab 1: Raw Data
    tab1, tab2, tab3, tab4 = st.tabs([
        "Data Asli",
        "Pola Osilasi",
        "Smoothing Tren",
        "Interpretasi"
    ])
    
    with tab1:
        st.subheader("Raw Data: Moose & Wolves")
        fig1 = plots.plot_raw_data(df_raw, save=False)
        st.pyplot(fig1)
        st.caption("**Gambar 1.** Data Asli: Populasi Moose (Prey) dan Wolves (Predator) dari tahun 1980-2019.")
        
        if st.button("Simpan", key="save_raw"):
            plots.plot_raw_data(df_raw, save=True)
            st.success("Disimpan sebagai 01_data_asli.png")
    
    with tab2:
        st.subheader("Pola Osilasi (Min-Max Scaling)")
        fig2 = plots.plot_oscillation_pattern(df_raw, save=False)
        st.pyplot(fig2)
        st.caption("**Gambar 2.** Pola Osilasi: Data yang dinormalisasi menggunakan Min-Max Scaling untuk melihat pola osilasi yang lebih jelas.")
        
        if st.button("Simpan", key="save_osc"):
            plots.plot_oscillation_pattern(df_raw, save=True)
            st.success("Disimpan sebagai 02_pola_osilasi.png")
    
    with tab3:
        st.subheader("Smoothing untuk Melihat Tren")
        window = st.slider("Moving Average Window", min_value=3, max_value=10, value=Config.DEFAULT_SMOOTHING_WINDOW, step=1)
        fig3 = plots.plot_smoothing_trend(df_raw, window=window, save=False)
        st.pyplot(fig3)
        st.caption(f"**Gambar 3.** Smoothing Tren: Data asli (transparan) dan moving average dengan window={window} untuk melihat tren osilasi.")
        
        if st.button("Simpan", key="save_smooth"):
            plots.plot_smoothing_trend(df_raw, window=window, save=True)
            st.success("Disimpan sebagai 03_smoothing_tren.png")
    
    with tab4:
        st.markdown("""
        ### Interpretasi EDA
        
        **1. Data Asli (Raw Data)**
        - Data menunjukkan variasi yang signifikan dalam populasi kedua spesies
        - Terdapat periode dimana populasi moose sangat tinggi (sekitar 2400) dan sangat rendah (sekitar 385)
        - Populasi serigala juga menunjukkan variasi besar (2-50 individu)
        
        **2. Pola Osilasi**
        - Setelah normalisasi, pola osilasi menjadi lebih jelas terlihat
        - Terdapat fase dimana prey tinggi dan predator rendah, kemudian sebaliknya
        - Pola ini konsisten dengan teori Lotka-Volterra
        
        **3. Smoothing Tren**
        - Moving average membantu mengurangi noise dalam data
        - Memungkinkan identifikasi tren jangka panjang
        - Window yang lebih besar memberikan smoothing yang lebih kuat namun kehilangan detail
        """)


# ============================================================================
# PAGE 3: LOTKA-VOLTERRA SIMULATOR
# ============================================================================
elif page == "Lotka-Volterra Simulator":
    st.title("Lotka-Volterra Simulator")
    
    with st.expander("Tentang Simulator", expanded=False):
        st.markdown("""
        **Lotka-Volterra Model:**
        
        ```
        dx/dt = αx - βxy  (Prey)
        dy/dt = δxy - γy  (Predator)
        ```
        
        - **α (alpha)**: Prey growth rate
        - **β (beta)**: Predation rate
        - **δ (delta)**: Predator growth efficiency
        - **γ (gamma)**: Predator death rate
        """)
    
    # Parameter controls
    st.subheader("Parameter Kontrol")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Growth & Predation Rates**")
        alpha = st.slider("α (alpha) - Prey growth", 0.001, 2.0, 1.0, 0.01, key="alpha_sim")
        beta = st.slider("β (beta) - Predation rate", 0.001, 30.0, 1.0, 0.1, key="beta_sim")
    
    with col2:
        st.markdown("**Predator Dynamics**")
        delta = st.slider("δ (delta) - Predator efficiency", 0.001, 30.0, 1.0, 0.1, key="delta_sim")
        gamma = st.slider("γ (gamma) - Predator death", 0.001, 2.0, 1.0, 0.01, key="gamma_sim")
    
    # Numeric inputs for fine-tuning
    st.markdown("**Fine-tune Parameters (Numeric Input)**")
    col3, col4, col5, col6 = st.columns(4)
    with col3:
        alpha_num = st.number_input("α", value=alpha, min_value=0.001, max_value=2.0, step=0.001, key="alpha_num")
    with col4:
        beta_num = st.number_input("β", value=beta, min_value=0.001, max_value=30.0, step=0.1, key="beta_num")
    with col5:
        delta_num = st.number_input("δ", value=delta, min_value=0.001, max_value=30.0, step=0.1, key="delta_num")
    with col6:
        gamma_num = st.number_input("γ", value=gamma, min_value=0.001, max_value=2.0, step=0.01, key="gamma_num")
    
    # Use numeric inputs if they differ from sliders
    params = {
        'alpha': alpha_num,
        'beta': beta_num,
        'delta': delta_num,
        'gamma': gamma_num
    }
    
    # Initial conditions
    st.subheader("Initial Conditions")
    col7, col8 = st.columns(2)
    with col7:
        x0 = st.number_input("x₀ (Initial Prey)", value=float(initial_conditions[0]), min_value=0.0, max_value=1.0, step=0.01)
    with col8:
        y0 = st.number_input("y₀ (Initial Predator)", value=float(initial_conditions[1]), min_value=0.0, max_value=1.0, step=0.01)
    
    # Solver controls
    st.subheader("Solver Settings")
    col9, col10 = st.columns(2)
    with col9:
        solver_method = st.selectbox("Method", ["RK45", "RK23", "DOP853", "Radau", "BDF"], index=0)
    with col10:
        dt_resolution = st.slider("Time Resolution (dt)", 0.1, 2.0, 1.0, 0.1)
        t_span_custom = np.arange(t_eval[0], t_eval[-1] + dt_resolution, dt_resolution)
    
    # Run simulation
    if st.button("Run Simulation", type="primary"):
        sim_result = models.simulate_lv(
            params,
            (x0, y0),
            t_eval,
            method=solver_method
        )
        
        if sim_result is not None:
            st.session_state.sim_results_manual = sim_result
            st.session_state.params_manual = params
            st.success("Simulation completed successfully!")
        else:
            st.error("Simulation failed. Please check parameters.")
    
    # Display results
    if 'sim_results_manual' in st.session_state:
        sim_result = st.session_state.sim_results_manual
        params_used = st.session_state.params_manual
        
        # Equilibrium point
        x_eq, y_eq = models.compute_equilibrium(
            params_used['alpha'],
            params_used['beta'],
            params_used['delta'],
            params_used['gamma']
        )
        
        st.markdown("---")
        st.subheader("Equilibrium Point")
        st.info(f"Equilibrium: (x*, y*) = ({x_eq:.4f}, {y_eq:.4f})")
        st.caption("Equilibrium point: (γ/δ, α/β)")
        
        # Overlay plot
        st.subheader("Simulation vs Data")
        fig_overlay = plots.plot_overlay_initial(
            df_scaled,
            sim_result[:, 0],
            sim_result[:, 1],
            save=False
        )
        st.pyplot(fig_overlay)
        st.caption("**Gambar 4.** Overlay Awal: Perbandingan data dengan simulasi menggunakan parameter yang dipilih.")
        
        if st.button("Simpan Overlay Awal", key="save_overlay_initial"):
            plots.plot_overlay_initial(df_scaled, sim_result[:, 0], sim_result[:, 1], save=True)
            st.success("Disimpan sebagai 04_overlay_awal.png")
        
        # Metrics
        from sklearn.metrics import mean_squared_error, mean_absolute_error
        mse_prey = mean_squared_error(prey_s, sim_result[:, 0])
        mae_prey = mean_absolute_error(prey_s, sim_result[:, 0])
        mse_pred = mean_squared_error(pred_s, sim_result[:, 1])
        mae_pred = mean_absolute_error(pred_s, sim_result[:, 1])
        
        st.subheader("Performance Metrics")
        col11, col12 = st.columns(2)
        with col11:
            st.metric("MSE Prey", f"{mse_prey:.6f}")
            st.metric("MAE Prey", f"{mae_prey:.6f}")
        with col12:
            st.metric("MSE Predator", f"{mse_pred:.6f}")
            st.metric("MAE Predator", f"{mae_pred:.6f}")


# ============================================================================
# PAGE 4: PARAMETER FITTING (AUTO TUNING)
# ============================================================================
elif page == "Parameter Fitting":
    st.title("Parameter Fitting (Auto Tuning)")
    
    with st.expander("Tentang Parameter Fitting", expanded=False):
        st.markdown("""
        **Differential Evolution (DE)** digunakan untuk mencari parameter optimal yang meminimalkan error antara data observasi dan simulasi model.
        
        **Loss Function:**
        ```
        Loss = w_prey * MSE(prey) + w_pred * MSE(predator)
        ```
        
        Dimana w_prey dan w_pred adalah bobot untuk menyeimbangkan kontribusi prey dan predator.
        """)
    
    # Bounds controls
    st.subheader("Parameter Bounds")
    st.markdown("Tentukan batas atas dan bawah untuk setiap parameter:")
    
    col1, col2 = st.columns(2)
    with col1:
        alpha_min = st.number_input("α min", value=0.001, min_value=0.0, step=0.001)
        alpha_max = st.number_input("α max", value=2.0, min_value=0.001, step=0.1)
        beta_min = st.number_input("β min", value=0.001, min_value=0.0, step=0.001)
        beta_max = st.number_input("β max", value=30.0, min_value=0.001, step=1.0)
    with col2:
        delta_min = st.number_input("δ min", value=0.001, min_value=0.0, step=0.001)
        delta_max = st.number_input("δ max", value=30.0, min_value=0.001, step=1.0)
        gamma_min = st.number_input("γ min", value=0.001, min_value=0.0, step=0.001)
        gamma_max = st.number_input("γ max", value=2.0, min_value=0.001, step=0.1)
    
    bounds = [
        (alpha_min, alpha_max),
        (beta_min, beta_max),
        (delta_min, delta_max),
        (gamma_min, gamma_max)
    ]
    
    # Weight controls
    st.subheader("Loss Function Weights")
    col3, col4 = st.columns(2)
    with col3:
        w_prey = st.number_input("w_prey (Prey weight)", value=1.0, min_value=0.1, step=0.1)
    with col4:
        w_pred = st.number_input("w_pred (Predator weight)", value=3.0, min_value=0.1, step=0.1)
    
    weights = {'w_prey': w_prey, 'w_pred': w_pred}
    
    # Quick Fit option
    st.subheader("Fitting Mode")
    use_quick_fit = st.checkbox("Quick Fit Mode (Lebih Cepat, Akurasi Sedikit Berkurang)", value=False, 
                                help="Quick Fit menggunakan iterasi dan populasi lebih kecil untuk hasil lebih cepat. Cocok untuk eksplorasi awal.")
    
    # DE settings
    st.subheader("Differential Evolution Settings")
    
    if use_quick_fit:
        # Quick fit defaults
        quick_maxiter = 100
        quick_popsize = 15
        st.info("Quick Fit Mode: Max Iterations = 100, Population Size = 15")
        col5, col6 = st.columns(2)
        with col5:
            maxiter = st.number_input("Max Iterations", value=quick_maxiter, min_value=50, max_value=200, step=25, 
                                     help="Quick Fit: 50-200 iterasi (default: 100)")
            popsize = st.number_input("Population Size", value=quick_popsize, min_value=10, max_value=30, step=5,
                                      help="Quick Fit: 10-30 individu (default: 15)")
        with col6:
            seed = st.number_input("Random Seed", value=Config.DE_SEED, min_value=0, step=1)
    else:
        # Full fit defaults
        col5, col6 = st.columns(2)
        with col5:
            maxiter = st.number_input("Max Iterations", value=Config.DE_MAXITER, min_value=50, max_value=1000, step=50,
                                     help="Full Fit: 50-1000 iterasi (default: 250)")
            popsize = st.number_input("Population Size", value=Config.DE_POPSIZE, min_value=10, max_value=100, step=5,
                                      help="Full Fit: 10-100 individu (default: 25)")
        with col6:
            seed = st.number_input("Random Seed", value=Config.DE_SEED, min_value=0, step=1)
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Run fitting
    if st.button("Run Differential Evolution", type="primary"):
        status_text.info("Starting Differential Evolution optimization...")
        progress_bar.progress(10)
        
        try:
            best_params = models.fit_de_params(
                prey_s,
                pred_s,
                t_eval,
                initial_conditions,
                bounds=bounds,
                weights=weights,
                maxiter=maxiter,
                popsize=popsize,
                seed=int(seed)
            )
            
            progress_bar.progress(90)
            
            # Run simulation with best params
            sim_results = models.simulate_lv(best_params, initial_conditions, t_eval)
            
            if sim_results is not None:
                st.session_state.best_params = best_params
                st.session_state.sim_results = sim_results
                st.session_state.fitting_complete = True
                
                # Auto-generate all plots after successful fitting
                status_text.info("Generating visualizations...")
                progress_bar.progress(95)
                
                try:
                    # Generate overlay final
                    plots.plot_overlay_final(df_scaled, sim_results[:, 0], sim_results[:, 1], save=True)
                    
                    # Generate phase portrait
                    plots.plot_phase_portrait(prey_s, pred_s, sim_results[:, 0], sim_results[:, 1], save=True)
                    
                    # Generate 3D trajectory
                    plots.plot_3d_trajectory_matplotlib(t_eval, sim_results[:, 0], sim_results[:, 1], save=True)
                    
                    # Generate overlay real scale
                    sim_prey_real = sim_results[:, 0] * scale_factor
                    sim_pred_real = sim_results[:, 1] * scale_factor
                    plots.plot_overlay_real_scale(df_raw, sim_prey_real, sim_pred_real, save=True)
                    
                    progress_bar.progress(100)
                    status_text.success("Fitting completed successfully! All visualizations generated.")
                    st.balloons()
                except Exception as e:
                    logger.warning(f"Error generating some plots: {e}")
                    progress_bar.progress(100)
                    status_text.success("Fitting completed successfully! (Some plots may need manual generation)")
                    st.balloons()
            else:
                st.error("Simulation with best parameters failed.")
                
        except Exception as e:
            st.error(f"Error during fitting: {e}")
            status_text.error("Fitting failed.")
    
    # Display results
    if st.session_state.fitting_complete:
        st.markdown("---")
        st.subheader("✅ Hasil Fitting")
        
        # Button to regenerate all plots
        if st.button("Regenerate All Visualizations", help="Generate semua plot dengan hasil fitting terbaru"):
            try:
                with st.spinner("Regenerating all visualizations..."):
                    plots.plot_overlay_final(df_scaled, sim_results[:, 0], sim_results[:, 1], save=True)
                    plots.plot_phase_portrait(prey_s, pred_s, sim_results[:, 0], sim_results[:, 1], save=True)
                    plots.plot_3d_trajectory_matplotlib(t_eval, sim_results[:, 0], sim_results[:, 1], save=True)
                    sim_prey_real = sim_results[:, 0] * scale_factor
                    sim_pred_real = sim_results[:, 1] * scale_factor
                    plots.plot_overlay_real_scale(df_raw, sim_prey_real, sim_pred_real, save=True)
                st.success("Semua visualisasi berhasil di-generate ulang!")
            except Exception as e:
                st.error(f"Error regenerating plots: {e}")
        
        best_params = st.session_state.best_params
        sim_results = st.session_state.sim_results
        
        # Parameter metrics
        col7, col8, col9, col10 = st.columns(4)
        col7.metric("Best α", f"{best_params['alpha']:.6f}")
        col8.metric("Best β", f"{best_params['beta']:.6f}")
        col9.metric("Best δ", f"{best_params['delta']:.6f}")
        col10.metric("Best γ", f"{best_params['gamma']:.6f}")
        
        # Equilibrium
        x_eq, y_eq = models.compute_equilibrium(
            best_params['alpha'],
            best_params['beta'],
            best_params['delta'],
            best_params['gamma']
        )
        st.info(f"**Equilibrium Point:** (x*, y*) = ({x_eq:.4f}, {y_eq:.4f})")
        
        # Tabs for visualizations
        tab1, tab2, tab3 = st.tabs(["Overlay", "Residuals", "Metrics"])
        
        with tab1:
            st.subheader("Overlay: Data vs Best-Fit Simulation")
            fig_overlay = plots.plot_overlay_final(
                df_scaled,
                sim_results[:, 0],
                sim_results[:, 1],
                save=False
            )
            st.pyplot(fig_overlay)
            st.caption("**Gambar 5.** Overlay FINAL: Perbandingan data dengan simulasi menggunakan parameter terbaik dari Differential Evolution.")
            
            if st.button("Simpan Overlay", key="save_overlay_final"):
                plots.plot_overlay_final(df_scaled, sim_results[:, 0], sim_results[:, 1], save=True)
                st.success("Disimpan sebagai 05_overlay_final.png")
            
            st.markdown("---")
            st.subheader("Overlay pada Skala Asli (Real Population)")
            sim_prey_real = sim_results[:, 0] * scale_factor
            sim_pred_real = sim_results[:, 1] * scale_factor
            fig_overlay_real = plots.plot_overlay_real_scale(
                df_raw,
                sim_prey_real,
                sim_pred_real,
                save=False
            )
            st.pyplot(fig_overlay_real)
            st.caption("**Gambar 8.** Overlay pada Skala Asli: Perbandingan data dan simulasi dalam skala populasi asli (tidak dinormalisasi).")
            
            if st.button("Simpan Overlay Skala Asli", key="save_overlay_real"):
                plots.plot_overlay_real_scale(df_raw, sim_prey_real, sim_pred_real, save=True)
                st.success("Disimpan sebagai 08_overlay_skala_asli.png")
        
        with tab2:
            st.subheader("Residuals: Data - Simulation")
            fig_res = plots.plot_residuals(
                prey_s, pred_s,
                sim_results[:, 0], sim_results[:, 1],
                df_scaled.index.values
            )
            st.pyplot(fig_res)
            st.caption("**Residuals Plot:** Menunjukkan perbedaan antara data observasi dan simulasi. Residual yang kecil dan acak menunjukkan model yang baik.")
        
        with tab3:
            st.subheader("Model Performance Metrics")
            
            # Scaled metrics
            metrics_prey = models.compute_metrics(prey_s, sim_results[:, 0], scaled=True)
            metrics_pred = models.compute_metrics(pred_s, sim_results[:, 1], scaled=True)
            
            st.markdown("**Scaled Data Metrics:**")
            col11, col12 = st.columns(2)
            with col11:
                st.metric("MSE Prey (scaled)", f"{metrics_prey['mse']:.6f}")
                st.metric("MAE Prey (scaled)", f"{metrics_prey['mae']:.6f}")
            with col12:
                st.metric("MSE Predator (scaled)", f"{metrics_pred['mse']:.6f}")
                st.metric("MAE Predator (scaled)", f"{metrics_pred['mae']:.6f}")
            
            # Real-scale metrics
            sim_prey_real = sim_results[:, 0] * scale_factor
            sim_pred_real = sim_results[:, 1] * scale_factor
            
            metrics_prey_real = models.compute_metrics(df_raw['prey'].values, sim_prey_real, scaled=False)
            metrics_pred_real = models.compute_metrics(df_raw['predator'].values, sim_pred_real, scaled=False)
            
            st.markdown("**Real-Scale Metrics:**")
            col13, col14 = st.columns(2)
            with col13:
                st.metric("MSE Prey (real)", f"{metrics_prey_real['mse']:.2f}")
                st.metric("MAE Prey (real)", f"{metrics_prey_real['mae']:.2f}")
            with col14:
                st.metric("MSE Predator (real)", f"{metrics_pred_real['mse']:.2f}")
                st.metric("MAE Predator (real)", f"{metrics_pred_real['mae']:.2f}")


# ============================================================================
# PAGE 5: PHASE PORTRAIT & DYNAMICS
# ============================================================================
elif page == "Phase Portrait & Dynamics":
    st.title("Phase Portrait & Dynamics")
    
    with st.expander("Tentang Phase Portrait", expanded=False):
        st.markdown("""
        **Phase Portrait** menunjukkan hubungan antara prey dan predator dalam ruang fase.
        
        - **Limit Cycle**: Dalam model Lotka-Volterra ideal, sistem akan mengikuti limit cycle (orbit tertutup)
        - **Data Real**: Data observasi menunjukkan noise dan variasi yang tidak sempurna mengikuti limit cycle
        - **3D Trajectory**: Menunjukkan evolusi sistem dalam ruang waktu-prey-predator
        """)
    
    if not st.session_state.fitting_complete:
        st.warning("Silakan jalankan Parameter Fitting terlebih dahulu untuk melihat phase portrait dan dynamics.")
    else:
        sim_results = st.session_state.sim_results
        
        tab1, tab2, tab3 = st.tabs(["Phase Portrait", "3D Trajectory", "Interpretasi"])
        
        with tab1:
            st.subheader("Phase Portrait: Data vs Simulation")
            
            # Vector field toggle
            show_vector_field = st.checkbox("Show Vector Field (Quiver)", value=False)
            
            fig_phase = plots.plot_phase_portrait(
                prey_s, pred_s,
                sim_results[:, 0], sim_results[:, 1],
                save=False
            )
            
            # Add vector field if requested
            if show_vector_field:
                # Create vector field
                best_params = st.session_state.best_params
                x_range = np.linspace(prey_s.min() * 0.8, prey_s.max() * 1.2, 15)
                y_range = np.linspace(pred_s.min() * 0.8, pred_s.max() * 1.2, 15)
                X, Y = np.meshgrid(x_range, y_range)
                
                U = best_params['alpha'] * X - best_params['beta'] * X * Y
                V = best_params['delta'] * X * Y - best_params['gamma'] * Y
                
                # Normalize for visualization
                norm = np.sqrt(U**2 + V**2)
                U = U / (norm + 1e-10)
                V = V / (norm + 1e-10)
                
                ax = fig_phase.axes[0]
                ax.quiver(X, Y, U, V, alpha=0.5, scale=20, width=0.003)
                fig_phase.tight_layout()
            
            st.pyplot(fig_phase)
            st.caption("**Gambar 6.** Phase Portrait: Perbandingan data observasi (titik) dengan simulasi model (garis). Limit cycle ideal ditunjukkan oleh orbit tertutup simulasi.")
            
            if st.button("Simpan Phase Portrait", key="save_phase"):
                plots.plot_phase_portrait(
                    prey_s, pred_s,
                    sim_results[:, 0], sim_results[:, 1],
                    save=True
                )
                st.success("Disimpan sebagai 06_phase_portrait.png")
        
        with tab2:
            st.subheader("3D Trajectory: Time–Prey–Predator")
            
            # Try Plotly first, fallback to matplotlib
            use_plotly = st.checkbox("Use Plotly (Interactive)", value=True)
            
            if use_plotly and plots.PLOTLY_AVAILABLE:
                fig_3d = plots.plot_3d_trajectory_plotly(t_eval, sim_results[:, 0], sim_results[:, 1])
                if fig_3d:
                    st.plotly_chart(fig_3d, use_container_width=True)
                    st.caption("**Gambar 7.** 3D Trajectory: Evolusi sistem dalam ruang waktu-prey-predator menggunakan Plotly interaktif.")
                else:
                    st.warning("Plotly plot failed, using matplotlib fallback.")
                    use_plotly = False
            
            if not use_plotly or not plots.PLOTLY_AVAILABLE:
                fig_3d = plots.plot_3d_trajectory_matplotlib(t_eval, sim_results[:, 0], sim_results[:, 1], save=False)
                st.pyplot(fig_3d)
                st.caption("**Gambar 7.** 3D Trajectory: Evolusi sistem dalam ruang waktu-prey-predator menggunakan Matplotlib.")
            
            if st.button("Simpan 3D Trajectory", key="save_3d"):
                plots.plot_3d_trajectory_matplotlib(t_eval, sim_results[:, 0], sim_results[:, 1], save=True)
                st.success("Disimpan sebagai 07_3d_trajectory.png")
        
        with tab3:
            st.markdown("""
            ### Interpretasi Phase Portrait & Dynamics
            
            **1. Phase Portrait**
            - Data observasi (titik biru) menunjukkan pola yang mirip dengan limit cycle, namun dengan noise
            - Simulasi model (garis oranye) menunjukkan limit cycle yang lebih halus dan teratur
            - Perbedaan antara data dan simulasi menunjukkan keterbatasan model dalam menangkap semua faktor eksternal
            
            **2. 3D Trajectory**
            - Trajectory 3D menunjukkan bagaimana sistem berevolusi seiring waktu
            - Pola spiral atau orbit menunjukkan sifat osilasi sistem
            - Visualisasi ini membantu memahami dinamika temporal sistem
            
            **3. Limit Cycle vs Real Data**
            - Model Lotka-Volterra ideal menghasilkan limit cycle yang sempurna
            - Data real menunjukkan variasi karena faktor eksternal (cuaca, penyakit, migrasi, dll)
            - Model tetap berguna untuk memahami mekanisme dasar interaksi predator-prey
            """)


# ============================================================================
# PAGE 6: EXPORT & GALLERY
# ============================================================================
elif page == "Export & Gallery":
    st.title("Export & Gallery")
    
    # Gallery section
    st.subheader("Visualization Gallery")
    
    viz_dir = Path(Config.VIZ_DIR)
    if viz_dir.exists():
        png_files = sorted(list(viz_dir.glob("*.png")))
        
        if png_files:
            st.info(f"Found {len(png_files)} visualization files in {Config.VIZ_DIR}/")
            
            # Display images in grid
            for i, img_path in enumerate(png_files):
                st.markdown(f"### {img_path.name}")
                
                # Get caption from config
                caption = Config.PLOT_NAMES.get(img_path.name, "Visualization")
                
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.image(str(img_path), use_container_width=True)
                    st.caption(f"**Gambar {i+1}.** {caption}")
                with col2:
                    with open(img_path, "rb") as f:
                        st.download_button(
                            label="Download",
                            data=f.read(),
                            file_name=img_path.name,
                            mime="image/png",
                            key=f"download_{i}"
                        )
        else:
            st.warning(f"No PNG files found in {Config.VIZ_DIR}/. Run simulations and save plots to generate visualizations.")
    else:
        st.warning(f"Directory {Config.VIZ_DIR}/ does not exist. It will be created when you save plots.")
    
    st.markdown("---")
    
    # Export Report Pack
    st.subheader("Export Report Pack")
    st.markdown("Generate a complete report package with all outputs:")
    
    if st.button("Generate Report Pack", type="primary"):
        with st.spinner("Generating report pack..."):
            # Ensure directory exists
            utils.ensure_dir(Config.VIZ_DIR)
            
            # Create temporary directory for report
            import tempfile
            import shutil
            
            with tempfile.TemporaryDirectory() as temp_dir:
                report_dir = Path(temp_dir) / "report_pack"
                report_dir.mkdir()
                
                # Copy all PNGs
                if viz_dir.exists():
                    for png_file in viz_dir.glob("*.png"):
                        shutil.copy(png_file, report_dir / png_file.name)
                
                # Save parameters JSON
                if st.session_state.best_params:
                    params_data = {
                        'best_parameters': st.session_state.best_params,
                        'initial_conditions': {
                            'x0': float(initial_conditions[0]),
                            'y0': float(initial_conditions[1])
                        },
                        'scale_factor': float(scale_factor),
                        'year_range': {
                            'min': int(df_raw.index.min()),
                            'max': int(df_raw.index.max())
                        }
                    }
                    utils.save_json(params_data, str(report_dir / "parameters.json"))
                
                # Save metrics JSON
                if st.session_state.sim_results is not None:
                    from sklearn.metrics import mean_squared_error, mean_absolute_error
                    sim_results = st.session_state.sim_results
                    
                    metrics_data = {
                        'scaled_metrics': {
                            'prey': {
                                'mse': float(mean_squared_error(prey_s, sim_results[:, 0])),
                                'mae': float(mean_absolute_error(prey_s, sim_results[:, 0]))
                            },
                            'predator': {
                                'mse': float(mean_squared_error(pred_s, sim_results[:, 1])),
                                'mae': float(mean_absolute_error(pred_s, sim_results[:, 1]))
                            }
                        },
                        'real_scale_metrics': {
                            'prey': {
                                'mse': float(mean_squared_error(df_raw['prey'].values, sim_results[:, 0] * scale_factor)),
                                'mae': float(mean_absolute_error(df_raw['prey'].values, sim_results[:, 0] * scale_factor))
                            },
                            'predator': {
                                'mse': float(mean_squared_error(df_raw['predator'].values, sim_results[:, 1] * scale_factor)),
                                'mae': float(mean_absolute_error(df_raw['predator'].values, sim_results[:, 1] * scale_factor))
                            }
                        }
                    }
                    utils.save_json(metrics_data, str(report_dir / "metrics.json"))
                
                # Save filtered CSV
                csv_path = report_dir / "filtered_data.csv"
                df_raw.to_csv(csv_path)
                
                # Create ZIP
                zip_path = utils.zip_outputs(str(report_dir), "report_pack.zip")
                
                # Read ZIP for download
                with open(zip_path, "rb") as f:
                    zip_data = f.read()
                
                st.success("Report pack generated successfully!")
                st.download_button(
                    label="Download Report Pack (ZIP)",
                    data=zip_data,
                    file_name="predator_prey_report_pack.zip",
                    mime="application/zip"
                )
    
    st.markdown("---")
    
    # Download current simulation
    st.subheader("Download Current Simulation")
    
    if st.session_state.sim_results is not None:
        sim_results = st.session_state.sim_results
        
        # Create DataFrame
        sim_df = pd.DataFrame({
            'year': df_raw.index.values,
            'prey_data': df_raw['prey'].values,
            'predator_data': df_raw['predator'].values,
            'prey_sim_scaled': sim_results[:, 0],
            'predator_sim_scaled': sim_results[:, 1],
            'prey_sim_real': sim_results[:, 0] * scale_factor,
            'predator_sim_real': sim_results[:, 1] * scale_factor
        })
        
        csv_sim = sim_df.to_csv(index=False)
        st.download_button(
            label="Download Simulation Results (CSV)",
            data=csv_sim,
            file_name="simulation_results.csv",
            mime="text/csv"
        )
    else:
        st.info("Run a simulation first to download results.")
