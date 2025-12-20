"""
Main Streamlit application file for Predator-Prey Analysis.
"""

import streamlit as st
import pandas as pd
import numpy as np

from src.data import load_data, filter_data_by_year
from src.plots import (
    plot_time_series, plot_time_series_twinx, plot_normalized, 
    plot_phase_diagram, plot_3d_trajectory, plot_decomposition,
    plot_rolling_stats, run_adf_test
)
from src.models import run_simulation, fit_parameters, plot_simulation_vs_data, plot_fit_vs_data

# --- Konfigurasi Halaman ---
st.set_page_config(
    page_title="Analisis Predator-Prey",
    page_icon="🐺",
    layout="wide"
)

# --- Judul dan Subjudul ---
st.title("Analisis Predator–Prey: Moose dan Serigala di Isle Royale")
st.subheader("Dataset tahunan 1980–2019 dari National Park Service (NPS)")

# --- Memuat Data ---
df_full = load_data()

# --- Sidebar untuk Kontrol Pengguna ---
st.sidebar.header("Pengaturan Visualisasi")

# Slider Rentang Tahun
min_year, max_year = int(df_full.index.min()), int(df_full.index.max())
start_year, end_year = st.sidebar.slider(
    "Pilih rentang tahun:",
    min_value=min_year,
    max_value=max_year,
    value=(min_year, max_year)
)

# Filter data berdasarkan rentang tahun
df_filtered = filter_data_by_year(df_full, (start_year, end_year))

# Checklist Grafik
with st.sidebar.expander("Pilih Grafik untuk Ditampilkan", expanded=True):
    show_ts = st.checkbox("Grafik Deret Waktu (satu sumbu)", True)
    show_ts_twinx = st.checkbox("Grafik Dua Sumbu (Moose vs Serigala)")
    show_normalized = st.checkbox("Grafik Ternormalisasi (0–1)")
    show_phase = st.checkbox("Diagram Fase (Moose vs Serigala)")
    show_3d = st.checkbox("Grafik 3D (Tahun–Moose–Serigala)")
    show_decomp_moose = st.checkbox("Dekomposisi Pola (Moose)")
    show_decomp_wolves = st.checkbox("Dekomposisi Pola (Serigala)")
    show_rolling_moose = st.checkbox("Statistik Bergulir (Moose)")
    show_adf_moose = st.checkbox("Uji Stasioneritas ADF (Moose)")

# Pengaturan Tambahan
st.sidebar.header("Pengaturan Tambahan")
decomp_period = st.sidebar.slider("Periode Dekomposisi (tahun)", 2, 15, 10)
rolling_window = st.sidebar.slider("Window Statistik Bergulir (tahun)", 2, 15, 5)
st.sidebar.info(
    "**Catatan:** Data ini adalah data tahunan. Istilah 'seasonal' dalam dekomposisi "
    "merujuk pada siklus berulang dalam beberapa tahun, bukan musim kalender."
)

# --- Simulasi Lotka-Volterra ---
st.sidebar.header("Simulasi Lotka–Volterra")
alpha = st.sidebar.slider('α (Laju pertumbuhan Moose)', 0.0, 1.0, 0.55, 0.01)
beta = st.sidebar.slider('β (Tingkat predasi Serigala)', 0.0, 0.1, 0.025, 0.001)
delta = st.sidebar.slider('δ (Efisiensi predasi)', 0.0, 0.1, 0.015, 0.001)
gamma = st.sidebar.slider('γ (Laju kematian Serigala)', 0.0, 2.0, 0.9, 0.01)

run_sim_button = st.sidebar.button("Jalankan Simulasi")

# --- Fitting Parameter ---
st.sidebar.header("Fitting Parameter Model")
fit_params_button = st.sidebar.button("Cari Parameter Terbaik")
use_filtered_range = st.sidebar.checkbox("Gunakan rentang tahun yang dipilih untuk fitting")


# --- Tampilan Halaman Utama dengan Tabs ---
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Ringkasan Data", "Visualisasi", "Simulasi LV", "Fitting Model", "Catatan"])

with tab1:
    st.header("Sekilas Data Populasi")
    st.write(f"Menampilkan data dari tahun {start_year} sampai {end_year}.")
    st.dataframe(df_filtered)
    if st.checkbox("Tampilkan statistik ringkas"):
        st.write(df_filtered.describe())

with tab2:
    st.header("Galeri Visualisasi Data")
    st.write("Grafik yang dipilih di sidebar akan muncul di sini.")

    if show_ts:
        st.subheader("Grafik Deret Waktu")
        fig = plot_time_series(df_filtered)
        st.pyplot(fig)
        st.caption("Grafik ini menunjukkan naik turunnya populasi Moose dan Serigala dari waktu ke waktu. Terlihat adanya siklus di mana puncak populasi Moose diikuti oleh puncak populasi Serigala.")

    if show_ts_twinx:
        st.subheader("Grafik Deret Waktu (Dua Sumbu)")
        fig = plot_time_series_twinx(df_filtered)
        st.pyplot(fig)
        st.caption("Dengan dua sumbu Y, kita bisa membandingkan tren kedua populasi dengan lebih jelas, meskipun skala angkanya berbeda jauh.")

    if show_normalized:
        st.subheader("Grafik Populasi Ternormalisasi")
        fig = plot_normalized(df_filtered)
        st.pyplot(fig)
        st.caption("Normalisasi membawa kedua data ke rentang 0-1. Ini membantu melihat seberapa dekat suatu populasi ke titik minimum atau maksimum historisnya dalam rentang waktu yang dipilih.")

    if show_phase:
        st.subheader("Diagram Fase")
        fig = plot_phase_diagram(df_filtered)
        st.pyplot(fig)
        st.caption("Diagram ini memplot populasi Moose vs. Serigala. Pola siklus yang berputar menunjukkan hubungan predator-prey klasik. Arah putaran (biasanya berlawanan arah jarum jam) menunjukkan keterlambatan respon predator terhadap perubahan mangsa.")

    if show_3d:
        st.subheader("Trajektori 3D")
        fig = plot_3d_trajectory(df_filtered)
        st.pyplot(fig)
        st.caption("Visualisasi 3D ini menambahkan dimensi waktu ke diagram fase, menunjukkan bagaimana siklus predator-prey berkembang dari tahun ke tahun.")

    if show_decomp_moose:
        st.subheader("Dekomposisi Pola Populasi Moose")
        fig, msg = plot_decomposition(df_filtered, 'Moose', decomp_period)
        st.pyplot(fig)
        st.caption(f"Grafik ini memecah data Moose menjadi tren jangka panjang, pola siklus (musiman), dan sisa (residual). {msg}")

    if show_decomp_wolves:
        st.subheader("Dekomposisi Pola Populasi Serigala")
        fig, msg = plot_decomposition(df_filtered, 'Wolves', decomp_period)
        st.pyplot(fig)
        st.caption(f"Sama seperti Moose, dekomposisi ini membantu kita memahami komponen-komponen yang membentuk fluktuasi populasi Serigala. {msg}")

    if show_rolling_moose:
        st.subheader("Statistik Bergulir untuk Moose")
        fig = plot_rolling_stats(df_filtered, 'Moose', rolling_window)
        st.pyplot(fig)
        st.caption(f"Rata-rata bergulir menghaluskan fluktuasi jangka pendek untuk menunjukkan tren. Standar deviasi bergulir menunjukkan periode volatilitas populasi.")

    if show_adf_moose:
        st.subheader("Uji Stasioneritas (ADF) untuk Moose")
        adf_stat, p_value, interpretation = run_adf_test(df_filtered['Moose'])
        if np.isnan(p_value):
            st.warning(interpretation)
        else:
            st.metric("ADF Statistic", f"{adf_stat:.4f}")
            st.metric("P-value", f"{p_value:.4f}")
            st.info(f"**Interpretasi:** {interpretation}")
            st.caption("Uji ini memeriksa apakah sebuah deret waktu 'stasioner' (rata-rata dan variansnya konstan dari waktu ke waktu). Ini adalah konsep penting dalam pemodelan deret waktu yang lebih lanjut.")

with tab3:
    st.header("Simulasi Model Lotka-Volterra")
    st.markdown("""
        Model Lotka-Volterra adalah model matematika sederhana untuk menggambarkan dinamika populasi predator-prey. 
        Parameter-parameter ini mengontrol interaksi mereka:
        - **α (alpha):** Laju pertumbuhan alami mangsa (Moose) tanpa adanya predator.
        - **β (beta):** Tingkat keberhasilan predator (Serigala) dalam memangsa.
        - **δ (delta):** Seberapa efisien mangsa yang dimakan diubah menjadi populasi predator baru.
        - **γ (gamma):** Laju kematian alami predator tanpa adanya mangsa.
    """)

    if run_sim_button:
        params = {'alpha': alpha, 'beta': beta, 'delta': delta, 'gamma': gamma}
        initial_conditions = (df_filtered['Moose'].iloc[0], df_filtered['Wolves'].iloc[0])
        t_eval = np.arange(len(df_filtered))
        
        sim_df = run_simulation(params, initial_conditions, t_eval)
        sim_df.index = df_filtered.index

        fig = plot_simulation_vs_data(df_filtered, sim_df)
        st.pyplot(fig)
        st.caption("Grafik ini membandingkan data asli (garis tebal dengan titik) dengan hasil simulasi (garis putus-putus) menggunakan parameter yang kamu atur di sidebar. Coba ubah parameter untuk melihat bagaimana dinamika siklus berubah.")
    else:
        st.info("Atur parameter di sidebar dan klik 'Jalankan Simulasi' untuk melihat hasilnya.")

with tab4:
    st.header("Mencari Parameter Model yang Paling Pas (Fitting)")
    st.write("Di sini, kita akan menggunakan metode optimasi untuk secara otomatis menemukan nilai parameter (α, β, δ, γ) yang membuat hasil simulasi paling mirip dengan data historis.")

    if fit_params_button:
        data_to_fit = df_filtered if use_filtered_range else df_full
        with st.spinner("Sedang menghitung parameter terbaik... Ini mungkin butuh beberapa saat."):
            fitted_params, errors = fit_parameters(data_to_fit)

        st.success("Parameter terbaik berhasil ditemukan!")
        st.balloons()
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Parameter Hasil Fitting:")
            st.json(fitted_params)
        with col2:
            st.subheader("Error (RMSE):")
            st.metric("Error Moose", f"{errors['RMSE_Moose']:.2f}")
            st.metric("Error Serigala", f"{errors['RMSE_Wolves']:.2f}")

        st.subheader("Perbandingan Data Asli vs. Model Hasil Fitting")
        fig = plot_fit_vs_data(data_to_fit, fitted_params)
        st.pyplot(fig)
        st.caption("Grafik ini menunjukkan seberapa baik model Lotka-Volterra dengan parameter yang dioptimalkan dapat meniru data nyata. Ketidakcocokan menunjukkan bahwa ada faktor-faktor lain di dunia nyata (misalnya, cuaca, penyakit, ketersediaan makanan lain) yang tidak diperhitungkan oleh model sederhana ini.")
    else:
        st.info("Klik 'Cari Parameter Terbaik' di sidebar untuk memulai proses fitting.")

with tab5:
    st.header("Catatan dan Sumber Data")
    st.markdown("Aplikasi ini dibuat untuk tujuan edukasi dalam mata kuliah Pemodelan dan Simulasi.")
    st.markdown("**Sumber Data:**")
    st.code("National Park Service (NPS) - Wolf & Moose Populations on Isle Royale", language="text")
    st.markdown("Data yang digunakan adalah populasi tahunan Serigala dan Moose di Isle Royale dari tahun 1980 hingga 2019.")
