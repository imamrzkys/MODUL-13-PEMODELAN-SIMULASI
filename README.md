# 🔬 Predator-Prey Analysis Lab

Aplikasi Streamlit interaktif untuk analisis dinamika predator-prey menggunakan model Lotka-Volterra. Aplikasi ini dirancang untuk analisis akademis dan penelitian dengan data populasi Moose dan Wolves di Isle Royale National Park (1980-2019).

## ✨ Features

### Multi-Page Interface
Aplikasi ini memiliki 6 halaman utama dengan navigasi sidebar:

1. **📊 Dashboard** - Ringkasan data, metrik utama, dan preview dataset
2. **🔍 Data Exploration (EDA)** - Analisis eksplorasi data dengan visualisasi dan smoothing
3. **⚙️ Lotka-Volterra Simulator** - Simulator interaktif dengan kontrol parameter manual
4. **🎯 Parameter Fitting (Auto Tuning)** - Optimasi otomatis menggunakan Differential Evolution
5. **🌀 Phase Portrait & Dynamics** - Visualisasi phase portrait dan 3D trajectory
6. **📦 Export & Gallery** - Galeri visualisasi dan export report pack

### Key Capabilities

- **Interactive Parameter Control**: Slider dan input numerik untuk fine-tuning parameter model
- **Automatic Parameter Fitting**: Differential Evolution untuk optimasi parameter optimal
- **Comprehensive Visualizations**: 
  - Raw time series
  - Oscillation patterns
  - Smoothing trends
  - Overlay comparisons (before/after tuning)
  - Phase portraits with optional vector fields
  - 3D trajectories (Plotly interactive or Matplotlib fallback)
  - Residual plots
- **Export Functionality**: 
  - Download individual plots
  - Export complete report pack (ZIP) dengan semua PNG, JSON parameters, metrics, dan CSV
  - Download simulation results as CSV
- **Production-Ready**: 
  - Error handling dan validation
  - Caching untuk performa
  - Consistent plot naming untuk report
  - Indonesian/English mixed language interface

## 📁 Project Structure

```
.
├── app.py                  # Main Streamlit application entrypoint
├── wolf_moose_nps.csv      # Dataset (Moose & Wolves, 1980-2019)
├── requirements.txt        # Python dependencies
├── README.md               # This file
├── TA13.ipynb             # Original Jupyter notebook (reference)
├── src/
│   ├── __init__.py
│   ├── data.py             # Data loading and processing functions
│   ├── models.py           # Lotka-Volterra simulation and fitting logic
│   ├── plots.py            # All visualization functions
│   └── utils.py            # Utility functions and configuration
└── visualizations/         # Directory for saved plots (auto-generated)
    ├── 01_data_asli.png
    ├── 02_pola_osilasi.png
    ├── 03_smoothing_tren.png
    ├── 04_overlay_awal.png
    ├── 05_overlay_final.png
    ├── 06_phase_portrait.png
    ├── 07_3d_trajectory.png
    └── 08_overlay_skala_asli.png
```

## 🚀 How to Run Locally

### Prerequisites

- Python 3.9 atau lebih tinggi
- Virtual environment tool (venv, conda, dll)

### Setup Instructions

1. **Clone atau download repository ini**

2. **Buat dan aktifkan virtual environment**

   **Windows:**
   ```bash
   python -m venv venv
   .\venv\Scripts\activate
   ```

   **macOS / Linux:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Pastikan file dataset ada**
   - File `wolf_moose_nps.csv` harus berada di root directory (sama dengan `app.py`)
   - Format CSV: `Year,Wolves,Moose`

5. **Jalankan aplikasi Streamlit**
   ```bash
   streamlit run app.py
   ```

   Aplikasi akan otomatis terbuka di browser default Anda di `http://localhost:8501`

## 📖 Usage Guide

### 1. Dashboard
- Lihat ringkasan data dan metrik utama
- Preview dataset (top/bottom 5 rows)
- Download filtered dataset

### 2. Data Exploration
- Visualisasi data asli
- Analisis pola osilasi dengan min-max scaling
- Smoothing dengan moving average (adjustable window)
- Interpretasi dan penjelasan setiap plot

### 3. Simulator
- Atur parameter model (α, β, δ, γ) dengan slider atau input numerik
- Set initial conditions (x₀, y₀)
- Pilih solver method (RK45, RK23, DOP853, Radau, BDF)
- Lihat overlay simulasi vs data
- Lihat equilibrium point dan metrics

### 4. Parameter Fitting
- Set bounds untuk setiap parameter
- Atur weights untuk loss function (w_prey, w_pred)
- Konfigurasi Differential Evolution (maxiter, popsize, seed)
- Jalankan optimasi dan lihat progress
- Hasil: best parameters, overlay plot, residuals, metrics (scaled & real-scale)

### 5. Phase Portrait & Dynamics
- Phase portrait: data vs simulation
- Optional vector field visualization
- 3D trajectory (Plotly interactive atau Matplotlib fallback)
- Interpretasi limit cycle vs real data

### 6. Export & Gallery
- Gallery semua plot yang tersimpan
- Download individual plots
- Generate dan download complete report pack (ZIP)
- Download simulation results sebagai CSV

## 🎯 Model: Lotka-Volterra

Model Lotka-Volterra menggambarkan interaksi predator-prey:

```
dx/dt = αx - βxy  (Prey)
dy/dt = δxy - γy  (Predator)
```

**Parameter:**
- **α (alpha)**: Prey growth rate
- **β (beta)**: Predation rate
- **δ (delta)**: Predator growth efficiency
- **γ (gamma)**: Predator death rate

**Equilibrium Point:** (γ/δ, α/β)

## 🔧 Configuration

Default configuration dapat diubah di `src/utils.py`:

- DE bounds untuk parameter fitting
- Default weights (w_prey, w_pred)
- Solver settings (rtol, atol)
- DE settings (maxiter, popsize, seed)
- Visualization directory

## 🐛 Troubleshooting

### FileNotFoundError untuk `wolf_moose_nps.csv`
- Pastikan file CSV berada di root directory (sama dengan `app.py`)
- Periksa nama file (case-sensitive)

### Plotly 3D plot tidak muncul
- Aplikasi akan otomatis fallback ke Matplotlib 3D jika Plotly tidak tersedia
- Pastikan Plotly terinstall: `pip install plotly`
- Di Streamlit, Plotly menggunakan `st.plotly_chart()`, bukan `fig.show()`

### Error saat fitting parameter
- Periksa bounds parameter (pastikan min < max)
- Pastikan initial conditions valid (non-negative)
- Coba kurangi maxiter atau popsize jika terlalu lama

### Import errors
- Pastikan semua dependencies terinstall: `pip install -r requirements.txt`
- Pastikan virtual environment aktif
- Periksa bahwa struktur folder `src/` lengkap dengan `__init__.py`

### Plot tidak tersimpan
- Directory `visualizations/` akan dibuat otomatis
- Pastikan ada permission write di directory tersebut

## 📊 Output Files

Aplikasi menghasilkan 8 plot standar untuk report:

1. `01_data_asli.png` - Data asli
2. `02_pola_osilasi.png` - Pola osilasi (min-max scaled)
3. `03_smoothing_tren.png` - Smoothing trend
4. `04_overlay_awal.png` - Overlay sebelum tuning
5. `05_overlay_final.png` - Overlay setelah tuning
6. `06_phase_portrait.png` - Phase portrait
7. `07_3d_trajectory.png` - 3D trajectory
8. `08_overlay_skala_asli.png` - Overlay pada skala asli

Report pack ZIP berisi:
- Semua PNG files
- `parameters.json` - Best-fit parameters
- `metrics.json` - Performance metrics (scaled & real-scale)
- `filtered_data.csv` - Dataset yang digunakan

## 📝 Notes

- Model logic tetap sama dengan notebook asli (`TA13.ipynb`)
- Aplikasi menggunakan bahasa Indonesia dan Inggris campuran sesuai kebutuhan
- Semua plot memiliki caption sesuai format report akademis
- Plotly 3D menggunakan `st.plotly_chart()` untuk kompatibilitas Streamlit
- Aplikasi menggunakan caching untuk performa yang lebih baik

## 📄 License

Project ini untuk keperluan akademis dan penelitian.

## 👤 Author

Dikembangkan untuk TUGAS 13 - Praktikum Pemodelan.

---

**Selamat menggunakan Predator-Prey Analysis Lab! 🔬**
