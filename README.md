# Analisis Predator-Prey: Moose dan Serigala (Isle Royale)

Ini adalah aplikasi web Streamlit untuk menganalisis, memvisualisasikan, dan memodelkan dinamika populasi antara Moose (mangsa) dan Serigala (predator) di Isle Royale, berdasarkan dataset dari National Park Service (1980–2019).

## Fitur

- **Visualisasi Interaktif**: Berbagai jenis plot untuk menganalisis data dari berbagai sudut, termasuk deret waktu, diagram fase, dan dekomposisi pola.
- **Simulasi Lotka-Volterra**: Jalankan simulasi model predator-prey klasik dengan parameter yang dapat disesuaikan secara real-time.
- **Fitting Parameter**: Temukan parameter model (α, β, δ, γ) yang paling cocok dengan data historis secara otomatis.
- **UI Modern**: Antarmuka pengguna yang bersih dengan tema gelap, dibangun dengan Streamlit.

## Teknologi

- Python 3.11+
- Streamlit
- Pandas
- NumPy
- Matplotlib
- SciPy
- Statsmodels

---

## Menjalankan Aplikasi Secara Lokal

1.  **Clone Repositori (jika ada)**

    ```bash
    git clone <URL_REPO_ANDA>
    cd <NAMA_DIREKTORI>
    ```

2.  **Buat dan Aktifkan Virtual Environment** (direkomendasikan)

    ```bash
    # Windows
    python -m venv venv
    .\venv\Scripts\activate

    # macOS / Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install Dependensi**

    Pastikan Anda sudah berada di direktori utama proyek, lalu jalankan:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Jalankan Aplikasi Streamlit**

    ```bash
    streamlit run app.py
    ```

    Aplikasi akan terbuka secara otomatis di browser Anda.

---

## Deploy ke Railway

Railway adalah platform yang memudahkan proses deploy aplikasi. Berikut cara deploy aplikasi ini:

1.  **Siapkan Akun Railway**

    - Buat akun di [railway.app](https://railway.app).
    - Hubungkan akun GitHub Anda.

2.  **Push Kode ke Repositori GitHub**

    Pastikan semua file (`app.py`, `requirements.txt`, `.streamlit/`, `src/`, `README.md`) sudah di-push ke repositori GitHub Anda.

3.  **Buat Proyek Baru di Railway**

    - Di dashboard Railway, klik **"New Project"**.
    - Pilih **"Deploy from GitHub repo"**.
    - Pilih repositori GitHub yang berisi aplikasi ini.

4.  **Konfigurasi Build & Start Command**

    Railway biasanya akan mendeteksi `requirements.txt` dan menginstal dependensi secara otomatis. Namun, Anda perlu mengatur **Start Command**.

    - Buka tab **"Settings"** di proyek Railway Anda.
    - Cari bagian **"Deploy"**.
    - Di kolom **"Start Command"**, masukkan perintah berikut:

      ```bash
      streamlit run app.py --server.port $PORT --server.address 0.0.0.0
      ```

    - **Penting:** Variabel `$PORT` disediakan secara otomatis oleh Railway. Perintah ini memastikan Streamlit berjalan di port yang benar yang diekspos oleh Railway.

5.  **Deploy**

    Railway akan memulai proses build dan deploy secara otomatis. Setelah selesai, Anda akan mendapatkan URL publik untuk mengakses aplikasi Anda (misalnya: `nama-proyek.up.railway.app`).
