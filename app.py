import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from datetime import timedelta
import pyswarms as ps
from io import BytesIO
from sklearn.tree import export_text
import time
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ===================== KONFIGURASI HALAMAN =====================
st.set_page_config(
    page_title="Klasifikasi Durasi Rawat Inap Pasien Skizofrenia",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🏥"
)

# ===================== STYLING CUSTOM =====================
st.markdown("""
<style>
    /* Global Styles */
    .main {
        background: linear-gradient(135deg, #e6f7ff 0%, #f0f8ff 100%);
        padding: 0;
    }
    
    .stApp {
        background: linear-gradient(135deg, #e6f7ff 0%, #f0f8ff 100%);
    }
    
    /* Header Styles */
    .main-header {
        background: linear-gradient(90deg, #2b5876 0%, #4b86b4 100%);
        padding: 2rem;
        border-radius: 0 0 20px 20px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }
    
    .main-title {
        font-size: 2.8rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        color: white;
    }
    
    .main-subtitle {
        font-size: 1.2rem;
        color: rgba(255, 255, 255, 0.9);
        margin-bottom: 1rem;
    }
    
    /* Card Styles */
    .custom-card {
        background: rgba(255, 255, 255, 0.9);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 1.5rem;
        box-shadow: 0 8px 32px rgba(31, 38, 135, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.5);
        margin-bottom: 1.5rem;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .custom-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px rgba(31, 38, 135, 0.2);
    }
    
    /* Button Styles */
    .stButton>button {
        background: linear-gradient(90deg, #4b86b4 0%, #63a4ff 100%);
        color: white;
        font-weight: 600;
        border-radius: 10px;
        padding: 0.7rem 1.5rem;
        border: none;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(75, 134, 180, 0.3);
    }
    
    .stButton>button:hover {
        background: linear-gradient(90deg, #2b5876 0%, #4b86b4 100%);
        transform: scale(1.05);
        box-shadow: 0 6px 20px rgba(43, 88, 118, 0.4);
    }
    
    /* Sidebar Styles */
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #2b5876 0%, #4b86b4 100%);
        color: white;
        padding: 2rem 1rem;
    }
    
    .sidebar-title {
        font-size: 1.5rem;
        font-weight: 600;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
    }
    
    /* Tab Styles */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: rgba(255, 255, 255, 0.5);
        border-radius: 10px 10px 0 0;
        padding: 1rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #4b86b4 0%, #63a4ff 100%);
        color: white;
    }
    
    /* Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, #4b86b4 0%, #63a4ff 100%);
        border-radius: 15px;
        padding: 1.5rem;
        color: white;
        text-align: center;
        box-shadow: 0 8px 20px rgba(75, 134, 180, 0.3);
    }
    
    .metric-title {
        font-size: 1rem;
        font-weight: 500;
        margin-bottom: 0.5rem;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
    }
    
    /* Animation Keyframes */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .fade-in {
        animation: fadeIn 1s ease forwards;
    }
    
    /* Progress Bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #4b86b4 0%, #63a4ff 100%);
    }
    
    /* File Uploader */
    .stFileUploader > div > div {
        border: 2px dashed #4b86b4;
        border-radius: 10px;
        background: rgba(255, 255, 255, 0.5);
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: linear-gradient(90deg, rgba(75, 134, 180, 0.1) 0%, rgba(99, 164, 255, 0.1) 100%);
        border-radius: 10px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ===================== JUDUL & HEADER =====================
st.markdown("""
<div class="main-header fade-in">
    <h1 class="main-title">🏥 Klasifikasi Durasi Rawat Inap Pasien Skizofrenia</h1>
    <p class="main-subtitle">Analisis Canggih dengan Algoritma C4.5 dan Particle Swarm Optimization (PSO)</p>
</div>
""", unsafe_allow_html=True)

# ===================== FUNGSI MAPE =====================
def calculate_mape(y_true, y_pred):
    """
    Menghitung Mean Absolute Percentage Error (MAPE)
    Rumus: (1/n) * Σ(|y_true - y_pred| / y_true) * 100%
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    # Hindari pembagian 0 dengan mengganti nilai 0 kecil (epsilon)
    epsilon = 1e-10  
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100
    return mape

# ===================== FUNGSI PRA_PROSES DATA =====================
@st.cache_data(show_spinner="Memproses data...")
def preprocess_data(df_raw):
    df = df_raw.copy()

    # 1. Pembersihan Kolom Tidak Relevan
    kolom_hapus = ['No', 'Nomor', 'ID'] 
    for kolom in kolom_hapus:
        if kolom in df.columns:
            df.drop(columns=[kolom], inplace=True)

    # 2. Konversi Tanggal
    if 'Tanggal Masuk' in df.columns and 'Tanggal Keluar' in df.columns:
        df['Tanggal Masuk'] = pd.to_datetime(df['Tanggal Masuk'])
        df['Tanggal Keluar'] = pd.to_datetime(df['Tanggal Keluar'])
        df['Durasi Rawat Inap (Hari)'] = (df['Tanggal Keluar'] - df['Tanggal Masuk']).dt.days
        df = df[df['Durasi Rawat Inap (Hari)'] >= 0]
    else:
        st.error("❌ Kolom tanggal tidak ditemukan. Pastikan ada kolom 'Tanggal Masuk' dan 'Tanggal Keluar'.")
        st.stop()

    # 3. Menangani Data Kosong
    df.dropna(inplace=True)
    
    # 4. Klasifikasi Durasi
    def classify_duration(days):
        if days <= 5: return 'Singkat'
        elif 6 <= days <= 10: return 'Sedang'
        else: return 'Lama'
    
    df['Kategori Durasi'] = df['Durasi Rawat Inap (Hari)'].apply(classify_duration)

    # 5. Encoding
    if 'Diagnosa' in df.columns:
        if 'Skizofrenia' in df['Diagnosa'].unique():
            df = df[df['Diagnosa'].str.contains('Skizofrenia', case=False, na=False)]
        df.drop(columns=['Diagnosa'], inplace=True)

    le = LabelEncoder()
    df['Kategori Durasi Encoded'] = le.fit_transform(df['Kategori Durasi'])
    
    # Inisialisasi label_encoder di session state
    if "label_encoder" not in st.session_state:
        st.session_state.label_encoder = LabelEncoder()
    df['Kategori Durasi Encoded'] = st.session_state.label_encoder.fit_transform(df['Kategori Durasi'])    

    return df

@st.cache_resource(show_spinner="Melatih model C4.5 dan mengoptimalkan dengan PSO...")
def train_models(X_train, X_test, y_train, y_test):
    # 1. Model dasar (baseline)
    dt_base = DecisionTreeClassifier(random_state=42)
    dt_base.fit(X_train, y_train)
    y_pred_base = dt_base.predict(X_test)

    # 2. Hitung akurasi dasar
    base_accuracy = accuracy_score(y_test, y_pred_base)

    # 3. Hitung MAPE untuk model dasar
    base_mape = calculate_mape(y_test, y_pred_base)

    # 4. Fungsi untuk optimasi PSO
    def f_objective(params):
        n_particles = params.shape[0]
        fitness = []
        for i in range(n_particles):
            max_depth = int(params[i, 0])          # Parameter 1: max_depth
            min_samples = int(params[i, 1])        # Parameter 2: min_samples_split
            
            # Batasan nilai parameter
            max_depth = max(1, min(max_depth, 20))  # Pastikan 1 <= max_depth <= 20
            min_samples = max(2, min(min_samples, 10))  # Pastikan 2 <= min_samples <= 10

            model = DecisionTreeClassifier(
                max_depth=max_depth,
                min_samples_split=min_samples,
                criterion='entropy',
                random_state=42
            )
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            fitness.append(-accuracy)  # Minimalkan negatif akurasi
        
        return np.array(fitness)
    
    # 5. Konfigurasi PSO
    bounds = (np.array([1, 2]), np.array([20, 10]))  # Batasan parameter
    options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
    optimizer = ps.single.GlobalBestPSO(n_particles=10, dimensions=2, options=options, bounds=bounds)

    # 6. Jalankan optimasi
    cost, pos = optimizer.optimize(f_objective, iters=50, verbose=False)

    # 7. Tetapkan nilai optimal
    best_max_depth = int(pos[0])        # Parameter optimal 1
    best_min_samples = int(pos[1])      # Parameter optimal 2

    # 8. Bangun Model dengan Parameter Optimal
    dt_optimized = DecisionTreeClassifier(
        max_depth=best_max_depth,
        min_samples_split=best_min_samples,
        criterion='entropy',
        random_state=42
    )
    dt_optimized.fit(X_train, y_train)
    y_pred_optim = dt_optimized.predict(X_test)

    # 9. Hitung akurasi model optimasi
    optim_accuracy = accuracy_score(y_test, y_pred_optim)

    # 10. Hitung MAPE untuk model optimasi
    optim_mape = calculate_mape(y_test, y_pred_optim)

    # 11. Kembalikan hasil
    return {
        "base_model": dt_base,
        "base_accuracy": base_accuracy,
        "base_mape": base_mape,
        "optimized_model": dt_optimized,
        "optim_accuracy": optim_accuracy,
        "optim_mape": optim_mape,
        "y_test": y_test,
        "y_pred_base": y_pred_base,
        "y_pred_optim": y_pred_optim,
        "best_params": {
            "max_depth": best_max_depth,
            "min_samples": best_min_samples
        }
    }

# ===================== SIDEBAR =====================
with st.sidebar:
    st.markdown('<h1 class="sidebar-title">📊 Menu Analisis</h1>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "📁 Unggah Data Pasien (Excel)", 
        type=["xlsx"],
        help="File harus mengandung kolom: Tanggal Masuk, Tanggal Keluar"
    )
    
    st.markdown("---")
    
    with st.expander("ℹ️ Panduan Penggunaan"):
        st.info("""
        1. Unggah file Excel dengan data pasien
        2. Data akan diproses secara otomatis
        3. Jelajahi berbagai tab untuk melihat analisis
        4. Gunakan fitur prediksi untuk kasus baru
        """)
    
    st.markdown("---")
    
    st.markdown("""
    <div style="padding: 1rem; background: rgba(255, 255, 255, 0.1); border-radius: 10px;">
        <h4 style="margin-bottom: 0.5rem;">📋 Informasi Dataset</h4>
        <p style="font-size: 0.9rem; margin-bottom: 0;">Pastikan dataset Anda mengandung:</p>
        <ul style="font-size: 0.8rem;">
            <li>Kolom Tanggal Masuk</li>
            <li>Kolom Tanggal Keluar</li>
            <li>Kolom Diagnosa</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("**Dikembangkan oleh:**")
    st.markdown("Putri Agustina Dewi")
    st.markdown("Teknik Informatika, Universitas Malikussaleh")
    st.markdown("© 2025")

# ===================== MAIN APP =====================
df_raw = None

if uploaded_file:
    try:
        with st.spinner('🔄 Memuat dan memproses data...'):
            progress_bar = st.progress(0)
            for i in range(100):
                time.sleep(0.01)
                progress_bar.progress(i + 1)
            
            df_raw = pd.read_excel(uploaded_file)
            st.sidebar.success("✅ File berhasil diunggah!")
            
            df_processed = preprocess_data(df_raw)
        
        # Verifikasi kolom
        required_cols = ['Durasi Rawat Inap (Hari)', 'Kategori Durasi Encoded']
        if not all(col in df_processed.columns for col in required_cols):
            st.error("❌ Format data tidak valid. Pastikan file memiliki kolom yang dibutuhkan.")
            st.stop()
            
        # Siapkan data training
        X = df_processed[['Durasi Rawat Inap (Hari)']]
        y = df_processed['Kategori Durasi Encoded']
        
        if len(df_processed) < 2:
            st.error("⚠ Data terlalu sedikit. Minimal diperlukan 2 sampel.")
            st.stop()
            
        if y.nunique() < 2:
            st.warning("⚠ Hanya ada 1 kategori durasi. Tidak bisa dilakukan klasifikasi.")
            st.stop()
            
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=0.2, 
            random_state=42, 
            stratify=y
        )
        
        with st.spinner("🧠 Melatih model dengan algoritma C4.5 dan PSO..."):
            model_results = train_models(X_train, X_test, y_train, y_test)
        
        # ===================== TABBED INTERFACE =====================
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Ikhtisar Data", 
            "📈 Perbandingan Model", 
            "🌳 Pohon Keputusan", 
            "🔍 Prediksi Baru", 
            "🏥 Estimasi Ruangan"
        ])
        
        with tab1:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown('<div class="custom-card"><h3>📋 Preview Data</h3>', unsafe_allow_html=True)
                st.dataframe(df_processed.head(), use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
            with col2:
                st.markdown('<div class="custom-card"><h3>📈 Statistik Data</h3>', unsafe_allow_html=True)
                st.metric("Total Sampel", len(df_processed))
                st.metric("Jumlah Fitur", len(df_processed.columns))
                st.metric("Rata-rata Durasi", f"{df_processed['Durasi Rawat Inap (Hari)'].mean():.1f} Hari")
                st.markdown('</div>', unsafe_allow_html=True)
            
            col3, col4 = st.columns(2)
            
            with col3:
                st.markdown('<div class="custom-card"><h3>📊 Distribusi Durasi</h3>', unsafe_allow_html=True)
                fig1 = px.histogram(
                    df_processed, 
                    x='Durasi Rawat Inap (Hari)',
                    nbins=15,
                    color_discrete_sequence=['#4b86b4']
                )
                fig1.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                )
                st.plotly_chart(fig1, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
            with col4:
                st.markdown('<div class="custom-card"><h3>🍩 Distribusi Kategori</h3>', unsafe_allow_html=True)
                kategori_count = df_processed['Kategori Durasi'].value_counts()
                fig2 = px.pie(
                    values=kategori_count.values,
                    names=kategori_count.index,
                    color_discrete_sequence=['#2b5876', '#4b86b4', '#63a4ff']
                )
                fig2.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                )
                st.plotly_chart(fig2, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="custom-card"><h3>📌 Informasi Data</h3>', unsafe_allow_html=True)
            col5, col6, col7 = st.columns(3)
            
            with col5:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Singkat (1-5 hari)</div>
                    <div class="metric-value">{len(df_processed[df_processed['Kategori Durasi'] == 'Singkat'])}</div>
                </div>
                """, unsafe_allow_html=True)
                
            with col6:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Sedang (6-10 hari)</div>
                    <div class="metric-value">{len(df_processed[df_processed['Kategori Durasi'] == 'Sedang'])}</div>
                </div>
                """, unsafe_allow_html=True)
                
            with col7:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Lama (>10 hari)</div>
                    <div class="metric-value">{len(df_processed[df_processed['Kategori Durasi'] == 'Lama'])}</div>
                </div>
                """, unsafe_allow_html=True)
                
            st.markdown('</div>', unsafe_allow_html=True)
        
        with tab2:
            st.markdown('<div class="custom-card"><h3>⚙️ Parameter Model Optimal</h3>', unsafe_allow_html=True)
            col8, col9 = st.columns(2)
            
            with col8:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Max Depth</div>
                    <div class="metric-value">{model_results['best_params']['max_depth']}</div>
                </div>
                """, unsafe_allow_html=True)
                
            with col9:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Min Samples Split</div>
                    <div class="metric-value">{model_results['best_params']['min_samples']}</div>
                </div>
                """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            col10, col11 = st.columns(2)
            
            with col10:
                st.markdown('<div class="custom-card"><h3>📊 Model C4.5 Dasar</h3>', unsafe_allow_html=True)
                st.metric("Akurasi", f"{model_results['base_accuracy']:.2%}", delta=None)
                st.metric("MAPE", f"{model_results['base_mape']:.2f}%", delta=None)
                
                # Visualisasi akurasi
                fig_base = go.Figure(go.Indicator(
                    mode = "gauge+number+delta",
                    value = model_results['base_accuracy'] * 100,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Akurasi Model Dasar (%)"},
                    gauge = {
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "#4b86b4"},
                        'steps': [
                            {'range': [0, 70], 'color': "lightgray"},
                            {'range': [70, 90], 'color': "gray"},
                            {'range': [90, 100], 'color': "lightblue"}
                        ],
                    }
                ))
                fig_base.update_layout(height=300)
                st.plotly_chart(fig_base, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
            with col11:
                st.markdown('<div class="custom-card"><h3>🚀 Model C4.5+PSO</h3>', unsafe_allow_html=True)
                improvement = model_results['optim_accuracy'] - model_results['base_accuracy']
                st.metric("Akurasi", f"{model_results['optim_accuracy']:.2%}", 
                         delta=f"{improvement:.2%}" if improvement > 0 else None)
                st.metric("MAPE", f"{model_results['optim_mape']:.2f}%", 
                         delta=f"{- (model_results['optim_mape'] - model_results['base_mape']):.2f}%" 
                         if model_results['optim_mape'] < model_results['base_mape'] else None)
                
                # Visualisasi akurasi
                fig_optim = go.Figure(go.Indicator(
                    mode = "gauge+number+delta",
                    value = model_results['optim_accuracy'] * 100,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Akurasi Model Optimasi (%)"},
                    gauge = {
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "#2b5876"},
                        'steps': [
                            {'range': [0, 70], 'color': "lightgray"},
                            {'range': [70, 90], 'color': "gray"},
                            {'range': [90, 100], 'color': "lightblue"}
                        ],
                    }
                ))
                fig_optim.update_layout(height=300)
                st.plotly_chart(fig_optim, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Laporan klasifikasi
            st.markdown('<div class="custom-card"><h3>📋 Laporan Klasifikasi</h3>', unsafe_allow_html=True)
            col12, col13 = st.columns(2)
            
            with col12:
                st.markdown("**Model Dasar**")
                st.code(
                    classification_report(
                        model_results['y_test'], 
                        model_results['y_pred_base'],
                        target_names=['Singkat', 'Sedang', 'Lama']
                    )
                )
                
            with col13:
                st.markdown("**Model Optimasi**")
                st.code(
                    classification_report(
                        model_results['y_test'], 
                        model_results['y_pred_optim'],
                        target_names=['Singkat', 'Sedang', 'Lama']
                    )
                )
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Download hasil prediksi
            output_df = X_test.copy()
            if "label_encoder" in st.session_state:
                output_df['Aktual'] = st.session_state['label_encoder'].inverse_transform(model_results['y_test'])
                output_df['Prediksi (Dasar)'] = st.session_state['label_encoder'].inverse_transform(model_results['y_pred_base'])
                output_df['Prediksi (Optimasi)'] = st.session_state['label_encoder'].inverse_transform(model_results['y_pred_optim'])
            else:
                st.error("Label encoder belum diinisialisasi.")
                st.stop()
            
            excel_buffer = BytesIO()
            output_df.to_excel(excel_buffer, index=False)
            
            st.markdown('<div class="custom-card">', unsafe_allow_html=True)
            st.download_button(
                label="📥 Unduh Hasil Prediksi (Excel)",
                data=excel_buffer,
                file_name="prediksi.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
            st.markdown('</div>', unsafe_allow_html=True)
        
        with tab3:
            st.markdown('<div class="custom-card"><h3>🌳 Visualisasi Pohon Keputusan</h3>', unsafe_allow_html=True)
            
            with st.expander("ℹ️ Tentang visualisasi ini"):
                st.write("""
                Pohon ini menunjukkan proses pengambilan keputusan model C4.5 yang dioptimasi.
                Setiap node menunjukkan kondisi pembagian berdasarkan durasi rawat inap.
                """)
            
            # Visualisasi pohon keputusan
            plt.figure(figsize=(20, 12))
            plot_tree(
                model_results['optimized_model'],
                feature_names=['Durasi (hari)'],
                class_names=['Singkat', 'Sedang', 'Lama'],
                filled=True,
                rounded=True,
                fontsize=10,
                max_depth=3,  # Menunjukkan 3 level pertama untuk kejelasan
                impurity=False,
                proportion=True
            )
            st.pyplot(plt)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Representasi teks
            st.markdown('<div class="custom-card"><h3>📝 Representasi Teks</h3>', unsafe_allow_html=True)
            tree_text = export_text(
                model_results['optimized_model'],
                feature_names=['Durasi'],
                max_depth=3
            )
            st.code(tree_text)
            
            st.download_button(
                "📥 Unduh Struktur Pohon (TXT)",
                tree_text,
                file_name="struktur_pohon.txt"
            )
            st.markdown('</div>', unsafe_allow_html=True)
        
        with tab4:
            st.markdown('<div class="custom-card"><h3>🔮 Prediksi Pasien Baru</h3>', unsafe_allow_html=True)
            st.markdown("""
            <div style='color:#666; margin-bottom:20px;'>
                 Masukkan tanggal masuk dan keluar pasien untuk memprediksi kategori durasi rawat inap.
            </div>
            """, unsafe_allow_html=True)

            with st.form("form_prediksi"):
                col14, col15 = st.columns(2)
                with col14:
                    tgl_masuk = st.date_input("Tanggal Masuk", key="tgl_masuk")
                with col15:
                    tgl_keluar = st.date_input("Tanggal Keluar", 
                                                value=tgl_masuk + timedelta(days=7),
                                                key="tgl_keluar")

                submit_button = st.form_submit_button("📊 Prediksi Kategori Durasi")

            if submit_button:
                durasi = (tgl_keluar - tgl_masuk).days
        
                if durasi < 0:
                    st.error("⛔ Tanggal keluar tidak boleh sebelum tanggal masuk!")
                else:
                    if 'label_encoder' in st.session_state and 'optimized_model' in model_results:
                        try:
                            # Prediksi kategori
                            prediksi = model_results['optimized_model'].predict([[durasi]])[0]
                            kategori = st.session_state['label_encoder'].inverse_transform([prediksi])[0]
                    
                            # Animasi hasil prediksi
                            with st.spinner('Memprediksi...'):
                                time.sleep(1)
                            
                            st.success(f"✅ Kategori Durasi Prediksi: **{kategori}**")
                    
                            # Visualisasi hasil prediksi
                            col16, col17 = st.columns([1, 2])
                            
                            with col16:
                                # Tampilkan indikator kategori
                                if kategori == 'Singkat':
                                    color = "#4b86b4"
                                    icon = "⏱️"
                                elif kategori == 'Sedang':
                                    color = "#2b5876"
                                    icon = "⏳"
                                else:
                                    color = "#1e3c5a"
                                    icon = "⌛"
                                
                                fig_pred = go.Figure(go.Indicator(
                                    mode = "number+delta",
                                    value = durasi,
                                    number = {'font': {'size': 40}, 'prefix': icon},
                                    title = {'text': "Durasi (Hari)", 'font': {'size': 20}},
                                    delta = {'reference': 5, 'position': "bottom"},
                                    domain = {'x': [0, 1], 'y': [0, 1]}
                                ))
                                
                                fig_pred.update_layout(
                                    height=300,
                                    paper_bgcolor=color,
                                    font={'color': "white"}
                                )
                                st.plotly_chart(fig_pred, use_container_width=True)
                            
                            with col17:
                                # Rekomendasi
                                st.markdown(f"""
                                <div style="background: linear-gradient(135deg, {color} 0%, {color}dd 100%); 
                                            padding: 1.5rem; border-radius: 15px; color: white;">
                                    <h3>📋 Rekomendasi Manajemen Ruangan</h3>
                                    <p><strong>Kategori: {kategori}</strong></p>
                                """, unsafe_allow_html=True)
                                
                                if kategori == 'Singkat':
                                    st.markdown("""
                                    - *Ruangan*: Gunakan ruangan dengan turnover cepat
                                    - *Perawatan*: Persiapan discharge mulai hari ke-3
                                    - *Sumber Daya*: 1 perawat per 5 pasien
                                    """)
                                elif kategori == 'Sedang':
                                    st.markdown("""
                                    - *Ruangan*: Ruang perawatan standar
                                    - *Perawatan*: Evaluasi mingguan
                                    - *Sumber Daya*: 1 perawat per 3 pasien
                                    """)
                                else:
                                    st.markdown("""
                                    - *Ruangan*: Ruang perawatan jangka panjang
                                    - *Perawatan*: Evaluasi harian
                                    - *Sumber Daya*: 1 perawat per 2 pasien
                                    """)
                                
                                st.markdown("</div>", unsafe_allow_html=True)
                    
                        except Exception as e:
                            st.error(f"⚠ Error dalam prediksi: {str(e)}")
                    else:
                        st.error("🔴 Sistem belum siap! Pastikan:")
                        st.write("1. Data pasien sudah diunggah")
                        st.write("2. Model sudah selesai dilatih")
                        st.stop()
            st.markdown('</div>', unsafe_allow_html=True)

        with tab5:
            st.markdown('<div class="custom-card"><h3>🏥 Estimasi Kebutuhan Ruangan</h3>', unsafe_allow_html=True)
            
            # Hitung distribusi pasien
            distribusi = df_processed['Kategori Durasi'].value_counts().reset_index()
            distribusi.columns = ['Kategori', 'Jumlah Pasien']
            
            # Visualisasi distribusi
            fig_dist = px.bar(
                distribusi,
                x='Kategori',
                y='Jumlah Pasien',
                color='Kategori',
                color_discrete_sequence=['#2b5876', '#4b86b4', '#63a4ff'],
                text='Jumlah Pasien'
            )
            fig_dist.update_layout(
                title='Distribusi Pasien Berdasarkan Kategori Durasi',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
            )
            st.plotly_chart(fig_dist, use_container_width=True)
            
            # Rekomendasi alokasi ruangan
            st.markdown("""
            <div style="background: linear-gradient(135deg, #4b86b4 0%, #2b5876 100%); 
                        padding: 1.5rem; border-radius: 15px; color: white; margin-top: 1rem;">
                <h3>📊 Rekomendasi Alokasi Ruangan</h3>
                <p>Berdasarkan data historis:</p>
            """, unsafe_allow_html=True)
            
            col18, col19, col20 = st.columns(3)
            
            with col18:
                singkat = distribusi[distribusi['Kategori'] == 'Singkat']['Jumlah Pasien'].values
                singkat_val = singkat[0] if len(singkat) > 0 else 0
                st.markdown(f"""
                <div style="text-align: center; padding: 1rem; background: rgba(255,255,255,0.2); border-radius: 10px;">
                    <h4>Singkat</h4>
                    <h2>{singkat_val}</h2>
                    <p>{singkat_val/len(df_processed)*100:.1f}% dari total</p>
                </div>
                """, unsafe_allow_html=True)
                
            with col19:
                sedang = distribusi[distribusi['Kategori'] == 'Sedang']['Jumlah Pasien'].values
                sedang_val = sedang[0] if len(sedang) > 0 else 0
                st.markdown(f"""
                <div style="text-align: center; padding: 1rem; background: rgba(255,255,255,0.2); border-radius: 10px;">
                    <h4>Sedang</h4>
                    <h2>{sedang_val}</h2>
                    <p>{sedang_val/len(df_processed)*100:.1f}% dari total</p>
                </div>
                """, unsafe_allow_html=True)
                
            with col20:
                lama = distribusi[distribusi['Kategori'] == 'Lama']['Jumlah Pasien'].values
                lama_val = lama[0] if len(lama) > 0 else 0
                st.markdown(f"""
                <div style="text-align: center; padding: 1rem; background: rgba(255,255,255,0.2); border-radius: 10px;">
                    <h4>Lama</h4>
                    <h2>{lama_val}</h2>
                    <p>{lama_val/len(df_processed)*100:.1f}% dari total</p>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("""
                <p style="margin-top: 1.5rem;"><strong>📋 Saran Alokasi:</strong></p>
                <ul>
                    <li>Ruangan singkat: 5-10% dari total kapasitas</li>
                    <li>Ruangan sedang: 15-20% dari total kapasitas</li>
                    <li>Ruangan panjang: 30-40% dari total kapasitas</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

    except Exception as e:
        st.error(f"""
        ❌ Terjadi kesalahan:
        {str(e)}
        """)
        st.write("Pastikan format file sesuai dan data lengkap.")

else:
    # Tampilan awal sebelum upload file
    col21, col22, col23 = st.columns([1, 2, 1])
    
    with col22:
        st.markdown("""
        <div style="text-align: center; padding: 3rem; background: rgba(255, 255, 255, 0.8); 
                    border-radius: 20px; margin-top: 2rem; box-shadow: 0 8px 32px rgba(31, 38, 135, 0.1);">
            <h2 style="color: #2b5876;">Selamat Datang</h2>
            <p style="color: #5d707f; font-size: 1.1rem;">
                Silakan unggah file data pasien untuk memulai analisis klasifikasi durasi rawat inap
            </p>
            <div style="font-size: 4rem; margin: 1.5rem 0;">📊</div>
            <p style="color: #5d707f;">
                Gunakan menu di sidebar untuk mengunggah file Excel Anda
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Fitur animasi
        st.markdown("""
        <div style="text-align: center; margin-top: 2rem;">
            <lottie-player src="https://assets1.lottiefiles.com/packages/lf20_ukaaZq.json"  
                background="transparent" speed="1" style="width: 300px; height: 300px; margin: 0 auto;" 
                loop autoplay>
            </lottie-player>
        </div>
        <script src="https://unpkg.com/@lottiefiles/lottie-player@latest/dist/lottie-player.js"></script>
        """, unsafe_allow_html=True)
