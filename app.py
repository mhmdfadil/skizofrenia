import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import hashlib
import time
from datetime import datetime
import psycopg2
from psycopg2 import sql
import os
from dotenv import load_dotenv
import asyncio

# Load environment variables
load_dotenv()

# Konfigurasi halaman
st.set_page_config(
    page_title="KLASIFIKASI DURASI RAWAT INAP PASIEN SKIZOFRENIA",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS untuk styling dengan background blue sky dan animasi
st.markdown("""
<style>
    /* Font Import */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;500;600;700&display=swap');
    
    /* Reset default Streamlit styles */
    .stApp {
        background: transparent !important;
    }
    
    .main .block-container {
        padding-top: 0;
        padding-bottom: 0;
        max-width: 1200px;
    }
    
    /* Global Styles */
    * {
        font-family: 'Poppins', sans-serif;
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }
    
    html, body, [data-testid="stAppViewContainer"] {
        height: 100%;
        overflow: auto;
        background: transparent !important;
    }
    
    /* Blue Sky Animated Background */
    .sky-background {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: linear-gradient(120deg, #a1c4fd 0%, #c2e9fb 50%, #fbc2eb 100%);


        z-index: -2;
        overflow: hidden;
    }

    
    .cloud {
        position: absolute;
        background: rgba(255, 255, 255, 0.9);
        border-radius: 50%;
        box-shadow: 0 0 30px rgba(255, 255, 255, 0.7);
        animation: float linear infinite;
        opacity: 0.8;
    }
    
    @keyframes float {
        0% {
            transform: translateX(-100px) translateY(0);
        }
        100% {
            transform: translateX(calc(100vw + 100px)) translateY(-20px);
        }
    }
    
    .main {
        position: relative;
        z-index: 1;
        min-height: 100vh;
    }
    
    /* Glass effect untuk login container */
    div[data-testid="stForm"] {
        background: rgba(255, 255, 255, 0.15);
        padding:40px;
        border-radius: 8px;
        margin: 20px 20px;
        margin-bottom:20px;
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.2);
        text-align: center;
    }

    
    /* Glassmorphism Effect */
    .glass {
        background: rgba(255, 255, 255, 0.2);
        backdrop-filter: blur(15px);
        -webkit-backdrop-filter: blur(15px);
        border-radius: 24px;
        border: 1px solid rgba(255, 255, 255, 0.25);
        box-shadow: 0 12px 40px 0 rgba(31, 38, 135, 0.2);
        padding: 40px;
        margin-bottom: 30px;
        transition: all 0.4s ease;
        overflow: hidden;
        position: relative;
    }
    
    .glass::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: linear-gradient(to bottom right, 
            rgba(255, 255, 255, 0.1), 
            rgba(255, 255, 255, 0.05));
        transform: rotate(30deg);
        z-index: -1;
    }
    
    .glass:hover {
        transform: translateY(-8px);
        box-shadow: 0 20px 50px rgba(31, 38, 135, 0.3);
    }
    
            
    
    
    /* Navigation Styles */
    .stTabs [data-baseweb="tab-list"] {
        gap: 16px;
        justify-content: end;
        background: transparent;
        padding: 0 20px;
        margin-bottom: 40px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 60px;
        white-space: pre;
        background: rgba(255, 255, 255, 0.39);
        border-radius: 16px;
        font-weight: 600;
        padding: 0 32px;
        color: #005;
        border: 1px solid rgba(255, 255, 255, 0.3);
        transition: all 0.3s ease;
        backdrop-filter: blur(10px);
        font-size: 1.1rem;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(255, 255, 255, 0.55);
        transform: translateY(-3px);
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
    }
    
    .stTabs [aria-selected="true"] {
        background: rgba(255, 255, 255, 0.9);
        color: #008080;
        box-shadow: 0 8px 20px rgba(13, 71, 161, 0.25);
        border: 1px solid rgba(255, 255, 255, 0.5);
    }
    
    /* Button Styles */
    .stButton button {
        background: linear-gradient(120deg, #a1c4fd 51%, #fbc2eb 100%);
        color: white;
        border: none;
        padding: 16px 40px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 18px;
        margin: 4px 2px;
        cursor: pointer;
        border-radius: 16px;
        transition: all 0.3s ease;
        font-weight: 600;
        box-shadow: 0 6px 15px rgba(13, 71, 161, 0.3);
        position: relative;
        overflow: hidden;
    }
    
    .stButton button::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(100deg, #a1c4fd 51%, #fbc2eb 100%);
        transition: 0.5s;
    }
    
    .stButton button:hover::before {
        left: 100%;
    }
    
    .stButton button:hover {
        background: linear-gradient(135deg, #a1c4fd 51%, #fbc2eb 100%);
        box-shadow: 0 10px 25px rgba(33, 150, 243, 0.4);
        transform: translateY(-3px);
    }
    
    .stButton button:active {
        transform: translateY(1px);
        box-shadow: 0 4px 10px rgba(33, 150, 243, 0.4);
    }
    
    /* Input Fields */
    .stTextInput>div>div>input {
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.4);
        padding: 16px;
        background: rgba(255, 255, 255, 0.85);
        transition: all 0.3s ease;
        font-size: 16px;
        color: #0d47a1;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.05);
    }
    
    .stTextInput>div>div>input:focus {
        background: rgba(255, 255, 255, 0.95);
        box-shadow: 0 0 0 3px rgba(33, 150, 243, 0.3);
        border: 1px solid rgba(33, 150, 243, 0.5);
    }
    
    .stTextInput>div>div>input::placeholder {
        color: #90a4ae;
    }
    
    /* Card Styles */
    .feature-card {
        background: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(12px);
        border-radius: 20px;
        padding: 30px;
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.1);
        height: 100%;
        transition: all 0.4s ease;
        border: 1px solid rgba(255, 255, 255, 0.2);
        text-align: center;
        position: relative;
        overflow: hidden;
    }
    
    .feature-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 5px;
        background: linear-gradient(90deg, #0d47a1, #1976d2);
    }
    
    .feature-card:hover {
        transform: translateY(-8px) scale(1.02);
        box-shadow: 0 15px 30px rgba(0, 0, 0, 0.15);
        background: rgba(255, 255, 255, 0.2);
    }
    
    .card-title {
        font-size: 1.5rem;
        font-weight: 700;
        color: #fff;
        margin-bottom: 20px;
        text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.2);
    }
    
    .card-content {
        font-size: 1.05rem;
        color: #e3f2fd;
        line-height: 1.6;
    }
    
    /* Success and Error Messages */
    .stAlert {
        border-radius: 16px;
        backdrop-filter: blur(10px);
        background: rgba(255, 255, 255, 0.2);
        border: 1px solid rgba(255, 255, 255, 0.3);
    }
    
    /* Footer */
   .footer {
            text-align: center;
            padding: 18px 10px;
            color: rgba(255, 255, 255, 0.9);
            margin-top: 40px;

            /* Glassmorphism Teal */
            background: linear-gradient(135deg, rgba(0,77,64,0.6), rgba(0,128,128,0.4));

            backdrop-filter: blur(12px) saturate(180%);
            -webkit-backdrop-filter: blur(12px) saturate(180%);
            border-radius: 20px;
            border-top: 1px solid rgba(255, 255, 255, 0.15);
            line-height: 0.5;

            /* Gradient halus ke teal gelap */
            background-image: linear-gradient(
                to right,
                rgba(0, 77, 77, 0.25),
                rgba(0, 128, 128, 0.15),
                rgba(0, 77, 77, 0.25)
            );

            /* Glow effect */
            box-shadow: 0 -4px 20px rgba(0, 77, 77, 0.3);
            text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.4);

            transition: all 0.3s ease-in-out;
        }

        .footer:hover {
            background: rgba(0, 128, 128, 0.2);
            background-image: linear-gradient(
                to right,
                rgba(0, 128, 128, 0.3),
                rgba(0, 128, 128, 0.2),
                rgba(0, 128, 128, 0.3)
            );
            box-shadow: 0 -6px 25px rgba(0, 77, 77, 0.5);
            transform: translateY(-2px);
        }

    
    /* Loading Animation */
    .loading-container {
        display: flex;
        justify-content: center;
        align-items: center;
        height: 100px;
    }
    
    .loading-spinner {
        width: 50px;
        height: 50px;
        border: 5px solid rgba(255, 255, 255, 0.3);
        border-radius: 50%;
        border-top: 5px solid #0d47a1;
        animation: spin 1s linear infinite;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    /* Login Form Specific Styles */
    .login-container {
        max-width: 500px;
        margin: 10px;
        padding: 50px 0;
    }
    
    .login-header {
        text-align: center;
        margin-bottom: 40px;
    }
    
    .login-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #004;
        margin-bottom: 10px;
        text-shadow: 1px 1px 4px rgba(0, 0, 0, 0.2);
    }
    
    .login-subtitle {
        font-size: 1.1rem;
        color: #e3f2fd;
        opacity: 0.9;
    }
    
    /* Responsive Design */
    @media (max-width: 768px) {
       
        
        .glass {
            padding: 25px;
            margin-bottom: 20px;
        }
        
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            padding: 0 20px;
            font-size: 1rem;
        }
        
        .feature-card {
            margin-bottom: 20px;
        }
    }
</style>
""", unsafe_allow_html=True)

# JavaScript untuk animasi awan dan efek interaktif
st.markdown("""
<script>
// Function to create clouds
function createClouds() {
    const sky = document.querySelector('.sky-background');
    if (!sky) return;
    
    // Clear existing clouds
    sky.innerHTML = '';
    
    // Create multiple clouds
    for (let i = 0; i < 7; i++) {
        const cloud = document.createElement('div');
        cloud.classList.add('cloud');
        
        // Random size and position
        const width = Math.random() * 120 + 80;
        const height = width * 0.5;
        const top = Math.random() * 60;
        const left = -width;
        const duration = Math.random() * 40 + 40;
        
        cloud.style.width = `${width}px`;
        cloud.style.height = `${height}px`;
        cloud.style.top = `${top}%`;
        cloud.style.left = `${left}px`;
        cloud.style.animationDuration = `${duration}s`;
        cloud.style.animationDelay = `${Math.random() * 20}s`;
        
        // Add some variation to cloud shape
        cloud.style.borderRadius = '50%';
        
        sky.appendChild(cloud);
    }
}

// Add ripple effect to buttons
function addRippleEffect() {
    const buttons = document.querySelectorAll('.stButton button');
    buttons.forEach(button => {
        button.addEventListener('click', function(e) {
            const x = e.pageX - this.offsetLeft;
            const y = e.pageY - this.offsetTop;
            
            const ripple = document.createElement('span');
            ripple.classList.add('ripple-effect');
            ripple.style.left = `${x}px`;
            ripple.style.top = `${y}px`;
            this.appendChild(ripple);
            
            setTimeout(() => {
                ripple.remove();
            }, 600);
        });
    });
}

// Run when page loads
window.addEventListener('load', function() {
    createClouds();
    addRippleEffect();
});

// Recreate clouds when page changes (for Streamlit)
document.addEventListener('DOMContentLoaded', function() {
    setTimeout(createClouds, 100);
    setTimeout(addRippleEffect, 500);
});
</script>
""", unsafe_allow_html=True)

# Add the sky background
st.markdown('<div class="sky-background"></div>', unsafe_allow_html=True)

# Fungsi hash password
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# Inisialisasi koneksi database
def init_db_connection():
    try:
        # Konfigurasi koneksi database
        conn = psycopg2.connect(
            user=os.getenv("ST_USER"),
            password=os.getenv("ST_PASSWORD"),
            host=os.getenv("ST_HOST"),
            port=os.getenv("ST_PORT"),
            dbname=os.getenv("ST_DATABASE")
        )
        
        return conn
    except Exception as e:
        st.error(f"❌ Gagal terhubung ke database: {str(e)}")
        return None

# Fungsi untuk membuat tabel jika belum ada
def create_tables(conn):
    try:
        cursor = conn.cursor()
        
        # Buat tabel users jika belum ada
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id SERIAL PRIMARY KEY,
                email VARCHAR(255) UNIQUE NOT NULL,
                password VARCHAR(255) NOT NULL,
                name VARCHAR(255) NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Buat tabel patients jika belum ada
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS patients (
                id SERIAL PRIMARY KEY,
                name VARCHAR(255) NOT NULL,
                age INTEGER NOT NULL,
                gender VARCHAR(10) NOT NULL,
                duration INTEGER NOT NULL,
                classification VARCHAR(50) NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        conn.commit()
        cursor.close()
        st.info("✅ Tabel database sudah siap")
    except Exception as e:
        st.error(f"❌ Gagal membuat tabel: {str(e)}")

# Fungsi untuk login dengan animasi loading
def login_user(email, password):
    try:
        hashed_password = hash_password(password)
        conn = init_db_connection()
        
        if conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id, email, password, name FROM users WHERE email = %s AND password = %s",
                (email, hashed_password)
            )
            user = cursor.fetchone()
            cursor.close()
            conn.close()
            
            if user:
                # Convert tuple to dictionary
                user_dict = {
                    "id": user[0],
                    "email": user[1],
                    "password": user[2],
                    "name": user[3]
                }
                return user_dict
            else:
                return None
        else:
            # Fallback ke data dummy jika koneksi database gagal
            dummy_users = [
                {"email": "admin@rsudmuyangkute.com", "password": hash_password("admin123"), "name": "Administrator"},
                {"email": "dokter@rsudmuyangkute.com", "password": hash_password("dokter123"), "name": "Dokter Spesialis"}
            ]
            
            for user in dummy_users:
                if user["email"] == email and user["password"] == hashed_password:
                    return user
            return None
    except Exception as e:
        st.error(f"Error logging in: {e}")
        return None

# Fungsi untuk menampilkan animasi loading
def show_loading_animation():
    st.markdown("""
    <div class="loading-container">
        <div class="loading-spinner"></div>
    </div>
    """, unsafe_allow_html=True)

# Fungsi untuk halaman login
def show_login_page():
    st.markdown("<div class='login-container'>", unsafe_allow_html=True)
    
    # Login header
    
    st.markdown("<h2 class='login-title' style='color:#006;'>🔐 Masuk ke Sistem</h2>", unsafe_allow_html=True)
    st.markdown("<p class='login-subtitle'>Silakan masukkan kredensial Anda untuk mengakses sistem</p>", unsafe_allow_html=True)
    


    with st.form("login_form"):
        (col1,) = st.columns([1])

        with col1:
            email = st.text_input("📧 Email", placeholder="Masukkan email Anda", key="login_email")
            password = st.text_input("🔒 Password", type="password", placeholder="Masukkan password Anda", key="login_password")
            
            submit_button = st.form_submit_button("🚀 Masuk", use_container_width=True)

            if submit_button:
                if email and password:
                    with st.spinner(""):
                        show_loading_animation()
                        time.sleep(1.5)
                        user = login_user(email, password)
                        if user:
                            st.session_state.user = user
                            st.session_state.page = "dashboard"
                            st.success("✅ Login berhasil! Mengalihkan...")
                            time.sleep(1.5)
                            st.rerun()
                        else:
                            st.error("❌ Email atau password salah")
                else:
                    st.warning("⚠️ Harap isi semua field")

    # Tutup div glass
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# Fungsi untuk halaman dashboard
def show_dashboard():
    st.markdown(f"<h2 style='text-align: center; color: #0d47a1;'>Selamat Datang, {st.session_state.user['name']}</h2>", unsafe_allow_html=True)
    st.markdown(f"<p style='text-align: center; color: #546e7a;'>{st.session_state.user['email']}</p>", unsafe_allow_html=True)
    
    # Logout button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🚪 Keluar", use_container_width=True):
            st.session_state.clear()
            st.success("Anda telah logout")
            time.sleep(1)
            st.rerun()
    
# Fungsi untuk halaman utama/beranda
def show_home_page():
    # Header section dengan efek paralaks
    st.markdown("""
    <div style="padding: 50px 0 40px 0; text-align: center; line-height:1.3;">
        <div style="font-size: 42px; font-weight: 800; color: #005;">
            KLASIFIKASI DURASI RAWAT INAP
        </div>
        <div style="font-size: 35px; margin-top: 4px; font-weight: 600; color: #006;">
            PASIEN SKIZOFRENIA DI RSUD MUYANG KUTE
        </div>
        <div style="font-size: 25px; margin-top: 2px; font-weight: 500; color: #006;">
            Menggunakan Kombinasi C4.5 dan Particle Swarm Optimization
        </div>
    </div>
    """, unsafe_allow_html=True)

    
    # Introduction section
    st.markdown("<div>", unsafe_allow_html=True)
    st.markdown("""
    <h3 class='glass' style='color: #0d47a1; text-align: center; margin-bottom: 30px;'>Tentang Sistem</h3>
    <p style='text-align: justify; line-height: 1.8; font-size: 1.1rem;'>
    Sistem ini dirancang untuk mengklasifikasikan durasi rawat inap pasien skizofrenia di RSUD Muyang Kute 
    dengan memanfaatkan kombinasi algoritma C4.5 dan Particle Swarm Optimization (PSO). Pendekatan ini 
    menghasilkan model prediksi yang akurat untuk membantu manajemen rumah sakit dalam perencanaan sumber daya 
    dan perawatan pasien yang lebih efektif.
    </p>
    """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Features section
    st.markdown("<h2 style='text-align: center; color: #fff; margin: 60px 0 40px 0; text-shadow: 1px 1px 3px rgba(0,0,0,0.2);'>Fitur Utama</h2>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
    
        st.markdown("""
        <div class='feature-card'>
        <div class='card-title'>🎯 Klasifikasi Akurat</div>
        <div class='card-content' style='text-align: justify; font-size:16px; color: #004d4d;'>
        Menggunakan kombinasi algoritma C4.5 dan PSO untuk menghasilkan 
        klasifikasi durasi rawat inap dengan akurasi tinggi.
        </div>
        """, unsafe_allow_html=True)
       
    
    with col2:
        st.markdown(
        """
            <div class='feature-card'>
            <div class='card-title'>⚙️ Optimasi Parameter </div>
            <div class='card-content' style='text-align: justify; font-size:16px; color: #004d4d;'>
            PSO digunakan untuk mengoptimasi parameter algoritma C4.5, 
            meningkatkan performa klasifikasi secara signifikan.
            </div>
            </div>
        """, unsafe_allow_html=True)
       
    
    with col3:
        st.markdown("""
        <div class='feature-card'>
        <div class='card-title'>📊 Dashboard</div>
        <div class='card-content' style='text-align: justify; font-size:16px; color: #004d4d;'>
        Melakukan perhitungan & visualisasi data yang interaktif dan informatif untuk 
        memudahkan analisis hasil klasifikasi.
        </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Methodology section
    st.markdown("<div  style='margin-top: 60px;'>", unsafe_allow_html=True)
    st.markdown("<h3 class='glass' style='color: #0d47a1; text-align: center; margin-bottom: 40px;'>Metodologi</h3>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
        """
        <div style="padding: 20px; background: rgba(13, 71, 161, 0.1); border-radius: 16px; margin-bottom: 10px;">
        <h4 style='color: #0d47a1; margin-bottom: 8px;'>Algoritma C4.5</h4>
        <p style='text-align: justify; color: #37474f;'>Algoritma C4.5 digunakan untuk membangun pohon keputusan yang dapat mengklasifikasikan 
        durasi rawat inap berdasarkan berbagai faktor klinis dan demografis pasien.</p>

        <h4 style='color: #0d47a1; margin-top: 10px; margin-bottom: 8px;'>Keunggulan:</h4>
        - Dapat menangani data numerik dan kategorikal  <br>
        - Melakukan pemilihan fitur otomatis  <br>
        - Mampu mengatasi nilai yang hilang  
        </div>
        """,
        unsafe_allow_html=True
    )

    
    with col2:
        st.markdown("""
            <div style="padding: 20px; background: rgba(13, 71, 161, 0.1); border-radius: 16px; margin-bottom: 10px;">
            <h4 style='color: #0d47a1; margin-bottom: 8px;'>Particle Swarm Optimization</h4>
            <p style='text-align: justify; color: #37474f;'>
            PSO digunakan untuk mengoptimasi parameter dari algoritma C4.5, 
            sehingga menghasilkan model klasifikasi dengan performa terbaik.</p>
            
            <h4 style='color: #0d47a1; margin-top: 10px; margin-bottom: 8px;'>Keunggulan:</h4> 
            - Konvergensi yang cepat  <br>
            - Menghindari terjebak di optimum lokal  <br>
            - Efisien dalam pencarian solusi optimal 
            
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Test database connection
    st.markdown("<div  style='margin-top: 60px;'>", unsafe_allow_html=True)
    st.markdown("<h3 class='glass' style='color: #0d47a1; text-align: center; margin-bottom: 30px;'>Status Sistem</h3>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        
        st.markdown("<div class='feature-card card-title'>🔌 Koneksi Database</div></div>", unsafe_allow_html=True)
        
        if st.button("Test Koneksi Database", use_container_width=True):
            conn = init_db_connection()
            if conn:
                st.success("✅ Terhubung ke database Supabase")
                create_tables(conn)
                conn.close()
            else:
                st.error("❌ Gagal terhubung ke database")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='feature-card card-title'>📈 Status Layanan", unsafe_allow_html=True)
        st.markdown("""
        <div class='card-content' style='line-height: 1; color:#005;' >
        <p>✅ Streamlit: Berjalan</p>
        <p>✅ Visualisasi: Aktif</p>
        <p>🔌 Database: Perlu diuji</p>
        <p>✅ Authentication: Siap</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Footer
    st.markdown("""
    <div class='footer'>
    <p style=font-size: 17px; opacity: 1;">© 2025 RSUD Muyang Kute - Sistem Klasifikasi Durasi Rawat Inap Pasien Skizofrenia</p>
    <p style="font-size: 14px; opacity: 1;">Dikembangkan dengan ❤️ PUTRI ALMUNAWARAH untuk pelayanan kesehatan yang lebih baik</p>
    </div>
    """, unsafe_allow_html=True)

# Main app logic
def main():
    # Initialize session state
    if "page" not in st.session_state:
        st.session_state.page = "home"
    if "user" not in st.session_state:
        st.session_state.user = None
    
    # Navigation
    if st.session_state.page == "home":
        # Navigation tabs
        tabs = st.tabs(["🏠 Beranda", "🔐 Masuk"])
        
        with tabs[0]:
            show_home_page()
        
        with tabs[1]:
            show_login_page()
    
    elif st.session_state.page == "dashboard":
        show_dashboard()

if __name__ == "__main__":
    main()
