"""
Streamlit Web Application for Deepfake Voice Detection
Modern cybersecurity-themed AI dashboard with dark UI, glassmorphism,
neon accents, and animated elements.
"""

import os
import sys
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import librosa
import librosa.display
import keras

# Use non-interactive backend for matplotlib
matplotlib.use('Agg')

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.preprocess import preprocess_audio
from src.features import extract_mel_spectrogram, prepare_cnn_input, pad_features, normalize_features


# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Deepfake Voice Detector — AI Security",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# CUSTOM CSS — Cybersecurity Dashboard Theme
# ============================================================
def inject_custom_css():
    """Inject all custom CSS for the futuristic dark theme."""
    st.markdown("""
    <style>
    /* ── Google Font ───────────────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

    /* ── Root variables ───────────────────────────────────── */
    :root {
        --bg-primary:   #0A0E1A;
        --bg-secondary: #111827;
        --bg-card:      rgba(17, 24, 39, 0.6);
        --accent-cyan:  #00D4FF;
        --accent-blue:  #3B82F6;
        --accent-purple:#8B5CF6;
        --glow-green:   #00FF88;
        --glow-red:     #FF3B5C;
        --text-primary: #E2E8F0;
        --text-muted:   #94A3B8;
        --border-glass: rgba(255,255,255,0.08);
    }

    /* ── Global overrides ────────────────────────────────── */
    html, body, [data-testid="stAppViewContainer"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        color: var(--text-primary);
    }

    /* Animated gradient background */
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #0A0E1A 0%, #0F172A 40%, #0A0E1A 70%, #111827 100%);
        background-size: 400% 400%;
        animation: gradientShift 20s ease infinite;
    }
    @keyframes gradientShift {
        0%   { background-position: 0% 50%; }
        50%  { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    /* Hide Streamlit header & footer */
    header[data-testid="stHeader"] {
        background: transparent !important;
    }
    footer { visibility: hidden; }

    /* Hide default sidebar toggle for cleaner look */
    [data-testid="collapsedControl"] { display: none; }

    /* ── Glass card component ────────────────────────────── */
    .glass-card {
        background: var(--bg-card);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid var(--border-glass);
        border-radius: 16px;
        padding: 2rem;
        margin-bottom: 1.5rem;
        transition: transform 0.35s cubic-bezier(.25,.8,.25,1),
                    box-shadow 0.35s cubic-bezier(.25,.8,.25,1),
                    border-color 0.35s ease;
        position: relative;
        overflow: hidden;
    }
    .glass-card::before {
        content: '';
        position: absolute;
        top: 0; left: -75%;
        width: 50%; height: 100%;
        background: linear-gradient(90deg, transparent, rgba(0,212,255,0.04), transparent);
        transition: left 0.6s ease;
        pointer-events: none;
    }
    .glass-card:hover {
        transform: translateY(-4px) scale(1.005);
        box-shadow: 0 12px 40px rgba(0, 212, 255, 0.12),
                    0 0 0 1px rgba(0, 212, 255, 0.1);
        border-color: rgba(0, 212, 255, 0.15);
    }
    .glass-card:hover::before {
        left: 125%;
    }

    /* ── Hero section ────────────────────────────────────── */
    .hero-section {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        text-align: center;
        padding: 3.5rem 1rem 2.5rem;
        width: 100%;
    }
    .hero-title {
        font-size: 3.2rem;
        font-weight: 900;
        letter-spacing: -1px;
        background: linear-gradient(135deg, #00D4FF 0%, #3B82F6 50%, #8B5CF6 100%);
        background-size: 200% 200%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.6rem;
        line-height: 1.15;
        animation: titleShimmer 4s ease-in-out infinite;
    }
    @keyframes titleShimmer {
        0%, 100% { background-position: 0% 50%; }
        50%      { background-position: 100% 50%; }
    }
    .hero-subtitle {
        font-size: 1.15rem;
        color: var(--text-muted);
        font-weight: 400;
        max-width: 650px;
        width: 100%;
        margin: 0 auto 1.5rem auto;
        line-height: 1.7;
        text-align: center;
    }

    /* ── Status indicator (pulsing dot) ──────────────────── */
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: rgba(0, 255, 136, 0.08);
        border: 1px solid rgba(0, 255, 136, 0.25);
        border-radius: 999px;
        padding: 6px 18px;
        font-size: 0.85rem;
        font-weight: 600;
        color: var(--glow-green);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        cursor: default;
    }
    .status-badge:hover {
        transform: scale(1.08);
        box-shadow: 0 0 20px rgba(0, 255, 136, 0.2);
    }
    .pulse-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: var(--glow-green);
        animation: pulse 2s ease-in-out infinite;
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; box-shadow: 0 0 0 0 rgba(0,255,136,0.5); }
        50%      { opacity: 0.6; box-shadow: 0 0 0 8px rgba(0,255,136,0); }
    }

    /* ── Section headings ────────────────────────────────── */
    .section-heading {
        font-size: 1.35rem;
        font-weight: 700;
        color: var(--text-primary);
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 10px;
        transition: color 0.3s ease;
        position: relative;
        padding-left: 4px;
    }
    .section-heading::after {
        content: '';
        position: absolute;
        bottom: -4px;
        left: 0;
        width: 0;
        height: 2px;
        background: linear-gradient(90deg, var(--accent-cyan), transparent);
        transition: width 0.4s ease;
    }
    .section-heading:hover::after {
        width: 100%;
    }
    .section-heading:hover {
        color: var(--accent-cyan);
    }
    .section-heading .icon {
        font-size: 1.3rem;
        transition: transform 0.3s ease;
    }
    .section-heading:hover .icon {
        transform: scale(1.2);
    }

    /* ── Upload area ─────────────────────────────────────── */
    [data-testid="stFileUploader"] {
        background: var(--bg-card);
        border: 2px dashed rgba(0, 212, 255, 0.3);
        border-radius: 16px;
        padding: 1.5rem;
        transition: all 0.4s cubic-bezier(.25,.8,.25,1);
        position: relative;
    }
    [data-testid="stFileUploader"]:hover {
        border-color: var(--accent-cyan);
        box-shadow: 0 0 30px rgba(0, 212, 255, 0.12),
                    0 0 60px rgba(0, 212, 255, 0.05);
        transform: scale(1.01);
        background: rgba(0, 212, 255, 0.03);
    }
    /* Upload label styling */
    [data-testid="stFileUploader"] label {
        font-weight: 600 !important;
        color: var(--text-primary) !important;
        transition: color 0.3s ease !important;
    }
    [data-testid="stFileUploader"]:hover label {
        color: var(--accent-cyan) !important;
    }

    /* ── Buttons ─────────────────────────────────────────── */
    .stButton > button {
        background: linear-gradient(135deg, var(--accent-cyan), var(--accent-blue)) !important;
        background-size: 200% 200% !important;
        color: #fff !important;
        font-weight: 600 !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.6rem 2rem !important;
        transition: all 0.35s cubic-bezier(.25,.8,.25,1) !important;
        box-shadow: 0 4px 16px rgba(0, 212, 255, 0.25) !important;
        position: relative !important;
        overflow: hidden !important;
    }
    .stButton > button::after {
        content: '' !important;
        position: absolute !important;
        top: 50% !important; left: 50% !important;
        width: 0 !important; height: 0 !important;
        border-radius: 50% !important;
        background: rgba(255,255,255,0.2) !important;
        transition: width 0.5s ease, height 0.5s ease, top 0.5s ease, left 0.5s ease !important;
        transform: translate(-50%, -50%) !important;
    }
    .stButton > button:hover {
        transform: translateY(-3px) scale(1.02) !important;
        box-shadow: 0 8px 28px rgba(0, 212, 255, 0.45) !important;
        background-position: 100% 50% !important;
    }
    .stButton > button:hover::after {
        width: 300px !important; height: 300px !important;
    }
    .stButton > button:active {
        transform: translateY(0px) scale(0.98) !important;
    }

    /* ── Prediction badges ───────────────────────────────── */
    .prediction-badge {
        text-align: center;
        padding: 2.5rem 2rem;
        border-radius: 20px;
        font-size: 2rem;
        font-weight: 800;
        margin: 1.5rem auto;
        max-width: 600px;
        animation: badgeAppear 0.9s cubic-bezier(.17,.67,.24,1.27);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        cursor: default;
    }
    .prediction-badge:hover {
        transform: scale(1.03);
    }
    @keyframes badgeAppear {
        0%   { opacity:0; transform: scale(0.7) translateY(30px); }
        60%  { opacity:1; transform: scale(1.04) translateY(-4px); }
        100% { transform: scale(1) translateY(0); }
    }
    .badge-real {
        background: rgba(0, 255, 136, 0.06);
        border: 2px solid rgba(0, 255, 136, 0.3);
        color: var(--glow-green);
        box-shadow: 0 0 40px rgba(0, 255, 136, 0.12),
                    inset 0 0 40px rgba(0, 255, 136, 0.04);
        animation: badgeAppear 0.9s cubic-bezier(.17,.67,.24,1.27),
                   glowPulseGreen 3s ease-in-out infinite 1s;
    }
    .badge-real:hover {
        box-shadow: 0 0 60px rgba(0, 255, 136, 0.2),
                    inset 0 0 60px rgba(0, 255, 136, 0.06);
    }
    @keyframes glowPulseGreen {
        0%, 100% { box-shadow: 0 0 40px rgba(0,255,136,0.12), inset 0 0 40px rgba(0,255,136,0.04); }
        50%      { box-shadow: 0 0 60px rgba(0,255,136,0.22), inset 0 0 50px rgba(0,255,136,0.08); }
    }
    .badge-fake {
        background: rgba(255, 59, 92, 0.06);
        border: 2px solid rgba(255, 59, 92, 0.3);
        color: var(--glow-red);
        box-shadow: 0 0 40px rgba(255, 59, 92, 0.12),
                    inset 0 0 40px rgba(255, 59, 92, 0.04);
        animation: badgeAppear 0.9s cubic-bezier(.17,.67,.24,1.27),
                   glowPulseRed 3s ease-in-out infinite 1s;
    }
    .badge-fake:hover {
        box-shadow: 0 0 60px rgba(255, 59, 92, 0.2),
                    inset 0 0 60px rgba(255, 59, 92, 0.06);
    }
    @keyframes glowPulseRed {
        0%, 100% { box-shadow: 0 0 40px rgba(255,59,92,0.12), inset 0 0 40px rgba(255,59,92,0.04); }
        50%      { box-shadow: 0 0 60px rgba(255,59,92,0.22), inset 0 0 50px rgba(255,59,92,0.08); }
    }
    .badge-label {
        font-size: 1.8rem;
        font-weight: 800;
        letter-spacing: 2px;
        text-transform: uppercase;
    }
    .badge-confidence {
        font-size: 1rem;
        font-weight: 500;
        margin-top: 0.5rem;
        opacity: 0.85;
    }

    /* ── Animated confidence bar ─────────────────────────── */
    .confidence-bar-container {
        background: rgba(255,255,255,0.05);
        border-radius: 12px;
        overflow: hidden;
        height: 28px;
        margin: 1rem 0;
        border: 1px solid var(--border-glass);
    }
    .confidence-bar-fill {
        height: 100%;
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
        font-size: 0.85rem;
        color: #fff;
        animation: barGrow 1.2s ease-out;
        transition: width 0.6s ease;
    }
    .bar-real {
        background: linear-gradient(90deg, #00FF88, #00D4FF);
    }
    .bar-fake {
        background: linear-gradient(90deg, #FF3B5C, #FF6B6B);
    }
    @keyframes barGrow {
        from { width: 0 !important; }
    }

    /* ── Fade-in animation ───────────────────────────────── */
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(24px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    .fade-in {
        animation: fadeInUp 0.6s ease-out;
    }

    /* ── Metric cards ────────────────────────────────────── */
    .metric-card {
        background: var(--bg-card);
        border: 1px solid var(--border-glass);
        border-radius: 14px;
        padding: 1.5rem;
        text-align: center;
        transition: transform 0.35s cubic-bezier(.25,.8,.25,1),
                    box-shadow 0.35s ease,
                    border-color 0.35s ease;
        position: relative;
        overflow: hidden;
    }
    .metric-card::after {
        content: '';
        position: absolute;
        bottom: 0; left: 0; right: 0;
        height: 3px;
        background: linear-gradient(90deg, var(--accent-cyan), var(--accent-blue), var(--accent-purple));
        background-size: 200% 100%;
        opacity: 0;
        transition: opacity 0.3s ease;
        animation: metricBarShift 3s ease infinite;
    }
    @keyframes metricBarShift {
        0%, 100% { background-position: 0% 50%; }
        50%      { background-position: 100% 50%; }
    }
    .metric-card:hover {
        transform: translateY(-6px) scale(1.02);
        box-shadow: 0 12px 32px rgba(0, 212, 255, 0.12);
        border-color: rgba(0, 212, 255, 0.2);
    }
    .metric-card:hover::after {
        opacity: 1;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #00D4FF, #3B82F6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        transition: transform 0.3s ease;
    }
    .metric-card:hover .metric-value {
        transform: scale(1.08);
    }
    .metric-label {
        font-size: 0.85rem;
        color: var(--text-muted);
        font-weight: 500;
        margin-top: 4px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* ── Info panel cards ────────────────────────────────── */
    .info-card {
        background: var(--bg-card);
        backdrop-filter: blur(12px);
        border: 1px solid var(--border-glass);
        border-radius: 16px;
        padding: 1.8rem;
        height: 100%;
        transition: transform 0.35s cubic-bezier(.25,.8,.25,1),
                    box-shadow 0.35s ease,
                    border-color 0.35s ease;
        position: relative;
        overflow: hidden;
    }
    .info-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0;
        width: 3px; height: 0;
        background: linear-gradient(180deg, var(--accent-cyan), var(--accent-purple));
        transition: height 0.4s ease;
    }
    .info-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 36px rgba(0, 212, 255, 0.1);
        border-color: rgba(0, 212, 255, 0.15);
    }
    .info-card:hover::before {
        height: 100%;
    }
    .info-card h3 {
        font-size: 1.1rem;
        font-weight: 700;
        color: var(--accent-cyan);
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 8px;
        transition: letter-spacing 0.3s ease;
    }
    .info-card:hover h3 {
        letter-spacing: 0.5px;
    }
    .info-card p, .info-card li {
        font-size: 0.92rem;
        color: var(--text-muted);
        line-height: 1.7;
        transition: color 0.3s ease;
    }
    .info-card:hover li {
        color: var(--text-primary);
    }
    .info-card ul {
        padding-left: 1.2rem;
    }

    /* ── Styled dataframe ────────────────────────────────── */
    [data-testid="stDataFrame"] {
        border-radius: 12px;
        overflow: hidden;
    }

    /* ── Audio player ────────────────────────────────────── */
    audio {
        width: 100%;
        border-radius: 12px;
    }

    /* ── Expander styling ────────────────────────────────── */
    [data-testid="stExpander"] {
        border: 1px solid var(--border-glass) !important;
        border-radius: 12px !important;
        background: var(--bg-card) !important;
        transition: border-color 0.3s ease, box-shadow 0.3s ease !important;
    }
    [data-testid="stExpander"]:hover {
        border-color: rgba(0, 212, 255, 0.2) !important;
        box-shadow: 0 4px 20px rgba(0, 212, 255, 0.06) !important;
    }

    /* ── Footer ──────────────────────────────────────────── */
    .app-footer {
        text-align: center;
        padding: 2rem 0;
        color: var(--text-muted);
        font-size: 0.85rem;
        border-top: 1px solid var(--border-glass);
        margin-top: 3rem;
    }
    .app-footer a {
        color: var(--accent-cyan);
        text-decoration: none;
        font-weight: 600;
        position: relative;
        transition: color 0.3s ease;
    }
    .app-footer a::after {
        content: '';
        position: absolute;
        bottom: -2px; left: 0;
        width: 0; height: 1px;
        background: var(--accent-cyan);
        transition: width 0.3s ease;
    }
    .app-footer a:hover {
        text-decoration: none;
        color: #fff;
    }
    .app-footer a:hover::after {
        width: 100%;
    }

    /* ── Floating particles (decorative) ─────────────────── */
    .particles-bg {
        position: fixed;
        top: 0; left: 0;
        width: 100%; height: 100%;
        pointer-events: none;
        z-index: 0;
        overflow: hidden;
    }
    .particle {
        position: absolute;
        width: 3px; height: 3px;
        background: rgba(0, 212, 255, 0.15);
        border-radius: 50%;
        animation: floatUp linear infinite;
    }
    .particle:nth-child(1)  { left: 10%; animation-duration: 18s; animation-delay: 0s; }
    .particle:nth-child(2)  { left: 25%; animation-duration: 22s; animation-delay: 3s; width:2px; height:2px; }
    .particle:nth-child(3)  { left: 40%; animation-duration: 16s; animation-delay: 6s; }
    .particle:nth-child(4)  { left: 55%; animation-duration: 24s; animation-delay: 1s; width:4px; height:4px; background:rgba(139,92,246,0.12); }
    .particle:nth-child(5)  { left: 70%; animation-duration: 20s; animation-delay: 4s; }
    .particle:nth-child(6)  { left: 85%; animation-duration: 19s; animation-delay: 7s; width:2px; height:2px; }
    .particle:nth-child(7)  { left: 15%; animation-duration: 21s; animation-delay: 9s; background:rgba(59,130,246,0.12); }
    .particle:nth-child(8)  { left: 60%; animation-duration: 17s; animation-delay: 2s; }
    .particle:nth-child(9)  { left: 90%; animation-duration: 23s; animation-delay: 5s; width:4px; height:4px; }
    .particle:nth-child(10) { left: 35%; animation-duration: 25s; animation-delay: 8s; background:rgba(139,92,246,0.1); }
    @keyframes floatUp {
        0%   { transform: translateY(100vh) rotate(0deg);   opacity: 0; }
        10%  { opacity: 1; }
        90%  { opacity: 1; }
        100% { transform: translateY(-10vh) rotate(360deg); opacity: 0; }
    }

    /* ── Divider animation ───────────────────────────────── */
    hr {
        border: none !important;
        height: 1px !important;
        background: linear-gradient(90deg, transparent, rgba(0,212,255,0.2), transparent) !important;
        margin: 2rem 0 !important;
    }

    /* ── Spinner override ────────────────────────────────── */
    .stSpinner > div {
        border-top-color: var(--accent-cyan) !important;
    }

    /* ── Scrollbar ───────────────────────────────────────── */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: var(--bg-primary); }
    ::-webkit-scrollbar-thumb { background: #334155; border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: #475569; }

    /* ── Loading animation (custom spinner) ──────────────── */
    .cyber-loader {
        display: flex;
        justify-content: center;
        align-items: center;
        gap: 6px;
        padding: 2rem;
    }
    .cyber-loader .bar {
        width: 4px;
        height: 24px;
        background: var(--accent-cyan);
        border-radius: 2px;
        animation: loaderPulse 1s ease-in-out infinite;
    }
    .cyber-loader .bar:nth-child(2) { animation-delay: 0.1s; }
    .cyber-loader .bar:nth-child(3) { animation-delay: 0.2s; }
    .cyber-loader .bar:nth-child(4) { animation-delay: 0.3s; }
    .cyber-loader .bar:nth-child(5) { animation-delay: 0.4s; }
    @keyframes loaderPulse {
        0%, 100% { transform: scaleY(0.4); opacity: 0.4; }
        50%      { transform: scaleY(1);   opacity: 1; }
    }

    /* ── Success / warning alerts ────────────────────────── */
    [data-testid="stAlert"] {
        border-radius: 12px !important;
    }

    </style>
    """, unsafe_allow_html=True)


# ============================================================
# MODEL LOADING (unchanged logic)
# ============================================================
@st.cache_resource
def load_model(model_path='models/deepfake_detector.h5'):
    """Load the trained model."""
    try:
        model = keras.saving.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None


# ============================================================
# DARK-THEMED MATPLOTLIB PLOTS
# ============================================================
def plot_waveform(audio, sr, title="Audio Waveform"):
    """Plot audio waveform with dark cybersecurity theme."""
    fig, ax = plt.subplots(figsize=(12, 4))
    fig.patch.set_facecolor('#0A0E1A')
    ax.set_facecolor('#0F172A')

    # Plot waveform with cyan color
    librosa.display.waveshow(audio, sr=sr, ax=ax, color='#00D4FF', alpha=0.85)

    ax.set_title(title, fontsize=14, fontweight='bold', color='#E2E8F0', pad=12)
    ax.set_xlabel("Time (s)", fontsize=11, color='#94A3B8')
    ax.set_ylabel("Amplitude", fontsize=11, color='#94A3B8')
    ax.tick_params(colors='#64748B', labelsize=9)
    ax.spines['bottom'].set_color('#1E293B')
    ax.spines['left'].set_color('#1E293B')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.1, color='#334155')
    plt.tight_layout()
    return fig


def plot_spectrogram(spectrogram, sr=16000, title="Mel Spectrogram"):
    """Plot mel spectrogram with dark cybersecurity theme."""
    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor('#0A0E1A')
    ax.set_facecolor('#0F172A')

    img = librosa.display.specshow(
        spectrogram,
        sr=sr,
        x_axis='time',
        y_axis='mel',
        ax=ax,
        cmap='magma'
    )

    ax.set_title(title, fontsize=14, fontweight='bold', color='#E2E8F0', pad=12)
    ax.set_xlabel("Time (s)", fontsize=11, color='#94A3B8')
    ax.set_ylabel("Mel Frequency", fontsize=11, color='#94A3B8')
    ax.tick_params(colors='#64748B', labelsize=9)
    ax.spines['bottom'].set_color('#1E293B')
    ax.spines['left'].set_color('#1E293B')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    cbar = fig.colorbar(img, ax=ax, format='%+2.0f dB', pad=0.02)
    cbar.ax.yaxis.set_tick_params(color='#64748B')
    cbar.outline.set_edgecolor('#1E293B')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='#64748B')

    plt.tight_layout()
    return fig


# ============================================================
# PREDICTION LOGIC (unchanged from original)
# ============================================================
def predict_audio(model, audio_file, target_width):
    """
    Process audio and make prediction using chunk-based analysis.

    Splits longer audio (up to 10 min) into 3-second chunks, predicts each
    chunk independently, and aggregates results.

    Args:
        model: Trained Keras model
        audio_file: Uploaded audio file
        target_width: Target width for padding

    Returns:
        tuple: (prediction, confidence, audio, sr, spectrogram, chunk_results)
    """
    import librosa as lr

    # Save uploaded file temporarily
    temp_path = "temp_audio.wav"
    with open(temp_path, "wb") as f:
        f.write(audio_file.getbuffer())

    sr = 16000
    chunk_duration = 3.0   # seconds per chunk (model's expected input)
    max_duration = 600.0   # max 10 minutes
    chunk_samples = int(sr * chunk_duration)

    # Load full audio (up to 10 minutes)
    full_audio, _ = lr.load(temp_path, sr=sr, duration=max_duration)

    # Trim silence from the full audio
    full_audio, _ = lr.effects.trim(full_audio, top_db=20)

    # If audio is shorter than one chunk, pad it
    if len(full_audio) < chunk_samples:
        full_audio = np.pad(full_audio, (0, chunk_samples - len(full_audio)))

    # Split into 3-second chunks
    num_chunks = len(full_audio) // chunk_samples
    chunks = [full_audio[i * chunk_samples:(i + 1) * chunk_samples] for i in range(num_chunks)]

    # Predict each chunk
    chunk_results = []
    all_spectrograms = []

    for i, chunk in enumerate(chunks):
        spectrogram = extract_mel_spectrogram(chunk, sr=sr)
        all_spectrograms.append(spectrogram)
        features = prepare_cnn_input(spectrogram)
        features = np.expand_dims(features, axis=0)
        features = pad_features(features, target_width)
        features = normalize_features(features)

        proba = model.predict(features, verbose=0)[0][0]
        chunk_pred = 1 if proba > 0.4 else 0
        chunk_conf = proba if chunk_pred == 1 else (1 - proba)

        chunk_results.append({
            'chunk': i + 1,
            'start': i * chunk_duration,
            'end': (i + 1) * chunk_duration,
            'prediction': 'Fake' if chunk_pred == 1 else 'Real',
            'confidence': chunk_conf,
            'raw_score': proba
        })

    # Aggregate: average raw scores across all chunks
    avg_score = np.mean([r['raw_score'] for r in chunk_results])
    prediction = 1 if avg_score > 0.4 else 0
    confidence = avg_score if prediction == 1 else (1 - avg_score)

    # Use the first chunk's spectrogram for display
    spectrogram = all_spectrograms[0]

    # Clean up
    if os.path.exists(temp_path):
        os.remove(temp_path)

    return prediction, confidence, full_audio, sr, spectrogram, chunk_results


# ============================================================
# UI COMPONENTS
# ============================================================

def render_hero():
    """Render the hero / header section."""
    # Floating particles background (decorative ambient animation)
    st.markdown("""
    <div class="particles-bg">
        <div class="particle"></div><div class="particle"></div>
        <div class="particle"></div><div class="particle"></div>
        <div class="particle"></div><div class="particle"></div>
        <div class="particle"></div><div class="particle"></div>
        <div class="particle"></div><div class="particle"></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="hero-section">
        <div class="hero-title">🛡️ Deepfake Voice Detection System</div>
        <p class="hero-subtitle">
            AI-powered cybersecurity tool that analyzes audio signals using deep learning
            to distinguish authentic human voices from AI-generated deepfakes — in real time.
        </p>
        <div class="status-badge">
            <span class="pulse-dot"></span>
            AI Engine Online
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_upload_section():
    """Render the file upload section and return the uploaded file."""
    st.markdown("""
    <div class="section-heading">
        <span class="icon">📤</span> Upload Audio for Analysis
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Drag and drop your audio file here",
        type=['wav', 'mp3', 'flac', 'ogg', 'm4a'],
        help="Supported formats: WAV, MP3, FLAC, OGG, M4A  •  Max duration: 10 minutes"
    )
    return uploaded_file


def render_visualizations(audio, sr, spectrogram):
    """Render waveform and spectrogram visualizations."""
    st.markdown("""
    <div class="section-heading fade-in">
        <span class="icon">📊</span> Audio Signal Analysis
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        fig_wave = plot_waveform(audio, sr, title="Waveform")
        st.pyplot(fig_wave)
        plt.close()
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        fig_spec = plot_spectrogram(spectrogram, sr, title="Mel Spectrogram")
        st.pyplot(fig_spec)
        plt.close()
        st.markdown('</div>', unsafe_allow_html=True)


def render_prediction_result(prediction, confidence):
    """Render the large prediction badge with animated confidence bar."""
    st.markdown("""
    <div class="section-heading fade-in">
        <span class="icon">🎯</span> Prediction Result
    </div>
    """, unsafe_allow_html=True)

    if prediction == 0:
        badge_class = "badge-real"
        bar_class = "bar-real"
        icon = "✅"
        label = "Real Voice"
    else:
        badge_class = "badge-fake"
        bar_class = "bar-fake"
        icon = "⚠️"
        label = "Deepfake Detected"

    pct = confidence * 100

    # Large prediction badge
    st.markdown(f"""
    <div class="prediction-badge {badge_class}">
        <div style="font-size:2.8rem; margin-bottom:0.3rem;">{icon}</div>
        <div class="badge-label">{label}</div>
        <div class="badge-confidence">Confidence: {pct:.1f}%</div>
    </div>
    """, unsafe_allow_html=True)

    # Animated confidence bar
    st.markdown(f"""
    <div style="max-width:600px; margin:0 auto;">
        <div class="confidence-bar-container">
            <div class="confidence-bar-fill {bar_class}" style="width:{pct:.1f}%;">
                {pct:.1f}%
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_metrics(prediction, confidence, chunk_results, audio, sr):
    """Render metric cards in a row."""
    st.markdown("""
    <div class="section-heading fade-in">
        <span class="icon">📈</span> Detailed Metrics
    </div>
    """, unsafe_allow_html=True)

    raw_score = confidence if prediction == 1 else (1 - confidence)
    duration = len(audio) / sr

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{"Real" if prediction == 0 else "Fake"}</div>
            <div class="metric-label">Prediction</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{confidence*100:.1f}%</div>
            <div class="metric-label">Confidence</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{raw_score:.4f}</div>
            <div class="metric-label">Raw Score</div>
        </div>
        """, unsafe_allow_html=True)

    with c4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{duration:.1f}s</div>
            <div class="metric-label">Duration</div>
        </div>
        """, unsafe_allow_html=True)


def render_chunk_analysis(chunk_results):
    """Render per-chunk analysis table for multi-chunk audio."""
    if len(chunk_results) <= 1:
        return

    st.markdown("""
    <div class="section-heading fade-in">
        <span class="icon">🧩</span> Per-Chunk Analysis
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="glass-card" style="padding:1.2rem 1.8rem;">
        <p style="color:var(--text-muted); margin-bottom:1rem;">
            Audio split into <strong style="color:#00D4FF;">{len(chunk_results)} chunks</strong> of 3 seconds each for granular analysis.
        </p>
    </div>
    """, unsafe_allow_html=True)

    import pandas as pd
    chunk_df = pd.DataFrame(chunk_results)
    chunk_df['Time Range'] = chunk_df.apply(
        lambda r: f"{r['start']:.0f}s – {r['end']:.0f}s", axis=1
    )
    chunk_df['Confidence'] = chunk_df['confidence'].apply(
        lambda c: f"{c*100:.1f}%"
    )
    display_df = chunk_df[['chunk', 'Time Range', 'prediction', 'Confidence']]
    display_df.columns = ['Chunk #', 'Time Range', 'Prediction', 'Confidence']
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    # Summary
    fake_chunks = sum(1 for r in chunk_results if r['prediction'] == 'Fake')
    real_chunks = sum(1 for r in chunk_results if r['prediction'] == 'Real')
    st.markdown(f"""
    <div style="text-align:center; padding:0.5rem; color:var(--text-muted);">
        🟢 Real chunks: <strong style="color:var(--glow-green);">{real_chunks}</strong>
        &nbsp;&nbsp;│&nbsp;&nbsp;
        🔴 Fake chunks: <strong style="color:var(--glow-red);">{fake_chunks}</strong>
    </div>
    """, unsafe_allow_html=True)


def render_info_panel():
    """Render the 3-column info panel at the bottom of the page."""
    st.markdown("""
    <div class="section-heading" style="margin-top:2.5rem;">
        <span class="icon">ℹ️</span> About the System
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("""
        <div class="info-card">
            <h3>🧠 How It Works</h3>
            <ul>
                <li>Audio is resampled to 16 kHz and trimmed</li>
                <li>Mel Spectrogram features are extracted</li>
                <li>A trained CNN classifies the signal</li>
                <li>Longer audio is split into 3 s chunks</li>
                <li>Results are aggregated for final verdict</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="info-card">
            <h3>📊 Model Performance</h3>
            <ul>
                <li>Architecture: CNN + Batch Norm</li>
                <li>Training data: Real &amp; synthetic voices</li>
                <li>Input: 128-band Mel Spectrogram</li>
                <li>Chunk-based analysis for long audio</li>
                <li>Real-time inference speed</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown("""
        <div class="info-card">
            <h3>⚙️ Tech Stack</h3>
            <ul>
                <li><strong>Framework:</strong> Streamlit</li>
                <li><strong>ML:</strong> TensorFlow / Keras</li>
                <li><strong>Audio:</strong> Librosa</li>
                <li><strong>Visualization:</strong> Matplotlib</li>
                <li><strong>Language:</strong> Python 3.10+</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)


def render_footer():
    """Render the application footer."""
    st.markdown("""
    <div class="app-footer">
        Built with 🛡️ for AI Security &nbsp;|&nbsp;
        <a href="https://github.com" target="_blank">GitHub</a> &nbsp;|&nbsp;
        Powered by TensorFlow, Librosa &amp; Streamlit
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# MAIN APPLICATION
# ============================================================
def main():
    """Main Streamlit application entry point."""

    # Inject custom CSS
    inject_custom_css()

    # ── Hero section ──────────────────────────────────────
    render_hero()

    st.markdown("---")

    # ── Upload section ────────────────────────────────────
    uploaded_file = render_upload_section()

    if uploaded_file is not None:
        # Show file info
        st.markdown(f"""
        <div class="glass-card fade-in" style="padding:1rem 1.5rem; display:flex; align-items:center; gap:12px;">
            <span style="font-size:1.4rem;">🎵</span>
            <div>
                <strong style="color:var(--accent-cyan);">{uploaded_file.name}</strong>
                <span style="color:var(--text-muted); margin-left:12px; font-size:0.9rem;">
                    ({uploaded_file.size / 1024:.1f} KB)
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Audio player
        st.audio(uploaded_file, format='audio/wav')

        # Load model
        with st.spinner("⚡ Loading AI model..."):
            model = load_model()

        if model is None:
            st.error("❌ Model not found! Train the model first with `python train.py`")
            st.stop()

        # Get target width from model input shape
        target_width = model.input_shape[2]

        # Custom loading animation
        analysis_placeholder = st.empty()
        analysis_placeholder.markdown("""
        <div class="glass-card" style="text-align:center;">
            <div class="cyber-loader">
                <div class="bar"></div>
                <div class="bar"></div>
                <div class="bar"></div>
                <div class="bar"></div>
                <div class="bar"></div>
            </div>
            <p style="color:var(--accent-cyan); font-weight:600;">Analyzing audio signal...</p>
        </div>
        """, unsafe_allow_html=True)

        try:
            prediction, confidence, audio, sr, spectrogram, chunk_results = predict_audio(
                model, uploaded_file, target_width
            )

            # Clear loading animation
            analysis_placeholder.empty()

            st.markdown("---")

            # ── Visualizations ─────────────────────────────
            render_visualizations(audio, sr, spectrogram)

            st.markdown("---")

            # ── Prediction result ──────────────────────────
            render_prediction_result(prediction, confidence)

            st.markdown("---")

            # ── Detailed metrics ───────────────────────────
            render_metrics(prediction, confidence, chunk_results, audio, sr)

            # ── Per-chunk analysis ─────────────────────────
            if len(chunk_results) > 1:
                st.markdown("---")
                render_chunk_analysis(chunk_results)

            # ── Audio details expander ─────────────────────
            with st.expander("🔍  Audio Details"):
                d1, d2, d3 = st.columns(3)
                d1.markdown(f"**Sample Rate:** {sr} Hz")
                d2.markdown(f"**Total Samples:** {len(audio):,}")
                d3.markdown(f"**Spectrogram Shape:** {spectrogram.shape}")

        except Exception as e:
            analysis_placeholder.empty()
            st.error(f"❌ Error processing audio: {str(e)}")
            st.exception(e)

    else:
        # Placeholder when no file uploaded
        st.markdown("""
        <div class="glass-card" style="text-align:center; padding:3rem 2rem;">
            <div style="font-size:3rem; margin-bottom:1rem;">🎙️</div>
            <p style="font-size:1.1rem; color:var(--text-muted); margin-bottom:0.5rem;">
                Upload an audio file to begin deepfake analysis
            </p>
            <p style="font-size:0.9rem; color:#64748B;">
                Supports WAV, MP3, FLAC, OGG, M4A  •  Up to 10 minutes
            </p>
        </div>
        """, unsafe_allow_html=True)

    # ── Info panel ────────────────────────────────────────
    st.markdown("---")
    render_info_panel()

    # ── Footer ────────────────────────────────────────────
    render_footer()


if __name__ == "__main__":
    main()
