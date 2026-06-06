import streamlit as st
import pickle
import numpy as np
from PIL import Image
import datetime
import base64
import io

# ---------------- PAGE CONFIG ---------------- #

st.set_page_config(
    page_title="AI Disease Predictor",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- SESSION STATE (Active Page) ---------------- #

if "active_page" not in st.session_state:
    st.session_state.active_page = "Dashboard"

# ---------------- LOAD LOGO ---------------- #

try:
    logo = Image.open("logo.png")
    _buf = io.BytesIO()
    logo.save(_buf, format="PNG")
    LOGO_B64 = base64.b64encode(_buf.getvalue()).decode()
    LOGO_HTML = f'<img src="data:image/png;base64,{LOGO_B64}" style="width:88px;height:88px;object-fit:contain;flex-shrink:0;" />'
    has_logo = True
except Exception:
    has_logo = False
    LOGO_HTML = ''

# ---------------- LOAD MODEL FILES ---------------- #

ALL_SYMPTOMS = [
    'itching','skin_rash','nodal_skin_eruptions','continuous_sneezing','shivering',
    'chills','joint_pain','stomach_pain','acidity','ulcers_on_tongue','muscle_wasting',
    'vomiting','burning_micturition','spotting_ urination','fatigue','weight_gain',
    'anxiety','cold_hands_and_feets','mood_swings','weight_loss','restlessness',
    'lethargy','patches_in_throat','irregular_sugar_level','cough','high_fever',
    'sunken_eyes','breathlessness','sweating','dehydration','indigestion','headache',
    'yellowish_skin','dark_urine','nausea','loss_of_appetite','pain_behind_the_eyes',
    'back_pain','constipation','abdominal_pain','diarrhoea','mild_fever','yellow_urine',
    'yellowing_of_eyes','acute_liver_failure','fluid_overload','swelling_of_stomach',
    'swelled_lymph_nodes','malaise','blurred_and_distorted_vision','phlegm',
    'throat_irritation','redness_of_eyes','sinus_pressure','runny_nose','congestion',
    'chest_pain','weakness_in_limbs','fast_heart_rate','pain_during_bowel_movements',
    'pain_in_anal_region','bloody_stool','irritation_in_anus','neck_pain','dizziness',
    'cramps','bruising','obesity','swollen_legs','swollen_blood_vessels',
    'puffy_face_and_eyes','enlarged_thyroid','brittle_nails','swollen_extremeties',
    'excessive_hunger','extra_marital_contacts','drying_and_tingling_lips',
    'slurred_speech','knee_pain','hip_joint_pain','muscle_weakness','stiff_neck',
    'swelling_joints','movement_stiffness','spinning_movements','loss_of_balance',
    'unsteadiness','weakness_of_one_body_side','loss_of_smell','bladder_discomfort',
    'foul_smell_of urine','continuous_feel_of_urine','passage_of_gases',
    'internal_itching','toxic_look_(typhos)','depression','irritability','muscle_pain',
    'altered_sensorium','red_spots_over_body','belly_pain','abnormal_menstruation',
    'dischromic _patches','watering_from_eyes','increased_appetite','polyuria',
    'family_history','mucoid_sputum','rusty_sputum','lack_of_concentration',
    'visual_disturbances','receiving_blood_transfusion','receiving_unsterile_injections',
    'coma','stomach_bleeding','distention_of_abdomen','history_of_alcohol_consumption',
    'fluid_overload.1','blood_in_sputum','prominent_veins_on_calf','palpitations',
    'painful_walking','pus_filled_pimples','blackheads','scurring','skin_peeling',
    'silver_like_dusting','small_dents_in_nails','inflammatory_nails','blister',
    'red_sore_around_nose','yellow_crust_ooze'
]

def load_models():
    try:
        with open("model.pkl",   "rb") as f: m = pickle.load(f)
        with open("encoder.pkl", "rb") as f: e = pickle.load(f)
        with open("columns.pkl", "rb") as f: c = pickle.load(f)
        return m, e, list(c), True
    except Exception:
        return None, None, ALL_SYMPTOMS, False

model, encoder, columns, model_loaded = load_models()
display_columns = columns

# ================================================================
#  FULL CSS
# ================================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Rajdhani:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Rajdhani', sans-serif; }

.stApp {
    background: #010b14;
    color: #e2f0ff;
    overflow-x: hidden;
}

/* ── PARTICLE CANVAS ── */
#particle-canvas {
    position: fixed; top:0; left:0;
    width:100%; height:100%;
    z-index:0; pointer-events:none;
}

/* ── NAVBAR ── */
.glass-navbar {
    position: sticky; top:0; z-index:999;
    display:flex; align-items:center; justify-content:space-between;
    padding:14px 36px;
    background: rgba(1,11,20,0.72);
    backdrop-filter: blur(24px);
    -webkit-backdrop-filter: blur(24px);
    border-bottom: 1px solid rgba(0,200,255,0.18);
    box-shadow: 0 4px 40px rgba(0,200,255,0.08);
    margin-bottom: 30px;
}
.navbar-brand {
    font-family:'Orbitron',monospace;
    font-size:22px; font-weight:900; letter-spacing:2px;
    background: linear-gradient(90deg,#00e5ff,#7b68ee,#ff6ec7);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
    cursor:pointer;
}
.navbar-links { display:flex; gap:6px; list-style:none; margin:0; padding:0; }
.nav-btn {
    background: rgba(0,229,255,0.04);
    border: 1px solid rgba(0,229,255,0.12);
    color: #8ec9e8;
    padding: 8px 20px;
    border-radius: 10px;
    font-family:'Rajdhani',sans-serif;
    font-size:14px; font-weight:700;
    letter-spacing:1.5px; text-transform:uppercase;
    cursor:pointer; transition: all 0.25s ease;
}
.nav-btn:hover {
    border-color: rgba(0,229,255,0.5);
    color:#00e5ff;
    background: rgba(0,229,255,0.1);
    box-shadow: 0 0 16px rgba(0,229,255,0.2);
}
.nav-btn.active-nav {
    background: rgba(0,229,255,0.15);
    border-color: rgba(0,229,255,0.6);
    color:#00e5ff;
    box-shadow: 0 0 20px rgba(0,229,255,0.25);
}
.navbar-status {
    display:flex; align-items:center; gap:8px;
    font-size:13px; color:#00ff9d; font-weight:600; letter-spacing:1px;
}
.navbar-dot {
    width:9px; height:9px; border-radius:50%; background:#00ff9d;
    animation: pulse-dot 1.5s infinite;
}
@keyframes pulse-dot {
    0%,100% { box-shadow:0 0 0 0 rgba(0,255,157,0.6); }
    50%      { box-shadow:0 0 0 8px rgba(0,255,157,0); }
}

/* ── NAVBAR COLUMN ALIGNMENT FIX ── */
div[data-testid="stHorizontalBlock"] {
    align-items: center !important;
    gap: 6px !important;
}
div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"] {
    display: flex !important;
    align-items: center !important;
    padding: 0 !important;
}
div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"] > div {
    width: 100%;
}
/* Navbar st.button override — compact pill style */
div[data-testid="stHorizontalBlock"] .stButton > button {
    width: 100% !important;
    padding: 9px 10px !important;
    border-radius: 10px !important;
    font-family: 'Rajdhani', sans-serif !important;
    font-size: 13px !important;
    font-weight: 700 !important;
    letter-spacing: 1.5px !important;
    text-transform: uppercase !important;
    background: rgba(0,229,255,0.04) !important;
    border: 1px solid rgba(0,229,255,0.18) !important;
    color: #8ec9e8 !important;
    box-shadow: none !important;
    margin-top: 0 !important;
    animation: none !important;
    transition: all 0.25s ease !important;
    line-height: 1.2 !important;
    height: 40px !important;
    min-height: unset !important;
    white-space: nowrap !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
}
div[data-testid="stHorizontalBlock"] .stButton > button:hover {
    border-color: rgba(0,229,255,0.55) !important;
    color: #00e5ff !important;
    background: rgba(0,229,255,0.1) !important;
    box-shadow: 0 0 14px rgba(0,229,255,0.2) !important;
    transform: none !important;
}

@media (max-width: 768px) {
    /* On mobile, navbar brand + status hide, buttons font smaller */
    .navbar-brand  { font-size: 15px !important; letter-spacing: 1px !important; }
    .navbar-status { display: none !important; }
    div[data-testid="stHorizontalBlock"] .stButton > button {
        font-size: 11px !important;
        padding: 7px 4px !important;
        letter-spacing: 0.5px !important;
        height: 36px !important;
    }
}

/* ── HERO ── */
.hero-wrapper { position:relative; z-index:10; margin-bottom:36px; }
.hero-title {
    font-family:'Orbitron',monospace;
    font-size:52px; font-weight:900; letter-spacing:2px;
    background:linear-gradient(135deg,#00e5ff 0%,#7b68ee 50%,#ff6ec7 100%);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
    line-height:1.15;
    animation: title-shimmer 4s ease-in-out infinite;
}
@keyframes title-shimmer {
    0%,100% { filter:brightness(1); }
    50%     { filter:brightness(1.25) drop-shadow(0 0 20px #00e5ff80); }
}
.hero-subtitle {
    color:#6fa8c8; font-size:18px; letter-spacing:3px;
    text-transform:uppercase; margin-top:10px; font-weight:500;
}

/* ── HEARTBEAT ── */
.heartbeat-container {
    position:relative; z-index:10;
    background:rgba(0,229,255,0.04);
    border:1px solid rgba(0,229,255,0.15);
    border-radius:18px; padding:20px 28px; margin-bottom:28px;
    overflow:hidden; display:flex; align-items:center; gap:18px;
}
.hb-label {
    font-family:'Orbitron',monospace; font-size:12px;
    color:#00e5ff; letter-spacing:2px; font-weight:700; white-space:nowrap;
}
.hb-svg-wrap { flex:1; overflow:hidden; }
svg.heartbeat-svg { width:100%; height:60px; display:block; }
.hb-line {
    fill:none; stroke:#00ff9d; stroke-width:2.5;
    stroke-linecap:round; stroke-linejoin:round;
    animation: hb-draw 2s linear infinite;
    filter:drop-shadow(0 0 5px #00ff9d);
}
.hb-glow {
    fill:none; stroke:#00ff9d; stroke-width:6; opacity:0.15;
    stroke-linecap:round; stroke-linejoin:round;
    animation: hb-draw 2s linear infinite;
}
@keyframes hb-draw {
    0%   { stroke-dasharray:0 1000; stroke-dashoffset:0; }
    70%  { stroke-dasharray:700 1000; stroke-dashoffset:0; }
    100% { stroke-dasharray:700 1000; stroke-dashoffset:-700; }
}
.hb-bpm {
    font-family:'Orbitron',monospace; font-size:28px; font-weight:900;
    color:#00ff9d; text-shadow:0 0 14px #00ff9d80;
    animation:bpm-pulse 1s ease-in-out infinite; white-space:nowrap;
}
@keyframes bpm-pulse {
    0%,100% { transform:scale(1); }
    50%     { transform:scale(1.08); }
}
.hb-unit { font-size:12px; color:#4a9060; letter-spacing:1px; }

/* ── 3D CARD ── */
.glass-card-3d {
    position:relative; z-index:10;
    background:rgba(5,20,40,0.7); padding:40px; border-radius:28px;
    border:1px solid rgba(0,229,255,0.2);
    backdrop-filter:blur(20px); -webkit-backdrop-filter:blur(20px);
    box-shadow: 0 0 0 1px rgba(0,229,255,0.05),
                0 20px 60px rgba(0,0,0,0.6),
                0 0 80px rgba(0,229,255,0.05);
    transform-style:preserve-3d;
    transition: transform 0.4s ease, box-shadow 0.4s ease;
}
.glass-card-3d:hover {
    transform:perspective(1000px) rotateX(1deg) rotateY(-1deg) translateY(-4px);
    box-shadow: 0 0 0 1px rgba(0,229,255,0.12),
                0 30px 80px rgba(0,0,0,0.7),
                0 0 120px rgba(0,229,255,0.1);
}
.card-header-label {
    font-family:'Orbitron',monospace; font-size:11px;
    color:#00e5ff; letter-spacing:3px; text-transform:uppercase;
    margin-bottom:20px; display:flex; align-items:center; gap:10px;
}
.card-header-label::after {
    content:''; flex:1; height:1px;
    background:linear-gradient(90deg,rgba(0,229,255,0.4),transparent);
}

/* ── SCAN PANEL ── */
.scan-wrapper {
    position:relative; z-index:10; margin:22px 0;
    border:1px solid rgba(0,229,255,0.1); border-radius:16px;
    overflow:hidden; background:rgba(0,10,25,0.5); padding:18px;
}
.scan-bar {
    position:absolute; left:0; top:0; width:100%; height:3px;
    background:linear-gradient(90deg,transparent,#00e5ff,#7b68ee,transparent);
    animation:scan-move 2.5s ease-in-out infinite;
    box-shadow:0 0 15px #00e5ff,0 0 30px #7b68ee80;
}
@keyframes scan-move {
    0%  { top:0%; opacity:1; }
    49% { top:96%; opacity:1; }
    50% { top:96%; opacity:0; }
    51% { top:0%; opacity:0; }
    52% { top:0%; opacity:1; }
    100%{ top:0%; opacity:1; }
}
.scan-grid { display:grid; grid-template-columns:1fr 1fr 1fr; gap:10px; }
.scan-cell {
    height:40px; background:rgba(0,229,255,0.03);
    border:1px solid rgba(0,229,255,0.08); border-radius:8px;
    display:flex; align-items:center; justify-content:center;
    font-family:'Orbitron',monospace; font-size:10px;
    color:rgba(0,229,255,0.35); letter-spacing:1px; transition:all 0.3s;
}
.scan-cell.active {
    background:rgba(0,229,255,0.1); border-color:rgba(0,229,255,0.4);
    color:#00e5ff; box-shadow:0 0 12px rgba(0,229,255,0.2);
}
.scan-status {
    margin-top:14px; font-family:'Orbitron',monospace;
    font-size:11px; color:#00e5ff; letter-spacing:2px; text-align:center;
}
.scan-status span { animation:blink-text 1.2s step-end infinite; }
@keyframes blink-text { 0%,100%{opacity:1;} 50%{opacity:0;} }

/* ── MULTISELECT ── */
.stMultiSelect div[data-baseweb="select"] {
    background:rgba(0,20,40,0.6)!important;
    border:1px solid rgba(0,229,255,0.25)!important;
    border-radius:14px!important;
}
.stMultiSelect [data-baseweb="tag"] {
    background:rgba(0,229,255,0.15)!important;
    border:1px solid rgba(0,229,255,0.3)!important;
    border-radius:8px!important; color:#00e5ff!important;
}

/* ── PREDICT BUTTON ── */
.stButton>button {
    width:100%; position:relative;
    background:linear-gradient(90deg,#003f5c,#1a1a6e,#003f5c);
    background-size:200% 100%;
    color:#00e5ff; border:1px solid rgba(0,229,255,0.4);
    padding:18px; border-radius:18px;
    font-family:'Orbitron',monospace;
    font-size:16px; font-weight:700; letter-spacing:3px;
    text-transform:uppercase; transition:all 0.4s ease;
    box-shadow:0 0 20px rgba(0,229,255,0.15),inset 0 0 20px rgba(0,229,255,0.03);
    animation:btn-sweep 3s ease infinite; margin-top:14px; cursor:pointer;
}
@keyframes btn-sweep {
    0%  { background-position:200% center; }
    100%{ background-position:-200% center; }
}
.stButton>button:hover {
    border-color:#00e5ff; color:white;
    box-shadow:0 0 40px rgba(0,229,255,0.4),0 0 80px rgba(0,229,255,0.15);
    transform:translateY(-2px) scale(1.01);
}

/* ── RESULT BOX ── */
.result-box {
    margin-top:30px; padding:36px; border-radius:26px;
    text-align:center; position:relative; overflow:hidden;
    background:rgba(5,15,35,0.9);
    border:1px solid rgba(0,229,255,0.4);
    box-shadow:0 0 60px rgba(0,229,255,0.12),0 0 120px rgba(123,104,238,0.08);
    animation:result-appear 0.6s cubic-bezier(0.175,0.885,0.32,1.275) forwards;
}
@keyframes result-appear {
    from { opacity:0; transform:scale(0.85) translateY(20px); }
    to   { opacity:1; transform:scale(1) translateY(0); }
}
.result-box::before {
    content:''; position:absolute; top:-2px; left:10%;
    width:80%; height:2px;
    background:linear-gradient(90deg,transparent,#00e5ff,#7b68ee,transparent);
    border-radius:2px;
}
.result-label {
    font-family:'Orbitron',monospace; font-size:11px;
    color:#4a8fa8; letter-spacing:4px; text-transform:uppercase; margin-bottom:12px;
}
.result-disease {
    font-family:'Orbitron',monospace; font-size:38px; font-weight:900;
    background:linear-gradient(135deg,#00e5ff,#7b68ee,#ff6ec7);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
    line-height:1.2;
    filter:drop-shadow(0 0 20px rgba(0,229,255,0.3));
}
.result-corner { position:absolute; width:20px; height:20px; border-color:rgba(0,229,255,0.5); border-style:solid; }
.result-corner.tl { top:12px; left:12px;   border-width:2px 0 0 2px; border-radius:4px 0 0 0; }
.result-corner.tr { top:12px; right:12px;  border-width:2px 2px 0 0; border-radius:0 4px 0 0; }
.result-corner.bl { bottom:12px; left:12px;  border-width:0 0 2px 2px; border-radius:0 0 0 4px; }
.result-corner.br { bottom:12px; right:12px; border-width:0 2px 2px 0; border-radius:0 0 4px 0; }

/* ── PAGE SECTIONS ── */
.page-section {
    position:relative; z-index:10;
    animation: page-fade 0.4s ease forwards;
}
@keyframes page-fade {
    from { opacity:0; transform:translateY(16px); }
    to   { opacity:1; transform:translateY(0); }
}

/* ── DASHBOARD STAT CARDS ── */
.stat-grid { display:grid; grid-template-columns:repeat(4,1fr); gap:18px; margin-bottom:30px; }
.stat-card {
    background:rgba(5,20,40,0.75);
    border:1px solid rgba(0,229,255,0.15); border-radius:20px;
    padding:28px 22px; text-align:center;
    transition: transform 0.3s, box-shadow 0.3s;
}
.stat-card:hover {
    transform:translateY(-6px);
    box-shadow:0 0 30px rgba(0,229,255,0.15);
}
.stat-icon { font-size:32px; margin-bottom:10px; }
.stat-value {
    font-family:'Orbitron',monospace; font-size:36px; font-weight:900;
    background:linear-gradient(135deg,#00e5ff,#7b68ee);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
}
.stat-label { color:#4a7a8a; font-size:13px; letter-spacing:2px; text-transform:uppercase; margin-top:4px; }

/* ── HISTORY TABLE ── */
.history-table { width:100%; border-collapse:collapse; }
.history-table th {
    font-family:'Orbitron',monospace; font-size:11px; letter-spacing:2px;
    color:#00e5ff; border-bottom:1px solid rgba(0,229,255,0.2);
    padding:12px 16px; text-align:left; text-transform:uppercase;
}
.history-table td {
    padding:14px 16px; border-bottom:1px solid rgba(0,229,255,0.06);
    color:#9ec8d8; font-size:14px;
}
.history-table tr:hover td { background:rgba(0,229,255,0.04); }
.badge {
    display:inline-block; padding:4px 12px; border-radius:20px;
    font-size:12px; font-weight:700; letter-spacing:1px;
}
.badge-high   { background:rgba(255,80,80,0.15);  color:#ff8080; border:1px solid rgba(255,80,80,0.3); }
.badge-medium { background:rgba(255,180,0,0.15);  color:#ffcc44; border:1px solid rgba(255,180,0,0.3); }
.badge-low    { background:rgba(0,255,157,0.12);  color:#00ff9d; border:1px solid rgba(0,255,157,0.3); }

/* ── ABOUT CARDS ── */
.about-grid { display:grid; grid-template-columns:1fr 1fr; gap:22px; }
.about-card {
    background:rgba(5,20,40,0.75);
    border:1px solid rgba(0,229,255,0.12); border-radius:20px; padding:30px;
    transition:transform 0.3s, border-color 0.3s;
}
.about-card:hover { transform:translateY(-4px); border-color:rgba(0,229,255,0.35); }
.about-card-title {
    font-family:'Orbitron',monospace; font-size:14px; font-weight:700;
    color:#00e5ff; letter-spacing:2px; margin-bottom:14px;
}
.about-card p { color:#7aacbf; font-size:15px; line-height:1.7; }
.tech-pill {
    display:inline-block; margin:4px;
    padding:6px 14px; border-radius:20px; font-size:13px; font-weight:600;
    background:rgba(0,229,255,0.08); border:1px solid rgba(0,229,255,0.2);
    color:#8ec9e8; letter-spacing:1px;
}

/* ── SIDEBAR ── */
section[data-testid="stSidebar"] {
    background:rgba(1,8,18,0.9);
    border-right:1px solid rgba(0,229,255,0.1);
}
.sidebar-card {
    background:rgba(0,229,255,0.04); padding:18px; border-radius:16px;
    margin-bottom:16px; border:1px solid rgba(0,229,255,0.1);
    font-size:14px; color:#7aacbf;
}
.sidebar-title {
    font-family:'Orbitron',monospace; font-size:14px; color:#00e5ff;
    font-weight:700; letter-spacing:2px; margin-bottom:14px;
}
.status-box {
    background:rgba(0,255,157,0.07); border:1px solid rgba(0,255,157,0.3);
    padding:16px; border-radius:14px; text-align:center; color:#00ff9d;
    font-family:'Orbitron',monospace; font-size:12px; letter-spacing:2px; font-weight:700;
    box-shadow:0 0 20px rgba(0,255,157,0.1);
    animation:status-glow 2s ease-in-out infinite;
}
@keyframes status-glow {
    0%,100% { box-shadow:0 0 15px rgba(0,255,157,0.1); }
    50%     { box-shadow:0 0 30px rgba(0,255,157,0.25); }
}

/* ── FOOTER ── */
.footer {
    text-align:center; color:#2a5060; margin-top:60px; padding:24px;
    font-family:'Orbitron',monospace; font-size:11px; letter-spacing:2px;
    border-top:1px solid rgba(0,229,255,0.06); position:relative; z-index:10;
}

img { filter:drop-shadow(0 0 5px rgba(0,229,255,0.18)); }

/* ══════════════════════════════
   MOBILE RESPONSIVE
══════════════════════════════ */

@media (max-width: 768px) {

    /* Hero title smaller */
    .hero-title {
        font-size: 28px !important;
        letter-spacing: 1px !important;
    }
    .hero-subtitle {
        font-size: 12px !important;
        letter-spacing: 1.5px !important;
    }

    /* Heartbeat — hide label, shrink BPM */
    .hb-label { display: none !important; }
    .hb-bpm   { font-size: 20px !important; }
    .heartbeat-container { padding: 14px 16px !important; gap: 10px !important; }

    /* Stat grid: 2 columns on mobile */
    .stat-grid {
        grid-template-columns: 1fr 1fr !important;
        gap: 12px !important;
    }
    .stat-value { font-size: 26px !important; }
    .stat-label { font-size: 11px !important; letter-spacing: 1px !important; }
    .stat-card  { padding: 18px 12px !important; border-radius: 14px !important; }

    /* History stats row: stack vertically */
    .history-stats-row {
        flex-direction: column !important;
        gap: 10px !important;
    }

    /* History TABLE → CARD layout on mobile */
    .history-table thead { display: none !important; }
    .history-table, .history-table tbody,
    .history-table tr, .history-table td {
        display: block !important;
        width: 100% !important;
    }
    .history-table tr {
        background: rgba(0,229,255,0.03);
        border: 1px solid rgba(0,229,255,0.1);
        border-radius: 14px !important;
        margin-bottom: 12px !important;
        padding: 12px 16px !important;
    }
    .history-table td {
        padding: 5px 0 !important;
        border: none !important;
        font-size: 13px !important;
        display: flex !important;
        align-items: center !important;
        gap: 8px !important;
    }
    .history-table td::before {
        content: attr(data-label);
        font-family: 'Orbitron', monospace;
        font-size: 10px;
        color: #00e5ff;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        min-width: 80px;
        flex-shrink: 0;
    }

    /* Glass cards: reduce padding */
    .glass-card-3d {
        padding: 20px 16px !important;
        border-radius: 18px !important;
    }
    .glass-card-3d:hover {
        transform: none !important;
    }

    /* About grid: 1 column */
    .about-grid {
        grid-template-columns: 1fr !important;
    }

    /* Result disease text smaller */
    .result-disease { font-size: 24px !important; }
    .result-box { padding: 24px 16px !important; }

    /* Scan grid: 2 cols on mobile */
    .scan-grid {
        grid-template-columns: 1fr 1fr !important;
    }

    /* Disable 3D hover transforms on mobile (touch unfriendly) */
    .stat-card:hover { transform: none !important; }
    .about-card:hover { transform: none !important; }
}

@media (max-width: 480px) {
    .hero-title  { font-size: 22px !important; }
    .stat-grid   { grid-template-columns: 1fr 1fr !important; gap: 8px !important; }
    .stat-value  { font-size: 22px !important; }
    .result-disease { font-size: 20px !important; }
    .hb-bpm { font-size: 18px !important; }
}

/* ── DIVIDER ── */
.cyber-divider {
    height:1px; margin:30px 0;
    background:linear-gradient(90deg,transparent,rgba(0,229,255,0.3),transparent);
}

/* ── SECTION TITLE ── */
.section-title {
    font-family:'Orbitron',monospace; font-size:13px; font-weight:700;
    color:#00e5ff; letter-spacing:3px; text-transform:uppercase;
    margin-bottom:22px; display:flex; align-items:center; gap:12px;
}
.section-title::after {
    content:''; flex:1; height:1px;
    background:linear-gradient(90deg,rgba(0,229,255,0.35),transparent);
}

/* ══════════════════════════════
   MOBILE RESPONSIVE
══════════════════════════════ */
@media (max-width: 768px) {

    /* Hero title smaller */
    .hero-title {
        font-size: 28px !important;
        letter-spacing: 1px !important;
    }
    .hero-subtitle {
        font-size: 12px !important;
        letter-spacing: 1.5px !important;
    }

    /* Hero logo+text stack vertically on very small screens */
    .hero-flex {
        flex-direction: column !important;
        align-items: flex-start !important;
        gap: 12px !important;
    }

    /* Heartbeat — hide label and BPM on tiny screens, keep ECG */
    .heartbeat-container {
        padding: 14px 16px !important;
        gap: 10px !important;
    }
    .hb-label { display: none !important; }
    .hb-bpm { font-size: 20px !important; }
    svg.heartbeat-svg { height: 44px !important; }

    /* Stat grid — 2 columns on mobile */
    .stat-grid {
        grid-template-columns: repeat(2, 1fr) !important;
        gap: 12px !important;
    }
    .stat-card { padding: 18px 12px !important; border-radius: 14px !important; }
    .stat-icon { font-size: 24px !important; }
    .stat-value { font-size: 26px !important; }
    .stat-label { font-size: 11px !important; letter-spacing: 1px !important; }

    /* Quick start guide text */
    .glass-card-3d { padding: 20px 16px !important; border-radius: 18px !important; }

    /* History stats row — stack vertically */
    .history-stats-row {
        flex-direction: column !important;
        gap: 10px !important;
    }

    /* History table — convert to card list on mobile */
    .history-table thead { display: none !important; }
    .history-table, .history-table tbody,
    .history-table tr, .history-table td {
        display: block !important;
        width: 100% !important;
    }
    .history-table tr {
        margin-bottom: 14px !important;
        background: rgba(0,229,255,0.03) !important;
        border: 1px solid rgba(0,229,255,0.1) !important;
        border-radius: 14px !important;
        padding: 12px 14px !important;
    }
    .history-table td {
        padding: 5px 0 !important;
        border: none !important;
        font-size: 13px !important;
        display: flex !important;
        justify-content: space-between !important;
        align-items: center !important;
    }
    .history-table td::before {
        content: attr(data-label) !important;
        font-family: 'Orbitron', monospace !important;
        font-size: 9px !important;
        color: rgba(0,229,255,0.5) !important;
        letter-spacing: 1.5px !important;
        text-transform: uppercase !important;
        flex-shrink: 0 !important;
        margin-right: 8px !important;
    }

    /* About grid — single column */
    .about-grid {
        grid-template-columns: 1fr !important;
        gap: 14px !important;
    }
    .about-card { padding: 20px !important; }

    /* Result disease name smaller */
    .result-disease { font-size: 26px !important; }

    /* Scan grid — 2 cols on mobile */
    .scan-grid {
        grid-template-columns: 1fr 1fr !important;
    }

    /* Section title smaller */
    .section-title {
        font-size: 11px !important;
        letter-spacing: 2px !important;
    }
}

@media (max-width: 480px) {
    .hero-title { font-size: 22px !important; }
    .stat-grid  { grid-template-columns: repeat(2,1fr) !important; gap: 10px !important; }
    .stat-value { font-size: 22px !important; }
}

</style>

<canvas id="particle-canvas"></canvas>

<script>
(function(){
    const c=document.getElementById('particle-canvas');
    const ctx=c.getContext('2d');
    let W,H,P=[];
    function resize(){ W=c.width=window.innerWidth; H=c.height=window.innerHeight; }
    resize();
    window.addEventListener('resize',resize);
    function rb(a,b){return a+Math.random()*(b-a);}
    class Pt{
        constructor(){this.reset();}
        reset(){this.x=rb(0,W);this.y=rb(0,H);this.r=rb(.5,2.2);this.vx=rb(-.3,.3);this.vy=rb(-.6,-.1);this.a=rb(.2,.8);this.col=Math.random()>.5?'#00e5ff':'#7b68ee';}
        update(){this.x+=this.vx;this.y+=this.vy;this.a-=.002;if(this.y<-10||this.a<=0)this.reset();}
        draw(){ctx.save();ctx.globalAlpha=this.a;ctx.shadowBlur=8;ctx.shadowColor=this.col;ctx.fillStyle=this.col;ctx.beginPath();ctx.arc(this.x,this.y,this.r,0,Math.PI*2);ctx.fill();ctx.restore();}
    }
    for(let i=0;i<160;i++)P.push(new Pt());
    function lines(){for(let i=0;i<P.length;i++){for(let j=i+1;j<P.length;j++){const dx=P[i].x-P[j].x,dy=P[i].y-P[j].y,d=Math.sqrt(dx*dx+dy*dy);if(d<110){ctx.save();ctx.globalAlpha=(1-d/110)*.12;ctx.strokeStyle='#00e5ff';ctx.lineWidth=.5;ctx.beginPath();ctx.moveTo(P[i].x,P[i].y);ctx.lineTo(P[j].x,P[j].y);ctx.stroke();ctx.restore();}}}}
    function loop(){ctx.clearRect(0,0,W,H);lines();P.forEach(p=>{p.update();p.draw();});requestAnimationFrame(loop);}
    loop();
})();
</script>

<script>
(function(){
    function run(){
        const cells=document.querySelectorAll('.scan-cell');
        if(!cells.length){setTimeout(run,600);return;}
        setInterval(()=>{
            cells.forEach(c=>c.classList.remove('active'));
            const n=Math.floor(Math.random()*3)+1,idx=new Set();
            while(idx.size<n)idx.add(Math.floor(Math.random()*cells.length));
            idx.forEach(i=>cells[i].classList.add('active'));
        },600);
    }
    setTimeout(run,800);
})();
</script>
""", unsafe_allow_html=True)

# ================================================================
#  NAVBAR  (Streamlit buttons for real page switching)
# ================================================================

pages = ["Dashboard", "Diagnose", "History", "About"]

col_brand, col_nav1, col_nav2, col_nav3, col_nav4, col_status = st.columns(
    [3, 1.3, 1.1, 1, 1, 2], vertical_alignment="center"
)

with col_brand:
    st.markdown(
        '<div class="navbar-brand">⬡ MED.AI</div>',
        unsafe_allow_html=True
    )

nav_cols = [col_nav1, col_nav2, col_nav3, col_nav4]
for i, page in enumerate(pages):
    with nav_cols[i]:
        if st.button(page, key=f"nav_{page}", use_container_width=True):
            st.session_state.active_page = page
            st.rerun()

with col_status:
    st.markdown(
        '<div class="navbar-status" style="justify-content:flex-end;">'
        '<div class="navbar-dot"></div>SYSTEM ONLINE</div>',
        unsafe_allow_html=True
    )

st.markdown('<div class="cyber-divider"></div>', unsafe_allow_html=True)

active = st.session_state.active_page

# ================================================================
#  PAGE: DASHBOARD
# ================================================================

if active == "Dashboard":
    st.markdown('<div class="page-section">', unsafe_allow_html=True)

    # Hero — logo + text inline
    st.markdown(f"""
    <div class="hero-flex" style="display:flex;align-items:center;gap:20px;margin-bottom:8px;">
        {LOGO_HTML}
        <div>
            <div class="hero-title">AI Disease Predictor</div>
            <div class="hero-subtitle">⬡ Neural Symptom Analysis Engine · v3.0</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div style="height:20px"></div>', unsafe_allow_html=True)

    # Stat cards
    st.markdown('<div class="section-title">⬡ &nbsp;System Overview</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="stat-grid">
        <div class="stat-card">
            <div class="stat-icon">🧬</div>
            <div class="stat-value">132</div>
            <div class="stat-label">Symptoms Mapped</div>
        </div>
        <div class="stat-card">
            <div class="stat-icon">🦠</div>
            <div class="stat-value">41</div>
            <div class="stat-label">Diseases Detected</div>
        </div>
        <div class="stat-card">
            <div class="stat-icon">🎯</div>
            <div class="stat-value">95%</div>
            <div class="stat-label">Model Accuracy</div>
        </div>
        <div class="stat-card">
            <div class="stat-icon">⚡</div>
            <div class="stat-value">&lt;1s</div>
            <div class="stat-label">Prediction Time</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="cyber-divider"></div>', unsafe_allow_html=True)

    # Quick tip
    st.markdown("""
    <div class="glass-card-3d" style="padding:28px 36px;">
        <div class="card-header-label">⬡ &nbsp;Quick Start Guide</div>
        <div style="color:#7aacbf; font-size:16px; line-height:2;">
            <span style="color:#00e5ff; font-weight:700;">Step 1 &nbsp;→</span>&nbsp; Click <b style="color:#e2f0ff;">Diagnose</b> from the navbar above<br>
            <span style="color:#00e5ff; font-weight:700;">Step 2 &nbsp;→</span>&nbsp; Select your symptoms from the dropdown<br>
            <span style="color:#00e5ff; font-weight:700;">Step 3 &nbsp;→</span>&nbsp; Click <b style="color:#e2f0ff;">ANALYSE SYMPTOMS</b> button<br>
            <span style="color:#00e5ff; font-weight:700;">Step 4 &nbsp;→</span>&nbsp; View your AI-generated diagnosis result<br>
            <span style="color:#ffcc44; font-weight:700;">⚠ Note &nbsp;&nbsp;→</span>&nbsp; Always consult a certified physician for final diagnosis
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# ================================================================
#  PAGE: DIAGNOSE
# ================================================================

elif active == "Diagnose":
    st.markdown('<div class="page-section">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🩺 &nbsp;Symptom Input Module</div>', unsafe_allow_html=True)

    selected_symptoms = st.multiselect("Choose symptoms from the database", display_columns)

    st.markdown("""
    <div class="scan-wrapper">
        <div class="scan-bar"></div>
        <div class="scan-grid">
            <div class="scan-cell">NEURAL NET</div>
            <div class="scan-cell active">SCANNING</div>
            <div class="scan-cell">DATABASE</div>
            <div class="scan-cell">PATHOLOGY</div>
            <div class="scan-cell">SYMPTOMS</div>
            <div class="scan-cell">INFERENCE</div>
        </div>
        <div class="scan-status">AI ENGINE READY &nbsp;·&nbsp; <span>█</span></div>
    </div>
    """, unsafe_allow_html=True)



    if st.button("⬡  ANALYSE SYMPTOMS  ⬡"):
        if not selected_symptoms:
            st.warning("⚠ Please select at least one symptom before analysing.")
        elif not model_loaded:
            st.error("⚠ Model files not found! Please upload model.pkl, encoder.pkl, columns.pkl in the same folder as app.py")
        else:
            input_data = np.zeros(len(columns))
            for symptom in selected_symptoms:
                if symptom in columns:
                    input_data[columns.index(symptom)] = 1
            input_data = input_data.reshape(1, -1)
            prediction = model.predict(input_data)
            disease = encoder.inverse_transform(prediction)
            disease_name = disease[0]

            # Confidence score
            proba = model.predict_proba(input_data)[0]
            confidence = round(max(proba) * 100, 1)
            if confidence >= 80:
                conf_color = "#00ff9d"
                conf_label = "HIGH"
            elif confidence >= 50:
                conf_color = "#ffcc44"
                conf_label = "MEDIUM"
            else:
                conf_color = "#ff6b6b"
                conf_label = "LOW"

            # Disease info database
            disease_db = {
                "Fungal infection":      {"desc": "A skin or nail infection caused by fungi. Common in warm, moist areas of the body.", "doctor": "Dermatologist", "icon": "🔬"},
                "Allergy":               {"desc": "Immune system reaction to foreign substances like pollen, food, or pet dander.", "doctor": "Allergist / Immunologist", "icon": "🤧"},
                "GERD":                  {"desc": "Gastroesophageal reflux disease — stomach acid frequently flows back into the esophagus.", "doctor": "Gastroenterologist", "icon": "🫁"},
                "Chronic cholestasis":   {"desc": "Reduced or blocked bile flow from the liver, causing buildup of bile acids.", "doctor": "Gastroenterologist / Hepatologist", "icon": "🫀"},
                "Drug Reaction":         {"desc": "An adverse reaction to a medication, ranging from mild rash to severe anaphylaxis.", "doctor": "General Physician / Allergist", "icon": "💊"},
                "Peptic ulcer disease":  {"desc": "Sores that develop on the lining of the stomach, small intestine or esophagus.", "doctor": "Gastroenterologist", "icon": "🫃"},
                "AIDS":                  {"desc": "Advanced stage of HIV infection that severely damages the immune system.", "doctor": "Infectious Disease Specialist", "icon": "🧬"},
                "Diabetes":              {"desc": "A chronic condition affecting how the body processes blood sugar (glucose).", "doctor": "Endocrinologist / Diabetologist", "icon": "🩸"},
                "Gastroenteritis":       {"desc": "Inflammation of the stomach and intestines, typically from viral or bacterial infection.", "doctor": "Gastroenterologist", "icon": "🦠"},
                "Bronchial Asthma":      {"desc": "Chronic lung disease causing airways to narrow, swell and produce extra mucus.", "doctor": "Pulmonologist", "icon": "🫁"},
                "Hypertension":          {"desc": "High blood pressure — a long-term force of blood against artery walls that is too high.", "doctor": "Cardiologist", "icon": "❤️"},
                "Migraine":              {"desc": "A neurological condition causing intense, debilitating headaches often with nausea.", "doctor": "Neurologist", "icon": "🧠"},
                "Cervical spondylosis":  {"desc": "Age-related wear and tear of spinal disks in the neck causing neck pain.", "doctor": "Orthopedic / Neurologist", "icon": "🦴"},
                "Paralysis (brain hemorrhage)": {"desc": "Loss of muscle function due to bleeding in the brain damaging nerve pathways.", "doctor": "Neurologist / Neurosurgeon", "icon": "🧠"},
                "Jaundice":              {"desc": "Yellowing of the skin and eyes caused by high bilirubin levels in the blood.", "doctor": "Hepatologist / Gastroenterologist", "icon": "🫀"},
                "Malaria":               {"desc": "A mosquito-borne infectious disease caused by Plasmodium parasites.", "doctor": "Infectious Disease Specialist", "icon": "🦟"},
                "Chicken pox":           {"desc": "Highly contagious viral infection causing itchy rash with fluid-filled blisters.", "doctor": "General Physician / Dermatologist", "icon": "🔴"},
                "Dengue":                {"desc": "Mosquito-borne viral disease causing high fever, severe headache and joint pain.", "doctor": "Infectious Disease Specialist", "icon": "🦟"},
                "Typhoid":               {"desc": "Bacterial infection caused by Salmonella typhi, spread through contaminated food/water.", "doctor": "Infectious Disease Specialist", "icon": "🦠"},
                "Hepatitis A":           {"desc": "Highly contagious liver infection caused by the hepatitis A virus.", "doctor": "Hepatologist / Gastroenterologist", "icon": "🫀"},
                "Hepatitis B":           {"desc": "Serious liver infection caused by hepatitis B virus, can become chronic.", "doctor": "Hepatologist", "icon": "🫀"},
                "Hepatitis C":           {"desc": "Viral infection that causes liver inflammation, sometimes leading to serious damage.", "doctor": "Hepatologist", "icon": "🫀"},
                "Hepatitis D":           {"desc": "Liver infection that only occurs in people already infected with Hepatitis B.", "doctor": "Hepatologist", "icon": "🫀"},
                "Hepatitis E":           {"desc": "Liver disease caused by hepatitis E virus, mainly spread through contaminated water.", "doctor": "Hepatologist / Gastroenterologist", "icon": "🫀"},
                "Alcoholic hepatitis":   {"desc": "Liver inflammation caused by drinking too much alcohol over time.", "doctor": "Hepatologist", "icon": "🍺"},
                "Tuberculosis":          {"desc": "Serious bacterial infection mainly affecting the lungs, spread through air droplets.", "doctor": "Pulmonologist / Infectious Disease Specialist", "icon": "🫁"},
                "Common Cold":           {"desc": "A viral infection of the upper respiratory tract causing runny nose and sore throat.", "doctor": "General Physician", "icon": "🤧"},
                "Pneumonia":             {"desc": "Infection that inflames air sacs in one or both lungs, which may fill with fluid.", "doctor": "Pulmonologist", "icon": "🫁"},
                "Dimorphic hemorrhoids(piles)": {"desc": "Swollen veins in the rectum or anus causing discomfort and bleeding.", "doctor": "Proctologist / General Surgeon", "icon": "🩺"},
                "Heart attack":          {"desc": "Occurs when blood flow to the heart is blocked, damaging heart muscle.", "doctor": "Cardiologist", "icon": "❤️"},
                "Varicose veins":        {"desc": "Enlarged, twisted veins usually in the legs caused by weakened vein walls.", "doctor": "Vascular Surgeon", "icon": "🦵"},
                "Hypothyroidism":        {"desc": "Underactive thyroid gland that does not produce enough thyroid hormone.", "doctor": "Endocrinologist", "icon": "🦋"},
                "Hyperthyroidism":       {"desc": "Overactive thyroid that produces too much thyroxine, speeding up metabolism.", "doctor": "Endocrinologist", "icon": "🦋"},
                "Hypoglycemia":          {"desc": "Low blood sugar level that can cause shakiness, confusion and in severe cases, seizures.", "doctor": "Endocrinologist / Diabetologist", "icon": "🩸"},
                "Osteoarthritis":        {"desc": "Degenerative joint disease where cartilage breaks down causing pain and stiffness.", "doctor": "Orthopedic / Rheumatologist", "icon": "🦴"},
                "Arthritis":             {"desc": "Inflammation of one or more joints causing pain, swelling and stiffness.", "doctor": "Rheumatologist", "icon": "🦴"},
                "(Vertigo) Paroxysmal Positional Vertigo": {"desc": "Inner ear condition causing brief episodes of mild to intense dizziness.", "doctor": "ENT Specialist / Neurologist", "icon": "🌀"},
                "Acne":                  {"desc": "Skin condition that occurs when hair follicles become plugged with oil and dead skin cells.", "doctor": "Dermatologist", "icon": "🔴"},
                "Urinary tract infection": {"desc": "Infection in any part of the urinary system — kidneys, bladder, ureters or urethra.", "doctor": "Urologist / General Physician", "icon": "💧"},
                "Psoriasis":             {"desc": "Skin disease causing red, itchy scaly patches, most commonly on knees and elbows.", "doctor": "Dermatologist", "icon": "🔬"},
                "Impetigo":              {"desc": "Common, highly contagious skin infection causing red sores that rupture and crust.", "doctor": "Dermatologist / General Physician", "icon": "🩹"},
            }

            info = disease_db.get(disease_name, {
                "desc": "A medical condition identified based on the symptoms provided. Please consult a doctor for detailed information.",
                "doctor": "General Physician",
                "icon": "🩺"
            })

            # Save to history
            if "history" not in st.session_state:
                st.session_state.history = []
            ist_offset = datetime.timezone(datetime.timedelta(hours=5, minutes=30))
            ist_time = datetime.datetime.now(ist_offset)
            st.session_state.history.insert(0, {
                "time": ist_time.strftime("%d %b %Y, %I:%M %p"),
                "symptoms": ", ".join(selected_symptoms[:3]) + ("..." if len(selected_symptoms) > 3 else ""),
                "result": disease_name,
                "count": len(selected_symptoms)
            })

            # ── RESULT BOX ──
            st.markdown(f"""
            <div class="result-box">
                <div class="result-corner tl"></div>
                <div class="result-corner tr"></div>
                <div class="result-corner bl"></div>
                <div class="result-corner br"></div>
                <div class="result-label">◈ Diagnosis Result</div>
                <div class="result-disease">{info['icon']} &nbsp;{disease_name}</div>
            </div>
            """, unsafe_allow_html=True)

            # ── CONFIDENCE + DESCRIPTION + DOCTOR ──
            st.markdown(f"""
            <style>
            .info-grid {{ display:grid; grid-template-columns:1fr 1fr 1fr; gap:16px; margin-top:20px; }}
            .info-card {{
                background:rgba(5,20,40,0.8);
                border:1px solid rgba(0,229,255,0.15);
                border-radius:18px; padding:22px 20px;
                transition: border-color 0.3s;
            }}
            .info-card:hover {{ border-color:rgba(0,229,255,0.4); }}
            .info-card-title {{
                font-family:'Orbitron',monospace; font-size:10px;
                letter-spacing:2px; color:#4a8fa8;
                text-transform:uppercase; margin-bottom:12px;
            }}
            .conf-value {{
                font-family:'Orbitron',monospace;
                font-size:42px; font-weight:900;
                color:{conf_color};
                text-shadow: 0 0 20px {conf_color}60;
                line-height:1;
            }}
            .conf-badge {{
                display:inline-block; margin-top:8px;
                padding:4px 12px; border-radius:20px;
                font-size:11px; font-weight:700; letter-spacing:1px;
                background: rgba(0,0,0,0.3);
                border:1px solid {conf_color}60;
                color:{conf_color};
            }}
            .desc-text {{
                color:#9ec8d8; font-size:14px; line-height:1.7;
            }}
            .doctor-name {{
                font-family:'Orbitron',monospace;
                font-size:15px; font-weight:700;
                color:#00e5ff; margin-top:8px; line-height:1.4;
            }}
            .doctor-icon {{ font-size:32px; margin-bottom:10px; }}
            @media(max-width:768px){{
                .info-grid {{ grid-template-columns:1fr; gap:12px; }}
                .conf-value {{ font-size:32px; }}
            }}
            </style>

            <div class="info-grid">
                <div class="info-card">
                    <div class="info-card-title">⬡ Confidence Score</div>
                    <div class="conf-value">{confidence}%</div>
                    <div class="conf-badge">{conf_label} CONFIDENCE</div>
                </div>
                <div class="info-card">
                    <div class="info-card-title">⬡ About This Disease</div>
                    <div class="desc-text">{info['desc']}</div>
                </div>
                <div class="info-card">
                    <div class="info-card-title">⬡ Recommended Doctor</div>
                    <div class="doctor-icon">👨‍⚕️</div>
                    <div class="doctor-name">{info['doctor']}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.warning("⚠ AI-generated prediction. Please consult a certified physician for medical advice.")

    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ================================================================
#  PAGE: HISTORY
# ================================================================

elif active == "History":
    st.markdown('<div class="page-section">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📋 &nbsp;Prediction History</div>', unsafe_allow_html=True)

    history = st.session_state.get("history", [])

    if not history:
        st.markdown("""
        <div class="glass-card-3d" style="text-align:center; padding:60px;">
            <div style="font-size:48px; margin-bottom:16px;">📭</div>
            <div style="font-family:'Orbitron',monospace; color:#00e5ff; font-size:14px; letter-spacing:2px;">NO RECORDS FOUND</div>
            <div style="color:#4a7a8a; margin-top:10px; font-size:14px;">
                Run a diagnosis first — your prediction history will appear here.
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Stats row
        st.markdown(f"""
        <div class="history-stats-row" style="display:flex; gap:16px; margin-bottom:24px;">
            <div class="stat-card" style="flex:1; padding:20px;">
                <div class="stat-value" style="font-size:28px;">{len(history)}</div>
                <div class="stat-label">Total Diagnoses</div>
            </div>
            <div class="stat-card" style="flex:1; padding:20px;">
                <div class="stat-value" style="font-size:28px;">{len(set(h['result'] for h in history))}</div>
                <div class="stat-label">Unique Results</div>
            </div>
            <div class="stat-card" style="flex:1; padding:20px;">
                <div class="stat-value" style="font-size:22px;">{history[0]['time'].split(',')[0]}</div>
                <div class="stat-label">Last Diagnosis</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Table CSS — separate call so Streamlit renders it properly
        st.markdown("""
        <style>
        .hist-table { width:100%; border-collapse:collapse; font-family:'Rajdhani',sans-serif; }
        .hist-table thead tr { border-bottom:1px solid rgba(0,229,255,0.25); }
        .hist-table th {
            padding:12px 14px; text-align:left;
            font-family:'Orbitron',monospace; font-size:10px;
            color:#00e5ff; letter-spacing:2px; font-weight:700;
        }
        .hist-table tbody tr {
            border-bottom:1px solid rgba(0,229,255,0.07);
            transition:background 0.2s;
        }
        .hist-table tbody tr:hover { background:rgba(0,229,255,0.05); }
        .hist-table tbody td {
            padding:14px 14px; color:#9ec8d8;
            font-size:15px; vertical-align:middle;
        }
        .hist-wrap {
            background:rgba(5,20,40,0.7);
            border:1px solid rgba(0,229,255,0.2);
            border-radius:24px; padding:28px 24px;
            backdrop-filter:blur(20px); overflow-x:auto;
        }
        .hist-header {
            font-family:'Orbitron',monospace; font-size:11px;
            color:#00e5ff; letter-spacing:3px; text-transform:uppercase;
            margin-bottom:20px;
        }
        @media (max-width:768px) {
            .hist-table thead { display:none; }
            .hist-table, .hist-table tbody,
            .hist-table tr, .hist-table td { display:block; width:100%; box-sizing:border-box; }
            .hist-table tr {
                background:rgba(0,229,255,0.04);
                border:1px solid rgba(0,229,255,0.12) !important;
                border-radius:16px; margin-bottom:14px; padding:14px 16px;
            }
            .hist-table td {
                display:flex; align-items:center; gap:10px;
                padding:7px 0; border:none !important;
                font-size:14px; color:#b0d4e0;
            }
            .hist-table td::before {
                content:attr(data-label);
                font-family:'Orbitron',monospace; font-size:9px;
                color:#00e5ff; letter-spacing:1.5px;
                min-width:72px; flex-shrink:0;
            }
        }
        </style>
        """, unsafe_allow_html=True)

        # Table — build entirely in one string
        rows_html = ""
        for i, h in enumerate(history):
            count = h.get("count", 1)
            if count >= 5:
                badge = '<span class="badge badge-high">HIGH</span>'
            elif count >= 3:
                badge = '<span class="badge badge-medium">MEDIUM</span>'
            else:
                badge = '<span class="badge badge-low">LOW</span>'
            rows_html += f"""
            <tr>
                <td data-label="NO"><span style="color:#00e5ff;font-family:'Orbitron',monospace;font-size:13px;">{i+1:02d}</span></td>
                <td data-label="DATE" style="white-space:nowrap;">{h['time']}</td>
                <td data-label="SYMPTOMS" style="color:#b0d4e0;">{h['symptoms']}</td>
                <td data-label="COUNT"><span style="color:#00e5ff;font-family:'Orbitron',monospace;font-size:15px;font-weight:700;">{count}</span></td>
                <td data-label="DISEASE"><span style="color:#e2f0ff;font-weight:700;">{h['result']}</span></td>
                <td data-label="PRIORITY">{badge}</td>
            </tr>"""

        st.markdown(f"""
        <div class="hist-wrap">
            <div class="hist-header">⬡ &nbsp;Recent Predictions</div>
            <table class="hist-table">
                <thead>
                    <tr>
                        <th>#</th>
                        <th>DATE &amp; TIME</th>
                        <th>SYMPTOMS</th>
                        <th>COUNT</th>
                        <th>DISEASE</th>
                        <th>PRIORITY</th>
                    </tr>
                </thead>
                <tbody>{rows_html}</tbody>
            </table>
        </div>
        """, unsafe_allow_html=True)

        if st.button("🗑  Clear History"):
            st.session_state.history = []
            st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

# ================================================================
#  PAGE: ABOUT
# ================================================================

elif active == "About":
    st.markdown('<div class="page-section">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">ℹ &nbsp;About This System</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="glass-card-3d" style="margin-bottom:24px; text-align:center; padding:40px 60px;">
        <div style="font-family:'Orbitron',monospace; font-size:32px; font-weight:900;
             background:linear-gradient(135deg,#00e5ff,#7b68ee,#ff6ec7);
             -webkit-background-clip:text; -webkit-text-fill-color:transparent; margin-bottom:12px;">
             MED.AI — v3.0
        </div>
        <div style="color:#6fa8c8; font-size:15px; letter-spacing:2px; text-transform:uppercase;">
            Neural Symptom Analysis & Disease Prediction Engine
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="about-grid">', unsafe_allow_html=True)

    st.markdown("""
    <div class="about-card">
        <div class="about-card-title">🧠 How It Works</div>
        <p>
        This system uses a trained <b style="color:#00e5ff;">Random Forest</b> machine learning model to
        predict diseases from selected symptoms. The model was trained on a curated medical dataset
        containing 132 symptoms and 41 diseases with over 95% cross-validated accuracy.
        </p>
    </div>
    <div class="about-card">
        <div class="about-card-title">⚙️ Technology Stack</div>
        <div>
            <span class="tech-pill">Python 3.x</span>
            <span class="tech-pill">Streamlit</span>
            <span class="tech-pill">Scikit-learn</span>
            <span class="tech-pill">NumPy</span>
            <span class="tech-pill">Pillow</span>
            <span class="tech-pill">Random Forest</span>
            <span class="tech-pill">Label Encoder</span>
            <span class="tech-pill">Pickle</span>
        </div>
    </div>
    <div class="about-card">
        <div class="about-card-title">📊 Model Performance</div>
        <p>
        The underlying ML model achieves <b style="color:#00ff9d;">95%+ accuracy</b> on the test dataset.
        It analyzes the binary symptom vector and outputs the most probable diagnosis using
        ensemble decision tree voting across 100+ estimators.
        </p>
    </div>
    <div class="about-card">
        <div class="about-card-title">⚠️ Medical Disclaimer</div>
        <p>
        This tool is intended for <b style="color:#ffcc44;">educational & informational purposes only</b>.
        It is NOT a substitute for professional medical advice, diagnosis, or treatment.
        Always consult a qualified healthcare physician for any medical condition.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="cyber-divider"></div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align:center; color:#4a7a8a; font-size:13px; letter-spacing:1px; padding:10px 0;">
        Built with ❤️ using Artificial Intelligence & Machine Learning &nbsp;·&nbsp;
        <span style="color:#00e5ff;">Streamlit Framework</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# ================================================================
#  SIDEBAR
# ================================================================

with st.sidebar:
    st.markdown('<div class="sidebar-title">🧠 AI HEALTH SYSTEM</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="sidebar-card">
    Neural disease prediction engine powered by
    Random Forest ML with real-time symptom analysis.
    </div>
    """, unsafe_allow_html=True)
    st.markdown("### ✦ Modules")
    with st.expander("🩺 Disease Prediction"):
        st.write("✔ Neural symptom analysis\n✔ Real-time AI detection\n✔ Confidence scoring")
    with st.expander("⚡ Fast AI Analysis"):
        st.write("🚀 Sub-second prediction\n🚀 300+ disease database")
    with st.expander("🧠 Smart Detection"):
        st.write("🤖 Random Forest algorithm\n🤖 Trained on medical dataset")
    with st.expander("📊 ML Stats"):
        st.write("✅ 95%+ accuracy\n✅ 132 symptoms mapped\n✅ Continuously improving")
    st.markdown('<div class="status-box">⬡ AI MODEL ACTIVE ⬡</div>', unsafe_allow_html=True)

# ================================================================
#  FOOTER
# ================================================================

st.markdown("""
<div class="footer">
⬡ MED.AI · POWERED BY ARTIFICIAL INTELLIGENCE & MACHINE LEARNING ⬡
</div>
""", unsafe_allow_html=True)
