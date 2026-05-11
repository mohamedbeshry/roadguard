import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
import time
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import Counter

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="RoadScan AI",
    page_icon="🚧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
}

.main { background-color: #0f1117; }

/* Header */
.hero-header {
    background: linear-gradient(135deg, #1a1f2e 0%, #16213e 50%, #0f3460 100%);
    border: 1px solid #e94560;
    border-radius: 16px;
    padding: 2.5rem 2rem;
    margin-bottom: 2rem;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.hero-header::before {
    content: '';
    position: absolute;
    top: -50%; left: -50%;
    width: 200%; height: 200%;
    background: radial-gradient(circle, rgba(233,69,96,0.08) 0%, transparent 60%);
    animation: pulse 4s ease-in-out infinite;
}
@keyframes pulse {
    0%, 100% { transform: scale(1); opacity: 0.5; }
    50% { transform: scale(1.1); opacity: 1; }
}
.hero-title {
    font-size: 3rem;
    font-weight: 700;
    color: #ffffff;
    margin: 0;
    letter-spacing: -1px;
}
.hero-title span { color: #e94560; }
.hero-subtitle {
    color: #8892a4;
    font-size: 1.1rem;
    margin-top: 0.5rem;
    font-weight: 300;
}

/* Metric cards */
.metric-card {
    background: #1a1f2e;
    border: 1px solid #2d3748;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    text-align: center;
    transition: border-color 0.3s;
}
.metric-card:hover { border-color: #e94560; }
.metric-value {
    font-size: 2rem;
    font-weight: 700;
    color: #e94560;
    font-family: 'JetBrains Mono', monospace;
}
.metric-label {
    font-size: 0.8rem;
    color: #8892a4;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 0.3rem;
}

/* Prediction result */
.pred-box {
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    margin: 1rem 0;
}
.pred-box.pothole  { background: linear-gradient(135deg, #2d1b1b, #4a1f1f); border: 2px solid #e74c3c; }
.pred-box.crack    { background: linear-gradient(135deg, #1b2a2d, #1f3a4a); border: 2px solid #3498db; }
.pred-box.manhole  { background: linear-gradient(135deg, #1b2d1b, #1f4a1f); border: 2px solid #2ecc71; }

.pred-class {
    font-size: 2.5rem;
    font-weight: 700;
    color: #ffffff;
    text-transform: uppercase;
    letter-spacing: 2px;
}
.pred-conf {
    font-size: 1.1rem;
    color: #8892a4;
    margin-top: 0.5rem;
    font-family: 'JetBrains Mono', monospace;
}

/* Confidence bars */
.conf-bar-container {
    background: #2d3748;
    border-radius: 8px;
    height: 12px;
    margin: 0.3rem 0;
    overflow: hidden;
}
.conf-bar {
    height: 100%;
    border-radius: 8px;
    transition: width 0.8s ease;
}

/* Upload zone */
.upload-zone {
    border: 2px dashed #2d3748;
    border-radius: 16px;
    padding: 3rem;
    text-align: center;
    color: #8892a4;
    transition: border-color 0.3s;
}
.upload-zone:hover { border-color: #e94560; }

/* Sidebar */
.sidebar-info {
    background: #1a1f2e;
    border: 1px solid #2d3748;
    border-radius: 10px;
    padding: 1rem;
    margin: 0.5rem 0;
    font-size: 0.85rem;
    color: #8892a4;
}

/* Model badge */
.model-badge {
    display: inline-block;
    background: #e94560;
    color: white;
    font-size: 0.75rem;
    padding: 0.2rem 0.7rem;
    border-radius: 20px;
    font-weight: 600;
    letter-spacing: 0.5px;
}

/* Section headers */
.section-header {
    font-size: 1.1rem;
    font-weight: 600;
    color: #ffffff;
    border-left: 3px solid #e94560;
    padding-left: 0.8rem;
    margin: 1.5rem 0 1rem 0;
}

/* History card */
.history-item {
    background: #1a1f2e;
    border: 1px solid #2d3748;
    border-radius: 10px;
    padding: 0.8rem 1rem;
    margin: 0.4rem 0;
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 0.85rem;
}
</style>
""", unsafe_allow_html=True)

# ── Constants ──────────────────────────────────────────────────
# Order MUST match flow_from_directory alphabetical sort:
# dataset/augmented → crack=0, manhole=1, pothole=2
CLASSES  = ["crack", "manhole", "pothole"]
IMG_SIZE = 128
CLASS_INFO = {
    "pothole": {
        "emoji": "⚠️",
        "color": "#e74c3c",
        "severity": "High",
        "desc": "Potholes are depressions in road surface. Immediate repair recommended to prevent vehicle damage and accidents.",
        "action": "Report to municipality — Priority repair needed"
    },
    "crack": {
        "emoji": "🔍",
        "color": "#3498db",
        "severity": "Medium",
        "desc": "Surface cracks indicate early-stage road deterioration. Sealing prevents water infiltration and further damage.",
        "action": "Schedule maintenance — Sealant application recommended"
    },
    "manhole": {
        "emoji": "🔵",
        "color": "#2ecc71",
        "severity": "Low",
        "desc": "Manhole cover detected. Check for proper seating, height alignment with road surface, and structural integrity.",
        "action": "Inspect cover condition and flush alignment"
    }
}

# ── Verified class mapping (alphabetical from flow_from_directory) ──
# dataset/augmented folders: crack/ manhole/ pothole/
# Alphabetical → crack=0, manhole=1, pothole=2
# CLASSES list must match this exactly
assert CLASSES == ["crack", "manhole", "pothole"], f"Wrong CLASSES order: {CLASSES}"

# ── Load models ────────────────────────────────────────────────
@st.cache_resource
def load_models():
    models_loaded = {}
    errors = []

    for name, path in [
        ("CNN Scratch",       "models/cnn_scratch_best.h5"),
        ("Transfer Learning", "models/tl_model_best.h5")
    ]:
        if os.path.exists(path):
            try:
                models_loaded[name] = tf.keras.models.load_model(path)
            except Exception as e:
                errors.append(f"{name}: {e}")
        else:
            errors.append(f"{name}: file not found at {path}")

    return models_loaded, errors

# ── Predict ────────────────────────────────────────────────────
def predict(model, img_array):
    start     = time.time()
    probs     = model.predict(img_array, verbose=0)[0]
    latency   = (time.time() - start) * 1000
    class_idx = np.argmax(probs)
    return CLASSES[class_idx], probs, latency

def preprocess(image: Image.Image):
    # Handle any image mode (RGBA, L, P, etc.) -> force RGB
    img = np.array(image.convert("RGB"))        # always 3-channel RGB
    # Match Cell 3 training pipeline exactly:
    # cv2.imread (BGR) -> cvtColor BGR2RGB -> /255
    # We simulate: RGB -> BGR -> resize -> RGB -> /255
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # RGB -> BGR
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))  # resize to 128x128
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR -> RGB
    img = img.astype("float32") / 255.0
    return img[np.newaxis]  # shape: (1, 128, 128, 3)

def confidence_bars(probs):
    colors = {"manhole": "#2ecc71", "pothole": "#e74c3c", "crack": "#3498db"}
    for i, cls in enumerate(CLASSES):
        pct = probs[i] * 100
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"""
            <div style="display:flex; align-items:center; gap:10px; margin:6px 0;">
                <span style="color:#8892a4; font-size:0.85rem; min-width:70px;">{cls}</span>
                <div class="conf-bar-container" style="flex:1;">
                    <div class="conf-bar" style="width:{pct:.1f}%; background:{colors[cls]};"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"<span style='font-family:JetBrains Mono,monospace; color:#fff; font-size:0.85rem;'>{pct:.1f}%</span>",
                        unsafe_allow_html=True)

# ── Session state ──────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []

# ── HERO ──────────────────────────────────────────────────────
st.markdown("""
<div class="hero-header">
    <div class="hero-title">🚧 Road<span>Scan</span> AI</div>
    <div class="hero-subtitle">
        Deep Learning–powered road damage detection · Potholes · Cracks · Manholes
    </div>
</div>
""", unsafe_allow_html=True)

# ── Stats row ─────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown("""<div class="metric-card">
        <div class="metric-value">82.1%</div>
        <div class="metric-label">CNN Accuracy</div>
    </div>""", unsafe_allow_html=True)
with c2:
    st.markdown("""<div class="metric-card">
        <div class="metric-value">78.2%</div>
        <div class="metric-label">TL Accuracy</div>
    </div>""", unsafe_allow_html=True)
with c3:
    st.markdown("""<div class="metric-card">
        <div class="metric-value">7,524</div>
        <div class="metric-label">Training Images</div>
    </div>""", unsafe_allow_html=True)
with c4:
    st.markdown("""<div class="metric-card">
        <div class="metric-value">3</div>
        <div class="metric-label">Damage Classes</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")

    # Load models
    models_dict, load_errors = load_models()

    # Errors logged silently — don't show yellow warnings to users
    if load_errors:
        for err in load_errors:
            print(f"[Model load warning] {err}")  # terminal only

    available = list(models_dict.keys())
    if not available:
        st.error("❌ No models found. Make sure .h5 files are in the models/ folder.")
        st.stop()

    selected_model = st.selectbox("🤖 Select Model", available)
    model = models_dict[selected_model]

    st.markdown(f"""<div class="sidebar-info">
        <b>Model:</b> {selected_model}<br>
        <b>Input:</b> 128×128 RGB<br>
        <b>Classes:</b> Manhole, Pothole, Crack<br>
        <b>Parameters:</b> {model.count_params():,}
    </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("## 📊 Class Guide")
    for cls, info in CLASS_INFO.items():
        st.markdown(f"""<div class="sidebar-info">
            {info['emoji']} <b style="color:{info['color']}">{cls.upper()}</b><br>
            Severity: <b>{info['severity']}</b><br>
            {info['desc'][:80]}...
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("## 📈 Session Stats")
    total = len(st.session_state.history)
    if total > 0:
        class_counts = Counter([h['class'] for h in st.session_state.history])
        st.markdown(f"**Total scanned:** {total}")
        for cls, cnt in class_counts.items():
            st.markdown(f"- {cls}: **{cnt}**")
        if st.button("🗑️ Clear History"):
            st.session_state.history = []
            st.rerun()
    else:
        st.markdown("*No images scanned yet*")

# ── Main layout ────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔍 Scan Image", "📊 Model Info", "📋 History"])

# ══════════════════════════════════════════════════════════════
# TAB 1 — Scan
# ══════════════════════════════════════════════════════════════
with tab1:
    col_upload, col_result = st.columns([1, 1], gap="large")

    with col_upload:
        st.markdown('<div class="section-header">Upload Road Image</div>',
                    unsafe_allow_html=True)

        uploaded = st.file_uploader(
            "Drag & drop or click to upload",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed"
        )

        if uploaded:
            image = Image.open(uploaded)
            st.image(image, caption="Uploaded image", use_container_width=True)

            # Image info
            st.markdown(f"""<div class="sidebar-info">
                📐 Size: {image.size[0]}×{image.size[1]} px &nbsp;|&nbsp;
                🎨 Mode: {image.mode} &nbsp;|&nbsp;
                📁 {uploaded.name}
            </div>""", unsafe_allow_html=True)

            scan_btn = st.button("🚀 Scan for Damage", type="primary",
                                 use_container_width=True)
        else:
            st.markdown("""<div class="upload-zone">
                <div style="font-size:3rem;">📸</div>
                <div style="font-size:1.1rem; margin-top:1rem;">Drop a road image here</div>
                <div style="font-size:0.85rem; margin-top:0.5rem;">JPG, JPEG, PNG supported</div>
            </div>""", unsafe_allow_html=True)
            scan_btn = False

    with col_result:
        st.markdown('<div class="section-header">Detection Result</div>',
                    unsafe_allow_html=True)

        if uploaded and scan_btn:
            with st.spinner("Analyzing road surface..."):
                img_array              = preprocess(image)
                pred_class, probs, lat = predict(model, img_array)
                info                   = CLASS_INFO[pred_class]
                confidence             = probs[CLASSES.index(pred_class)] * 100

            # Result box
            st.markdown(f"""
            <div class="pred-box {pred_class}">
                <div style="font-size:3rem;">{info['emoji']}</div>
                <div class="pred-class">{pred_class}</div>
                <div class="pred-conf">Confidence: {confidence:.1f}%</div>
                <div style="margin-top:0.8rem;">
                    <span style="background:{info['color']}22; color:{info['color']};
                    border:1px solid {info['color']}; border-radius:20px;
                    padding:0.2rem 0.8rem; font-size:0.8rem; font-weight:600;">
                        Severity: {info['severity']}
                    </span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Confidence bars
            st.markdown('<div class="section-header">Class Probabilities</div>',
                        unsafe_allow_html=True)
            confidence_bars(probs)

            # Info & action
            st.markdown('<div class="section-header">Assessment</div>',
                        unsafe_allow_html=True)
            st.info(info['desc'])
            st.success(f"✅ **Recommended Action:** {info['action']}")

            # Latency
            st.markdown(f"""<div class="sidebar-info" style="text-align:center;">
                ⚡ Inference time: <b style="color:#e94560; font-family:JetBrains Mono,monospace;">{lat:.1f} ms</b>
                &nbsp;|&nbsp; Model: <b>{selected_model}</b>
            </div>""", unsafe_allow_html=True)

            # Save to history
            st.session_state.history.append({
                "file"      : uploaded.name,
                "class"     : pred_class,
                "confidence": confidence,
                "model"     : selected_model,
                "latency"   : lat,
                "time"      : time.strftime("%H:%M:%S")
            })

        elif not uploaded:
            st.markdown("""<div style="text-align:center; padding:4rem 2rem; color:#8892a4;">
                <div style="font-size:4rem; opacity:0.3;">🛣️</div>
                <div style="margin-top:1rem; font-size:1rem;">
                    Upload an image to begin scanning
                </div>
            </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# TAB 2 — Model Info
# ══════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-header">Model Comparison</div>',
                unsafe_allow_html=True)

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("""<div class="metric-card" style="text-align:left;">
            <div style="font-size:1rem; font-weight:700; color:#fff; margin-bottom:1rem;">
                🧠 CNN from Scratch
            </div>
            <table style="width:100%; font-size:0.85rem; color:#8892a4;">
                <tr><td>Test Accuracy</td><td style="color:#e94560; font-weight:700; text-align:right;">82.11%</td></tr>
                <tr><td>F1-Score</td><td style="color:#fff; text-align:right;">0.8194</td></tr>
                <tr><td>Precision</td><td style="color:#fff; text-align:right;">0.8269</td></tr>
                <tr><td>Recall</td><td style="color:#fff; text-align:right;">0.8211</td></tr>
                <tr><td>Parameters</td><td style="color:#fff; text-align:right;">8,913,603</td></tr>
                <tr><td>Training Time</td><td style="color:#fff; text-align:right;">129.97 min</td></tr>
                <tr><td>Input Size</td><td style="color:#fff; text-align:right;">128×128</td></tr>
            </table>
        </div>""", unsafe_allow_html=True)

    with col_b:
        st.markdown("""<div class="metric-card" style="text-align:left;">
            <div style="font-size:1rem; font-weight:700; color:#fff; margin-bottom:1rem;">
                🔁 Transfer Learning (MobileNetV2)
            </div>
            <table style="width:100%; font-size:0.85rem; color:#8892a4;">
                <tr><td>Test Accuracy</td><td style="color:#3498db; font-weight:700; text-align:right;">78.21%</td></tr>
                <tr><td>F1-Score</td><td style="color:#fff; text-align:right;">0.7813</td></tr>
                <tr><td>Precision</td><td style="color:#fff; text-align:right;">0.7881</td></tr>
                <tr><td>Recall</td><td style="color:#fff; text-align:right;">0.7821</td></tr>
                <tr><td>Parameters</td><td style="color:#fff; text-align:right;">3,166,915</td></tr>
                <tr><td>Training Time</td><td style="color:#fff; text-align:right;">2 phases</td></tr>
                <tr><td>Input Size</td><td style="color:#fff; text-align:right;">128×128</td></tr>
            </table>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-header">Per-class Performance (CNN Scratch)</div>',
                unsafe_allow_html=True)

    per_class = {
        "manhole" : {"precision": 0.75, "recall": 0.90, "f1": 0.82, "support": 376},
        "pothole" : {"precision": 0.81, "recall": 0.68, "f1": 0.74, "support": 377},
        "crack"   : {"precision": 0.92, "recall": 0.88, "f1": 0.90, "support": 376},
    }
    colors_cls = {"manhole": "#2ecc71", "pothole": "#e74c3c", "crack": "#3498db"}

    for cls, m in per_class.items():
        c1, c2, c3, c4 = st.columns([2, 2, 2, 1])
        with c1:
            st.markdown(f"<span style='color:{colors_cls[cls]}; font-weight:700;'>{cls.upper()}</span>",
                        unsafe_allow_html=True)
        with c2:
            st.progress(m['precision'], text=f"Precision {m['precision']:.0%}")
        with c3:
            st.progress(m['recall'], text=f"Recall {m['recall']:.0%}")
        with c4:
            st.markdown(f"<span style='color:#e94560; font-weight:700;'>F1: {m['f1']:.2f}</span>",
                        unsafe_allow_html=True)

    st.markdown('<div class="section-header">Dataset Summary</div>',
                unsafe_allow_html=True)
    d1, d2, d3, d4 = st.columns(4)
    for col, label, val in zip(
        [d1, d2, d3, d4],
        ["Total Images", "After Augmentation", "Train Split", "Test Split"],
        ["4,601", "7,524", "5,266", "1,129"]
    ):
        with col:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-value" style="font-size:1.5rem;">{val}</div>
                <div class="metric-label">{label}</div>
            </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# TAB 3 — History
# ══════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-header">Scan History</div>',
                unsafe_allow_html=True)

    if not st.session_state.history:
        st.markdown("""<div style="text-align:center; padding:3rem; color:#8892a4;">
            No scans yet. Upload an image in the Scan tab to get started.
        </div>""", unsafe_allow_html=True)
    else:
        # Summary
        total    = len(st.session_state.history)
        avg_conf = np.mean([h['confidence'] for h in st.session_state.history])
        avg_lat  = np.mean([h['latency']    for h in st.session_state.history])
        counts   = Counter([h['class']      for h in st.session_state.history])

        s1, s2, s3 = st.columns(3)
        with s1:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-value">{total}</div>
                <div class="metric-label">Total Scans</div>
            </div>""", unsafe_allow_html=True)
        with s2:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-value">{avg_conf:.1f}%</div>
                <div class="metric-label">Avg Confidence</div>
            </div>""", unsafe_allow_html=True)
        with s3:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-value">{avg_lat:.0f}ms</div>
                <div class="metric-label">Avg Latency</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # History table
        colors_cls = {"manhole": "#2ecc71", "pothole": "#e74c3c", "crack": "#3498db"}
        for h in reversed(st.session_state.history):
            c = h['class']
            st.markdown(f"""
            <div class="history-item">
                <span style="color:{colors_cls[c]}; font-weight:600;">
                    {CLASS_INFO[c]['emoji']} {c.upper()}
                </span>
                <span style="color:#8892a4;">{h['file']}</span>
                <span style="color:#fff; font-family:JetBrains Mono,monospace;">
                    {h['confidence']:.1f}%
                </span>
                <span style="color:#8892a4; font-size:0.8rem;">{h['model']}</span>
                <span style="color:#8892a4; font-size:0.8rem;">{h['time']}</span>
            </div>""", unsafe_allow_html=True)

        # Pie chart
        if len(counts) > 1:
            st.markdown('<div class="section-header">Detection Distribution</div>',
                        unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(5, 4),
                                   facecolor='#1a1f2e')
            ax.set_facecolor('#1a1f2e')
            wedge_colors = [colors_cls[c] for c in counts.keys()]
            ax.pie(counts.values(), labels=counts.keys(),
                   colors=wedge_colors, autopct='%1.0f%%',
                   textprops={'color': 'white', 'fontsize': 11},
                   wedgeprops={'edgecolor': '#0f1117', 'linewidth': 2})
            st.pyplot(fig, use_container_width=False)