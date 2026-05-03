import streamlit as st
import numpy as np
import pandas as pd
import pickle
import math
import os

# ──────────────────────────────────────────────
# Numpy pickle compatibility
# ──────────────────────────────────────────────
try:
    from numpy.random import _pickle as np_pickle
    import numpy.random as npr

    for name, cls in list(np_pickle.BitGenerators.items()):
        if isinstance(name, str):
            attr = getattr(npr, name, None)
            if attr is not None:
                np_pickle.BitGenerators[attr] = cls

    def _compat_bit_generator_ctor(bit_generator_name='MT19937'):
        if bit_generator_name in np_pickle.BitGenerators:
            return np_pickle.BitGenerators[bit_generator_name]()
        if isinstance(bit_generator_name, type):
            name = bit_generator_name.__name__
            if name in np_pickle.BitGenerators:
                return np_pickle.BitGenerators[name]()
        raise ValueError(str(bit_generator_name) + ' is not a known '
                         'BitGenerator module.')

    np_pickle.__bit_generator_ctor = _compat_bit_generator_ctor
except Exception:
    pass

# ──────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Irrigation System",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# Custom CSS
# ──────────────────────────────────────────────
st.markdown("""
<style>
  [data-testid="stAppViewContainer"] { background: #0f1923; color: #e0e6ef; }
  [data-testid="stSidebar"] { background: #13202e; }
  .metric-card {
    background: linear-gradient(135deg, #1a2d3d 0%, #162434 100%);
    border: 1px solid #2a4a6b;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    margin: 6px 0;
  }
  .metric-card .label { font-size: 12px; color: #7a9ab8; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 6px; }
  .metric-card .value { font-size: 28px; font-weight: 700; }
  .decision-water-now  { color: #ff4e42; }
  .decision-water-later { color: #f5a623; }
  .decision-no-water  { color: #4caf50; }
  .fuzzy-bar-bg { background: #1e3347; border-radius: 6px; height: 22px; overflow: hidden; margin: 4px 0; }
  .fuzzy-bar-fill { height: 100%; border-radius: 6px; transition: width 0.5s; }
  .section-header {
    font-size: 14px; font-weight: 600; color: #7a9ab8;
    text-transform: uppercase; letter-spacing: 1.2px;
    border-bottom: 1px solid #2a4a6b;
    padding-bottom: 6px; margin: 18px 0 10px;
  }
  .stSlider > div > div > div > div { background: #2a6b9e !important; }
  .rule-card {
    background: #1a2d3d; border-left: 3px solid #2a6b9e;
    border-radius: 6px; padding: 8px 12px;
    margin: 4px 0; font-size: 13px; color: #c0d4e8;
  }
  .rule-card.active { border-left-color: #4caf50; color: #a5d6a7; background: #1a2e1e; }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Load models
# ──────────────────────────────────────────────
@st.cache_resource
def load_models():
    base = os.path.dirname(__file__)
    with open(os.path.join(base, "models.pkl"), "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_ts():
    base = os.path.dirname(__file__)
    df = pd.read_csv(os.path.join(base, "ts_data.csv"), parse_dates=["timestamp"])
    return df

m = load_models()
ts_df = load_ts()


# ──────────────────────────────────────────────
# Fuzzy Logic Engine (pure numpy)
# ──────────────────────────────────────────────
def trimf(x, a, b, c):
    """Triangular membership function."""
    if x <= a or x >= c:
        return 0.0
    if x <= b:
        return (x - a) / (b - a) if b != a else 1.0
    return (c - x) / (c - b) if c != b else 1.0

def trapmf(x, a, b, c, d):
    """Trapezoidal membership function."""
    if x <= a or x >= d:
        return 0.0
    if x <= b:
        return (x - a) / (b - a) if b != a else 1.0
    if x <= c:
        return 1.0
    return (d - x) / (d - c) if d != c else 1.0

def compute_memberships(moisture, temperature, humidity, rainfall, predicted_moisture):
    """Compute fuzzy membership degrees for all linguistic variables."""
    # Soil Moisture (%)
    sm = moisture
    ms = {
        "very_low":  trapmf(sm, 0, 0, 10, 20),
        "low":       trimf(sm, 10, 22, 34),
        "medium":    trimf(sm, 28, 38, 50),
        "high":      trimf(sm, 44, 55, 65),
        "very_high": trapmf(sm, 58, 65, 100, 100),
    }
    # Temperature (°C)
    t = temperature
    mt = {
        "cool":  trapmf(t, -10, -10, 15, 25),
        "warm":  trimf(t, 18, 28, 38),
        "hot":   trapmf(t, 32, 40, 60, 60),
    }
    # Humidity (%)
    h = humidity
    mh = {
        "low":    trapmf(h, 0, 0, 30, 50),
        "medium": trimf(h, 35, 55, 75),
        "high":   trapmf(h, 65, 80, 100, 100),
    }
    # Rainfall (mm — daily)
    r = rainfall
    mr = {
        "none":     trapmf(r, 0, 0, 1, 5),
        "light":    trimf(r, 2, 10, 25),
        "moderate": trimf(r, 18, 40, 65),
        "heavy":    trapmf(r, 55, 80, 9999, 9999),
    }
    # Predicted future moisture
    pm = predicted_moisture
    mp = {
        "declining": trapmf(pm, 0, 0, 25, 38),
        "stable":    trimf(pm, 30, 40, 52),
        "rising":    trapmf(pm, 46, 58, 100, 100),
    }
    return ms, mt, mh, mr, mp

def fuzzy_decision(moisture, temperature, humidity, rainfall, predicted_moisture):
    ms, mt, mh, mr, mp = compute_memberships(moisture, temperature, humidity, rainfall, predicted_moisture)

    rules = []

    # WATER NOW rules (high evapotranspiration, low soil reserves, and drying forecast)
    rules.append(("Moisture very low",
                  min(ms["very_low"], 1.0), "water_now", ms["very_low"]))
    rules.append(("Moisture very low + low humidity",
                  min(ms["very_low"], mh["low"]), "water_now",
                  min(ms["very_low"], mh["low"])))
    rules.append(("Moisture low + hot + predicted declining",
                  min(ms["low"], mt["hot"], mp["declining"]), "water_now",
                  min(ms["low"], mt["hot"], mp["declining"])))
    rules.append(("Moisture low + low humidity + no rain",
                  min(ms["low"], mh["low"], mr["none"]), "water_now",
                  min(ms["low"], mh["low"], mr["none"])))
    rules.append(("Moisture low + hot + dry + no rain",
                  min(ms["low"], mt["hot"], mh["low"], mr["none"]), "water_now",
                  min(ms["low"], mt["hot"], mh["low"], mr["none"])))
    rules.append(("Moisture low + predicted stable + hot + low humidity",
                  min(ms["low"], mp["stable"], mt["hot"], mh["low"]), "water_now",
                  min(ms["low"], mp["stable"], mt["hot"], mh["low"])))
    rules.append(("Moisture medium + declining + hot + low humidity",
                  min(ms["medium"], mp["declining"], mt["hot"], mh["low"]), "water_now",
                  min(ms["medium"], mp["declining"], mt["hot"], mh["low"])))
    rules.append(("Moisture medium + no rain + predicted declining",
                  min(ms["medium"], mr["none"], mp["declining"]), "water_now",
                  min(ms["medium"], mr["none"], mp["declining"])))

    # WATER LATER rules (moderate soil moisture, lower evapotranspiration, and incoming rain)
    rules.append(("Moisture medium + warm + predicted stable",
                  min(ms["medium"], mt["warm"], mp["stable"]), "water_later",
                  min(ms["medium"], mt["warm"], mp["stable"])))
    rules.append(("Moisture low + light rain coming",
                  min(ms["low"], mr["light"]), "water_later",
                  min(ms["low"], mr["light"])))
    rules.append(("Moisture low + moderate rain + rising",
                  min(ms["low"], mr["moderate"], mp["rising"]), "water_later",
                  min(ms["low"], mr["moderate"], mp["rising"])))
    rules.append(("Moisture medium + warm + light rain + stable",
                  min(ms["medium"], mt["warm"], mr["light"], mp["stable"]), "water_later",
                  min(ms["medium"], mt["warm"], mr["light"], mp["stable"])))
    rules.append(("Moisture low + predicted rising + warm + moderate rain",
                  min(ms["low"], mp["rising"], mt["warm"], mr["moderate"]), "water_later",
                  min(ms["low"], mp["rising"], mt["warm"], mr["moderate"])))
    rules.append(("Moisture medium + high humidity + predicted stable",
                  min(ms["medium"], mh["high"], mp["stable"]), "water_later",
                  min(ms["medium"], mh["high"], mp["stable"])))
    rules.append(("Moisture medium + declining + cool",
                  min(ms["medium"], mp["declining"], mt["cool"]), "water_later",
                  min(ms["medium"], mp["declining"], mt["cool"])))

    # NO WATER rules
    rules.append(("Heavy rainfall",
                  mr["heavy"], "no_water", mr["heavy"]))
    rules.append(("Moisture high",
                  ms["high"], "no_water", ms["high"]))
    rules.append(("Moisture very high",
                  ms["very_high"], "no_water", ms["very_high"]))
    rules.append(("Moisture rising + moderate rain",
                  min(mp["rising"], mr["moderate"]), "no_water",
                  min(mp["rising"], mr["moderate"])))
    rules.append(("High humidity + rising moisture",
                  min(mh["high"], mp["rising"]), "no_water",
                  min(mh["high"], mp["rising"])))

    # Aggregate by output class
    agg = {"water_now": 0.0, "water_later": 0.0, "no_water": 0.0}
    for _, strength, cls, _ in rules:
        agg[cls] = max(agg[cls], strength)

    total = sum(agg.values()) or 1.0
    probs = {k: v / total for k, v in agg.items()}

    best = max(probs, key=probs.get)
    label_map = {
        "water_now":   ("💧 Water Now",   "decision-water-now"),
        "water_later": ("⏳ Water Later", "decision-water-later"),
        "no_water":    ("✅ No Water Needed", "decision-no-water"),
    }
    decision_label, decision_css = label_map[best]
    confidence = probs[best] * 100

    return decision_label, decision_css, confidence, probs, rules, ms, mt, mh, mr, mp


# ──────────────────────────────────────────────
# Build feature vector
# ──────────────────────────────────────────────
def build_features(crop, soil, temp, humidity, rainfall, moisture,
                   sunlight, wind, ph, oc, ec, prev_irr,
                   month, moisture_lag1, moisture_lag2, le_crop, le_soil):
    crop_enc = le_crop.transform([crop])[0]
    soil_enc = le_soil.transform([soil])[0]
    sin_m = math.sin(2 * math.pi * month / 12)
    cos_m = math.cos(2 * math.pi * month / 12)
    temp_dryness_stress = temp / (moisture + 1)
    heat_humidity_index = temp - 0.55 * (1 - humidity / 100) * (temp - 14.5)
    rain_moisture_deficit = moisture - rainfall / 50.0
    evap_demand = temp * (1 - humidity / 100)
    # Match the exact feature order used during training.
    return np.array([[temp,
                      humidity,
                      rainfall,
                      moisture,
                      crop_enc,
                      soil_enc,
                      temp_dryness_stress,
                      heat_humidity_index,
                      rain_moisture_deficit,
                      evap_demand,
                      sunlight,
                      wind,
                      ph,
                      oc,
                      ec,
                      moisture_lag1,
                      moisture_lag2,
                      sin_m,
                      cos_m]])


# ──────────────────────────────────────────────
# Sidebar Inputs
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌱 Smart Irrigation")
    st.markdown("---")
    st.markdown('<div class="section-header">Crop & Soil</div>', unsafe_allow_html=True)

    crop_options = sorted(m["le_crop"].classes_.tolist())
    soil_options = sorted(m["le_soil"].classes_.tolist())
    crop = st.selectbox("Crop Type", crop_options)
    soil = st.selectbox("Soil Type", soil_options)

    st.markdown('<div class="section-header">Environmental Conditions</div>', unsafe_allow_html=True)
    temp      = st.slider("Temperature (°C)", 5.0, 50.0, 28.0, 0.5)
    humidity  = st.slider("Humidity (%)", 10.0, 100.0, 60.0, 1.0)
    rainfall  = st.slider("Rainfall (mm)", 0.0, 300.0, 10.0, 1.0)

    st.markdown('<div class="section-header">Soil Properties</div>', unsafe_allow_html=True)
    moisture  = st.slider("Current Soil Moisture (%)", 8.0, 65.0, 35.0, 0.5)
    
    month         = 6
    sunlight      = 8.0
    wind          = 12.0
    ph            = 6.8
    oc            = 1.0
    ec            = 1.5
    prev_irr      = 0.0
    moisture_lag1 = moisture
    moisture_lag2 = moisture

    run = st.button("🔍  Analyse & Recommend", use_container_width=True)


# ──────────────────────────────────────────────
# Main Panel
# ──────────────────────────────────────────────
st.markdown("# 🌾 Smart Irrigation System")
st.caption("Neural Network prediction · Fuzzy Logic decision · Explainable AI")

if not run:
    st.info("👈  Configure parameters in the sidebar, then click **Analyse & Recommend**.")

    # Show dataset overview
    st.markdown("### 📊 Dataset Overview")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total Records", "10,000")
    with c2:
        st.metric("Crop Types", "6")
    with c3:
        st.metric("Soil Types", "4")
    with c4:
        st.metric("Date Range", "Jan 2024 – Mar 2025")

else:
    # ── Prediction ──────────────────────────────
    X = build_features(crop, soil, temp, humidity, rainfall, moisture,
                       sunlight, wind, ph, oc, ec, prev_irr,
                       month, moisture_lag1, moisture_lag2,
                       m["le_crop"], m["le_soil"])
    X_sc = m["scaler"].transform(X)
    pred_moisture = float(m["nn_moisture"].predict(X_sc)[0])
    pred_moisture = np.clip(pred_moisture, 8.0, 65.0)

    # NN classification confidence
    nn_proba = m["nn_irr"].predict_proba(X_sc)[0]
    nn_classes = m["le_irr"].classes_

    # Fuzzy decision
    decision, decision_css, fuzzy_conf, probs, rules, ms, mt, mh, mr, mp = \
        fuzzy_decision(moisture, temp, humidity, rainfall, pred_moisture)

    # Blend NN + Fuzzy confidence
    class_map = {"High": "water_now", "Low": "no_water", "Medium": "water_later"}
    blended_conf = 0.0
    for i, cls in enumerate(nn_classes):
        fuzzy_key = class_map[cls]
        blended_conf += nn_proba[i] * probs.get(fuzzy_key, 0.0)
    final_conf = min(100.0, (fuzzy_conf * 0.6 + blended_conf * 100 * 0.4))

    # ── Row 1: Key Metrics ───────────────────────
    st.markdown("### 📋 Recommendation")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="metric-card">
          <div class="label">Decision</div>
          <div class="value {decision_css}">{decision}</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="metric-card">
          <div class="label">Confidence</div>
          <div class="value" style="color:#4fc3f7">{final_conf:.1f}%</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="metric-card">
          <div class="label">Predicted Moisture</div>
          <div class="value" style="color:#81c784">{pred_moisture:.1f}%</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        delta = pred_moisture - moisture
        arrow = "▲" if delta > 0 else "▼"
        col = "#ef9a9a" if delta < -3 else "#a5d6a7" if delta > 3 else "#fff59d"
        st.markdown(f"""
        <div class="metric-card">
          <div class="label">Moisture Trend</div>
          <div class="value" style="color:{col}">{arrow} {abs(delta):.1f}%</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── Row 2: Fuzzy + Time Series ───────────────
    left, right = st.columns([1, 1.6])

    with left:
        st.markdown("### 🧠 Fuzzy Logic Engine")

        # Membership values
        def bar_html(label, value, color):
            pct = int(value * 100)
            return f"""
            <div style="display:flex;align-items:center;margin:3px 0;">
              <span style="width:160px;font-size:12px;color:#a0bcd4">{label}</span>
              <div class="fuzzy-bar-bg" style="flex:1">
                <div class="fuzzy-bar-fill" style="width:{pct}%;background:{color}"></div>
              </div>
              <span style="width:40px;text-align:right;font-size:12px;color:#e0e6ef">{value:.2f}</span>
            </div>"""

        html = ""
        # Moisture
        html += '<div class="section-header">Soil Moisture Membership</div>'
        for k, v in ms.items():
            html += bar_html(k.replace("_", " ").title(), v, "#2196f3")
        # Temperature
        html += '<div class="section-header">Temperature Membership</div>'
        for k, v in mt.items():
            html += bar_html(k.title(), v, "#ff9800")
        # Humidity
        html += '<div class="section-header">Humidity Membership</div>'
        for k, v in mh.items():
            html += bar_html(k.title(), v, "#9c27b0")
        # Rainfall
        html += '<div class="section-header">Rainfall Membership</div>'
        for k, v in mr.items():
            html += bar_html(k.title(), v, "#00bcd4")
        # Predicted Moisture
        html += '<div class="section-header">Predicted Moisture Trend</div>'
        for k, v in mp.items():
            html += bar_html(k.title(), v, "#8bc34a")

        # Output probabilities
        html += '<div class="section-header">Output Aggregation</div>'
        out_colors = {"water_now": "#ef5350", "water_later": "#ffa726", "no_water": "#66bb6a"}
        for k, v in probs.items():
            html += bar_html(k.replace("_", " ").title(), v, out_colors[k])

        st.markdown(html, unsafe_allow_html=True)

        # Fired rules
        st.markdown("### 📜 Active Rules")
        fired = [(r[0], r[1], r[2]) for r in rules if r[1] > 0.01]
        fired.sort(key=lambda x: -x[1])
        for name, strength, cls in fired[:8]:
            cls_label = {"water_now": "🔴 Water Now", "water_later": "🟡 Water Later",
                         "no_water": "🟢 No Water"}[cls]
            is_active = "active" if strength > 0.3 else ""
            st.markdown(f"""
            <div class="rule-card {is_active}">
              <b>{cls_label}</b> | α={strength:.2f}<br>
              <span style="font-size:11px;color:#8ab0cc">IF {name}</span>
            </div>""", unsafe_allow_html=True)

    with right:
        st.markdown("### 📈 Time Series Analysis")

        # Filter time series for selected crop+soil
        subset = ts_df[(ts_df["Crop_Type"] == crop) & (ts_df["Soil_Type"] == soil)].copy()
        subset = subset.sort_values("timestamp")

        if len(subset) < 5:
            st.warning("Not enough time series data for this crop/soil combination.")
        else:
            # Simple rolling smoothing as "predicted"
            subset["Smoothed"] = subset["Soil_Moisture"].rolling(7, min_periods=1).mean()

            try:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
                import matplotlib.patches as mpatches

                fig, axes = plt.subplots(2, 1, figsize=(8, 6),
                                          facecolor="#0f1923", tight_layout=True)
                fig.patch.set_facecolor("#0f1923")

                # Subplot 1 – Moisture over time
                ax1 = axes[0]
                ax1.set_facecolor("#13202e")
                ax1.plot(subset["timestamp"], subset["Soil_Moisture"],
                         color="#4fc3f7", lw=1, alpha=0.7, label="Actual")
                ax1.plot(subset["timestamp"], subset["Smoothed"],
                         color="#ff9800", lw=1.5, linestyle="--", label="Smoothed (7-day)")
                ax1.axhline(moisture, color="#ef5350", lw=1, linestyle=":", label=f"Current: {moisture:.1f}%")
                ax1.axhline(pred_moisture, color="#66bb6a", lw=1, linestyle=":", label=f"Predicted: {pred_moisture:.1f}%")
                ax1.set_title(f"Soil Moisture – {crop} / {soil}", color="#e0e6ef", fontsize=11)
                ax1.set_ylabel("Moisture (%)", color="#7a9ab8", fontsize=9)
                ax1.tick_params(colors="#7a9ab8", labelsize=8)
                ax1.legend(fontsize=8, facecolor="#1a2d3d", labelcolor="#e0e6ef",
                           edgecolor="#2a4a6b")
                for spine in ax1.spines.values():
                    spine.set_color("#2a4a6b")

                # Subplot 2 – Irrigation Need distribution
                ax2 = axes[1]
                ax2.set_facecolor("#13202e")
                colors_map = {"Low": "#66bb6a", "Medium": "#ffa726", "High": "#ef5350"}
                counts = subset["Irrigation_Need"].value_counts()
                bars = ax2.bar(counts.index, counts.values,
                               color=[colors_map.get(c, "#888") for c in counts.index],
                               edgecolor="#2a4a6b", linewidth=0.8)
                for bar, val in zip(bars, counts.values):
                    ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                             str(val), ha="center", va="bottom",
                             color="#e0e6ef", fontsize=9)
                ax2.set_title("Irrigation Need Distribution", color="#e0e6ef", fontsize=11)
                ax2.set_ylabel("Count", color="#7a9ab8", fontsize=9)
                ax2.tick_params(colors="#7a9ab8", labelsize=9)
                for spine in ax2.spines.values():
                    spine.set_color("#2a4a6b")

                st.pyplot(fig)
                plt.close(fig)
            except Exception as e:
                st.error(f"Chart error: {e}")

        # NN confidence breakdown
        st.markdown("### 🤖 Neural Network Class Probabilities")
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig2, ax = plt.subplots(figsize=(5, 2.5), facecolor="#0f1923")
            ax.set_facecolor("#13202e")
            colors_nn = {"High": "#ef5350", "Low": "#66bb6a", "Medium": "#ffa726"}
            ax.barh(nn_classes, nn_proba,
                    color=[colors_nn.get(c, "#888") for c in nn_classes],
                    edgecolor="#2a4a6b")
            for i, v in enumerate(nn_proba):
                ax.text(v + 0.01, i, f"{v*100:.1f}%", va="center",
                        color="#e0e6ef", fontsize=9)
            ax.set_xlim(0, 1.15)
            ax.set_xlabel("Probability", color="#7a9ab8", fontsize=9)
            ax.set_title("NN Output (High=Urgent, Medium=Later, Low=None)",
                         color="#e0e6ef", fontsize=9)
            ax.tick_params(colors="#7a9ab8", labelsize=9)
            for spine in ax.spines.values():
                spine.set_color("#2a4a6b")
            st.pyplot(fig2)
            plt.close(fig2)
        except Exception as e:
            st.warning(f"NN chart skipped: {e}")

    # ── Model Info ───────────────────────────────
    with st.expander("ℹ️ Model Architecture & Details"):
        st.markdown("""
**Neural Network (Regressor — Soil Moisture Prediction)**
- Architecture: 16 → 128 → 64 → 32 → 1
- Trained R²: **0.9984**  |  Activation: ReLU  |  Optimizer: Adam (early stopping)

**Neural Network (Classifier — Irrigation Class)**
- Architecture: 16 → 128 → 64 → 32 → 3 (softmax)
- Accuracy: **~69%**  |  Classes: High / Medium / Low

**Fuzzy Logic System**
- Linguistic variables: Soil Moisture, Temperature, Humidity, Rainfall, Predicted Moisture
- Membership functions: Triangular + Trapezoidal
- Rule base: 13 IF-THEN rules
- Inference: Mamdani-style (max-min), centroid defuzzification simplified to ratio

**Final Decision**
- Blended: 60% Fuzzy confidence + 40% Neural Network probability
        """)