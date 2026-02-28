import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

# ── Load artifacts ────────────────────────────────────────────────
model    = joblib.load("xgb_model.pkl")
le_brand = joblib.load("le_brand.pkl")
le_city  = joblib.load("le_city.pkl")
le_owner = joblib.load("le_owner.pkl")
features = joblib.load("features.pkl")

explainer = shap.TreeExplainer(model)

# ── Page config ───────────────────────────────────────────────────
st.set_page_config(page_title="🏍️ Bike Price Predictor", layout="wide")

st.markdown("""
<div style='background: linear-gradient(135deg, #1a1a2e, #0f3460);
            padding: 2rem; border-radius: 12px; text-align: center; margin-bottom: 2rem;'>
    <h1 style='color: white; margin: 0;'>🏍️ Used Bike Price Predictor</h1>
    <p style='color: #aaa; margin: 0.5rem 0 0;'>
        ML model trained on 32,000+ listings · XGBoost · R² = 0.9359 · MAPE = 5.31%
    </p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar inputs ────────────────────────────────────────────────
st.sidebar.header("🔧 Enter Bike Details")

brand = st.sidebar.selectbox("Brand", sorted(le_brand.classes_))
city  = st.sidebar.selectbox("City",  sorted(le_city.classes_))
owner = st.sidebar.selectbox("Owner", sorted(le_owner.classes_))

age        = st.sidebar.slider("Bike Age (years)", 0, 20, 3)
kms_driven = st.sidebar.slider("KMs Driven", 0, 200000, 20000, step=500)
power      = st.sidebar.slider("Engine CC", 50, 650, 150, step=5)

predict_btn = st.sidebar.button("🔮 Predict Price", use_container_width=True, type="primary")

# ── Helper: build input ───────────────────────────────────────────
def build_input(brand, city, owner, age, kms_driven, power):
    try: brand_enc = le_brand.transform([brand])[0]
    except: brand_enc = 0
    try: city_enc = le_city.transform([city])[0]
    except: city_enc = 0
    try: owner_enc = le_owner.transform([owner])[0]
    except: owner_enc = 0

    return pd.DataFrame([[
        kms_driven, owner_enc, age, power,
        brand_enc, city_enc,
        kms_driven / (age + 1),
        power / (age + 1),
        age ** 2,
        kms_driven ** 2,
        power * age,
        int(owner == "First Owner"),
        int(kms_driven > 50000),
        int(age <= 2),
        int(power > 300),
    ]], columns=features)

# ── Main ──────────────────────────────────────────────────────────
if predict_btn:
    input_df = build_input(brand, city, owner, age, kms_driven, power)
    price    = float(model.predict(input_df)[0])
    price    = max(5000, price)
    margin   = price * 0.10

    # ── Price card ────────────────────────────────────────────────
    col_l, col_m, col_r = st.columns([1, 2, 1])
    with col_m:
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #1a1a2e, #0f3460);
                    padding: 2rem; border-radius: 16px; text-align: center;'>
            <p style='color: #aaa; margin: 0;'>Estimated Market Price</p>
            <h1 style='color: #e94560; font-size: 3rem; margin: 0.5rem 0;'>
                Rs {price:,.0f}
            </h1>
            <p style='color: #888;'>
                Confidence Range: Rs {price - margin:,.0f} - Rs {price + margin:,.0f}
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Metrics row ───────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Brand",  brand)
    c2.metric("Age",    f"{age} yrs")
    c3.metric("KMs",    f"{kms_driven:,}")
    c4.metric("Engine", f"{power} cc")
    c5.metric("Owner",  owner)

    st.markdown("---")

    # ── SHAP explanation ──────────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🧠 Why this price? (SHAP Explanation)")
        shap_vals = explainer.shap_values(input_df)[0]
        shap_df = pd.DataFrame({
            "Feature":    features,
            "SHAP Value": shap_vals
        }).sort_values("SHAP Value", key=abs, ascending=True).tail(10)

        fig, ax = plt.subplots(figsize=(8, 5))
        colors = ["#e94560" if v > 0 else "#2ecc71" for v in shap_df["SHAP Value"]]
        ax.barh(shap_df["Feature"], shap_df["SHAP Value"], color=colors)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title("Feature contributions to predicted price", fontsize=11)
        ax.set_xlabel("SHAP Value (impact on price)")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        st.markdown("### 📊 Feature Impact (SHAP Bar)")
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        colors2 = ["#e94560" if v > 0 else "#2ecc71" for v in shap_df["SHAP Value"]]
        ax2.barh(shap_df["Feature"], shap_df["SHAP Value"], color=colors2)
        ax2.axvline(0, color="black", linewidth=0.8)
        ax2.set_title("Red = increases price | Green = decreases price", fontsize=10)
        ax2.set_xlabel("SHAP Value (impact on price)")
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()

    # ── SHAP explanation text ─────────────────────────────────────
    st.markdown("---")
    st.markdown("### 💡 Key Price Drivers")

    top_pos = shap_df[shap_df["SHAP Value"] > 0].sort_values(
        "SHAP Value", ascending=False).head(2)
    top_neg = shap_df[shap_df["SHAP Value"] < 0].sort_values(
        "SHAP Value").head(2)

    c1, c2 = st.columns(2)
    with c1:
        st.success("**Factors pushing price UP** 📈")
        for _, row in top_pos.iterrows():
            st.write(f"• **{row['Feature']}** → +Rs {abs(row['SHAP Value']):,.0f}")
    with c2:
        st.error("**Factors pushing price DOWN** 📉")
        for _, row in top_neg.iterrows():
            st.write(f"• **{row['Feature']}** → -Rs {abs(row['SHAP Value']):,.0f}")

else:
    # ── Landing state ─────────────────────────────────────────────
    st.markdown("""
    <div style='text-align: center; padding: 3rem; color: #888;'>
        <p style='font-size: 4rem; margin: 0;'>🏍️</p>
        <h3>Fill in the bike details on the left and click Predict</h3>
        <p>Get an instant price estimate with AI-powered explanations</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🏆 Model Performance")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Algorithm",      "XGBoost")
    c2.metric("R² Score",       "0.9359")
    c3.metric("RMSE",           "Rs 22,123")
    c4.metric("MAPE",           "5.31%")
    c5.metric("Training Rows",  "32,671")

    st.markdown("### 📦 Data Sources")
    c1, c2, c3 = st.columns(3)
    c1.metric("Kaggle Records",  "32,648")
    c2.metric("Scraped Records", "34")
    c3.metric("Features Used",   "15")