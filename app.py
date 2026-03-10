import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import shap
import matplotlib.pyplot as plt

# ── Configuração da página ─────────────────────────────────
st.set_page_config(
    page_title="Previsão de Risco — Insurtech",
    page_icon="🏥",
    layout="wide"
)

# ── Carregar modelos ───────────────────────────────────────
@st.cache_resource
def carregar_modelos():
    with open('data/processed/xgb_regressor.pkl', 'rb') as f:
        xgb = pickle.load(f)
    with open('data/processed/rf_classifier.pkl', 'rb') as f:
        rf = pickle.load(f)
    with open('data/processed/risk_thresholds.json', 'r') as f:
        cortes = json.load(f)
    return xgb, rf, cortes

xgb, rf, cortes = carregar_modelos()

# ── Header ─────────────────────────────────────────────────
st.title("🏥 Previsão de Risco e Custo — Seguradora de Saúde")
st.markdown("**Ferramenta de apoio à decisão para analistas de risco e atuários**")
st.divider()

# ── Sidebar — inputs ───────────────────────────────────────
st.sidebar.header("👤 Perfil do Beneficiário")

age      = st.sidebar.slider("Idade", 18, 64, 35)
bmi      = st.sidebar.slider("IMC (Índice de Massa Corporal)", 15.0, 53.0, 28.0, 0.1)
children = st.sidebar.selectbox("Número de filhos/dependentes", [0,1,2,3,4,5])
smoker   = st.sidebar.radio("Fumante?", ["Não", "Sim"])
sex      = st.sidebar.radio("Sexo", ["Masculino", "Feminino"])
region   = st.sidebar.selectbox("Região", ["northeast","northwest","southeast","southwest"])

st.sidebar.divider()
st.sidebar.markdown("📌 Cortes de risco:")
st.sidebar.markdown(f"- Baixo/Médio: **${cortes['p33']:,.0f}**")
st.sidebar.markdown(f"- Médio/Alto:  **${cortes['p66']:,.0f}**")

# ── Preparar features ──────────────────────────────────────
def preparar_features(age, bmi, children, smoker, sex, region):
    smoker_enc = 1 if smoker == "Sim" else 0
    sex_enc    = 1 if sex == "Masculino" else 0
    is_obese   = 1 if bmi > 30 else 0
    smoker_obese = 1 if (smoker_enc == 1 and is_obese == 1) else 0

    region_northwest = 1 if region == "northwest" else 0
    region_southeast = 1 if region == "southeast" else 0
    region_southwest = 1 if region == "southwest" else 0

    if age <= 18:   ag = '0-18'
    elif age <= 25: ag = '19-25'
    elif age <= 39: ag = '26-39'
    elif age <= 49: ag = '40-49'
    elif age <= 59: ag = '50-59'
    else:           ag = '60+'

    age_groups = {'0-18':0,'19-25':0,'26-39':0,'40-49':0,'50-59':0,'60+':0}
    age_groups[ag] = 1

    features = {
        'age'              : age,
        'bmi'              : bmi,
        'children'         : children,
        'is_obese'         : is_obese,
        'smoker_obese'     : smoker_obese,
        'smoker_enc'       : smoker_enc,
        'sex_enc'          : sex_enc,
        'region_northwest' : region_northwest,
        'region_southeast' : region_southeast,
        'region_southwest' : region_southwest,
        'age_19-25'        : age_groups['19-25'],
        'age_26-39'        : age_groups['26-39'],
        'age_40-49'        : age_groups['40-49'],
        'age_50-59'        : age_groups['50-59'],
        'age_60+'          : age_groups['60+'],
    }
    return pd.DataFrame([features])

# ── Previsão ───────────────────────────────────────────────
X_input = preparar_features(age, bmi, children, smoker, sex, region)

custo_log  = xgb.predict(X_input)[0]
custo_prev = np.exp(custo_log)
risco_num  = rf.predict(X_input)[0]
risco_prob = rf.predict_proba(X_input)[0]

risco_label = {0: "🟢 Baixo", 1: "🟡 Médio", 2: "🔴 Alto"}
risco_cor   = {0: "green",    1: "orange",    2: "red"}

# ── Layout principal ───────────────────────────────────────
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        label="💰 Custo Anual Estimado",
        value=f"${custo_prev:,.2f}",
        delta=f"{'Acima' if custo_prev > cortes['p66'] else 'Abaixo'} da média"
    )

with col2:
    st.metric(
        label="⚠️ Classe de Risco",
        value=risco_label[risco_num]
    )

with col3:
    st.metric(
        label="📊 Probabilidade Alto Custo",
        value=f"{risco_prob[2]*100:.1f}%"
    )

st.divider()

# ── Probabilidades por classe ──────────────────────────────
col4, col5 = st.columns(2)

with col4:
    st.subheader("📊 Probabilidade por Classe de Risco")
    classes = ['Baixo Custo', 'Médio Custo', 'Alto Custo']
    cores   = ['#2ecc71', '#f39c12', '#e74c3c']
    fig, ax = plt.subplots(figsize=(6, 3))
    bars = ax.barh(classes, risco_prob * 100, color=cores, edgecolor='white')
    ax.set_xlabel('Probabilidade (%)')
    ax.set_xlim(0, 100)
    for bar, val in zip(bars, risco_prob * 100):
        ax.text(val + 1, bar.get_y() + bar.get_height()/2,
                f'{val:.1f}%', va='center', fontsize=10, fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig)

with col5:
    st.subheader("🔍 Fatores de Risco do Perfil")
    fatores = []
    if smoker == "Sim":
        fatores.append("🚬 Fumante — maior fator de risco (+3.5x custo)")
    if bmi > 30:
        fatores.append(f"⚖️ Obeso (IMC {bmi:.1f}) — risco elevado")
    if smoker == "Sim" and bmi > 30:
        fatores.append("🔥 Fumante obeso — perfil crítico ($51k/ano médio)")
    if age >= 50:
        fatores.append(f"👴 Idade {age} anos — faixa de alto custo")
    if not fatores:
        fatores.append("✅ Nenhum fator de risco crítico identificado")
    for f in fatores:
        st.markdown(f"- {f}")

st.divider()

# ── SHAP explicação individual ─────────────────────────────
st.subheader("🧠 Explicação da Previsão (SHAP)")
st.markdown("*Quais fatores mais influenciaram esse resultado?*")

try:
    explainer   = shap.Explainer(xgb)
    shap_values = explainer(X_input)
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    shap.waterfall_plot(shap_values[0], show=False)
    plt.tight_layout()
    st.pyplot(fig2)
except Exception as e:
    st.info("Gráfico SHAP não disponível para este perfil.")

st.divider()
st.markdown("""
**📌 Sobre o modelo:**
- **Regressão:** XGBoost treinado em 99.989 beneficiários | R² = 88.3% | MAE = $3.514
- **Classificação:** Random Forest | Acurácia = 72% | Falsos negativos críticos = 0
- **Dataset:** Sintético baseado no Medical Cost Personal Dataset (Kaggle)
""")