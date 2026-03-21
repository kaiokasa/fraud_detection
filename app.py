import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix, roc_curve
)
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="🔍",
    layout="wide"
)

st.markdown("""
<style>
.fraud-badge {
    background: #fee2e2; color: #991b1b;
    padding: 6px 16px; border-radius: 8px;
    font-weight: 600; font-size: 1.2rem;
    display: inline-block;
}
.safe-badge {
    background: #dcfce7; color: #166534;
    padding: 6px 16px; border-radius: 8px;
    font-weight: 600; font-size: 1.2rem;
    display: inline-block;
}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    train = pd.read_csv("creditcard_train_scaled.csv")
    test  = pd.read_csv("creditcard_test_scaled.csv")
    return train, test


@st.cache_resource
def load_model():
    model = joblib.load("best_xgb_model.pkl")
    model.set_params(device="cpu")
    return model

st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Page", ["Vue d'ensemble", "Performance du modèle", "Prédiction live"],
    label_visibility="collapsed"
)
st.sidebar.markdown("---")
st.sidebar.markdown("**Modèle :** XGBoost (tuné)")
st.sidebar.markdown("**F1 :** 0.872 · **ROC-AUC :** 0.981")
st.sidebar.caption("Projet personnel — Kaggle dataset\nOmar Bouchentouf · ECE Paris MSc DA&IA")

try:
    train_data, test_data = load_data()
except FileNotFoundError:
    st.error("CSV introuvables. Place les fichiers dans le même dossier que app.py.")
    st.stop()

try:
    model = load_model()
except FileNotFoundError:
    st.error("best_xgb_model.pkl introuvable. Sauvegarde le modèle depuis le notebook.")
    st.stop()

X_train = train_data.drop(columns="Class")
y_train = train_data["Class"]
X_test  = test_data.drop(columns="Class")
y_test  = test_data["Class"]
amount_col = next((c for c in X_test.columns if "amount" in c.lower()), X_test.columns[0])

# ── PAGE 1 ───────────────────────────────────────────────────────────────────
if page == "Vue d'ensemble":
    st.title("Vue d'ensemble — Credit Card Fraud Detection")
    st.caption("Dataset Kaggle · transactions anonymisées · features PCA V1–V28 + Amount")

    full       = pd.concat([train_data, test_data])
    total      = len(full)
    n_fraud    = int(full["Class"].sum())
    n_legit    = total - n_fraud
    fraud_rate = round(n_fraud / total * 100, 3)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total transactions", f"{total:,}")
    c2.metric("Légitimes", f"{n_legit:,}")
    c3.metric("Frauduleuses", f"{n_fraud:,}")
    c4.metric("Taux de fraude", f"{fraud_rate}%")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Distribution fraude / légitime")
        dist = full["Class"].value_counts().reset_index()
        dist.columns = ["Classe", "Count"]
        dist["Classe"] = dist["Classe"].map({0: "Légitime", 1: "Fraude"})
        fig = px.pie(dist, names="Classe", values="Count",
                     color="Classe",
                     color_discrete_map={"Légitime": "#22c55e", "Fraude": "#ef4444"},
                     hole=0.45)
        fig.update_traces(textposition="outside", textinfo="percent+label")
        fig.update_layout(showlegend=False, margin=dict(t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader(f"Distribution de {amount_col} par classe")
        sample = full.sample(min(5000, len(full)), random_state=42).copy()
        sample["Classe"] = sample["Class"].map({0: "Légitime", 1: "Fraude"})
        fig2 = px.box(sample, x="Classe", y=amount_col,
                      color="Classe",
                      color_discrete_map={"Légitime": "#22c55e", "Fraude": "#ef4444"},
                      points=False)
        fig2.update_layout(showlegend=False, margin=dict(t=20, b=20))
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    st.subheader("Déséquilibre des classes dans le train set")
    col3, col4 = st.columns(2)
    with col3:
        before = pd.DataFrame({
            "Classe": ["Légitime", "Fraude"],
            "Count":  [int((y_train==0).sum()), int((y_train==1).sum())]
        })
        fig3 = px.bar(before, x="Classe", y="Count", color="Classe",
                      color_discrete_map={"Légitime": "#22c55e", "Fraude": "#ef4444"},
                      title="Avant SMOTE", text="Count")
        fig3.update_traces(textposition="outside")
        fig3.update_layout(showlegend=False)
        st.plotly_chart(fig3, use_container_width=True)
    with col4:
        st.markdown("")
        st.markdown("")
        st.markdown("")
        st.info("""
**Problème :** Dataset fortement déséquilibré — les fraudes représentent moins de 0.2% des transactions.

**Solution :** SMOTE (Synthetic Minority Over-sampling Technique) appliqué avant l'entraînement pour rééquilibrer les classes.
        """)

# ── PAGE 2 ───────────────────────────────────────────────────────────────────
elif page == "Performance du modèle":
    st.title("Performance du modèle — XGBoost (tuné)")

    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    precision = precision_score(y_test, y_pred)
    recall    = recall_score(y_test, y_pred)
    f1        = f1_score(y_test, y_pred)
    roc_auc   = roc_auc_score(y_test, y_proba)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Precision", f"{precision:.3f}")
    c2.metric("Recall",    f"{recall:.3f}")
    c3.metric("F1 (fraud)",f"{f1:.3f}")
    c4.metric("ROC-AUC",   f"{roc_auc:.3f}")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Matrice de confusion")
        cm = confusion_matrix(y_test, y_pred)
        fig_cm = px.imshow(
            cm, text_auto=True,
            x=["Prédit: Légitime", "Prédit: Fraude"],
            y=["Réel: Légitime",   "Réel: Fraude"],
            color_continuous_scale="Blues", aspect="auto"
        )
        fig_cm.update_layout(margin=dict(t=10))
        st.plotly_chart(fig_cm, use_container_width=True)

    with col2:
        st.subheader("Courbe ROC")
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        fig_roc = go.Figure()
        fig_roc.add_shape(type="line", x0=0, y0=0, x1=1, y1=1,
                          line=dict(dash="dash", color="gray", width=1))
        fig_roc.add_trace(go.Scatter(
            x=fpr, y=tpr,
            name=f"XGBoost (AUC={roc_auc:.3f})",
            line=dict(color="#3b82f6", width=2),
            fill="tozeroy", fillcolor="rgba(59,130,246,0.08)"
        ))
        fig_roc.update_layout(
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            margin=dict(t=10)
        )
        st.plotly_chart(fig_roc, use_container_width=True)

    st.markdown("---")
    st.subheader("Importance des features (Top 20)")
    importance = pd.DataFrame({
        "Feature":    X_test.columns,
        "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=False).head(20)

    fig_imp = px.bar(importance, x="Importance", y="Feature",
                     orientation="h", color="Importance",
                     color_continuous_scale="Blues")
    fig_imp.update_layout(
        coloraxis_showscale=False,
        yaxis=dict(autorange="reversed"),
        margin=dict(l=10, r=20)
    )
    st.plotly_chart(fig_imp, use_container_width=True)

# ── PAGE 3 ───────────────────────────────────────────────────────────────────
elif page == "Prédiction live":
    st.title("Prédiction live")
    st.caption("Simule une transaction et obtiens une prédiction instantanée.")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Paramètres")
        profile = st.selectbox(
            "Profil de transaction",
            ["Transaction normale", "Transaction suspecte", "Aléatoire"]
        )
        amount_val = st.slider(
            f"Valeur de {amount_col}",
            min_value=-3.0, max_value=10.0, value=0.5, step=0.1
        )

    sample_legit = X_test[y_test == 0].sample(1, random_state=42).values.flatten()
    sample_fraud = X_test[y_test == 1].sample(1, random_state=42).values.flatten()

    if profile == "Transaction normale":
        features = sample_legit.copy()
    elif profile == "Transaction suspecte":
        features = sample_fraud.copy()
    else:
        features = X_test.sample(1, random_state=np.random.randint(0, 999)).values.flatten()

    amount_idx         = list(X_test.columns).index(amount_col)
    features[amount_idx] = amount_val

    input_df   = pd.DataFrame([features], columns=X_test.columns)
    proba      = model.predict_proba(input_df)[0][1]
    prediction = int(proba >= 0.5)

    with col2:
        st.subheader("Résultat")
        if prediction == 1:
            st.markdown('<div class="fraud-badge">FRAUDE DÉTECTÉE</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="safe-badge">TRANSACTION LÉGITIME</div>', unsafe_allow_html=True)

        st.markdown(f"**Probabilité de fraude : `{proba:.1%}`**")

        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=round(proba * 100, 1),
            number={"suffix": "%"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar":  {"color": "#ef4444" if prediction == 1 else "#22c55e"},
                "steps": [
                    {"range": [0,  30], "color": "#dcfce7"},
                    {"range": [30, 60], "color": "#fef9c3"},
                    {"range": [60,100], "color": "#fee2e2"},
                ],
                "threshold": {"line": {"color": "black", "width": 3}, "value": 50}
            },
            title={"text": "Score de risque"}
        ))
        fig_gauge.update_layout(height=250, margin=dict(t=40, b=0))
        st.plotly_chart(fig_gauge, use_container_width=True)

    st.markdown("---")
    st.subheader("Valeurs des features")
    feat_df = pd.DataFrame({
        "Feature": X_test.columns,
        "Valeur":  [round(float(v), 4) for v in features]
    })
    st.dataframe(feat_df, use_container_width=True, hide_index=True, height=300)
