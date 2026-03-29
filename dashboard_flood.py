"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  Dashboard Streamlit — Prédiction de la Probabilité d'Inondation            ║
║  Projet Fil Conducteur | Africa Tech Up Tour                                 ║
║  Auteur : Judicaël Karol DOBOEVI — ENSGMM, Bénin                            ║
╚══════════════════════════════════════════════════════════════════════════════╝

Lancement :
    pip install streamlit lightgbm scikit-learn pandas numpy matplotlib seaborn plotly
    streamlit run dashboard_flood.py
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
import lightgbm as lgb
import io

# ─────────────────────────────────────────────────────────────────────────────
# Configuration de la page
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Flood Prediction Dashboard",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé
st.markdown("""
<style>
    /* Header principal */
    .main-header {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 50%, #1a7abf 100%);
        padding: 2rem;
        border-radius: 12px;
        text-align: center;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    .main-header h1 { color: white; font-size: 2.2rem; margin: 0; }
    .main-header p  { color: #c8ddf5; font-size: 1rem; margin: 0.4rem 0 0; }

    /* Cartes métriques */
    .metric-card {
        background: linear-gradient(135deg, #f8f9ff 0%, #e8f0fe 100%);
        border: 1px solid #d2e3fc;
        border-radius: 10px;
        padding: 1.2rem;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        transition: transform 0.2s;
    }
    .metric-card:hover { transform: translateY(-3px); }
    .metric-value { font-size: 2rem; font-weight: 800; color: #1a56db; }
    .metric-label { font-size: 0.85rem; color: #555; margin-top: 0.3rem; }

    /* Cartes de section */
    .section-card {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 10px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    }

    /* Badge */
    .badge-success { background:#d1fae5; color:#065f46; padding:3px 10px; border-radius:20px; font-size:0.8rem; font-weight:600; }
    .badge-info    { background:#dbeafe; color:#1e40af; padding:3px 10px; border-radius:20px; font-size:0.8rem; font-weight:600; }

    /* Sidebar */
    .sidebar-info { background:#f0f4ff; border-radius:8px; padding:1rem; border-left:4px solid #1a56db; }

    /* Footer */
    .footer { text-align:center; color:#9ca3af; font-size:0.78rem; margin-top:2rem; padding-top:1rem; border-top:1px solid #e5e7eb; }

    div[data-testid="stMetric"] { background: #f8f9ff; border-radius:8px; padding:1rem; border:1px solid #d2e3fc; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# DÉFINITIONS DES FEATURES
# ─────────────────────────────────────────────────────────────────────────────
NUM_COLS = [
    'MonsoonIntensity', 'TopographyDrainage', 'RiverManagement',
    'Deforestation', 'Urbanization', 'ClimateChange', 'DamsQuality',
    'Siltation', 'AgriculturalPractices', 'Encroachments',
    'IneffectiveDisasterPreparedness', 'DrainageSystems',
    'CoastalVulnerability', 'Landslides', 'Watersheds',
    'DeterioratingInfrastructure', 'PopulationScore',
    'WetlandLoss', 'InadequatePlanning', 'PoliticalFactors'
]

FEATURE_LABELS = {
    'MonsoonIntensity': 'Intensité des Moussons',
    'TopographyDrainage': 'Drainage Topographique',
    'RiverManagement': 'Gestion des Rivières',
    'Deforestation': 'Déforestation',
    'Urbanization': 'Urbanisation',
    'ClimateChange': 'Changement Climatique',
    'DamsQuality': 'Qualité des Barrages',
    'Siltation': 'Envasement',
    'AgriculturalPractices': 'Pratiques Agricoles',
    'Encroachments': 'Empiètements',
    'IneffectiveDisasterPreparedness': 'Préparation aux Catastrophes',
    'DrainageSystems': 'Systèmes de Drainage',
    'CoastalVulnerability': 'Vulnérabilité Côtière',
    'Landslides': 'Glissements de Terrain',
    'Watersheds': 'Bassins Versants',
    'DeterioratingInfrastructure': 'Infrastructure Dégradée',
    'PopulationScore': 'Score de Population',
    'WetlandLoss': 'Perte de Zones Humides',
    'InadequatePlanning': 'Planification Inadéquate',
    'PoliticalFactors': 'Facteurs Politiques'
}

# ─────────────────────────────────────────────────────────────────────────────
# FONCTIONS UTILITAIRES
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def generate_demo_data(n=5000, seed=42):
    """Génère des données synthétiques réalistes pour la démo."""
    rng = np.random.default_rng(seed)
    data = {}
    for col in NUM_COLS:
        data[col] = rng.integers(0, 11, size=n)
    df = pd.DataFrame(data)
    # FloodProbability synthétique corrélée aux features
    signal = (
        0.04 * df['MonsoonIntensity'] +
        0.03 * df['TopographyDrainage'] +
        0.035 * df['ClimateChange'] +
        0.025 * df['Deforestation'] +
        0.02 * df['Urbanization'] +
        0.018 * df['Siltation'] +
        0.015 * df['Encroachments'] +
        0.012 * df['CoastalVulnerability'] +
        sum(0.008 * df[c] for c in NUM_COLS[7:])
    )
    noise = rng.normal(0, 0.03, n)
    fp = 0.25 + signal + noise
    df['FloodProbability'] = np.clip(fp, 0.1, 0.9)
    return df


def create_features_demo(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """Version simplifiée du feature engineering pour la démo."""
    scaler = StandardScaler()
    df = df.copy()
    df['ClimateAnthropogenicInteraction'] = (
        (df['MonsoonIntensity'] + df['ClimateChange']) *
        (df['Deforestation'] + df['Urbanization'])
    )
    df['InfrastructurePreventionInteraction'] = (
        (df['DamsQuality'] + df['DrainageSystems']) *
        (df['RiverManagement'] + df['InadequatePlanning'])
    )
    df['row_sum']    = df[cols].sum(axis=1)
    df['row_mean']   = df[cols].mean(axis=1)
    df['row_std']    = df[cols].std(axis=1)
    df['row_max']    = df[cols].max(axis=1)
    df['row_min']    = df[cols].min(axis=1)
    df['row_range']  = df['row_max'] - df['row_min']
    df['row_skew']   = df[cols].skew(axis=1)
    df['row_kurt']   = df[cols].kurt(axis=1)
    df['row_cv']     = df['row_std'] / (df['row_mean'] + 1e-8)
    for pct in [25, 50, 75]:
        df[f'q{pct}'] = df[cols].quantile(pct / 100, axis=1)
    df[cols] = scaler.fit_transform(df[cols])
    return df


@st.cache_resource
def train_model_demo():
    """Entraîne un modèle demo sur données synthétiques."""
    df = generate_demo_data(n=8000)
    df_feat = create_features_demo(df, NUM_COLS)
    feature_cols = [c for c in df_feat.columns if c not in ['FloodProbability']]
    X = df_feat[feature_cols]
    y = df_feat['FloodProbability']

    params = {
        'n_estimators': 300, 'learning_rate': 0.05,
        'num_leaves': 63, 'max_depth': 6,
        'verbosity': -1, 'random_state': 42
    }
    model = lgb.LGBMRegressor(**params)
    model.fit(X, y)

    oof = np.zeros(len(X))
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    val_r2s = []
    for tr_idx, va_idx in kf.split(X, y):
        m = lgb.LGBMRegressor(**params)
        m.fit(X.iloc[tr_idx], y.iloc[tr_idx])
        oof[va_idx] = m.predict(X.iloc[va_idx])
        val_r2s.append(r2_score(y.iloc[va_idx], oof[va_idx]))

    return model, feature_cols, val_r2s, oof, y.values, df


def predict_single(model, feature_cols, values_dict):
    """Prédit pour une observation unique saisie manuellement."""
    df_single = pd.DataFrame([values_dict])
    df_feat = create_features_demo(df_single, NUM_COLS)
    # Aligner les colonnes
    for c in feature_cols:
        if c not in df_feat.columns:
            df_feat[c] = 0
    X = df_feat[feature_cols]
    return model.predict(X)[0]


def flood_risk_label(prob):
    if prob < 0.35:
        return "🟢 Faible", "#16a34a"
    elif prob < 0.55:
        return "🟡 Modéré", "#d97706"
    elif prob < 0.70:
        return "🟠 Élevé", "#ea580c"
    else:
        return "🔴 Très élevé", "#dc2626"


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class='sidebar-info'>
        <b>🎓 Projet Fil Conducteur</b><br>
        Formation Data Scientist<br>
        Africa Tech Up Tour
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**👤 Auteur**")
    st.markdown("Judicaël Karol **DOBOEVI**")
    st.markdown("1ère année — Génie Math. & Modélisation")
    st.markdown("🏛️ ENSGMM, Bénin")
    st.markdown("---")

    st.markdown("**🧭 Navigation**")
    page = st.radio(
        "Choisir une section :",
        ["🏠 Accueil", "📊 Analyse EDA", "🤖 Modèle & Performance",
         "🔮 Prédiction Interactive", "📐 Feature Engineering"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("**📦 Dataset**")
    st.caption("Kaggle Playground Series S4E5")
    st.caption("1 117 957 observations | 20 features")
    st.markdown("**🏆 Score public**")
    st.markdown("<span class='badge-success'>R² = 0.86931</span>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# CHARGEMENT DU MODÈLE
# ─────────────────────────────────────────────────────────────────────────────
model, feature_cols, val_r2s, oof_preds, y_true, df_demo = train_model_demo()

# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class='main-header'>
    <h1>🌊 Prédiction de la Probabilité d'Inondation</h1>
    <p>Modèle LightGBM · Validation croisée K-Fold · Projet Fil Conducteur – Africa Tech Up Tour</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE : ACCUEIL
# ─────────────────────────────────────────────────────────────────────────────
if page == "🏠 Accueil":

    st.markdown("## Bienvenue sur le tableau de bord 👋")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        <div class='metric-card'>
            <div class='metric-value'>0.869</div>
            <div class='metric-label'>Score R² public (Kaggle)</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class='metric-card'>
            <div class='metric-value'>20</div>
            <div class='metric-label'>Variables explicatives</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class='metric-card'>
            <div class='metric-value'>1.1M+</div>
            <div class='metric-label'>Observations d'entraînement</div>
        </div>""", unsafe_allow_html=True)
    with col4:
        st.markdown("""
        <div class='metric-card'>
            <div class='metric-value'>~50</div>
            <div class='metric-label'>Features créées (Engineering)</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    col_a, col_b = st.columns([3, 2])
    with col_a:
        st.markdown("""
        <div class='section-card'>
        <h3>🎯 Contexte du projet</h3>
        <p>Les inondations sont parmi les catastrophes naturelles les plus dévastatrices, 
        particulièrement en Afrique subsaharienne. Ce projet vise à construire un 
        <b>modèle de régression supervisé</b> capable de prédire la probabilité 
        d'occurrence d'une inondation à partir de 20 variables environnementales, 
        climatiques et socio-économiques.</p>
        <p>Le modèle utilise <b>LightGBM</b> (Light Gradient Boosting Machine), 
        couplé à une validation croisée K-Fold (K=5) pour garantir la robustesse 
        des performances.</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class='section-card'>
        <h3>🗺️ Pipeline Méthodologique</h3>
        </div>
        """, unsafe_allow_html=True)

        steps = [
            ("📥 Chargement des données", "Train : 1 117 957 lignes × 22 colonnes"),
            ("🔍 Analyse Exploratoire (EDA)", "Distribution, statistiques, corrélations"),
            ("⚙️ Feature Engineering", "~50 nouvelles variables créées"),
            ("🌟 Importance des Variables", "Critère Gain de LightGBM"),
            ("🔄 Validation Croisée K-Fold", "K=5, stratégie Shuffle"),
            ("📈 Évaluation", "R² = 0.869 sur données publiques"),
        ]
        for i, (title, desc) in enumerate(steps, 1):
            st.markdown(f"**{i}. {title}** — *{desc}*")

    with col_b:
        st.markdown("""
        <div class='section-card'>
        <h3>🌿 Variables explicatives</h3>
        </div>
        """, unsafe_allow_html=True)

        categories = {
            "🌧️ Climatiques": ["MonsoonIntensity", "ClimateChange"],
            "🏔️ Géographiques": ["TopographyDrainage", "Landslides", "Watersheds", "CoastalVulnerability"],
            "🌳 Environnementales": ["Deforestation", "Siltation", "WetlandLoss", "AgriculturalPractices"],
            "🏙️ Anthropiques": ["Urbanization", "Encroachments", "PopulationScore"],
            "🏗️ Infrastructure": ["DamsQuality", "DrainageSystems", "DeterioratingInfrastructure", "RiverManagement"],
            "📋 Gouvernance": ["IneffectiveDisasterPreparedness", "InadequatePlanning", "PoliticalFactors"]
        }
        for cat, cols in categories.items():
            with st.expander(cat, expanded=False):
                for c in cols:
                    st.markdown(f"• {FEATURE_LABELS.get(c, c)}")

        st.markdown("""
        <div class='section-card'>
        <h3>🏆 Résultats</h3>
        <table style='width:100%; font-size:0.9rem;'>
        <tr><td>Algorithme</td><td><b>LightGBM GBDT</b></td></tr>
        <tr><td>Validation</td><td><b>K-Fold (K=5)</b></td></tr>
        <tr><td>R² (OOF)</td><td><b>≈ 0.869</b></td></tr>
        <tr><td>RMSE</td><td><b>≈ 0.043</b></td></tr>
        <tr><td>Score Kaggle</td><td><b>0.86931</b></td></tr>
        </table>
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE : EDA
# ─────────────────────────────────────────────────────────────────────────────
elif page == "📊 Analyse EDA":
    st.markdown("## 📊 Analyse Exploratoire des Données")
    st.info("ℹ️ Les graphiques ci-dessous utilisent un échantillon synthétique de 5 000 observations simulant les distributions du dataset Kaggle réel.")

    tab1, tab2, tab3 = st.tabs(["📈 Variable cible", "📦 Variables explicatives", "🔥 Corrélations"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            fig = px.histogram(
                df_demo, x='FloodProbability', nbins=60,
                title="Distribution de FloodProbability",
                color_discrete_sequence=['#1a56db'],
                labels={'FloodProbability': "Probabilité d'inondation"}
            )
            fig.add_vline(x=df_demo['FloodProbability'].mean(), line_dash="dash",
                          line_color="red", annotation_text=f"Moy={df_demo['FloodProbability'].mean():.3f}")
            fig.update_layout(plot_bgcolor='#f9fafb', paper_bgcolor='white', height=360)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig2 = px.box(
                df_demo, y='FloodProbability',
                title="Boîte à moustaches — FloodProbability",
                color_discrete_sequence=['#1a56db'],
            )
            fig2.update_layout(plot_bgcolor='#f9fafb', paper_bgcolor='white', height=360)
            st.plotly_chart(fig2, use_container_width=True)

        cols_stats = st.columns(5)
        stats = {
            "Min": df_demo['FloodProbability'].min(),
            "Max": df_demo['FloodProbability'].max(),
            "Moyenne": df_demo['FloodProbability'].mean(),
            "Médiane": df_demo['FloodProbability'].median(),
            "Écart-type": df_demo['FloodProbability'].std()
        }
        for col, (label, val) in zip(cols_stats, stats.items()):
            col.metric(label, f"{val:.4f}")

    with tab2:
        selected_cols = st.multiselect(
            "Sélectionner les variables à visualiser :",
            NUM_COLS,
            default=NUM_COLS[:8],
            format_func=lambda x: FEATURE_LABELS.get(x, x)
        )

        if selected_cols:
            ncols = min(3, len(selected_cols))
            nrows = (len(selected_cols) + ncols - 1) // ncols
            fig3, axes = plt.subplots(nrows, ncols, figsize=(16, nrows * 3.5))
            if nrows == 1 and ncols == 1:
                axes = np.array([[axes]])
            elif nrows == 1:
                axes = axes.reshape(1, -1)
            elif ncols == 1:
                axes = axes.reshape(-1, 1)

            for i, col in enumerate(selected_cols):
                r, c = divmod(i, ncols)
                axes[r, c].hist(df_demo[col], bins=15, color='#1a56db', edgecolor='white', alpha=0.85)
                axes[r, c].set_title(FEATURE_LABELS.get(col, col), fontsize=9, fontweight='bold')
                axes[r, c].set_xlabel("Valeur", fontsize=8)
                axes[r, c].tick_params(labelsize=7)

            for j in range(len(selected_cols), nrows * ncols):
                r, c = divmod(j, ncols)
                axes[r, c].set_visible(False)

            plt.tight_layout()
            st.pyplot(fig3)
        else:
            st.warning("Veuillez sélectionner au moins une variable.")

    with tab3:
        corr_df = df_demo[NUM_COLS + ['FloodProbability']].corr()

        fig4 = px.imshow(
            corr_df,
            color_continuous_scale='RdBu_r',
            zmin=-0.3, zmax=1,
            title="Matrice de Corrélation",
            text_auto='.2f',
            aspect='auto',
            width=750, height=700
        )
        fig4.update_layout(paper_bgcolor='white')
        st.plotly_chart(fig4, use_container_width=True)

        # Corrélation avec la cible
        corr_target = corr_df['FloodProbability'].drop('FloodProbability').sort_values()
        fig5 = px.bar(
            x=corr_target.values,
            y=[FEATURE_LABELS.get(c, c) for c in corr_target.index],
            orientation='h',
            title="Corrélation avec FloodProbability",
            color=corr_target.values,
            color_continuous_scale='Blues',
            labels={'x': 'Corrélation de Pearson', 'y': 'Variable'}
        )
        fig5.update_layout(plot_bgcolor='#f9fafb', paper_bgcolor='white', height=550)
        st.plotly_chart(fig5, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE : MODÈLE & PERFORMANCE
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🤖 Modèle & Performance":
    st.markdown("## 🤖 Modèle LightGBM & Performances")

    col1, col2, col3, col4 = st.columns(4)
    r2_mean = np.mean(val_r2s)
    rmse_val = np.sqrt(mean_squared_error(y_true, oof_preds))
    r2_oof = r2_score(y_true, oof_preds)

    col1.metric("R² Moyen (CV)", f"{r2_mean:.4f}")
    col2.metric("R² OOF Global", f"{r2_oof:.4f}")
    col3.metric("RMSE (OOF)", f"{rmse_val:.4f}")
    col4.metric("Score Kaggle Public", "0.86931 🏆")

    tab1, tab2, tab3 = st.tabs(["📊 R² par Fold", "🔍 Résidus", "🌟 Importance"])

    with tab1:
        fig = go.Figure()
        colors = ['#1a56db' if v >= r2_mean else '#93c5fd' for v in val_r2s]
        fig.add_trace(go.Bar(
            x=[f"Fold {i+1}" for i in range(len(val_r2s))],
            y=val_r2s,
            marker_color=colors,
            text=[f"{v:.4f}" for v in val_r2s],
            textposition='outside',
            name='R² par Fold'
        ))
        fig.add_hline(y=r2_mean, line_dash="dash", line_color="red",
                      annotation_text=f"Moyenne = {r2_mean:.4f}",
                      annotation_position="top right")
        fig.update_layout(
            title="R² de Validation par Fold (K-Fold, K=5)",
            yaxis_title="R²", xaxis_title="Fold",
            plot_bgcolor='#f9fafb', paper_bgcolor='white',
            yaxis=dict(range=[min(val_r2s)-0.01, 1.0]), height=400
        )
        st.plotly_chart(fig, use_container_width=True)

        summary_df = pd.DataFrame({
            'Fold': [f"Fold {i+1}" for i in range(len(val_r2s))],
            'R² Validation': [f"{v:.5f}" for v in val_r2s]
        })
        summary_df.loc[len(summary_df)] = ['**Moyenne ± Écart-type**',
                                             f"**{r2_mean:.5f} ± {np.std(val_r2s):.5f}**"]
        st.table(summary_df)

    with tab2:
        residuals = y_true - oof_preds

        col1, col2 = st.columns(2)
        with col1:
            fig_r1 = px.scatter(
                x=oof_preds[:3000], y=residuals[:3000],
                labels={'x': 'Valeurs prédites', 'y': 'Résidus'},
                title="Résidus vs Prédictions (3k points)",
                opacity=0.3, color_discrete_sequence=['#1a56db']
            )
            fig_r1.add_hline(y=0, line_dash="dash", line_color="red", line_width=2)
            fig_r1.update_layout(plot_bgcolor='#f9fafb', paper_bgcolor='white', height=380)
            st.plotly_chart(fig_r1, use_container_width=True)

        with col2:
            fig_r2 = px.histogram(
                x=residuals, nbins=80,
                labels={'x': 'Résidus', 'y': 'Fréquence'},
                title="Distribution des Résidus",
                color_discrete_sequence=['#1a56db']
            )
            fig_r2.add_vline(x=0, line_dash="dash", line_color="red", line_width=2)
            fig_r2.update_layout(plot_bgcolor='#f9fafb', paper_bgcolor='white', height=380)
            st.plotly_chart(fig_r2, use_container_width=True)

        fig_r3 = px.scatter(
            x=y_true[:3000], y=oof_preds[:3000],
            labels={'x': 'Valeurs réelles', 'y': 'Valeurs prédites'},
            title="Valeurs Réelles vs Prédites (3k points)",
            opacity=0.3, color_discrete_sequence=['#1a56db']
        )
        lims = [min(y_true.min(), oof_preds.min()), max(y_true.max(), oof_preds.max())]
        fig_r3.add_shape(type='line', x0=lims[0], y0=lims[0], x1=lims[1], y1=lims[1],
                         line=dict(color='red', dash='dash', width=2))
        fig_r3.update_layout(plot_bgcolor='#f9fafb', paper_bgcolor='white', height=420)
        st.plotly_chart(fig_r3, use_container_width=True)

        col3, col4, col5 = st.columns(3)
        col3.metric("Résidu moyen", f"{np.mean(residuals):.5f}")
        col4.metric("Résidu médian", f"{np.median(residuals):.5f}")
        col5.metric("Écart-type des résidus", f"{np.std(residuals):.5f}")

    with tab3:
        imp_df = pd.DataFrame({
            'feature': feature_cols,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False).head(20)
        imp_df['feature_label'] = imp_df['feature'].apply(
            lambda x: FEATURE_LABELS.get(x, x)
        )

        fig_imp = px.bar(
            imp_df, x='importance', y='feature_label',
            orientation='h', title="Top 20 Variables — Importance (Gain)",
            color='importance', color_continuous_scale='Blues',
            labels={'importance': 'Importance (Gain)', 'feature_label': 'Variable'}
        )
        fig_imp.update_layout(
            plot_bgcolor='#f9fafb', paper_bgcolor='white',
            yaxis=dict(autorange='reversed'), height=600
        )
        st.plotly_chart(fig_imp, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE : PRÉDICTION INTERACTIVE
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🔮 Prédiction Interactive":
    st.markdown("## 🔮 Simulateur de Risque d'Inondation")
    st.info("Ajustez les valeurs des 20 variables pour obtenir une estimation de la probabilité d'inondation.")

    with st.form("prediction_form"):
        st.markdown("### ⚙️ Paramètres environnementaux et socio-économiques")

        values = {}
        groups = [
            ("🌧️ Facteurs Climatiques", ['MonsoonIntensity', 'ClimateChange']),
            ("🏔️ Facteurs Géographiques", ['TopographyDrainage', 'Landslides', 'Watersheds', 'CoastalVulnerability']),
            ("🌳 Facteurs Environnementaux", ['Deforestation', 'Siltation', 'WetlandLoss', 'AgriculturalPractices']),
            ("🏙️ Facteurs Anthropiques", ['Urbanization', 'Encroachments', 'PopulationScore']),
            ("🏗️ Infrastructure", ['DamsQuality', 'DrainageSystems', 'DeterioratingInfrastructure', 'RiverManagement']),
            ("📋 Gouvernance", ['IneffectiveDisasterPreparedness', 'InadequatePlanning', 'PoliticalFactors'])
        ]

        for group_name, group_cols in groups:
            st.markdown(f"**{group_name}**")
            n = len(group_cols)
            cols = st.columns(min(n, 4))
            for i, col_name in enumerate(group_cols):
                label = FEATURE_LABELS.get(col_name, col_name)
                with cols[i % min(n, 4)]:
                    values[col_name] = st.slider(
                        label, min_value=0, max_value=10, value=5, key=col_name
                    )
            st.markdown("")

        submitted = st.form_submit_button("🚀 Prédire la probabilité d'inondation", use_container_width=True)

    if submitted:
        prob = predict_single(model, feature_cols, values)
        prob = float(np.clip(prob, 0.0, 1.0))
        risk_label, risk_color = flood_risk_label(prob)

        st.markdown("---")
        col1, col2, col3 = st.columns([1, 2, 1])

        with col2:
            st.markdown(f"""
            <div style='text-align:center; background:linear-gradient(135deg, #f0f9ff, #e0f2fe);
                         border-radius:16px; padding:2rem; border:2px solid {risk_color};
                         box-shadow:0 4px 20px rgba(0,0,0,0.1);'>
                <h2 style='color:{risk_color}; font-size:3rem; margin:0;'>{prob:.1%}</h2>
                <p style='font-size:1.3rem; margin:0.5rem 0; font-weight:bold;'>Probabilité d'Inondation</p>
                <span style='font-size:1.5rem;'>{risk_label}</span>
            </div>
            """, unsafe_allow_html=True)

        # Jauge
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=prob * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Risque d'Inondation (%)", 'font': {'size': 16}},
            delta={'reference': 50},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1},
                'bar': {'color': risk_color},
                'bgcolor': "white",
                'steps': [
                    {'range': [0, 35], 'color': '#d1fae5'},
                    {'range': [35, 55], 'color': '#fef3c7'},
                    {'range': [55, 70], 'color': '#fed7aa'},
                    {'range': [70, 100], 'color': '#fee2e2'},
                ],
                'threshold': {
                    'line': {'color': "darkred", 'width': 4},
                    'thickness': 0.8, 'value': prob * 100
                }
            }
        ))
        fig_gauge.update_layout(paper_bgcolor='white', height=300, margin=dict(t=40, b=0))
        st.plotly_chart(fig_gauge, use_container_width=True)

        # Variables les plus influentes
        st.markdown("#### 📊 Contribution des facteurs saisis")
        contrib = {FEATURE_LABELS.get(k, k): v for k, v in values.items()}
        contrib_df = pd.DataFrame(list(contrib.items()), columns=['Variable', 'Valeur'])
        contrib_df = contrib_df.sort_values('Valeur', ascending=False)

        fig_contrib = px.bar(
            contrib_df, x='Valeur', y='Variable',
            orientation='h', color='Valeur',
            color_continuous_scale='Blues',
            title="Valeurs saisies par variable (0–10)",
            range_x=[0, 10]
        )
        fig_contrib.update_layout(plot_bgcolor='#f9fafb', paper_bgcolor='white', height=520)
        st.plotly_chart(fig_contrib, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE : FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────
elif page == "📐 Feature Engineering":
    st.markdown("## 📐 Ingénierie des Variables")

    st.markdown("""
    Le feature engineering est l'étape qui a le plus contribué à l'amélioration du score.
    Voici les familles de nouvelles variables créées :
    """)

    col1, col2 = st.columns(2)

    with col1:
        with st.expander("🔗 Interactions métier", expanded=True):
            st.markdown("""
            **ClimateAnthropogenicInteraction**
            ```python
            (MonsoonIntensity + ClimateChange) ×
            (Deforestation + Urbanization + AgriculturalPractices + Encroachments)
            ```
            Capture l'effet combiné du stress climatique et des perturbations humaines.

            **InfrastructurePreventionInteraction**
            ```python
            (DamsQuality + DrainageSystems + DeterioratingInfrastructure) ×
            (RiverManagement + IneffectiveDisasterPreparedness + InadequatePlanning)
            ```
            Mesure l'efficacité combinée des infrastructures et de la gouvernance.
            """)

        with st.expander("📊 Statistiques de ligne", expanded=True):
            features_stats = {
                'row_sum': 'Somme des 20 variables',
                'row_mean': 'Moyenne arithmétique',
                'row_std': 'Écart-type',
                'row_max': 'Valeur maximale',
                'row_min': 'Valeur minimale',
                'row_range': 'Amplitude (max − min)',
                'row_median': 'Médiane',
                'row_mode': 'Mode (valeur la plus fréquente)',
                'row_cv': 'Coefficient de variation (std/mean)',
                'harmonic_mean': 'Moyenne harmonique',
                'geometric_mean': 'Moyenne géométrique',
            }
            for feat, desc in features_stats.items():
                st.markdown(f"• **`{feat}`** : {desc}")

    with col2:
        with st.expander("📈 Moments statistiques", expanded=True):
            st.markdown("""
            • **`row_skew`** : Asymétrie de la distribution (Skewness)
            • **`row_kurt`** : Aplatissement (Kurtosis)
            • **`2nd_moment`** : 2ème moment (moyenne des carrés)
            • **`3rd_moment`** : 3ème moment (moyenne des cubes)
            • **`row_zscore`** : Z-score moyen normalisé
            • **`entropy`** : Entropie de Shannon de la distribution des valeurs
            • **`skewness_75`** : Skewness basée sur le 3ème quartile
            • **`skewness_25`** : Skewness basée sur le 1er quartile
            """)

        with st.expander("📉 Quantiles de ligne", expanded=True):
            st.markdown("""
            Percentiles calculés sur les 20 features originales pour chaque observation :
            - **`q10`** à **`q90`** (par pas de 10%)
            
            Ces quantiles capturent la forme de la distribution des valeurs pour chaque ligne.
            """)

        with st.expander("🔢 Comptages de valeurs", expanded=True):
            st.markdown("""
            Pour chaque valeur entière `v` de 0 à 15 :
            ```python
            cnt_v = nombre de colonnes ayant exactement la valeur v
            ```
            Ces features encodent implicitement le profil de risque global
            (ex. : beaucoup de valeurs élevées ↔ risque élevé).
            """)

    # Visualisation de l'impact
    st.markdown("---")
    st.markdown("### 📊 Nombre de features créées par famille")

    families = {
        'Interactions métier': 2,
        'Statistiques de ligne': 11,
        'Moments statistiques': 8,
        'Quantiles (10%–90%)': 9,
        'Moyennes harm./géom.': 2,
        'Comptages (cnt_v)': 16,
        'Features originales normalisées': 20
    }
    fig = px.bar(
        x=list(families.keys()), y=list(families.values()),
        title=f"Répartition des {sum(families.values())} features par famille",
        color=list(families.values()),
        color_continuous_scale='Blues',
        labels={'x': 'Famille de features', 'y': 'Nombre de features'}
    )
    fig.update_layout(plot_bgcolor='#f9fafb', paper_bgcolor='white', height=380,
                      xaxis_tickangle=-30, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class='footer'>
    🌊 Flood Prediction Dashboard · Judicaël Karol DOBOEVI · ENSGMM, Bénin<br>
    Projet Fil Conducteur — Formation Data Scientist · Africa Tech Up Tour<br>
    LightGBM · K-Fold Cross-Validation · Score R² : 0.869
</div>
""", unsafe_allow_html=True)
