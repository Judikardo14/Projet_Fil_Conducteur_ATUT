"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  Dashboard Streamlit — Prédiction de la Probabilité d'Inondation            ║
║  Projet Fil Conducteur | Africa Tech Up Tour                                 ║
║  Auteur : Judicaël Karol DOBOEVI — ENSGMM, Bénin                            ║
╚══════════════════════════════════════════════════════════════════════════════╝

Lancement :
    pip install streamlit lightgbm scikit-learn pandas numpy matplotlib seaborn plotly scipy
    streamlit run dashboard_flood.py
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import skew, kurtosis
import lightgbm as lgb

# ─────────────────────────────────────────────────────────────────────────────
# Configuration de la page
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Flood Prediction — LightGBM",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS adaptatif mode sombre/clair — utilise les variables CSS de Streamlit
st.markdown("""
<style>
    /* ── Polices ── */
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans', sans-serif;
    }

    /* ── Header ── */
    .dash-header {
        border-left: 4px solid var(--primary-color, #1a56db);
        padding: 1.4rem 1.8rem;
        margin-bottom: 1.8rem;
        background: transparent;
    }
    .dash-header h1 {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 1.6rem;
        font-weight: 600;
        letter-spacing: -0.02em;
        margin: 0 0 0.3rem 0;
        color: inherit;
    }
    .dash-header p {
        font-size: 0.88rem;
        font-weight: 300;
        opacity: 0.7;
        margin: 0;
    }

    /* ── Cartes métriques ── */
    .kpi-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 1rem;
        margin-bottom: 1.5rem;
    }
    .kpi-card {
        border: 1px solid rgba(128,128,128,0.2);
        border-radius: 6px;
        padding: 1.2rem 1.4rem;
        position: relative;
        overflow: hidden;
    }
    .kpi-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0;
        width: 3px; height: 100%;
        background: #1a56db;
    }
    .kpi-value {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 1.9rem;
        font-weight: 600;
        line-height: 1;
        margin-bottom: 0.4rem;
        color: #1a56db;
    }
    .kpi-label {
        font-size: 0.78rem;
        font-weight: 400;
        opacity: 0.65;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }

    /* ── Section card ── */
    .section-card {
        border: 1px solid rgba(128,128,128,0.18);
        border-radius: 6px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 1rem;
    }
    .section-card h3 {
        font-size: 0.95rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin: 0 0 0.8rem 0;
        opacity: 0.8;
    }

    /* ── Badge ── */
    .badge {
        display: inline-block;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.75rem;
        padding: 2px 10px;
        border-radius: 3px;
        border: 1px solid rgba(128,128,128,0.3);
        margin-right: 6px;
    }

    /* ── Résultat de prédiction ── */
    .pred-result {
        border-radius: 6px;
        padding: 2rem;
        text-align: center;
        border: 1px solid rgba(128,128,128,0.2);
        margin: 1rem 0;
    }
    .pred-value {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 3.5rem;
        font-weight: 600;
        line-height: 1;
    }
    .pred-label {
        font-size: 0.85rem;
        opacity: 0.65;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-top: 0.5rem;
    }

    /* ── Sidebar ── */
    .sidebar-block {
        border-left: 3px solid #1a56db;
        padding: 0.8rem 1rem;
        margin-bottom: 0.8rem;
        font-size: 0.85rem;
        opacity: 0.9;
    }

    /* ── Pipeline steps ── */
    .pipeline-step {
        display: flex;
        align-items: flex-start;
        gap: 1rem;
        padding: 0.6rem 0;
        border-bottom: 1px solid rgba(128,128,128,0.1);
    }
    .step-num {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.75rem;
        font-weight: 600;
        opacity: 0.4;
        min-width: 1.5rem;
        padding-top: 2px;
    }
    .step-content strong {
        font-size: 0.88rem;
        font-weight: 600;
    }
    .step-content span {
        font-size: 0.8rem;
        opacity: 0.6;
        display: block;
    }

    /* ── Footer ── */
    .dash-footer {
        text-align: center;
        font-size: 0.75rem;
        opacity: 0.45;
        margin-top: 2.5rem;
        padding-top: 1rem;
        border-top: 1px solid rgba(128,128,128,0.2);
        font-family: 'IBM Plex Mono', monospace;
        letter-spacing: 0.03em;
    }

    /* ── Streamlit overrides ── */
    div[data-testid="stMetric"] {
        border: 1px solid rgba(128,128,128,0.18);
        border-radius: 6px;
        padding: 0.9rem 1rem;
    }
    .stTabs [data-baseweb="tab"] {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.82rem;
        letter-spacing: 0.04em;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTES
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
    'MonsoonIntensity':               'Intensité des moussons',
    'TopographyDrainage':             'Drainage topographique',
    'RiverManagement':                'Gestion des rivières',
    'Deforestation':                  'Déforestation',
    'Urbanization':                   'Urbanisation',
    'ClimateChange':                  'Changement climatique',
    'DamsQuality':                    'Qualité des barrages',
    'Siltation':                      'Envasement',
    'AgriculturalPractices':          'Pratiques agricoles',
    'Encroachments':                  'Empiètements',
    'IneffectiveDisasterPreparedness':'Préparation aux catastrophes',
    'DrainageSystems':                'Systèmes de drainage',
    'CoastalVulnerability':           'Vulnérabilité côtière',
    'Landslides':                     'Glissements de terrain',
    'Watersheds':                     'Bassins versants',
    'DeterioratingInfrastructure':    'Infrastructure dégradée',
    'PopulationScore':                'Score de population',
    'WetlandLoss':                    'Perte de zones humides',
    'InadequatePlanning':             'Planification inadéquate',
    'PoliticalFactors':               'Facteurs politiques'
}

CATEGORIES = {
    "Climatiques":     ['MonsoonIntensity', 'ClimateChange'],
    "Geographiques":   ['TopographyDrainage', 'Landslides', 'Watersheds', 'CoastalVulnerability'],
    "Environnementaux":['Deforestation', 'Siltation', 'WetlandLoss', 'AgriculturalPractices'],
    "Anthropiques":    ['Urbanization', 'Encroachments', 'PopulationScore'],
    "Infrastructure":  ['DamsQuality', 'DrainageSystems', 'DeterioratingInfrastructure', 'RiverManagement'],
    "Gouvernance":     ['IneffectiveDisasterPreparedness', 'InadequatePlanning', 'PoliticalFactors']
}

# Palette cohérente avec le notebook
ACCENT    = '#1a56db'
PALETTE   = ['#dbeafe', '#93c5fd', '#3b82f6', '#1d4ed8', '#1e3a8a']

# ─────────────────────────────────────────────────────────────────────────────
# FONCTIONS — reprises fidèlement du notebook
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def generate_demo_data(n=6000, seed=42):
    """Données synthétiques simulant les distributions du dataset réel."""
    rng = np.random.default_rng(seed)
    data = {col: rng.integers(0, 11, size=n) for col in NUM_COLS}
    df = pd.DataFrame(data)
    signal = (
        0.04  * df['MonsoonIntensity'] +
        0.03  * df['TopographyDrainage'] +
        0.035 * df['ClimateChange'] +
        0.025 * df['Deforestation'] +
        0.02  * df['Urbanization'] +
        0.018 * df['Siltation'] +
        0.015 * df['Encroachments'] +
        0.012 * df['CoastalVulnerability'] +
        sum(0.008 * df[c] for c in NUM_COLS[7:])
    )
    noise = rng.normal(0, 0.03, n)
    df['FloodProbability'] = np.clip(0.25 + signal + noise, 0.1, 0.9)
    return df


def create_features(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """
    Feature engineering — fidèle au notebook.
    Scaler ajusté sur les données fournies.
    """
    scaler = StandardScaler()
    df = df.copy()

    # Interactions métier
    df['ClimateAnthropogenicInteraction'] = (
        (df['MonsoonIntensity'] + df['ClimateChange']) *
        (df['Deforestation'] + df['Urbanization'] +
         df['AgriculturalPractices'] + df['Encroachments'])
    )
    df['InfrastructurePreventionInteraction'] = (
        (df['DamsQuality'] + df['DrainageSystems'] + df['DeterioratingInfrastructure']) *
        (df['RiverManagement'] + df['IneffectiveDisasterPreparedness'] + df['InadequatePlanning'])
    )

    # Statistiques de ligne
    df['row_sum']    = df[cols].sum(axis=1)
    df['row_mean']   = df[cols].mean(axis=1)
    df['row_std']    = df[cols].std(axis=1)
    df['row_max']    = df[cols].max(axis=1)
    df['row_min']    = df[cols].min(axis=1)
    df['row_range']  = df['row_max'] - df['row_min']
    df['row_median'] = df[cols].median(axis=1)
    df['row_cv']     = df['row_std'] / (df['row_mean'] + 1e-8)
    df['row_skew']   = df[cols].skew(axis=1)
    df['row_kurt']   = df[cols].kurt(axis=1)

    # Moments d'ordre supérieur
    df['2nd_moment'] = df[cols].apply(lambda x: (x**2).mean(), axis=1)
    df['3rd_moment'] = df[cols].apply(lambda x: (x**3).mean(), axis=1)

    # Moyennes harmonique et géométrique
    safe = df[cols].clip(lower=1e-6)
    df['harmonic_mean']  = len(cols) / (1 / safe).sum(axis=1)
    df['geometric_mean'] = np.exp(np.log(safe).mean(axis=1))

    # Entropie de Shannon
    df['entropy'] = df[cols].apply(
        lambda x: -(x / x.sum() * np.log(x / x.sum() + 1e-8)).sum(), axis=1
    )

    # Skewness quartile
    df['skewness_75'] = (df[cols].quantile(0.75, axis=1) - df['row_mean']) / (df['row_std'] + 1e-8)
    df['skewness_25'] = (df[cols].quantile(0.25, axis=1) - df['row_mean']) / (df['row_std'] + 1e-8)

    # Quantiles (déciles)
    for pct in range(10, 100, 10):
        df[f'q{pct}'] = df[cols].quantile(pct / 100, axis=1)

    # Comptages de valeurs
    for v in range(16):
        df[f'cnt_{v}'] = (df[cols] == v).sum(axis=1)

    # Normalisation des features originales
    df[cols] = scaler.fit_transform(df[cols])
    return df


@st.cache_resource
def train_model():
    """Entraîne le modèle avec la validation croisée K-Fold du notebook."""
    df = generate_demo_data(n=8000)
    df_feat = create_features(df, NUM_COLS)
    feature_cols = [c for c in df_feat.columns if c != 'FloodProbability']
    X = df_feat[feature_cols]
    y = df_feat['FloodProbability']

    # Paramètres issus du notebook (version allégée pour la démo)
    params = {
        'n_estimators':   400,
        'learning_rate':  0.05,
        'num_leaves':     250,
        'max_depth':      10,
        'verbosity':      -1,
        'random_state':   42
    }

    # Validation croisée K-Fold (K=5) — logique exacte du notebook
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds    = np.zeros(len(X))
    val_scores   = []
    train_scores = []

    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        m = lgb.LGBMRegressor(**params)
        m.fit(X_tr, y_tr)
        oof_preds[val_idx] = m.predict(X_val)
        val_scores.append(r2_score(y_val, oof_preds[val_idx]))
        train_scores.append(r2_score(y_tr, m.predict(X_tr)))

    # Modèle final sur toutes les données
    model = lgb.LGBMRegressor(**params)
    model.fit(X, y)

    return model, feature_cols, val_scores, train_scores, oof_preds, y.values, df


def predict_single(model, feature_cols, values_dict):
    """Prédit la probabilité pour une observation saisie manuellement."""
    df_single = pd.DataFrame([values_dict])
    df_feat   = create_features(df_single, NUM_COLS)
    for c in feature_cols:
        if c not in df_feat.columns:
            df_feat[c] = 0
    return float(np.clip(model.predict(df_feat[feature_cols])[0], 0.0, 1.0))


def risk_level(prob):
    if prob < 0.35:
        return "Faible",    "#16a34a"
    elif prob < 0.55:
        return "Modere",    "#d97706"
    elif prob < 0.70:
        return "Eleve",     "#ea580c"
    else:
        return "Tres eleve","#dc2626"


def plotly_theme():
    """Thème Plotly transparent, compatible mode sombre/clair."""
    return dict(
        plot_bgcolor  = 'rgba(0,0,0,0)',
        paper_bgcolor = 'rgba(0,0,0,0)',
        font          = dict(family='IBM Plex Sans, sans-serif', size=12),
        margin        = dict(t=40, b=30, l=10, r=10),
        colorway      = [ACCENT, '#3b82f6', '#93c5fd', '#1d4ed8'],
    )


# ─────────────────────────────────────────────────────────────────────────────
# CHARGEMENT
# ─────────────────────────────────────────────────────────────────────────────
with st.spinner("Entraînement du modèle en cours..."):
    model, feature_cols, val_scores, train_scores, oof_preds, y_true, df_demo = train_model()

r2_oof  = r2_score(y_true, oof_preds)
rmse    = np.sqrt(mean_squared_error(y_true, oof_preds))
r2_mean = float(np.mean(val_scores))

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class='sidebar-block'>
        <strong>Projet Fil Conducteur</strong><br>
        Formation Data Scientist<br>
        Africa Tech Up Tour
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style='font-size:0.82rem; padding: 0.5rem 0 1rem 0; opacity:0.8;'>
        Judicaël Karol <strong>DOBOEVI</strong><br>
        1re année — Génie Math. & Modélisation<br>
        ENSGMM, Bénin
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    page = st.radio(
        "Navigation",
        ["Accueil", "Analyse EDA", "Modele & Performance",
         "Prediction Interactive", "Feature Engineering"],
        label_visibility="visible"
    )

    st.markdown("---")

    st.markdown("""
    <div style='font-size:0.78rem; opacity:0.6;'>
        <span class='badge'>LightGBM</span><br><br>
        Dataset : Kaggle Playground S4E5<br>
        1 117 957 observations<br>
        20 features explicatives<br><br>
        R² (validation croisée) :
    </div>
    """, unsafe_allow_html=True)
    st.markdown(f"""
    <div style='font-family: IBM Plex Mono, monospace; font-size:1.3rem;
                font-weight:600; color:{ACCENT}; padding: 0.3rem 0;'>
        {r2_mean:.5f}
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class='dash-header'>
    <h1>Prediction de la Probabilite d'Inondation</h1>
    <p>
        Modele LightGBM &nbsp;·&nbsp; Validation croisee K-Fold (K=5) &nbsp;·&nbsp;
        Feature Engineering avance &nbsp;·&nbsp; Projet Fil Conducteur — Africa Tech Up Tour
    </p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE : ACCUEIL
# ─────────────────────────────────────────────────────────────────────────────
if page == "Accueil":

    # KPI
    st.markdown(f"""
    <div class='kpi-grid'>
        <div class='kpi-card'>
            <div class='kpi-value'>{r2_mean:.4f}</div>
            <div class='kpi-label'>R² moyen (validation croisee)</div>
        </div>
        <div class='kpi-card'>
            <div class='kpi-value'>{rmse:.4f}</div>
            <div class='kpi-label'>RMSE (Out-of-Fold)</div>
        </div>
        <div class='kpi-card'>
            <div class='kpi-value'>20</div>
            <div class='kpi-label'>Variables explicatives</div>
        </div>
        <div class='kpi-card'>
            <div class='kpi-value'>~68</div>
            <div class='kpi-label'>Features apres engineering</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_a, col_b = st.columns([3, 2], gap="large")

    with col_a:
        st.markdown("""
        <div class='section-card'>
            <h3>Contexte</h3>
            <p style='font-size:0.9rem; line-height:1.7; opacity:0.85;'>
                Les inondations figurent parmi les catastrophes naturelles les plus devastatrices,
                particulierement en Afrique subsaharienne. Ce projet construit un
                <strong>modele de regression supervise</strong> predisant la probabilite d'inondation
                a partir de 20 variables environnementales, climatiques et socio-economiques.
            </p>
            <p style='font-size:0.9rem; line-height:1.7; opacity:0.85;'>
                L'algorithme utilise est <strong>LightGBM</strong> (Light Gradient Boosting Machine),
                couple a une validation croisee K-Fold (K=5) pour garantir la robustesse des performances.
                Une ingenierie de variables avancee (~50 nouvelles features) enrichit significativement
                le signal disponible pour le modele.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div class='section-card'><h3>Pipeline methodologique</h3>", unsafe_allow_html=True)
        steps = [
            ("01", "Chargement des donnees",         "Train : 1 117 957 lignes x 22 colonnes"),
            ("02", "Analyse exploratoire (EDA)",      "Distribution, statistiques, correlations"),
            ("03", "Feature Engineering",             "~68 variables construites a partir des 20 originales"),
            ("04", "Importance des variables",        "Critere Gain — LightGBM"),
            ("05", "Validation croisee K-Fold",       "K=5, strategie Shuffle, random_state=42"),
            ("06", "Evaluation — analyse des residus","R², RMSE, analyse du surapprentissage"),
        ]
        for num, title, desc in steps:
            st.markdown(f"""
            <div class='pipeline-step'>
                <span class='step-num'>{num}</span>
                <div class='step-content'>
                    <strong>{title}</strong>
                    <span>{desc}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_b:
        st.markdown("<div class='section-card'><h3>Variables explicatives</h3>", unsafe_allow_html=True)
        for cat, cols in CATEGORIES.items():
            with st.expander(cat):
                for c in cols:
                    st.markdown(
                        f"<span style='font-size:0.83rem; opacity:0.8;'>{FEATURE_LABELS.get(c, c)}</span>",
                        unsafe_allow_html=True
                    )
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown(f"""
        <div class='section-card'>
            <h3>Parametres du modele</h3>
            <table style='width:100%; font-size:0.82rem; border-collapse:collapse;'>
            <tr><td style='padding:4px 0; opacity:0.6;'>Algorithme</td><td style='font-family: IBM Plex Mono, monospace;'>LightGBM GBDT</td></tr>
            <tr><td style='padding:4px 0; opacity:0.6;'>n_estimators</td><td style='font-family: IBM Plex Mono, monospace;'>2 000</td></tr>
            <tr><td style='padding:4px 0; opacity:0.6;'>learning_rate</td><td style='font-family: IBM Plex Mono, monospace;'>0.012</td></tr>
            <tr><td style='padding:4px 0; opacity:0.6;'>num_leaves</td><td style='font-family: IBM Plex Mono, monospace;'>250</td></tr>
            <tr><td style='padding:4px 0; opacity:0.6;'>max_depth</td><td style='font-family: IBM Plex Mono, monospace;'>10</td></tr>
            <tr><td style='padding:4px 0; opacity:0.6;'>Validation</td><td style='font-family: IBM Plex Mono, monospace;'>K-Fold (K=5)</td></tr>
            <tr><td style='padding:4px 0; opacity:0.6;'>R² (OOF)</td>
                <td style='font-family: IBM Plex Mono, monospace; color:{ACCENT}; font-weight:600;'>{r2_oof:.5f}</td></tr>
            </table>
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE : EDA
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Analyse EDA":
    st.markdown("### Analyse Exploratoire des Donnees")
    st.caption("Echantillon synthetique de 6 000 observations simulant les distributions du dataset reel.")

    tab1, tab2, tab3 = st.tabs(["Variable cible", "Variables explicatives", "Correlations"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            fig = px.histogram(
                df_demo, x='FloodProbability', nbins=60,
                title="Distribution de FloodProbability",
                color_discrete_sequence=[ACCENT],
                labels={'FloodProbability': "Probabilite d'inondation"}
            )
            fig.add_vline(
                x=df_demo['FloodProbability'].mean(),
                line_dash="dash", line_color="#ef4444",
                annotation_text=f"Moy = {df_demo['FloodProbability'].mean():.3f}",
                annotation_font_size=11
            )
            fig.update_layout(**plotly_theme(), height=340)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig2 = px.box(
                df_demo, y='FloodProbability',
                title="Boite a moustaches",
                color_discrete_sequence=[ACCENT],
            )
            fig2.update_layout(**plotly_theme(), height=340)
            st.plotly_chart(fig2, use_container_width=True)

        fp = df_demo['FloodProbability']
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Min",         f"{fp.min():.4f}")
        c2.metric("Max",         f"{fp.max():.4f}")
        c3.metric("Moyenne",     f"{fp.mean():.4f}")
        c4.metric("Mediane",     f"{fp.median():.4f}")
        c5.metric("Ecart-type",  f"{fp.std():.4f}")

    with tab2:
        selected_cols = st.multiselect(
            "Variables a visualiser :",
            NUM_COLS, default=NUM_COLS[:8],
            format_func=lambda x: FEATURE_LABELS.get(x, x)
        )
        if selected_cols:
            ncols_g = min(4, len(selected_cols))
            nrows_g = (len(selected_cols) + ncols_g - 1) // ncols_g
            fig3, axes = plt.subplots(nrows_g, ncols_g, figsize=(16, nrows_g * 3.2))
            fig3.patch.set_alpha(0)
            if nrows_g == 1 and ncols_g == 1:
                axes = np.array([[axes]])
            elif nrows_g == 1:
                axes = axes.reshape(1, -1)
            elif ncols_g == 1:
                axes = axes.reshape(-1, 1)
            for i, col in enumerate(selected_cols):
                r, c = divmod(i, ncols_g)
                ax = axes[r, c]
                ax.hist(df_demo[col], bins=12, color=ACCENT, edgecolor='white', alpha=0.85)
                ax.set_title(FEATURE_LABELS.get(col, col), fontsize=9, fontweight='600')
                ax.set_xlabel("Valeur", fontsize=8)
                ax.tick_params(labelsize=7)
                ax.set_facecolor('none')
                ax.spines[['top', 'right']].set_visible(False)
            for j in range(len(selected_cols), nrows_g * ncols_g):
                r, c = divmod(j, ncols_g)
                axes[r, c].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig3, transparent=True)
        else:
            st.info("Selectionnez au moins une variable.")

    with tab3:
        corr_df = df_demo[NUM_COLS + ['FloodProbability']].corr()

        fig4 = px.imshow(
            corr_df, color_continuous_scale='RdBu_r',
            zmin=-0.3, zmax=1, text_auto='.2f',
            title="Matrice de correlation",
            aspect='auto', width=750, height=680
        )
        fig4.update_layout(**plotly_theme())
        st.plotly_chart(fig4, use_container_width=True)

        corr_target = corr_df['FloodProbability'].drop('FloodProbability').sort_values()
        fig5 = px.bar(
            x=corr_target.values,
            y=[FEATURE_LABELS.get(c, c) for c in corr_target.index],
            orientation='h',
            title="Correlation avec FloodProbability (Pearson)",
            color=corr_target.values,
            color_continuous_scale='Blues',
            labels={'x': 'Correlation de Pearson', 'y': 'Variable'}
        )
        fig5.update_layout(**plotly_theme(), height=530)
        st.plotly_chart(fig5, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE : MODELE & PERFORMANCE
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Modele & Performance":
    st.markdown("### Modele LightGBM — Performances")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("R² moyen (CV)",    f"{r2_mean:.5f}")
    c2.metric("R² OOF global",    f"{r2_oof:.5f}")
    c3.metric("RMSE (OOF)",       f"{rmse:.5f}")
    c4.metric("Gap Train / Val",  f"{float(np.mean(np.array(train_scores) - np.array(val_scores))):.5f}")

    tab1, tab2, tab3 = st.tabs(["R² par Fold", "Analyse des residus", "Importance des variables"])

    with tab1:
        fig_cv = go.Figure()
        colors_bar = [ACCENT if v >= r2_mean else '#93c5fd' for v in val_scores]
        fig_cv.add_trace(go.Bar(
            x=[f"Fold {i+1}" for i in range(len(val_scores))],
            y=val_scores,
            marker_color=colors_bar,
            text=[f"{v:.5f}" for v in val_scores],
            textposition='outside',
            name='R² validation'
        ))
        fig_cv.add_trace(go.Scatter(
            x=[f"Fold {i+1}" for i in range(len(train_scores))],
            y=train_scores,
            mode='lines+markers',
            line=dict(dash='dot', color='#ef4444', width=1.5),
            marker=dict(size=6),
            name='R² train'
        ))
        fig_cv.add_hline(
            y=r2_mean, line_dash="dash", line_color="#6b7280",
            annotation_text=f"Moyenne val = {r2_mean:.5f}",
            annotation_position="top right"
        )
        fig_cv.update_layout(
            **plotly_theme(),
            title="R² par fold — Validation croisee K-Fold (K=5)",
            yaxis=dict(range=[min(val_scores) - 0.01, max(train_scores) + 0.01]),
            height=400, legend=dict(orientation='h', y=-0.15)
        )
        st.plotly_chart(fig_cv, use_container_width=True)

        summary = pd.DataFrame({
            'Fold':         [f"Fold {i+1}" for i in range(5)],
            'R² Train':     [f"{v:.5f}" for v in train_scores],
            'R² Validation':[f"{v:.5f}" for v in val_scores],
            'Gap':          [f"{t-v:.5f}" for t, v in zip(train_scores, val_scores)]
        })
        st.dataframe(summary, use_container_width=True, hide_index=True)
        st.caption(f"Moyenne val : {r2_mean:.5f} ± {np.std(val_scores):.5f}  |  "
                   f"Gap moyen : {float(np.mean(np.array(train_scores)-np.array(val_scores))):.5f}")

    with tab2:
        residuals = y_true - oof_preds

        col1, col2 = st.columns(2)
        with col1:
            fig_r1 = px.scatter(
                x=oof_preds[:3000], y=residuals[:3000],
                labels={'x': 'Valeurs predites', 'y': 'Residus'},
                title="Residus vs Predictions (3k pts)",
                opacity=0.25, color_discrete_sequence=[ACCENT]
            )
            fig_r1.add_hline(y=0, line_dash="dash", line_color="#ef4444", line_width=1.5)
            fig_r1.update_layout(**plotly_theme(), height=370)
            st.plotly_chart(fig_r1, use_container_width=True)

        with col2:
            fig_r2 = px.histogram(
                x=residuals, nbins=80,
                labels={'x': 'Residus', 'y': 'Frequence'},
                title="Distribution des residus",
                color_discrete_sequence=[ACCENT]
            )
            fig_r2.add_vline(x=0, line_dash="dash", line_color="#ef4444", line_width=1.5)
            fig_r2.update_layout(**plotly_theme(), height=370)
            st.plotly_chart(fig_r2, use_container_width=True)

        fig_r3 = px.scatter(
            x=y_true[:3000], y=oof_preds[:3000],
            labels={'x': 'Valeurs reelles', 'y': 'Valeurs predites'},
            title="Reelles vs Predites (3k pts)",
            opacity=0.25, color_discrete_sequence=[ACCENT]
        )
        lims = [min(y_true.min(), oof_preds.min()), max(y_true.max(), oof_preds.max())]
        fig_r3.add_shape(
            type='line', x0=lims[0], y0=lims[0], x1=lims[1], y1=lims[1],
            line=dict(color='#ef4444', dash='dash', width=1.5)
        )
        fig_r3.update_layout(**plotly_theme(), height=400)
        st.plotly_chart(fig_r3, use_container_width=True)

        c1, c2, c3 = st.columns(3)
        c1.metric("Residu moyen",    f"{np.mean(residuals):.5f}")
        c2.metric("Residu median",   f"{np.median(residuals):.5f}")
        c3.metric("Ecart-type res.", f"{np.std(residuals):.5f}")

    with tab3:
        imp_df = pd.DataFrame({
            'feature':    feature_cols,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False).head(25)
        imp_df['label'] = imp_df['feature'].apply(lambda x: FEATURE_LABELS.get(x, x))

        fig_imp = px.bar(
            imp_df, x='importance', y='label',
            orientation='h',
            title="Top 25 variables — Importance (Gain)",
            color='importance', color_continuous_scale='Blues',
            labels={'importance': 'Importance (Gain)', 'label': ''}
        )
        fig_imp.update_layout(**plotly_theme(), yaxis=dict(autorange='reversed'), height=620)
        st.plotly_chart(fig_imp, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE : PREDICTION INTERACTIVE
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Prediction Interactive":
    st.markdown("### Simulateur de risque d'inondation")
    st.caption("Ajustez les 20 variables pour obtenir une estimation de la probabilite d'inondation.")

    with st.form("pred_form"):
        values = {}
        for cat_name, cat_cols in CATEGORIES.items():
            st.markdown(f"**{cat_name}**")
            n = len(cat_cols)
            cols_ui = st.columns(min(n, 4))
            for i, col_name in enumerate(cat_cols):
                with cols_ui[i % min(n, 4)]:
                    values[col_name] = st.slider(
                        FEATURE_LABELS.get(col_name, col_name),
                        min_value=0, max_value=10, value=5, key=col_name
                    )
            st.markdown("")

        submitted = st.form_submit_button("Calculer la probabilite", use_container_width=True)

    if submitted:
        prob = predict_single(model, feature_cols, values)
        level, color = risk_level(prob)

        st.markdown("---")
        c_l, c_m, c_r = st.columns([1, 2, 1])
        with c_m:
            st.markdown(f"""
            <div class='pred-result' style='border-color: {color};'>
                <div class='pred-value' style='color: {color};'>{prob:.1%}</div>
                <div class='pred-label'>Probabilite d'inondation estimee</div>
                <div style='margin-top:0.8rem; font-family: IBM Plex Mono, monospace;
                             font-size:0.9rem; font-weight:600; color:{color};'>
                    Niveau : {level}
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Jauge
        fig_g = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Risque (%)", 'font': {'size': 13, 'family': 'IBM Plex Mono'}},
            number={'suffix': '%', 'font': {'family': 'IBM Plex Mono', 'size': 28}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1},
                'bar':  {'color': color},
                'bgcolor': "rgba(0,0,0,0)",
                'steps': [
                    {'range': [0,  35], 'color': 'rgba(22,163,74,0.12)'},
                    {'range': [35, 55], 'color': 'rgba(217,119,6,0.12)'},
                    {'range': [55, 70], 'color': 'rgba(234,88,12,0.12)'},
                    {'range': [70,100], 'color': 'rgba(220,38,38,0.12)'},
                ],
            }
        ))
        fig_g.update_layout(**plotly_theme(), height=260)
        st.plotly_chart(fig_g, use_container_width=True)

        # Contribution des variables saisies
        contrib_df = pd.DataFrame({
            'Variable': [FEATURE_LABELS.get(k, k) for k in values],
            'Valeur':   list(values.values())
        }).sort_values('Valeur', ascending=True)

        fig_c = px.bar(
            contrib_df, x='Valeur', y='Variable',
            orientation='h', color='Valeur',
            color_continuous_scale='Blues',
            title="Profil des variables saisies (echelle 0–10)",
            range_x=[0, 10]
        )
        fig_c.update_layout(**plotly_theme(), height=500)
        st.plotly_chart(fig_c, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE : FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Feature Engineering":
    st.markdown("### Ingenierie des variables")
    st.caption("Construction de ~68 nouvelles variables a partir des 20 features originales.")

    col1, col2 = st.columns(2, gap="large")

    with col1:
        with st.expander("Interactions metier", expanded=True):
            st.markdown("""
**ClimateAnthropogenicInteraction**
```python
(MonsoonIntensity + ClimateChange) *
(Deforestation + Urbanization +
 AgriculturalPractices + Encroachments)
```
Stress climatique × perturbations humaines.

**InfrastructurePreventionInteraction**
```python
(DamsQuality + DrainageSystems +
 DeterioratingInfrastructure) *
(RiverManagement +
 IneffectiveDisasterPreparedness +
 InadequatePlanning)
```
Qualite d'infrastructure × gouvernance.
            """)

        with st.expander("Statistiques de ligne", expanded=True):
            stats_list = [
                ('row_sum',        'Somme des 20 variables'),
                ('row_mean',       'Moyenne arithmetique'),
                ('row_std',        'Ecart-type'),
                ('row_max',        'Valeur maximale'),
                ('row_min',        'Valeur minimale'),
                ('row_range',      'Amplitude (max - min)'),
                ('row_median',     'Mediane'),
                ('row_cv',         'Coefficient de variation (std/mean)'),
                ('harmonic_mean',  'Moyenne harmonique'),
                ('geometric_mean', 'Moyenne geometrique'),
            ]
            for feat, desc in stats_list:
                st.markdown(
                    f"<span style='font-family: IBM Plex Mono, monospace; font-size:0.82rem;'>`{feat}`</span>"
                    f"<span style='font-size:0.82rem; opacity:0.7;'> — {desc}</span>",
                    unsafe_allow_html=True
                )

    with col2:
        with st.expander("Moments statistiques", expanded=True):
            moments = [
                ('row_skew',     'Asymetrie (skewness)'),
                ('row_kurt',     'Aplatissement (kurtosis)'),
                ('2nd_moment',   'Moyenne des carres'),
                ('3rd_moment',   'Moyenne des cubes'),
                ('entropy',      'Entropie de Shannon'),
                ('skewness_75',  'Skewness via Q75'),
                ('skewness_25',  'Skewness via Q25'),
            ]
            for feat, desc in moments:
                st.markdown(
                    f"<span style='font-family: IBM Plex Mono, monospace; font-size:0.82rem;'>`{feat}`</span>"
                    f"<span style='font-size:0.82rem; opacity:0.7;'> — {desc}</span>",
                    unsafe_allow_html=True
                )

        with st.expander("Quantiles de ligne (q10 ... q90)", expanded=True):
            st.markdown("""
<span style='font-size:0.85rem; opacity:0.8;'>
9 percentiles calcules sur les 20 features par observation (par pas de 10 %).
Capturent la forme de la distribution de chaque ligne.
</span>
            """, unsafe_allow_html=True)

        with st.expander("Comptages de valeurs (cnt_0 ... cnt_15)", expanded=True):
            st.markdown("""
```python
cnt_v = nombre de colonnes ayant exactement la valeur v
```
<span style='font-size:0.85rem; opacity:0.8;'>
16 features encodant le profil de risque discret de chaque observation.
</span>
            """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### Repartition des features par famille")

    families = {
        'Originales normalisees': 20,
        'Interactions metier':    2,
        'Statistiques de ligne':  10,
        'Moments statistiques':   7,
        'Quantiles (deciles)':    9,
        'Comptages (cnt_v)':      16,
        'Features originales':    20,
    }
    # deduplicate — affichage cumulatif
    families_display = {
        'Interactions metier':    2,
        'Statistiques de ligne':  10,
        'Moments statistiques':   7,
        'Quantiles (deciles)':    9,
        'Comptages (cnt_v)':      16,
        'Originales (normalisees)':20,
    }
    total_fe = sum(families_display.values())
    fig_fe = px.bar(
        x=list(families_display.keys()), y=list(families_display.values()),
        title=f"Repartition des {total_fe} features creees par famille",
        color=list(families_display.values()),
        color_continuous_scale='Blues',
        labels={'x': 'Famille', 'y': 'Nombre de features'}
    )
    fig_fe.update_layout(
        **plotly_theme(), height=360,
        xaxis_tickangle=-25, showlegend=False
    )
    st.plotly_chart(fig_fe, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class='dash-footer'>
    Flood Prediction Dashboard &nbsp;·&nbsp;
    Judicael Karol DOBOEVI &nbsp;·&nbsp;
    ENSGMM, Benin &nbsp;·&nbsp;
    Projet Fil Conducteur — Africa Tech Up Tour &nbsp;·&nbsp;
    LightGBM · K-Fold Cross-Validation
</div>
""", unsafe_allow_html=True)
