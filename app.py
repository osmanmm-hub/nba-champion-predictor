import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import shap
import pickle
import json
import warnings
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title='NBA Championship Predictor',
    page_icon='🏀',
    layout='wide',
    initial_sidebar_state='collapsed'
)

# ── Mobile-friendly CSS ───────────────────────────────────
st.markdown("""
<style>
    .block-container { padding: 1rem 1rem 1rem 1rem; }
    @media (max-width: 768px) {
        .block-container { padding: 0.5rem; }
        h1 { font-size: 1.4rem !important; }
        h2 { font-size: 1.2rem !important; }
        h3 { font-size: 1rem !important; }
    }

    /* Tab bar background */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #e8eaf0;
        border-radius: 8px;
        padding: 4px 6px;
        gap: 2px;
        flex-wrap: wrap;
    }

    /* Each tab — all same background */
    .stTabs [data-baseweb="tab"] {
        background-color: #e8eaf0;
        border-radius: 6px;
        border: none !important;
        color: #666666;
        font-size: 0.80rem;
        font-weight: 600;
        padding: 7px 12px;
        transition: all 0.15s ease;
        white-space: nowrap;
    }

    /* Hover */
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #d0d4de;
        color: #111111;
    }

    /* Active tab — same bg, bold text + thick underline */
    .stTabs [aria-selected="true"] {
        background-color: #e8eaf0 !important;
        color: #000000 !important;
        font-weight: 900 !important;
        border-bottom: 3px solid #17408B !important;
        border-radius: 0px !important;
    }

    /* Remove default indicator lines */
    .stTabs [data-baseweb="tab-highlight"] { display: none !important; }
    .stTabs [data-baseweb="tab-border"] { display: none !important; }
</style>
""", unsafe_allow_html=True)

# ── Conference mappings ───────────────────────────────────
EASTERN = [
    'Atlanta Hawks','Boston Celtics','Brooklyn Nets','Charlotte Hornets',
    'Chicago Bulls','Cleveland Cavaliers','Detroit Pistons','Indiana Pacers',
    'Miami Heat','Milwaukee Bucks','New York Knicks','Orlando Magic',
    'Philadelphia 76ers','Toronto Raptors','Washington Wizards'
]
WESTERN = [
    'Dallas Mavericks','Denver Nuggets','Golden State Warriors','Houston Rockets',
    'Los Angeles Clippers','Los Angeles Lakers','Memphis Grizzlies','Minnesota Timberwolves',
    'New Orleans Pelicans','Oklahoma City Thunder','Phoenix Suns','Portland Trail Blazers',
    'Sacramento Kings','San Antonio Spurs','Utah Jazz'
]
CHAMPIONS = {
    '1999-00':'Los Angeles Lakers','2000-01':'Los Angeles Lakers',
    '2001-02':'Los Angeles Lakers','2002-03':'San Antonio Spurs',
    '2003-04':'Detroit Pistons',   '2004-05':'San Antonio Spurs',
    '2005-06':'Miami Heat',        '2006-07':'San Antonio Spurs',
    '2007-08':'Boston Celtics',    '2008-09':'Los Angeles Lakers',
    '2009-10':'Los Angeles Lakers','2010-11':'Dallas Mavericks',
    '2011-12':'Miami Heat',        '2012-13':'Miami Heat',
    '2013-14':'San Antonio Spurs', '2014-15':'Golden State Warriors',
    '2015-16':'Cleveland Cavaliers','2016-17':'Golden State Warriors',
    '2017-18':'Golden State Warriors','2018-19':'Toronto Raptors',
    '2019-20':'Los Angeles Lakers','2020-21':'Milwaukee Bucks',
    '2021-22':'Golden State Warriors','2022-23':'Denver Nuggets',
    '2023-24':'Boston Celtics',    '2024-25':'Oklahoma City Thunder',
}

# ── Load models & data ────────────────────────────────────
@st.cache_resource
def load_models():
    with open('model_xgb.pkl','rb') as f: xgb = pickle.load(f)
    with open('model_rf.pkl', 'rb') as f: rf  = pickle.load(f)
    with open('model_lr.pkl', 'rb') as f: lr  = pickle.load(f)
    with open('scaler.pkl',   'rb') as f: sc  = pickle.load(f)
    return xgb, rf, lr, sc

@st.cache_data
def load_data():
    hist    = pd.read_csv('nba_historical.csv')
    current = pd.read_csv('nba_current_2526.csv')
    preds   = pd.read_csv('championship_predictions.csv')
    results = pd.read_csv('model_results.csv', index_col=0)
    with open('feature_list.json') as f: features = json.load(f)
    return hist, current, preds, results, features

xgb_model, rf_model, lr_model, scaler = load_models()
df_hist, df_current, df_pred, model_results, FEATURES = load_data()

def get_conf(t):
    if t in EASTERN: return 'East'
    if t in WESTERN: return 'West'
    return 'Unknown'

df_pred['conference'] = df_pred['team'].apply(get_conf)
df_hist['conference'] = df_hist['TEAM_NAME'].apply(get_conf)
last_updated = datetime.now().strftime('%B %d, %Y at %I:%M %p')

@st.cache_data
def get_test_split():
    X = df_hist[FEATURES]; y = df_hist['champion']
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    return Xtr, Xte, scaler.transform(Xte), ytr, yte

X_train, X_test, X_test_sc, y_train, y_test = get_test_split()

# ── Tabs ─────────────────────────────────────────────────
tab1,tab2,tab3,tab4,tab5,tab6,tab7 = st.tabs([
    'Executive Summary',
    'Descriptive Analytics',
    'Model Performance',
    'Explainability & Prediction',
    'Season Replay',
    'Head-to-Head',
    'Accuracy & Injuries'
])

# ═══════════════════════════════════════════════════════════
# TAB 1 — Executive Summary
# ═══════════════════════════════════════════════════════════
with tab1:
    col_logo, col_title = st.columns([1, 10])
    with col_logo:
        st.image(
            'https://upload.wikimedia.org/wikipedia/en/thumb/0/03/National_Basketball_Association_logo.svg/400px-National_Basketball_Association_logo.svg.png',
            width=24
        )
    with col_title:
        st.title('NBA Championship Predictor — 2025-26')
    st.caption(f'Data last refreshed: {last_updated}')

    # ── Project Summary first ─────────────────────────────
    st.markdown("## About This Project")
    st.markdown("""
    As a longtime basketball fan and a big Kobe Bryant fan, I have always been curious
    about what makes a team win a championship. For this project, I used team statistics
    from the NBA Stats API (nba_api) covering approximately 25 seasons from 2000 to 2025.
    Each row in the dataset represents one NBA team in one season, and the model predicts
    whether that team won the championship that year (1 = champion, 0 = not champion).
    The features include key team statistics such as win percentage, points per game,
    rebounds, assists, turnovers, shooting efficiency, and net points per game — which
    measures how much a team outscores its opponents on average across the season.
    """)
    st.markdown("""
    Predicting an NBA champion is exciting but also very difficult. As fans, we often
    rely on opinions or narratives, but data and statistics help reveal clearer patterns
    behind winning teams. This type of analysis can be useful for fans, analysts, and
    even front offices trying to understand what drives championship success. The problem
    is especially challenging because only 1 out of 30 teams wins the championship each
    year, creating a severe class imbalance in the data. A model that simply predicted
    "no champion" for every team would be right 97% of the time — but completely useless.
    Addressing this imbalance through class weights and choosing the right evaluation
    metrics (AUC-ROC rather than accuracy) was a key part of building a meaningful model.
    """)
    st.markdown("""
    Based on the model's predictions using live 2025-26 season stats, the Detroit Pistons
    currently have the highest championship probability, followed by the Oklahoma City
    Thunder and the San Antonio Spurs. Among the four models tested — Logistic Regression,
    Random Forest, XGBoost, and a Neural Network — XGBoost performed the best and produced
    the most accurate predictions as measured by AUC-ROC. The results consistently show
    that win percentage, net points per game (Plus/Minus), and shooting efficiency are
    the most important factors in predicting a champion. This supports the idea that
    balanced teams that dominate their opponents — not just teams that get hot at the
    right time — are far more likely to win championships.
    """)

    st.markdown("---")

    # ── Key stats ─────────────────────────────────────────
    st.markdown("## Key Stats")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric('Years of Training Data', '25')
    col2.metric('Teams in Current Season', str(len(df_current)))
    col3.metric('Features Used', str(len(FEATURES)))
    col4.metric('Best Model AUC', f"{model_results['AUC-ROC'].max():.3f}")

    st.markdown("---")

    # ── Championship Predictions second ───────────────────
    st.markdown("## 2025-26 Championship Predictions")
    st.warning("Predictions are based on regular season stats only and do not account for injuries, player availability, or playoff matchup dynamics.")
    st.markdown("Current ranking of all 30 teams by predicted championship probability.")

    top3 = df_pred.nlargest(3, 'champ_prob_pct')
    c1, c2, c3 = st.columns(3)
    c1.metric('Favorite',  top3.iloc[0]['team'], f"{top3.iloc[0]['champ_prob_pct']:.1f}%")
    c2.metric('2nd Place', top3.iloc[1]['team'], f"{top3.iloc[1]['champ_prob_pct']:.1f}%")
    c3.metric('3rd Place', top3.iloc[2]['team'], f"{top3.iloc[2]['champ_prob_pct']:.1f}%")

    top10 = df_pred.nlargest(10, 'champ_prob_pct').sort_values('champ_prob_pct')
    fig = px.bar(
        top10, x='champ_prob_pct', y='team', orientation='h',
        color='champ_prob_pct', color_continuous_scale='YlOrRd',
        labels={'champ_prob_pct': 'Championship %', 'team': ''},
        text='champ_prob_pct'
    )
    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig.update_layout(
        coloraxis_showscale=False, height=400,
        margin=dict(l=10, r=60, t=20, b=20),
        yaxis=dict(tickfont=dict(size=12)),
        xaxis_title='Championship Probability (%)'
    )
    st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════
# TAB 2 — Descriptive Analytics
# ═══════════════════════════════════════════════════════════
with tab2:
    st.title('Descriptive Analytics')
    champs     = df_hist[df_hist['champion'] == 1]
    non_champs = df_hist[df_hist['champion'] == 0]

    # 2.1 Target distribution
    st.subheader('Target Variable Distribution')
    fig = px.bar(
        x=['Non-Champion', 'Champion'],
        y=[len(non_champs), len(champs)],
        color=['Non-Champion', 'Champion'],
        color_discrete_map={'Champion': '#FDB927', 'Non-Champion': '#17408B'},
        labels={'x': '', 'y': 'Count'},
        text=[len(non_champs), len(champs)]
    )
    fig.update_traces(textposition='outside')
    fig.update_layout(showlegend=False, height=350, margin=dict(t=20))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("""
    Out of 26 seasons, only 26 team-seasons are labeled as champions (one per year) versus
    roughly 690 non-champion team-seasons. This extreme class imbalance — about 1 champion
    per 30 teams — is the central challenge of this prediction task and was addressed using
    class weights in all models.
    """)

    # 2.2 Win % distribution
    st.subheader('Win Percentage: Champions vs All Teams')
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=non_champs['W_PCT'], name='Non-Champions',
                               marker_color='#17408B', opacity=0.6, nbinsx=20))
    fig.add_trace(go.Histogram(x=champs['W_PCT'], name='Champions',
                               marker_color='#FDB927', opacity=0.9, nbinsx=10))
    fig.update_layout(
        barmode='overlay', height=380,
        xaxis_title='Win Percentage', yaxis_title='Count',
        legend=dict(orientation='h', yanchor='bottom', y=1.02),
        margin=dict(t=30)
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("""
    Champions cluster tightly in the 0.60–0.80 win percentage range while non-champions
    spread broadly from 0.20 to 0.75. No championship team in the last 25 years has finished
    the regular season below a 50% win rate, making win percentage one of the strongest
    single predictors in the dataset.
    """)

    # 2.3 Plus/Minus boxplot
    st.subheader('Net Points Per Game (Plus/Minus)')
    fig = go.Figure()
    fig.add_trace(go.Box(y=non_champs['PLUS_MINUS'], name='Non-Champions',
                         marker_color='#17408B', boxmean=True))
    fig.add_trace(go.Box(y=champs['PLUS_MINUS'], name='Champions',
                         marker_color='#FDB927', boxmean=True))
    fig.add_hline(y=0, line_dash='dash', line_color='red',
                  annotation_text='Break even')
    fig.update_layout(height=400, yaxis_title='Net Points Per Game',
                      margin=dict(t=20))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("""
    Champions have a median Plus/Minus of around +6 to +8 points per game, meaning they
    outscore opponents by a wide margin. Non-champions are centered near zero. This confirms
    that true contenders dominate opponents rather than just winning close games.
    """)

    # 2.4 3PT% trend
    st.subheader('3-Point Shooting % Over Time — Champions Only')
    champ_sorted = champs.sort_values('SEASON')
    fig = px.line(
        champ_sorted, x='SEASON', y='FG3_PCT',
        markers=True, labels={'SEASON': 'Season', 'FG3_PCT': '3PT%'},
        color_discrete_sequence=['#17408B']
    )
    fig.add_trace(go.Scatter(
        x=champ_sorted['SEASON'],
        y=np.poly1d(np.polyfit(range(len(champ_sorted)), champ_sorted['FG3_PCT'], 1))(range(len(champ_sorted))),
        mode='lines', name='Trend', line=dict(dash='dash', color='red')
    ))
    fig.update_layout(height=380, margin=dict(t=20),
                      xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("""
    There is a clear upward trend in 3-point shooting percentage among championship teams
    since around 2014, coinciding with the start of the three-point revolution. Recent
    champions like Golden State (2017-18) and Boston (2023-24) shot notably better from
    three than champions of the early 2000s.
    """)

    # 2.5 Scatter — Win % vs Plus/Minus
    st.subheader('Win % vs Plus/Minus — All Team-Seasons')
    df_scatter = df_hist.copy()
    df_scatter['Label'] = df_scatter['champion'].map({1: 'Champion', 0: 'Non-Champion'})
    fig = px.scatter(
        df_scatter, x='W_PCT', y='PLUS_MINUS',
        color='Label',
        color_discrete_map={'Champion': '#FDB927', 'Non-Champion': '#17408B'},
        hover_data=['TEAM_NAME', 'SEASON'],
        opacity=0.6,
        labels={'W_PCT': 'Win %', 'PLUS_MINUS': 'Plus/Minus'},
    )
    fig.update_layout(height=420, margin=dict(t=20),
                      legend=dict(orientation='h', yanchor='bottom', y=1.02))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("""
    Champions (gold) cluster in the top-right corner — high win percentage AND high
    plus/minus. Hover over any point to see which team and season it represents.
    Non-champions that fall in the top-right are typically the runner-up or a strong
    playoff team that fell short.
    """)

    # 2.6 Correlation heatmap
    st.subheader('Correlation Heatmap')
    corr = df_hist[FEATURES + ['champion']].corr().round(2)
    fig = px.imshow(
        corr, text_auto=True, aspect='auto',
        color_continuous_scale='RdBu_r', zmin=-1, zmax=1,
        labels=dict(color='Correlation')
    )
    fig.update_layout(height=500, margin=dict(t=20))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("""
    Win percentage and Plus/Minus show the strongest positive correlation with the champion
    label. Turnovers are negatively correlated — teams that turn the ball over frequently
    rarely win championships. Wins and Win% are highly correlated with each other as expected.
    """)

    # 2.7 Champion averages
    st.subheader('Average Stats: Champions vs Non-Champions')
    compare = pd.DataFrame({
        'Champions':     champs[FEATURES].mean(),
        'Non-Champions': non_champs[FEATURES].mean(),
    }).round(3)
    compare['Edge'] = (compare['Champions'] - compare['Non-Champions']).round(3)
    st.dataframe(compare, use_container_width=True)
    st.caption("""
    Champions average about 10 more wins, a 12% higher win percentage, and nearly
    7 more net points per game than the average non-champion. The negative edge on
    turnovers confirms champion-caliber teams protect the ball better than the rest.
    """)

# ═══════════════════════════════════════════════════════════
# TAB 3 — Model Performance
# ═══════════════════════════════════════════════════════════
with tab3:
    st.title('Model Performance')
    st.markdown("Five models trained on 25 years of NBA data. All evaluated on a 25% held-out test set.")

    with st.expander("What do these metrics mean?"):
        c1, c2 = st.columns(2)
        with c1:
            st.info("**AUC-ROC** — How well does the model rank strong teams above weak ones? 1.0 = perfect, 0.5 = coin flip.")
            st.info("**Precision** — When the model flags a contender, how often is it actually right?")
        with c2:
            st.info("**Recall** — Of all real contenders, how many did the model catch?")
            st.info("**F1** — The combined grade balancing precision and recall.")

    st.subheader('Model Comparison Table')
    st.dataframe(model_results.round(4).style.highlight_max(axis=0, color='#d4edda'),
                 use_container_width=True)

    # AUC bar chart — interactive
    st.subheader('AUC-ROC by Model')
    fig = px.bar(
        model_results.reset_index(),
        x='Model' if 'Model' in model_results.reset_index().columns else model_results.reset_index().columns[0],
        y='AUC-ROC',
        color='AUC-ROC', color_continuous_scale='Blues',
        text='AUC-ROC'
    )
    fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
    fig.update_layout(coloraxis_showscale=False, height=380,
                      yaxis_range=[0, 1.15], margin=dict(t=20))
    st.plotly_chart(fig, use_container_width=True)

    # ROC curves — interactive
    st.subheader('ROC Curves — All Models')
    st.markdown("Hover over any curve to see the exact true/false positive rate at that threshold.")
    fig = go.Figure()
    model_map = {
        'Logistic Regression': (lr_model,  X_test_sc),
        'Random Forest':       (rf_model,  X_test),
        'XGBoost':             (xgb_model, X_test),
    }
    palette = ['#17408B', '#C9082A', '#FDB927']
    for (name, (model, X_use)), color in zip(model_map.items(), palette):
        try:
            proba = model.predict_proba(X_use)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, proba)
            auc_val = auc(fpr, tpr)
            fig.add_trace(go.Scatter(
                x=fpr, y=tpr, mode='lines', name=f'{name} (AUC={auc_val:.3f})',
                line=dict(width=2.5, color=color),
                hovertemplate='FPR: %{x:.3f}<br>TPR: %{y:.3f}<extra>' + name + '</extra>'
            ))
        except Exception as e:
            st.warning(f'Could not plot {name}: {e}')
    fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', name='Random (AUC=0.5)',
                             line=dict(dash='dash', color='gray', width=1)))
    fig.update_layout(
        height=480, xaxis_title='False Positive Rate', yaxis_title='True Positive Rate',
        legend=dict(orientation='h', yanchor='bottom', y=-0.3),
        margin=dict(t=20)
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("""
    All three models clearly outperform random guessing (dashed line). XGBoost's curve
    sits closest to the top-left corner, meaning it catches more real contenders while
    producing fewer false alarms.
    """)

    # Hyperparameters
    st.subheader('Best Hyperparameters from Cross-Validation')
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Logistic Regression**")
        try:
            p = lr_model.get_params()
            st.json({'C': p.get('C'), 'class_weight': p.get('class_weight'), 'max_iter': p.get('max_iter')})
        except: st.json({'note': 'See notebook'})
    with c2:
        st.markdown("**Random Forest**")
        try:
            p = rf_model.get_params()
            st.json({'n_estimators': p.get('n_estimators'), 'max_depth': p.get('max_depth'), 'min_samples_leaf': p.get('min_samples_leaf')})
        except: st.json({'note': 'See notebook'})
    with c3:
        st.markdown("**XGBoost**")
        try:
            p = xgb_model.get_params()
            st.json({'n_estimators': p.get('n_estimators'), 'max_depth': p.get('max_depth'), 'learning_rate': p.get('learning_rate')})
        except: st.json({'note': 'See notebook'})

    st.markdown("""
    **Which model performed best?** XGBoost achieved the highest AUC-ROC and F1 score.
    Logistic Regression was surprisingly competitive, suggesting win percentage and
    plus/minus are strong linear predictors on their own. The trade-off is interpretability —
    Logistic Regression is easy to explain, while XGBoost requires SHAP analysis.
    """)

# ═══════════════════════════════════════════════════════════
# TAB 4 — Explainability & Interactive Prediction
# ═══════════════════════════════════════════════════════════
with tab4:
    st.title('Explainability & Interactive Prediction')
    subtab1, subtab2 = st.tabs(['SHAP Analysis', 'Interactive Prediction'])

    with subtab1:
        st.subheader('Why Does the Model Make These Predictions?')
        st.markdown("Red = pushes probability up. Blue = pushes it down. Each dot is one team-season.")

        explainer = shap.TreeExplainer(xgb_model)
        X_hist_feat = df_hist[FEATURES]
        shap_values = explainer.shap_values(X_hist_feat)
        sv = shap_values[1] if isinstance(shap_values, list) else shap_values

        # Beeswarm (matplotlib — SHAP doesn't support Plotly natively)
        st.markdown("**Summary Beeswarm Plot**")
        plt.figure(figsize=(9, 5))
        shap.summary_plot(sv, X_hist_feat, feature_names=FEATURES, show=False)
        plt.tight_layout()
        st.pyplot(plt.gcf(), use_container_width=True)
        plt.clf()
        st.caption("Win percentage and Plus/Minus dominate. High values (red) push the model toward predicting a championship. High turnovers (blue on TOV) push it away.")

        # Bar — interactive with Plotly
        st.markdown("**Feature Importance (Mean Absolute SHAP) — Interactive**")
        mean_shap = np.abs(sv).mean(axis=0)
        shap_df = pd.DataFrame({'Feature': FEATURES, 'Mean |SHAP|': mean_shap})
        shap_df = shap_df.sort_values('Mean |SHAP|', ascending=True)
        fig = px.bar(shap_df, x='Mean |SHAP|', y='Feature', orientation='h',
                     color='Mean |SHAP|', color_continuous_scale='Oranges',
                     labels={'Mean |SHAP|': 'Mean |SHAP Value|'})
        fig.update_layout(coloraxis_showscale=False, height=420, margin=dict(t=20))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Win percentage is the most important feature by a clear margin. Plus/Minus and Wins follow closely. Turnovers rank mid-table but are negative predictors.")

        # Waterfall for top team
        st.markdown("**Waterfall — Top 2025-26 Favorite**")
        top_team_name = df_pred.nlargest(1, 'champ_prob_pct').iloc[0]['team']
        top_prob = df_pred.nlargest(1, 'champ_prob_pct').iloc[0]['champ_prob_pct']
        top_row = df_current[df_current['TEAM_NAME'] == top_team_name]
        if not top_row.empty:
            X_top = top_row[[f for f in FEATURES if f in top_row.columns]]
            sv_top = explainer.shap_values(X_top)
            sv_top_use = sv_top[1][0] if isinstance(sv_top, list) else sv_top[0]
            ev = explainer.expected_value
            if isinstance(ev, (list, np.ndarray)): ev = float(ev[1])
            st.markdown(f"**{top_team_name}** — {top_prob:.1f}% championship probability")
            shap.waterfall_plot(shap.Explanation(
                values=sv_top_use, base_values=ev,
                data=X_top.iloc[0].values, feature_names=list(X_top.columns)
            ), show=False)
            st.pyplot(plt.gcf(), use_container_width=True)
            plt.clf()
            st.caption("Each bar shows exactly why this team is ranked #1. Red bars are strengths. Blue bars are weaknesses dragging the probability down slightly.")

    with subtab2:
        st.subheader('Build Your Own Team — Live Prediction')
        st.markdown("Adjust the sliders and the model updates the championship probability instantly.")

        model_choice = st.selectbox('Model:', ['XGBoost (Best)', 'Random Forest', 'Logistic Regression'])
        league_avg = df_hist[FEATURES].mean()

        c1, c2, c3 = st.columns(3)
        with c1:
            w_pct = st.slider('Win %', 0.10, 0.90, float(round(league_avg['W_PCT'], 2)), 0.01,
                              help='0.61 = 50 wins in an 82-game season')
            pts   = st.slider('Points/Game', 90.0, 130.0, float(round(league_avg['PTS'], 1)), 0.5)
            pm    = st.slider('Plus/Minus', -15.0, 15.0, float(round(league_avg['PLUS_MINUS'], 1)), 0.5,
                              help='Positive = outscoring opponents on average')
        with c2:
            fg3   = st.slider('3PT%', 0.28, 0.45, float(round(league_avg['FG3_PCT'], 3)), 0.005)
            ast   = st.slider('Assists/Game', 15.0, 35.0, float(round(league_avg['AST'], 1)), 0.5)
            tov   = st.slider('Turnovers/Game', 8.0, 20.0, float(round(league_avg['TOV'], 1)), 0.5,
                              help='Lower is better')
        with c3:
            reb   = st.slider('Rebounds/Game', 35.0, 55.0, float(round(league_avg['REB'], 1)), 0.5)
            fgpct = st.slider('FG%', 0.40, 0.55, float(round(league_avg['FG_PCT'], 3)), 0.005)

        user_input = league_avg.copy()
        user_input.update({'W_PCT': w_pct, 'PTS': pts, 'PLUS_MINUS': pm,
                           'FG3_PCT': fg3, 'AST': ast, 'TOV': tov,
                           'REB': reb, 'FG_PCT': fgpct,
                           'W': round(w_pct * 82), 'L': 82 - round(w_pct * 82)})
        X_user    = pd.DataFrame([user_input[FEATURES]])
        X_user_sc = scaler.transform(X_user)

        sel_model = xgb_model if 'XGBoost' in model_choice else (rf_model if 'Forest' in model_choice else lr_model)
        X_inp     = X_user_sc if 'Logistic' in model_choice else X_user

        raw_prob = sel_model.predict_proba(X_inp)[0][1]
        curr_X   = df_current[[f for f in FEATURES if f in df_current.columns]]
        curr_inp = scaler.transform(curr_X) if 'Logistic' in model_choice else curr_X
        combined = np.append(sel_model.predict_proba(curr_inp)[:, 1], raw_prob)
        norm_prob = raw_prob / combined.sum() * 100

        st.markdown("---")
        r1, r2, r3 = st.columns(3)
        r1.metric('Championship Probability', f'{norm_prob:.1f}%')
        r2.metric('Raw Score', f'{raw_prob:.4f}')
        r3.metric('Est. Record', f"{int(round(w_pct*82))}–{82-int(round(w_pct*82))}")

        top3_cutoff = df_pred.nlargest(3,'champ_prob_pct')['champ_prob_pct'].min()
        if norm_prob >= top3_cutoff:
            st.success('This team profile ranks in the Top 3 championship favorites!')
        elif norm_prob >= df_pred['champ_prob_pct'].quantile(0.75):
            st.info('This team profile is a legitimate contender.')
        else:
            st.warning('This team profile is not considered a championship favorite.')

        # Gauge chart
        fig = go.Figure(go.Indicator(
            mode='gauge+number',
            value=norm_prob,
            title={'text': 'Championship Probability (%)'},
            gauge={
                'axis': {'range': [0, 30]},
                'bar': {'color': '#FDB927'},
                'steps': [
                    {'range': [0, 5],  'color': '#f8d7da'},
                    {'range': [5, 15], 'color': '#fff3cd'},
                    {'range': [15, 30],'color': '#d4edda'},
                ],
                'threshold': {'line': {'color': 'red', 'width': 3}, 'value': top3_cutoff}
            }
        ))
        fig.update_layout(height=280, margin=dict(t=30, b=10))
        st.plotly_chart(fig, use_container_width=True)

        # SHAP waterfall for custom input
        if 'Logistic' not in model_choice:
            exp = shap.TreeExplainer(sel_model)
            sv_u = exp.shap_values(X_user)
            sv_u_use = sv_u[1][0] if isinstance(sv_u, list) else sv_u[0]
            ev_u = exp.expected_value
            if isinstance(ev_u, (list, np.ndarray)): ev_u = float(ev_u[1])
            shap.waterfall_plot(shap.Explanation(
                values=sv_u_use, base_values=ev_u,
                data=X_user.iloc[0].values, feature_names=FEATURES
            ), show=False)
            st.pyplot(plt.gcf(), use_container_width=True)
            plt.clf()

# ═══════════════════════════════════════════════════════════
# TAB 5 — Season Replay
# ═══════════════════════════════════════════════════════════
with tab5:
    st.title('Season Replay')
    st.markdown("Select any past season to see how the model would have ranked every team.")
    st.warning("In-sample predictions — the model was trained on this data.")

    seasons = sorted(df_hist['SEASON'].unique())
    sel_season = st.select_slider('Season:', options=seasons, value=seasons[-1])
    df_s = df_hist[df_hist['SEASON'] == sel_season].copy()
    actual_champ = CHAMPIONS.get(sel_season, 'Unknown')

    if not df_s.empty:
        X_s = df_s[[f for f in FEATURES if f in df_s.columns]]
        proba_s = xgb_model.predict_proba(X_s)[:, 1]
        df_s['champ_prob'] = (proba_s / proba_s.sum() * 100).round(1)
        df_s['Champion'] = df_s['TEAM_NAME'].apply(lambda x: '🏆 ' + x if x == actual_champ else x)
        df_s = df_s.sort_values('champ_prob', ascending=False).reset_index(drop=True)

        champ_rank = df_s[df_s['TEAM_NAME'] == actual_champ].index[0] + 1
        c1, c2, c3 = st.columns(3)
        c1.metric('Season', sel_season)
        c2.metric('Actual Champion', actual_champ)
        c3.metric('Model Rank', f'#{champ_rank}',
                  delta='Correct!' if champ_rank == 1 else f'Missed — #{champ_rank}',
                  delta_color='normal' if champ_rank == 1 else 'inverse')

        df_s_plot = df_s.sort_values('champ_prob', ascending=True)
        df_s_plot['color'] = df_s_plot['TEAM_NAME'].apply(
            lambda x: '#FDB927' if x == actual_champ else '#17408B')

        fig = px.bar(
            df_s_plot, x='champ_prob', y='Champion', orientation='h',
            text='champ_prob',
            color='color', color_discrete_map='identity',
            labels={'champ_prob': 'Championship Probability (%)', 'Champion': ''},
        )
        fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig.update_layout(showlegend=False, height=600,
                          margin=dict(l=10, r=60, t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(
            df_s[['TEAM_NAME','W','W_PCT','champ_prob']].rename(
                columns={'TEAM_NAME':'Team','W':'Wins','W_PCT':'Win %','champ_prob':'Model Prob (%)'}
            ), use_container_width=True
        )

# ═══════════════════════════════════════════════════════════
# TAB 6 — Head-to-Head
# ═══════════════════════════════════════════════════════════
with tab6:
    st.title('Head-to-Head Comparison')
    all_teams = sorted(df_pred['team'].tolist())
    c1, c2 = st.columns(2)
    with c1: team_a = st.selectbox('Team A:', all_teams, index=0)
    with c2: team_b = st.selectbox('Team B:', all_teams, index=1)

    if team_a == team_b:
        st.warning('Select two different teams.')
    else:
        row_a = df_current[df_current['TEAM_NAME'] == team_a]
        row_b = df_current[df_current['TEAM_NAME'] == team_b]
        prob_a = df_pred[df_pred['team'] == team_a]['champ_prob_pct'].values[0]
        prob_b = df_pred[df_pred['team'] == team_b]['champ_prob_pct'].values[0]

        m1, m2 = st.columns(2)
        m1.metric(team_a, f'{prob_a:.1f}%', f'{prob_a-prob_b:+.1f}%')
        m2.metric(team_b, f'{prob_b:.1f}%', f'{prob_b-prob_a:+.1f}%')

        if not row_a.empty and not row_b.empty:
            key_stats = [s for s in ['W_PCT','PTS','PLUS_MINUS','FG3_PCT','TOV','AST'] if s in row_a.columns]
            vals_a = [float(row_a[s].values[0]) for s in key_stats]
            vals_b = [float(row_b[s].values[0]) for s in key_stats]

            fig = go.Figure()
            fig.add_trace(go.Bar(name=team_a, x=key_stats, y=vals_a,
                                 marker_color='#17408B', text=[f'{v:.3f}' for v in vals_a],
                                 textposition='outside'))
            fig.add_trace(go.Bar(name=team_b, x=key_stats, y=vals_b,
                                 marker_color='#FDB927', text=[f'{v:.3f}' for v in vals_b],
                                 textposition='outside'))
            fig.update_layout(barmode='group', height=420,
                              legend=dict(orientation='h', yanchor='bottom', y=1.02),
                              margin=dict(t=30))
            st.plotly_chart(fig, use_container_width=True)

            # SHAP side by side
            st.subheader('What drives each team\'s probability?')
            exp = shap.TreeExplainer(xgb_model)
            X_a = row_a[[f for f in FEATURES if f in row_a.columns]]
            X_b = row_b[[f for f in FEATURES if f in row_b.columns]]
            sv_a = exp.shap_values(X_a); sv_b = exp.shap_values(X_b)
            sv_a = sv_a[1] if isinstance(sv_a, list) else sv_a
            sv_b = sv_b[1] if isinstance(sv_b, list) else sv_b

            feat_names = list(X_a.columns)
            shap_a = pd.DataFrame({'Feature': feat_names, 'SHAP': sv_a[0]})
            shap_b = pd.DataFrame({'Feature': feat_names, 'SHAP': sv_b[0]})

            fig = make_subplots(rows=1, cols=2,
                                subplot_titles=[f'{team_a} ({prob_a:.1f}%)', f'{team_b} ({prob_b:.1f}%)'])
            for i, (sdf, color_pos, color_neg) in enumerate([
                (shap_a, '#FDB927', '#17408B'),
                (shap_b, '#FDB927', '#17408B')
            ]):
                sdf_s = sdf.sort_values('SHAP')
                colors = ['#C9082A' if v > 0 else '#17408B' for v in sdf_s['SHAP']]
                fig.add_trace(go.Bar(
                    x=sdf_s['SHAP'], y=sdf_s['Feature'], orientation='h',
                    marker_color=colors, showlegend=False
                ), row=1, col=i+1)
            fig.update_layout(height=420, margin=dict(t=40))
            st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════
# TAB 7 — Accuracy & Injuries
# ═══════════════════════════════════════════════════════════
with tab7:
    acc_tab, inj_tab = st.tabs(['Historical Accuracy', 'Injury Tracker'])

    with acc_tab:
        st.title('Historical Accuracy')
        st.markdown("Did the model correctly rank the eventual champion as #1?")
        st.warning("In-sample results — the model was trained on this data.")

        accuracy_rows = []
        for season, champ in CHAMPIONS.items():
            df_s2 = df_hist[df_hist['SEASON'] == season].copy()
            if df_s2.empty: continue
            X_s2 = df_s2[[f for f in FEATURES if f in df_s2.columns]]
            proba2 = xgb_model.predict_proba(X_s2)[:, 1]
            df_s2['prob'] = proba2
            df_s2_sorted = df_s2.sort_values('prob', ascending=False).reset_index(drop=True)
            champ_rows = df_s2_sorted[df_s2_sorted['TEAM_NAME'] == champ]
            if champ_rows.empty: continue
            rank = int(champ_rows.index[0]) + 1
            top_pick = df_s2_sorted.iloc[0]['TEAM_NAME']
            wpc = df_s2[df_s2['TEAM_NAME'] == champ]['W_PCT'].values
            accuracy_rows.append({
                'Season': season,
                'Actual Champion': champ,
                "Win %": round(float(wpc[0]), 3) if len(wpc) > 0 else None,
                "Model's #1 Pick": top_pick,
                'Champion Rank': rank,
                'Correct': rank == 1
            })

        acc_df = pd.DataFrame(accuracy_rows)
        n_correct = acc_df['Correct'].sum()
        n_total   = len(acc_df)

        c1, c2, c3 = st.columns(3)
        c1.metric('Seasons Evaluated', str(n_total))
        c2.metric('Champion Ranked #1', f'{n_correct} / {n_total}')
        c3.metric('Accuracy', f'{n_correct/n_total*100:.0f}%')

        # Interactive accuracy chart
        acc_df['Result'] = acc_df['Correct'].map({True: 'Correct', False: 'Missed'})
        fig = px.bar(
            acc_df, x='Season', y='Champion Rank',
            color='Result',
            color_discrete_map={'Correct': '#28a745', 'Missed': '#dc3545'},
            hover_data=['Actual Champion', "Model's #1 Pick", 'Win %'],
            labels={'Champion Rank': 'Rank Given to Champion'},
            text='Champion Rank'
        )
        fig.add_hline(y=1, line_dash='dash', line_color='gold', line_width=2,
                      annotation_text='Perfect = Rank 1')
        fig.update_traces(textposition='outside')
        fig.update_layout(height=420, xaxis_tickangle=-45, margin=dict(t=20))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Green bars = seasons the model correctly ranked the champion #1. Hover over any bar to see details. Shorter bars are better.")

        st.dataframe(
            acc_df[['Season','Actual Champion','Win %',"Model's #1 Pick",'Champion Rank','Result']],
            use_container_width=True
        )

    with inj_tab:
        st.title('Injury Tracker')
        st.markdown("The model does not account for injuries. Check current player availability before interpreting predictions.")
        st.warning("A team ranked highly here may have key players currently out. Always cross-reference with injury reports.")

        try:
            import urllib.request, json as json_lib
            url = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries"
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=8) as resp:
                data = json_lib.loads(resp.read().decode())

            rows = []
            for team_entry in data.get('injuries', []):
                tn = team_entry.get('team', {}).get('displayName', 'Unknown')
                conf = get_conf(tn)
                for player in team_entry.get('injuries', []):
                    athlete = player.get('athlete', {})
                    rows.append({
                        'Team': tn,
                        'Conference': conf,
                        'Player': athlete.get('displayName', 'Unknown'),
                        'Position': athlete.get('position', {}).get('abbreviation', 'N/A') if isinstance(athlete.get('position'), dict) else 'N/A',
                        'Status': player.get('status', 'Unknown'),
                        'Injury': player.get('shortComment', 'N/A')[:60],
                    })

            if rows:
                inj_df = pd.DataFrame(rows)
                teams_list = ['All Teams'] + sorted(inj_df['Team'].unique().tolist())
                sel_team = st.selectbox('Filter by team:', teams_list)
                if sel_team != 'All Teams':
                    inj_df = inj_df[inj_df['Team'] == sel_team]

                # Conference breakdown
                if sel_team == 'All Teams':
                    conf_counts = inj_df['Conference'].value_counts().reset_index()
                    conf_counts.columns = ['Conference', 'Count']
                    c1, c2 = st.columns(2)
                    with c1:
                        fig = px.bar(conf_counts, x='Conference', y='Count',
                                     color='Conference',
                                     color_discrete_map={'East':'#17408B','West':'#C9082A','Unknown':'gray'},
                                     text='Count', title='Injuries by Conference')
                        fig.update_traces(textposition='outside')
                        fig.update_layout(showlegend=False, height=280, margin=dict(t=40))
                        st.plotly_chart(fig, use_container_width=True)
                    with c2:
                        status_counts = inj_df['Status'].value_counts().reset_index()
                        status_counts.columns = ['Status', 'Count']
                        fig = px.bar(status_counts, x='Status', y='Count',
                                     color='Status', text='Count', title='Injuries by Status')
                        fig.update_traces(textposition='outside')
                        fig.update_layout(showlegend=False, height=280, margin=dict(t=40))
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    status_counts = inj_df['Status'].value_counts().reset_index()
                    status_counts.columns = ['Status', 'Count']
                    fig = px.bar(status_counts, x='Status', y='Count',
                                 color='Status', text='Count')
                    fig.update_traces(textposition='outside')
                    fig.update_layout(showlegend=False, height=280, margin=dict(t=20))
                    st.plotly_chart(fig, use_container_width=True)

                def highlight_status(row):
                    if 'Out' in str(row['Status']): return ['background-color: #f8d7da'] * len(row)
                    if 'Doubtful' in str(row['Status']): return ['background-color: #fff3cd'] * len(row)
                    return [''] * len(row)

                st.dataframe(
                    inj_df[['Team','Conference','Player','Position','Status','Injury']].style.apply(highlight_status, axis=1),
                    use_container_width=True
                )
                st.caption(f"Red = Out, Yellow = Doubtful. Showing {len(inj_df)} injuries across {inj_df['Team'].nunique()} teams.")
            else:
                st.info('No injury data returned from ESPN right now.')

        except Exception:
            st.info("Live injury data unavailable. Check these sources directly:")
            st.markdown("- [ESPN NBA Injuries](https://www.espn.com/nba/injuries)")
            st.markdown("- [NBA.com Injury Report](https://www.nba.com/injuries)")
            st.markdown("- [CBS Sports](https://www.cbssports.com/nba/injuries/)")
