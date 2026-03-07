import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import shap
import pickle
import json
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title='NBA Championship Predictor', layout='wide')

@st.cache_resource
def load_models():
    with open('model_xgb.pkl', 'rb') as f: xgb_model = pickle.load(f)
    with open('model_rf.pkl',  'rb') as f: rf_model  = pickle.load(f)
    with open('scaler.pkl',    'rb') as f: scaler    = pickle.load(f)
    return xgb_model, rf_model, scaler

@st.cache_data
def load_data():
    hist    = pd.read_csv('nba_historical.csv')
    current = pd.read_csv('nba_current_2526.csv')
    preds   = pd.read_csv('championship_predictions.csv')
    results = pd.read_csv('model_results.csv', index_col=0)
    with open('feature_list.json') as f: features = json.load(f)
    return hist, current, preds, results, features

xgb_model, rf_model, scaler = load_models()
df_hist, df_current, df_pred, model_results, FEATURES = load_data()

tab1, tab2, tab3, tab4 = st.tabs([
    'Overview',
    'Historical Data',
    'Model Results',
    '2025-26 Predictions'
])

# ── TAB 1: Overview ──────────────────────────────────────
with tab1:
    st.title('NBA Championship Predictor — 2025-26 Season')

    st.markdown("""
    This app uses machine learning to estimate which NBA team is most likely
    to win the 2025-26 championship based on how they are currently performing.

    **How it works:**
    We collected team stats from every NBA season between 2000 and 2025 and labeled
    which team won the championship each year. We then trained four different models
    to learn what a championship-caliber team looks like. Those models are now applied
    to live 2025-26 stats to generate championship probabilities for all 30 teams.

    **What the models look at:**
    Win percentage, scoring, rebounding, assists, turnovers, shooting efficiency,
    and net points per game (how much a team outscores opponents on average).

    **Key insight:**
    Win percentage and net points per game are by far the strongest predictors.
    Teams that consistently dominate their opponents — not just win close games —
    are the ones that tend to go deep in the playoffs.
    """)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric('Years of Training Data', '25')
    col2.metric('Teams in Current Season', str(len(df_current)))
    col3.metric('Stats Used as Features', str(len(FEATURES)))
    col4.metric('Best Model AUC Score', f"{model_results['AUC-ROC'].max():.3f}")

    st.markdown("""
    ---
    **What is AUC?**
    AUC (Area Under the Curve) measures how well a model can distinguish
    championship teams from non-championship teams. A score of 1.0 is perfect,
    0.5 is no better than random guessing. Our best model scores above 0.85,
    meaning it is quite good at ranking true contenders above average teams.
    """)

# ── TAB 2: Historical Data ───────────────────────────────
with tab2:
    st.title('What Do Championship Teams Look Like?')

    st.markdown("""
    The table below compares the average stats of NBA champions versus all other teams
    from 2000 to 2025. The "Edge" column shows how much better champions were on average.
    """)

    champs     = df_hist[df_hist['champion'] == 1]
    non_champs = df_hist[df_hist['champion'] == 0]

    compare = pd.DataFrame({
        'Champions':     champs[FEATURES].mean(),
        'Non-Champions': non_champs[FEATURES].mean(),
    }).round(3)
    compare['Edge'] = (compare['Champions'] - compare['Non-Champions']).round(3)
    st.dataframe(compare)

    st.subheader('Win Percentage Distribution')
    st.markdown('Champions almost always finish the regular season with a high win percentage. Very few title winners had a win rate below 60%.')

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(non_champs['W_PCT'], bins=20, alpha=0.6, color='steelblue', label='All Other Teams')
    ax.hist(champs['W_PCT'], bins=8, alpha=0.9, color='gold', edgecolor='black', label='Champions')
    ax.set_title('Regular Season Win % — Champions vs All Teams (2000-2025)')
    ax.set_xlabel('Win Percentage')
    ax.set_ylabel('Number of Teams')
    ax.legend()
    st.pyplot(fig)

    st.subheader('Every NBA Champion Since 2000')
    champ_display = champs[['SEASON','TEAM_NAME','W','L','W_PCT','PTS','PLUS_MINUS','FG3_PCT']].sort_values('SEASON', ascending=False)
    champ_display.columns = ['Season','Team','Wins','Losses','Win %','Points Per Game','Net Points','3PT %']
    st.dataframe(champ_display.reset_index(drop=True))

# ── TAB 3: Model Results ─────────────────────────────────
with tab3:
    st.title('Model Comparison')

    st.markdown("""
    We trained four different models and compared their performance.
    Each model was tested on data it had never seen before.

    **How to read this table:**
    - **AUC-ROC** — how well the model ranks contenders above non-contenders (higher is better)
    - **F1** — balance between catching real contenders and avoiding false alarms
    - **Precision** — of teams the model flagged as contenders, how many actually were?
    - **Recall** — of actual champions, how many did the model correctly identify?
    """)

    st.dataframe(model_results.round(4))

    st.subheader('AUC-ROC by Model')
    fig, ax = plt.subplots(figsize=(9, 4))
    colors = ['steelblue', 'darkorange', 'gold', 'gray']
    bars = ax.bar(model_results.index, model_results['AUC-ROC'],
                  color=colors, edgecolor='black')
    ax.set_ylim(0, 1.1)
    ax.set_title('AUC-ROC Score by Model')
    ax.set_ylabel('AUC-ROC')
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.01,
                f'{bar.get_height():.3f}', ha='center', fontsize=11)
    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("""
    **Why XGBoost tends to win:**
    XGBoost is a boosting algorithm — it builds many small decision trees in sequence,
    each one correcting the mistakes of the previous one. This makes it very good at
    finding subtle patterns in tabular data like team stats.
    """)

# ── TAB 4: 2025-26 Predictions ───────────────────────────
with tab4:
    st.title('2025-26 NBA Championship Predictions')

    games_played = int(df_current['GP'].mean()) if 'GP' in df_current.columns else 'unknown'
    st.markdown(f"""
    These predictions are based on current 2025-26 regular season stats
    (approximately **{games_played} games played** per team so far this season).
    As the season progresses and more games are played, predictions will become more reliable.
    """)

    st.subheader('Championship Probability — All 30 Teams')
    df_sorted = df_pred.sort_values('champ_prob_pct', ascending=True)

    fig, ax = plt.subplots(figsize=(10, 12))
    bar_colors = [
        'gold'      if i >= len(df_sorted) - 3 else
        'steelblue' if i >= len(df_sorted) - 8 else
        'lightgray'
        for i in range(len(df_sorted))
    ]
    ax.barh(df_sorted['team'], df_sorted['champ_prob_pct'],
            color=bar_colors, edgecolor='black', linewidth=0.5)
    ax.set_xlabel('Championship Probability (%)')
    ax.set_title('2025-26 NBA Championship Probability', fontsize=14)
    ax.legend(handles=[
        mpatches.Patch(color='gold',      label='Top 3 favorites'),
        mpatches.Patch(color='steelblue', label='Contenders (top 8)'),
        mpatches.Patch(color='lightgray', label='Long shots'),
    ])
    plt.tight_layout()
    st.pyplot(fig)

    st.subheader('Full Rankings')
    display_df = df_pred[['team','W','L','W_PCT','champ_prob_pct']].copy()
    display_df.columns = ['Team','Wins','Losses','Win %','Championship Probability (%)']
    st.dataframe(display_df)

    st.subheader('Why is a specific team ranked where they are?')
    st.markdown("""
    Select any team below to see a breakdown of which stats are helping or hurting
    their championship chances. Bars pointing right mean that stat is boosting
    their probability. Bars pointing left mean it is dragging it down.
    """)

    selected_team = st.selectbox('Select a team:', df_pred['team'].tolist())
    team_row = df_current[df_current['TEAM_NAME'] == selected_team]

    if not team_row.empty:
        X_team = team_row[[f for f in FEATURES if f in team_row.columns]]

        explainer = shap.TreeExplainer(xgb_model)
        sv = explainer.shap_values(X_team)
        sv_use = sv[1] if isinstance(sv, list) else sv
        ev_use = explainer.expected_value
        if isinstance(ev_use, (list, np.ndarray)):
            ev_use = float(ev_use[1])

        prob = df_pred[df_pred['team'] == selected_team]['champ_prob_pct'].values[0]
        st.metric('Championship Probability', f'{prob:.1f}%')

        shap.waterfall_plot(
            shap.Explanation(
                values=sv_use[0],
                base_values=ev_use,
                data=X_team.iloc[0].values,
                feature_names=list(X_team.columns)
            ), show=False
        )
        st.pyplot(plt.gcf())
        plt.clf()

        st.subheader(f'{selected_team} — Current Season Stats')
        stats_show = team_row[['TEAM_NAME','W','L','W_PCT'] + FEATURES].T
        stats_show.columns = ['Value']
        st.dataframe(stats_show)
