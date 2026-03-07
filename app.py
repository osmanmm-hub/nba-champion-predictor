import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import shap
import pickle
import json
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')

st.set_page_config(page_title='NBA Championship Predictor', layout='wide')

# ── Conference mappings ───────────────────────────────────
EASTERN = [
    'Atlanta Hawks', 'Boston Celtics', 'Brooklyn Nets', 'Charlotte Hornets',
    'Chicago Bulls', 'Cleveland Cavaliers', 'Detroit Pistons', 'Indiana Pacers',
    'Miami Heat', 'Milwaukee Bucks', 'New York Knicks', 'Orlando Magic',
    'Philadelphia 76ers', 'Toronto Raptors', 'Washington Wizards'
]
WESTERN = [
    'Dallas Mavericks', 'Denver Nuggets', 'Golden State Warriors', 'Houston Rockets',
    'Los Angeles Clippers', 'Los Angeles Lakers', 'Memphis Grizzlies', 'Minnesota Timberwolves',
    'New Orleans Pelicans', 'Oklahoma City Thunder', 'Phoenix Suns', 'Portland Trail Blazers',
    'Sacramento Kings', 'San Antonio Spurs', 'Utah Jazz'
]

CHAMPIONS = {
    '1999-00': 'Los Angeles Lakers', '2000-01': 'Los Angeles Lakers',
    '2001-02': 'Los Angeles Lakers', '2002-03': 'San Antonio Spurs',
    '2003-04': 'Detroit Pistons',    '2004-05': 'San Antonio Spurs',
    '2005-06': 'Miami Heat',         '2006-07': 'San Antonio Spurs',
    '2007-08': 'Boston Celtics',     '2008-09': 'Los Angeles Lakers',
    '2009-10': 'Los Angeles Lakers', '2010-11': 'Dallas Mavericks',
    '2011-12': 'Miami Heat',         '2012-13': 'Miami Heat',
    '2013-14': 'San Antonio Spurs',  '2014-15': 'Golden State Warriors',
    '2015-16': 'Cleveland Cavaliers','2016-17': 'Golden State Warriors',
    '2017-18': 'Golden State Warriors','2018-19': 'Toronto Raptors',
    '2019-20': 'Los Angeles Lakers', '2020-21': 'Milwaukee Bucks',
    '2021-22': 'Golden State Warriors','2022-23': 'Denver Nuggets',
    '2023-24': 'Boston Celtics',     '2024-25': 'Oklahoma City Thunder',
}

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

def get_conference(team):
    if team in EASTERN: return 'East'
    if team in WESTERN: return 'West'
    return 'Unknown'

df_pred['conference'] = df_pred['team'].apply(get_conference)
df_hist['conference'] = df_hist['TEAM_NAME'].apply(get_conference)

last_updated = datetime.now().strftime('%B %d, %Y at %I:%M %p')

# ── Tabs ─────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    'Overview',
    'Historical Data',
    'Model Results',
    '2025-26 Predictions',
    'Season Replay',
    'Head-to-Head',
])

# ── TAB 1: Overview ──────────────────────────────────────
with tab1:
    st.title('NBA Championship Predictor — 2025-26 Season')
    st.caption(f'Data last refreshed: {last_updated}')

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

    st.warning(
        "These predictions are based on regular season stats only and do not account "
        "for injuries, player availability, or playoff matchup dynamics. Use them as "
        "a data-driven starting point, not a guarantee."
    )

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

    fig, ax = plt.subplots(figsize=(9, 4))
    colors = ['steelblue', 'darkorange', 'gold', 'gray']
    bars = ax.bar(model_results.index, model_results['AUC-ROC'], color=colors, edgecolor='black')
    ax.set_ylim(0, 1.1)
    ax.set_title('AUC-ROC Score by Model')
    ax.set_ylabel('AUC-ROC')
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
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
    st.caption(f'Data last refreshed: {last_updated}')

    st.info(
        f'About {games_played} games played per team so far this season. '
        'Predictions become more reliable as more games are played. '
        'Only 8 teams per conference qualify for the playoffs — historically, '
        'teams below a 55% win rate rarely win the championship.'
    )

    st.warning(
        "Predictions are based on regular season stats only and do not account for "
        "injuries, player availability, or playoff matchup dynamics."
    )

    # Conference filter
    conf_filter = st.radio(
        'Show teams from:',
        options=['All Teams', 'Eastern Conference', 'Western Conference'],
        horizontal=True
    )

    if conf_filter == 'Eastern Conference':
        df_filtered = df_pred[df_pred['conference'] == 'East'].copy()
    elif conf_filter == 'Western Conference':
        df_filtered = df_pred[df_pred['conference'] == 'West'].copy()
    else:
        df_filtered = df_pred.copy()

    df_sorted = df_filtered.sort_values('champ_prob_pct', ascending=True)

    # Mobile-friendly: use vertical bar chart on small screens, horizontal on large
    use_mobile = st.checkbox('Mobile-friendly chart (vertical bars)', value=False)

    if use_mobile:
        fig, ax = plt.subplots(figsize=(max(8, len(df_sorted) * 0.5), 5))
        bar_colors = [
            'gold'      if i >= len(df_sorted) - 3 else
            'steelblue' if i >= len(df_sorted) - 8 else
            'lightgray'
            for i in range(len(df_sorted))
        ]
        ax.bar(df_sorted['team'], df_sorted['champ_prob_pct'],
               color=bar_colors[::-1], edgecolor='black', linewidth=0.5)
        ax.set_ylabel('Championship Probability (%)')
        plt.xticks(rotation=45, ha='right', fontsize=8)
    else:
        fig, ax = plt.subplots(figsize=(10, max(6, len(df_sorted) * 0.4)))
        bar_colors = [
            'gold'      if i >= len(df_sorted) - 3 else
            'steelblue' if i >= len(df_sorted) - 8 else
            'lightgray'
            for i in range(len(df_sorted))
        ]
        ax.barh(df_sorted['team'], df_sorted['champ_prob_pct'],
                color=bar_colors, edgecolor='black', linewidth=0.5)
        ax.set_xlabel('Championship Probability (%)')

    ax.set_title(f'2025-26 NBA Championship Probability — {conf_filter}', fontsize=13)
    ax.legend(handles=[
        mpatches.Patch(color='gold',      label='Top 3 favorites'),
        mpatches.Patch(color='steelblue', label='Contenders (top 8)'),
        mpatches.Patch(color='lightgray', label='Long shots'),
    ])
    plt.tight_layout()
    st.pyplot(fig)

    st.subheader('Full Rankings Table')
    display_df = df_filtered[['team','conference','W','L','W_PCT','champ_prob_pct']].copy()
    display_df.columns = ['Team','Conference','Wins','Losses','Win %','Championship Probability (%)']
    st.dataframe(display_df.sort_values('Championship Probability (%)', ascending=False).reset_index(drop=True))

    st.markdown('---')
    st.subheader('Why is a specific team ranked where they are?')
    st.markdown("""
    Type a team name below to filter, then select it to see which stats are
    helping or hurting their championship chances. Bars pointing right boost
    their probability. Bars pointing left drag it down.
    """)

    search = st.text_input('Search for a team:', placeholder='e.g. Oklahoma, Boston, Lakers')
    all_teams = df_pred['team'].sort_values().tolist()
    filtered_teams = [t for t in all_teams if search.lower() in t.lower()] if search else all_teams

    if filtered_teams:
        selected_team = st.selectbox('Select a team:', filtered_teams)
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
            conf = df_pred[df_pred['team'] == selected_team]['conference'].values[0]
            tw = int(df_current[df_current['TEAM_NAME'] == selected_team]['W'].values[0])
            tl = int(df_current[df_current['TEAM_NAME'] == selected_team]['L'].values[0])

            col1, col2, col3 = st.columns(3)
            col1.metric('Championship Probability', f'{prob:.1f}%')
            col2.metric('Conference', conf)
            col3.metric('Record', f'{tw}–{tl}')

            shap.waterfall_plot(shap.Explanation(
                values=sv_use[0], base_values=ev_use,
                data=X_team.iloc[0].values, feature_names=list(X_team.columns)
            ), show=False)
            st.pyplot(plt.gcf())
            plt.clf()

            st.subheader(f'{selected_team} — Current Season Stats')
            stats_show = team_row[['TEAM_NAME','W','L','W_PCT'] + FEATURES].T
            stats_show.columns = ['Value']
            st.dataframe(stats_show)
    else:
        st.warning('No teams found matching your search.')

# ── TAB 5: Season Replay ─────────────────────────────────
with tab5:
    st.title('Season Replay — What Would the Model Have Predicted?')
    st.markdown("""
    Select any past season to see what championship probability the model would have
    assigned to every team, and check whether the actual champion was ranked #1.
    This shows how well the model performs on seasons it was trained on.
    """)

    st.warning(
        "Note: Because the model was trained on all historical seasons, these are "
        "in-sample predictions — the model has already seen this data. "
        "Think of this as a sanity check, not a true backtest."
    )

    seasons = sorted(df_hist['SEASON'].unique())
    selected_season = st.select_slider('Select a season:', options=seasons, value=seasons[-1])

    df_season = df_hist[df_hist['SEASON'] == selected_season].copy()
    actual_champion = CHAMPIONS.get(selected_season, 'Unknown')

    if not df_season.empty:
        X_season = df_season[[f for f in FEATURES if f in df_season.columns]]
        season_proba = xgb_model.predict_proba(X_season)[:, 1]
        season_proba_norm = season_proba / season_proba.sum() * 100

        df_season_pred = pd.DataFrame({
            'team': df_season['TEAM_NAME'].values,
            'W': df_season['W'].values,
            'W_PCT': df_season['W_PCT'].values,
            'champ_prob_pct': season_proba_norm.round(1),
            'actual_champion': (df_season['TEAM_NAME'] == actual_champion).astype(int).values
        }).sort_values('champ_prob_pct', ascending=False).reset_index(drop=True)

        predicted_winner = df_season_pred.iloc[0]['team']
        predicted_rank_of_champ = df_season_pred[df_season_pred['team'] == actual_champion].index[0] + 1

        col1, col2, col3 = st.columns(3)
        col1.metric('Season', selected_season)
        col2.metric('Actual Champion', actual_champion)
        col3.metric(
            'Model Ranked Champion',
            f'#{predicted_rank_of_champ}',
            delta='Correct!' if predicted_rank_of_champ == 1 else f'Predicted #{predicted_rank_of_champ}',
            delta_color='normal' if predicted_rank_of_champ == 1 else 'inverse'
        )

        # Chart
        df_s_sorted = df_season_pred.sort_values('champ_prob_pct', ascending=True)
        bar_colors_s = ['gold' if t == actual_champion else
                        'steelblue' if p >= df_s_sorted['champ_prob_pct'].quantile(0.75) else
                        'lightgray'
                        for t, p in zip(df_s_sorted['team'], df_s_sorted['champ_prob_pct'])]

        fig, ax = plt.subplots(figsize=(10, max(6, len(df_s_sorted) * 0.38)))
        ax.barh(df_s_sorted['team'], df_s_sorted['champ_prob_pct'],
                color=bar_colors_s, edgecolor='black', linewidth=0.5)
        ax.set_xlabel('Model-Assigned Championship Probability (%)')
        ax.set_title(f'{selected_season} Season — Model Predictions vs Reality', fontsize=13)
        ax.legend(handles=[
            mpatches.Patch(color='gold',      label=f'Actual Champion ({actual_champion})'),
            mpatches.Patch(color='steelblue', label='Top quartile'),
            mpatches.Patch(color='lightgray', label='Rest of league'),
        ])
        plt.tight_layout()
        st.pyplot(fig)

        st.subheader('Full Rankings Table')
        st.dataframe(df_season_pred[['team','W','W_PCT','champ_prob_pct','actual_champion']].rename(columns={
            'team': 'Team', 'W': 'Wins', 'W_PCT': 'Win %',
            'champ_prob_pct': 'Model Probability (%)', 'actual_champion': 'Won Title'
        }))

# ── TAB 6: Head-to-Head ──────────────────────────────────
with tab6:
    st.title('Head-to-Head Team Comparison')
    st.markdown("""
    Select any two current teams to compare their stats and championship
    probabilities side by side.
    """)

    all_teams_sorted = sorted(df_pred['team'].tolist())

    col1, col2 = st.columns(2)
    with col1:
        team_a = st.selectbox('Team A:', all_teams_sorted, index=0)
    with col2:
        team_b = st.selectbox('Team B:', all_teams_sorted, index=1)

    if team_a == team_b:
        st.warning('Please select two different teams.')
    else:
        row_a = df_current[df_current['TEAM_NAME'] == team_a]
        row_b = df_current[df_current['TEAM_NAME'] == team_b]
        prob_a = df_pred[df_pred['team'] == team_a]['champ_prob_pct'].values[0]
        prob_b = df_pred[df_pred['team'] == team_b]['champ_prob_pct'].values[0]
        conf_a = df_pred[df_pred['team'] == team_a]['conference'].values[0]
        conf_b = df_pred[df_pred['team'] == team_b]['conference'].values[0]

        # Summary metrics
        st.subheader('Championship Probability')
        m1, m2 = st.columns(2)
        m1.metric(team_a, f'{prob_a:.1f}%',
                  delta=f'{prob_a - prob_b:+.1f}% vs {team_b}')
        m2.metric(team_b, f'{prob_b:.1f}%',
                  delta=f'{prob_b - prob_a:+.1f}% vs {team_a}')

        # Side-by-side stats table
        st.subheader('Stats Comparison')
        stats_cols = ['TEAM_NAME', 'W', 'L', 'W_PCT'] + FEATURES

        if not row_a.empty and not row_b.empty:
            stats_a = row_a[stats_cols].iloc[0]
            stats_b = row_b[stats_cols].iloc[0]

            compare_df = pd.DataFrame({
                team_a: stats_a,
                team_b: stats_b,
            }).drop('TEAM_NAME')

            compare_df[team_a] = pd.to_numeric(compare_df[team_a], errors='coerce')
            compare_df[team_b] = pd.to_numeric(compare_df[team_b], errors='coerce')
            compare_df[f'Edge ({team_a})'] = (compare_df[team_a] - compare_df[team_b]).round(3)
            st.dataframe(compare_df.round(3))

            # Visual bar comparison for key stats
            st.subheader('Key Stats — Side by Side')
            key_stats = ['W_PCT', 'PTS', 'PLUS_MINUS', 'FG3_PCT', 'TOV', 'AST']
            key_stats = [s for s in key_stats if s in compare_df.index]

            vals_a = [float(compare_df.loc[s, team_a]) for s in key_stats]
            vals_b = [float(compare_df.loc[s, team_b]) for s in key_stats]

            x = np.arange(len(key_stats))
            width = 0.35

            fig, ax = plt.subplots(figsize=(11, 5))
            bars_a = ax.bar(x - width/2, vals_a, width, label=team_a,
                            color='steelblue', edgecolor='black')
            bars_b = ax.bar(x + width/2, vals_b, width, label=team_b,
                            color='gold', edgecolor='black')
            ax.set_xticks(x)
            ax.set_xticklabels(key_stats, fontsize=11)
            ax.set_title(f'{team_a} vs {team_b} — Key Stats', fontsize=13)
            ax.legend()
            plt.tight_layout()
            st.pyplot(fig)

            # SHAP comparison
            st.subheader('What is driving each team\'s championship probability?')
            st.markdown('Red bars boost the probability. Blue bars drag it down.')

            X_a = row_a[[f for f in FEATURES if f in row_a.columns]]
            X_b = row_b[[f for f in FEATURES if f in row_b.columns]]

            explainer = shap.TreeExplainer(xgb_model)

            sv_a = explainer.shap_values(X_a)
            sv_b = explainer.shap_values(X_b)
            sv_a = sv_a[1] if isinstance(sv_a, list) else sv_a
            sv_b = sv_b[1] if isinstance(sv_b, list) else sv_b
            ev = explainer.expected_value
            if isinstance(ev, (list, np.ndarray)): ev = float(ev[1])

            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            for ax_i, (sv_use, X_use, team_name, prob) in enumerate([
                (sv_a, X_a, team_a, prob_a),
                (sv_b, X_b, team_b, prob_b),
            ]):
                sorted_idx = np.argsort(np.abs(sv_use[0]))[::-1]
                feature_names = list(X_use.columns)
                colors = ['tomato' if v > 0 else 'steelblue' for v in sv_use[0][sorted_idx]]
                axes[ax_i].barh(
                    [feature_names[i] for i in sorted_idx][::-1],
                    sv_use[0][sorted_idx][::-1],
                    color=colors[::-1]
                )
                axes[ax_i].axvline(0, color='black', lw=0.8)
                axes[ax_i].set_title(f'{team_name}\n{prob:.1f}% championship probability', fontsize=11)

            plt.suptitle('SHAP Feature Contributions — Head-to-Head', fontsize=13)
            plt.tight_layout()
            st.pyplot(fig)
