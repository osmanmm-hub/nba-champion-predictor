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

st.set_page_config(page_title='🏀 NBA Champion Predictor', page_icon='🏀', layout='wide')

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

tab1, tab2, tab3, tab4 = st.tabs(['📋 Executive Summary','📊 Historical Analysis','🤖 Model Performance','🏆 Live 2025-26 Predictions'])

with tab1:
    st.title('🏀 NBA Championship Predictor — 2025-26 Season')
    st.markdown('## Executive Summary')
    st.markdown('This app predicts the **probability of each NBA team winning the 2025-26 Championship** using live team stats from the NBA Stats API.')
    st.markdown('### Methodology')
    st.markdown('- Training data: Historical team stats from 2000-2024 (25 seasons x 30 teams)')
    st.markdown('- Target: Did this team win the championship that season?')
    st.markdown('- Models: Logistic Regression, Random Forest, XGBoost, Neural Network')
    st.markdown('### Key Findings')
    st.markdown('- Win % and plus/minus are the strongest predictors of championship success')
    st.markdown('- 3-point shooting efficiency has become more important since 2015')
    st.markdown('- Turnovers are a strong negative predictor')
    col1, col2, col3, col4 = st.columns(4)
    col1.metric('Seasons of Training Data', '25')
    col2.metric('Teams Scored', str(len(df_current)))
    col3.metric('Features Used', str(len(FEATURES)))
    col4.metric('Best Model AUC', f"{model_results['AUC-ROC'].max():.3f}")

with tab2:
    st.title('📊 Historical NBA Champion Analysis')
    champs = df_hist[df_hist['champion'] == 1]
    non_champs = df_hist[df_hist['champion'] == 0]
    st.subheader('Champion vs Non-Champion Averages')
    compare = pd.DataFrame({'Champions': champs[FEATURES].mean(), 'Non-Champions': non_champs[FEATURES].mean()}).round(3)
    compare['Edge'] = (compare['Champions'] - compare['Non-Champions']).round(3)
    st.dataframe(compare)
    st.subheader('Win % Distribution')
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(non_champs['W_PCT'], bins=20, alpha=0.6, color='steelblue', label='All Teams')
    ax.hist(champs['W_PCT'], bins=8, alpha=0.9, color='gold', edgecolor='black', label='Champions')
    ax.set_title('Win % — Champions vs All Teams (2000-2025)')
    ax.legend()
    st.pyplot(fig)
    st.subheader('Historical Champions')
    champ_display = champs[['SEASON','TEAM_NAME','W','L','W_PCT','PTS','PLUS_MINUS','FG3_PCT']].sort_values('SEASON', ascending=False)
    champ_display.columns = ['Season','Team','W','L','Win%','PPG','+/-','3PT%']
    st.dataframe(champ_display.reset_index(drop=True))

with tab3:
    st.title('🤖 Model Performance')
    st.subheader('Metrics Comparison')
    st.dataframe(model_results.round(4))
    st.subheader('AUC-ROC Comparison')
    fig, ax = plt.subplots(figsize=(9, 4))
    colors = ['steelblue', 'darkorange', 'gold', 'purple']
    bars = ax.bar(model_results.index, model_results['AUC-ROC'], color=colors, edgecolor='black')
    ax.set_ylim(0, 1.1)
    ax.set_title('AUC-ROC by Model')
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()+0.01, f'{bar.get_height():.3f}', ha='center')
    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()
    st.pyplot(fig)
    st.info('AUC-ROC is the primary metric — we care about ranking champion-caliber teams higher than average teams.')

with tab4:
    st.title('🏆 Live 2025-26 Championship Predictions')
    games_played = int(df_current['GP'].mean()) if 'GP' in df_current.columns else '?'
    st.info(f'📡 Live data — ~{games_played} games played per team in 2025-26 season')
    st.subheader('Championship Probability — All 30 Teams')
    df_sorted = df_pred.sort_values('champ_prob_pct', ascending=True)
    fig, ax = plt.subplots(figsize=(10, 12))
    bar_colors = ['gold' if i >= len(df_sorted)-3 else 'steelblue' if i >= len(df_sorted)-8 else 'lightgray' for i in range(len(df_sorted))]
    ax.barh(df_sorted['team'], df_sorted['champ_prob_pct'], color=bar_colors, edgecolor='black', linewidth=0.5)
    ax.set_xlabel('Championship Probability (%)')
    ax.set_title('2025-26 NBA Championship Probability', fontsize=14)
    gold_patch = mpatches.Patch(color='gold', label='Top 3 favorites')
    blue_patch = mpatches.Patch(color='steelblue', label='Top 8 contenders')
    gray_patch = mpatches.Patch(color='lightgray', label='Long shots')
    ax.legend(handles=[gold_patch, blue_patch, gray_patch])
    plt.tight_layout()
    st.pyplot(fig)
    st.subheader('Full Rankings Table')
    display_df = df_pred[['team','W','L','W_PCT','champ_prob_pct']].copy()
    display_df.columns = ['Team','W','L','Win%','Champ Prob %']
    st.dataframe(display_df)
    st.subheader('🔍 SHAP Explanation for Any Team')
    selected_team = st.selectbox('Select a team:', df_pred['team'].tolist())
    team_row = df_current[df_current['TEAM_NAME'] == selected_team]
    if not team_row.empty:
        X_team = team_row[[f for f in FEATURES if f in team_row.columns]]
        explainer = shap.TreeExplainer(xgb_model)
        sv = explainer.shap_values(X_team)
        sv_use = sv[1] if isinstance(sv, list) else sv
        ev_use = explainer.expected_value
        if isinstance(ev_use, (list, np.ndarray)): ev_use = float(ev_use[1])
        prob = df_pred[df_pred['team'] == selected_team]['champ_prob_pct'].values[0]
        st.metric(f'{selected_team} Championship Probability', f'{prob:.1f}%')
        shap.waterfall_plot(shap.Explanation(values=sv_use[0], base_values=ev_use, data=X_team.iloc[0].values, feature_names=list(X_team.columns)), show=False)
        st.pyplot(plt.gcf())
        plt.clf()
        st.subheader(f'{selected_team} — Current Stats')
        stats_show = team_row[['TEAM_NAME','W','L','W_PCT'] + FEATURES].T
        stats_show.columns = ['Value']
        st.dataframe(stats_show)