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
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, roc_auc_score
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

# ── Load models & data ────────────────────────────────────
@st.cache_resource
def load_models():
    with open('model_xgb.pkl', 'rb') as f: xgb_model = pickle.load(f)
    with open('model_rf.pkl',  'rb') as f: rf_model  = pickle.load(f)
    with open('model_lr.pkl',  'rb') as f: lr_model  = pickle.load(f)
    with open('scaler.pkl',    'rb') as f: scaler    = pickle.load(f)
    return xgb_model, rf_model, lr_model, scaler

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

def get_conference(team):
    if team in EASTERN: return 'East'
    if team in WESTERN: return 'West'
    return 'Unknown'

df_pred['conference'] = df_pred['team'].apply(get_conference)
df_hist['conference'] = df_hist['TEAM_NAME'].apply(get_conference)

last_updated = datetime.now().strftime('%B %d, %Y at %I:%M %p')

# ── Prepare train/test split for ROC curves ───────────────
@st.cache_data
def get_test_split():
    X = df_hist[FEATURES]
    y = df_hist['champion']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    X_test_sc = scaler.transform(X_test)
    return X_train, X_test, X_test_sc, y_train, y_test

X_train, X_test, X_test_sc, y_train, y_test = get_test_split()

# ── Tabs matching assignment rubric ──────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    'Executive Summary',
    'Descriptive Analytics',
    'Model Performance',
    'Explainability & Prediction',
    'Season Replay',
    'Head-to-Head',
])

# ═══════════════════════════════════════════════════════════
# TAB 1 — Executive Summary
# ═══════════════════════════════════════════════════════════
with tab1:
    st.title('NBA Championship Predictor — 2025-26')
    st.caption(f'Data last refreshed: {last_updated}')

    st.warning(
        "Predictions are based on regular season stats only and do not account for "
        "injuries, player availability, or playoff matchups."
    )

    st.markdown("## About This Project")

    st.markdown("""
    As a longtime basketball fan and a big Kobe Bryant fan, I have always been interested
    in what really makes a team win a championship. For this project, I used team data
    from the NBA Stats API through `nba_api`, covering about 25 NBA seasons from 2000 to 2025.
    Each row in the dataset represents one NBA team in one season, and the goal is to predict
    whether that team won the championship or not. The model uses team stats such as win percentage,
    points per game, rebounds, assists, turnovers, shooting efficiency, and net points per game.
    """)

    st.markdown("""
    Predicting an NBA champion is exciting but also very difficult. Fans often rely on opinions
    or narratives, but using statistics helps reveal clearer patterns behind winning teams.
    This type of analysis can be useful for fans, analysts, and even front offices trying to
    understand what drives championship success. The challenge is that only 1 out of 30 teams
    wins the championship each year, which creates a big imbalance in the data.
    """)

    st.markdown("""
    Based on the model's predictions using current 2025–26 season stats, the Detroit Pistons
    currently show the highest championship probability, followed by the Oklahoma City Thunder
    and the San Antonio Spurs. Out of the models tested, XGBoost performed the best based on
    AUC score. The results also show that win percentage, net points per game, and shooting
    efficiency are the most important factors in predicting a champion.
    """)

    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric('Seasons of Training Data', '25')
    col2.metric('Teams in Current Season', str(len(df_current)))
    col3.metric('Features Used', str(len(FEATURES)))
    col4.metric('Best Model AUC', f"{model_results['AUC-ROC'].max():.3f}")

    st.markdown("---")
    st.markdown("""
    **Features used in all models:**
    Win/Loss record, Win Percentage, Points Per Game, Rebounds, Assists,
    Turnovers, Steals, Blocks, Field Goal %, 3-Point %, Free Throw %,
    and Plus/Minus (how much a team outscores opponents on average).

    **Models trained:**
    Logistic Regression (baseline), Decision Tree, Random Forest, XGBoost, Neural Network.

    **Key finding:**
    Win percentage and net points per game (Plus/Minus) are the strongest predictors
    of championship success — teams that dominate opponents consistently are far more
    likely to win a title than teams that simply win close games.
    """)

# ═══════════════════════════════════════════════════════════
# TAB 2 — Descriptive Analytics
# ═══════════════════════════════════════════════════════════
with tab2:
    st.title('Descriptive Analytics')
    st.markdown("Exploring what the data looks like before any modeling.")

    champs     = df_hist[df_hist['champion'] == 1]
    non_champs = df_hist[df_hist['champion'] == 0]

    # 2.1 Target distribution
    st.subheader('Target Variable Distribution')
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(['Non-Champion', 'Champion'], [len(non_champs), len(champs)],
           color=['steelblue', 'gold'], edgecolor='black')
    ax.set_title('Class Balance: Champions vs Non-Champions (2000–2025)')
    ax.set_ylabel('Number of Team-Seasons')
    for i, v in enumerate([len(non_champs), len(champs)]):
        ax.text(i, v + 2, str(v), ha='center', fontsize=11)
    plt.tight_layout()
    st.pyplot(fig)
    st.caption("""
    Out of 26 seasons, only 26 team-seasons are labeled as champions (one per year) compared
    to roughly 690 non-champion team-seasons. This extreme class imbalance — about 1 champion
    per 30 teams — is the central challenge of this prediction task and was addressed using
    class weights in all models.
    """)

    # 2.2 Win % distribution
    st.subheader('Win Percentage: Champions vs All Teams')
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(non_champs['W_PCT'], bins=20, alpha=0.6, color='steelblue', label='Non-Champions')
    ax.hist(champs['W_PCT'], bins=8, alpha=0.9, color='gold', edgecolor='black', label='Champions')
    ax.set_title('Regular Season Win % Distribution (2000–2025)')
    ax.set_xlabel('Win Percentage')
    ax.set_ylabel('Count')
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)
    st.caption("""
    Champions are clustered tightly in the 0.60–0.80 win percentage range, while non-champions
    spread broadly from 0.20 to 0.75. No championship team in the last 25 years has finished
    the regular season below a 50% win rate, making win percentage one of the strongest single
    predictors in the dataset.
    """)

    # 2.3 Plus/Minus boxplot
    st.subheader('Net Points Per Game (Plus/Minus): Champions vs Rest')
    fig, ax = plt.subplots(figsize=(8, 5))
    data = [non_champs['PLUS_MINUS'].values, champs['PLUS_MINUS'].values]
    bp = ax.boxplot(data, patch_artist=True, labels=['Non-Champions', 'Champions'])
    bp['boxes'][0].set_facecolor('steelblue')
    bp['boxes'][1].set_facecolor('gold')
    ax.set_title('Plus/Minus Distribution: Champions vs Non-Champions')
    ax.set_ylabel('Net Points Per Game')
    ax.axhline(0, color='red', linestyle='--', lw=1, label='Break even')
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)
    st.caption("""
    Champions have a median Plus/Minus of around +6 to +8 points per game, meaning they
    outscore opponents by a wide margin on average. Non-champions are centered near zero.
    This confirms that true contenders do not just win — they dominate, which is why
    Plus/Minus is the second-strongest predictor in the model.
    """)

    # 2.4 3PT% trend over time
    st.subheader('3-Point Shooting % Over Time — Champions Only')
    champ_sorted = champs.sort_values('SEASON')
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(range(len(champ_sorted)), champ_sorted['FG3_PCT'],
            marker='o', lw=2, color='steelblue', markerfacecolor='gold', markersize=8)
    ax.set_xticks(range(len(champ_sorted)))
    ax.set_xticklabels([s[:4] for s in champ_sorted['SEASON']], rotation=45, fontsize=8)
    ax.set_title('Championship Team 3PT% by Season (2000–2025)')
    ax.set_ylabel('3-Point FG%')
    z = np.polyfit(range(len(champ_sorted)), champ_sorted['FG3_PCT'], 1)
    p = np.poly1d(z)
    ax.plot(range(len(champ_sorted)), p(range(len(champ_sorted))),
            'r--', lw=1.5, label='Trend')
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)
    st.caption("""
    There is a clear upward trend in 3-point shooting percentage among championship teams
    since around 2014, coinciding with the start of the "three-point revolution" in the NBA.
    Recent champions like Golden State (2017-18) and Boston (2023-24) shot notably better
    from three than champions of the early 2000s, reflecting how the game has changed.
    """)

    # 2.5 Correlation heatmap
    st.subheader('Correlation Heatmap')
    import seaborn as sns
    fig, ax = plt.subplots(figsize=(12, 9))
    corr = df_hist[FEATURES + ['champion']].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdYlBu_r',
                center=0, linewidths=0.5, ax=ax)
    ax.set_title('Feature Correlation Matrix', fontsize=14)
    plt.tight_layout()
    st.pyplot(fig)
    st.caption("""
    Win percentage (W_PCT) and Plus/Minus show the strongest positive correlation with
    the champion label. Turnovers (TOV) are negatively correlated — teams that turn the
    ball over frequently rarely win championships. Wins (W) and Win% are highly correlated
    with each other (as expected), suggesting some redundancy between those two features.
    """)

    # 2.6 Champion averages table
    st.subheader('Average Stats: Champions vs Non-Champions')
    compare = pd.DataFrame({
        'Champions':     champs[FEATURES].mean(),
        'Non-Champions': non_champs[FEATURES].mean(),
    }).round(3)
    compare['Edge (Champion)'] = (compare['Champions'] - compare['Non-Champions']).round(3)
    st.dataframe(compare)
    st.caption("""
    Champions average about 10 more wins, a 12% higher win percentage, and nearly
    7 more net points per game than the average non-champion. The negative edge on
    turnovers confirms that champion-caliber teams protect the ball better than the rest.
    """)

# ═══════════════════════════════════════════════════════════
# TAB 3 — Model Performance
# ═══════════════════════════════════════════════════════════
with tab3:
    st.title('Model Performance')
    st.markdown("""
    We trained five models on 25 years of historical NBA team stats (2000–2025).
    All models were evaluated on a held-out 25% test set using random_state=42.
    """)

    # Metric explanations
    with st.expander("What do these metrics mean? (click to expand)"):
        col1, col2 = st.columns(2)
        with col1:
            st.info("""
**AUC-ROC — How good is the model at sorting teams correctly?**

Think of it like a teacher predicting which students will pass before the final exam.
AUC measures how often the model correctly places a strong team above a weak one.
- **1.0** = perfect every time
- **0.5** = no better than a coin flip
- **Our best model: ~0.85+**
            """)
            st.info("""
**Precision — When the model flags a contender, how often is it right?**

If the model flags 10 teams as contenders and 8 actually were, precision is 80%.
High precision means the model does not cry wolf.
            """)
        with col2:
            st.info("""
**Recall — Did the model catch every real contender?**

Out of all truly championship-caliber teams, how many did the model find?
If 8 were real contenders and the model caught 5, recall is 63%.
High recall means the model does not miss anyone important.
            """)
            st.info("""
**F1 — The combined overall grade**

F1 balances precision and recall into one score.
It only goes up when the model is both accurate and thorough.
            """)

    # Model comparison table
    st.subheader('Model Comparison Table')
    st.dataframe(model_results.round(4).style.highlight_max(axis=0, color='#d4edda'))

    # AUC bar chart
    st.subheader('AUC-ROC Comparison')
    fig, ax = plt.subplots(figsize=(9, 4))
    colors = ['steelblue', 'darkorange', 'gold', 'purple', 'gray'][:len(model_results)]
    bars = ax.bar(model_results.index, model_results['AUC-ROC'],
                  color=colors, edgecolor='black')
    ax.set_ylim(0, 1.15)
    ax.set_title('AUC-ROC Score by Model')
    ax.set_ylabel('AUC-ROC')
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{bar.get_height():.3f}', ha='center', fontsize=11)
    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()
    st.pyplot(fig)
    st.caption("""
    XGBoost achieved the highest AUC-ROC, meaning it is best at correctly ranking
    championship-caliber teams above average teams. Logistic Regression serves as
    the baseline — all other models are compared against it.
    """)

    # ROC Curves
    st.subheader('ROC Curves — All Models')
    st.markdown("""
    The ROC curve plots the true positive rate vs. false positive rate at every
    possible decision threshold. A model that hugs the top-left corner is better.
    The diagonal dashed line represents random guessing.
    """)

    fig, ax = plt.subplots(figsize=(9, 7))

    model_map = {
        'Logistic Regression': (lr_model,  X_test_sc, False),
        'Random Forest':       (rf_model,  X_test,    False),
        'XGBoost':             (xgb_model, X_test,    False),
    }

    roc_colors = ['steelblue', 'darkorange', 'gold', 'purple']
    for (name, (model, X_use, _)), color in zip(model_map.items(), roc_colors):
        try:
            proba = model.predict_proba(X_use)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, proba)
            auc_val = auc(fpr, tpr)
            ax.plot(fpr, tpr, lw=2.5, color=color, label=f'{name} (AUC = {auc_val:.3f})')
        except Exception as e:
            st.warning(f'Could not plot ROC for {name}: {e}')

    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random guessing (AUC = 0.5)')
    ax.set_xlabel('False Positive Rate', fontsize=13)
    ax.set_ylabel('True Positive Rate', fontsize=13)
    ax.set_title('ROC Curves — All Models', fontsize=15)
    ax.legend(loc='lower right', fontsize=10)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    plt.tight_layout()
    st.pyplot(fig)
    st.caption("""
    All three models clearly outperform random guessing (the dashed diagonal line).
    XGBoost's curve sits closest to the top-left corner, confirming it achieves the
    best balance between catching real contenders and avoiding false positives.
    """)

    # Best hyperparameters
    st.subheader('Best Hyperparameters from Cross-Validation')
    st.markdown("""
    Each model was tuned using 5-fold GridSearchCV on the training set.
    The parameters below produced the highest cross-validation AUC score.
    """)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Logistic Regression**")
        try:
            lr_params = lr_model.get_params()
            st.json({'C': lr_params.get('C', 'N/A'),
                     'class_weight': lr_params.get('class_weight', 'N/A'),
                     'max_iter': lr_params.get('max_iter', 'N/A')})
        except:
            st.json({'C': 'See notebook', 'class_weight': 'balanced'})

    with col2:
        st.markdown("**Random Forest**")
        try:
            rf_params = rf_model.get_params()
            st.json({'n_estimators': rf_params.get('n_estimators', 'N/A'),
                     'max_depth': rf_params.get('max_depth', 'N/A'),
                     'min_samples_leaf': rf_params.get('min_samples_leaf', 'N/A'),
                     'class_weight': rf_params.get('class_weight', 'N/A')})
        except:
            st.json({'n_estimators': 'See notebook', 'max_depth': 'See notebook'})

    with col3:
        st.markdown("**XGBoost**")
        try:
            xgb_params = xgb_model.get_params()
            st.json({'n_estimators': xgb_params.get('n_estimators', 'N/A'),
                     'max_depth': xgb_params.get('max_depth', 'N/A'),
                     'learning_rate': xgb_params.get('learning_rate', 'N/A'),
                     'scale_pos_weight': round(float(xgb_params.get('scale_pos_weight', 0)), 1)})
        except:
            st.json({'n_estimators': 'See notebook', 'max_depth': 'See notebook'})

    st.markdown("""
    **Which model performed best?**
    XGBoost achieved the highest AUC-ROC and F1 score on the held-out test set.
    This is expected — boosting algorithms iteratively correct their own mistakes,
    making them well-suited for imbalanced tabular data like this dataset.
    Logistic Regression, despite being the simplest model, performed surprisingly
    well, which suggests that win percentage and plus/minus are strong linear
    predictors of championship success on their own.
    The trade-off is interpretability: Logistic Regression is easy to explain to
    a non-technical audience, while XGBoost requires SHAP to understand its decisions.
    """)

# ═══════════════════════════════════════════════════════════
# TAB 4 — Explainability & Interactive Prediction
# ═══════════════════════════════════════════════════════════
with tab4:
    st.title('Explainability & Interactive Prediction')

    subtab1, subtab2 = st.tabs(['SHAP Analysis', 'Interactive Prediction'])

    # ── SHAP Analysis ──────────────────────────────────────
    with subtab1:
        st.subheader('SHAP Analysis — Why Does the Model Make These Predictions?')
        st.markdown("""
        SHAP (SHapley Additive exPlanations) breaks down each prediction to show exactly
        how much each stat contributed — positively or negatively — to the final score.
        Red = pushes the probability up. Blue = pushes it down.
        """)

        explainer = shap.TreeExplainer(xgb_model)
        X_hist_feat = df_hist[FEATURES]
        shap_values = explainer.shap_values(X_hist_feat)
        sv = shap_values[1] if isinstance(shap_values, list) else shap_values

        # Beeswarm
        st.markdown("**Summary Plot (Beeswarm) — Feature Importance and Direction**")
        st.markdown("""
        Each dot is one team-season. Position left or right shows whether that stat
        pushed the prediction toward champion (right) or away (left).
        Color shows whether the feature value was high (red) or low (blue).
        """)
        plt.figure()
        shap.summary_plot(sv, X_hist_feat, feature_names=FEATURES, show=False)
        plt.tight_layout()
        st.pyplot(plt.gcf())
        plt.clf()
        st.caption("""
        Win percentage (W_PCT) and Plus/Minus dominate the chart — high values of both
        push the model strongly toward predicting a championship. Turnovers (TOV) work
        in reverse: high turnover rates pull the prediction away from a championship outcome.
        """)

        # Bar plot
        st.markdown("**Bar Plot — Overall Feature Importance (Mean Absolute SHAP)**")
        plt.figure()
        shap.summary_plot(sv, X_hist_feat, feature_names=FEATURES,
                          plot_type='bar', show=False)
        plt.tight_layout()
        st.pyplot(plt.gcf())
        plt.clf()
        st.caption("""
        Win percentage is the single most important feature by a clear margin, followed
        by Plus/Minus and Wins. 3-point shooting percentage (FG3_PCT) ranks in the middle,
        reflecting the growing importance of three-point shooting in the modern NBA.
        """)

        # Waterfall for top predicted team
        st.markdown("**Waterfall Plot — Top 2025-26 Championship Favorite**")
        st.markdown("""
        This chart explains the prediction for the single team with the highest
        championship probability this season. Each bar shows how much one stat
        moved the prediction up or down from the baseline average.
        """)

        top_team = df_pred.sort_values('champ_prob_pct', ascending=False).iloc[0]
        top_team_name = top_team['team']
        top_team_prob = top_team['champ_prob_pct']

        top_row = df_current[df_current['TEAM_NAME'] == top_team_name]
        if not top_row.empty:
            X_top = top_row[[f for f in FEATURES if f in top_row.columns]]
            sv_top = explainer.shap_values(X_top)
            sv_top_use = sv_top[1][0] if isinstance(sv_top, list) else sv_top[0]
            ev = explainer.expected_value
            if isinstance(ev, (list, np.ndarray)): ev = float(ev[1])

            st.markdown(f"**{top_team_name}** — {top_team_prob:.1f}% championship probability")

            shap.waterfall_plot(
                shap.Explanation(
                    values=sv_top_use,
                    base_values=ev,
                    data=X_top.iloc[0].values,
                    feature_names=list(X_top.columns)
                ), show=False
            )
            st.pyplot(plt.gcf())
            plt.clf()
            st.caption("""
            The waterfall shows the exact reason this team is ranked #1. Each red bar
            is a stat where they outperform what a typical team looks like, pushing
            their championship probability higher. Blue bars represent weaknesses
            that slightly reduce their probability.
            """)

    # ── Interactive Prediction ──────────────────────────────
    with subtab2:
        st.subheader('Interactive Prediction — Build Your Own Team')
        st.markdown("""
        Use the sliders below to set team stats and see what championship probability
        the model predicts. Adjust a few key stats or build a hypothetical team from scratch.
        Stats not shown use the league average automatically.
        """)

        # Model selector
        model_choice = st.selectbox(
            'Select which model to use for prediction:',
            ['XGBoost (Best)', 'Random Forest', 'Logistic Regression']
        )

        # League averages as defaults
        league_avg = df_hist[FEATURES].mean()

        st.markdown("**Adjust the key stats below:**")

        col1, col2, col3 = st.columns(3)

        with col1:
            w_pct = st.slider(
                'Win Percentage',
                min_value=0.10, max_value=0.90,
                value=float(round(league_avg['W_PCT'], 2)),
                step=0.01,
                help='Fraction of games won. 0.61 = 50 wins in an 82-game season.'
            )
            pts = st.slider(
                'Points Per Game',
                min_value=90.0, max_value=130.0,
                value=float(round(league_avg['PTS'], 1)),
                step=0.5,
                help='Average points scored per game.'
            )
            plus_minus = st.slider(
                'Plus/Minus (Net Points Per Game)',
                min_value=-15.0, max_value=15.0,
                value=float(round(league_avg['PLUS_MINUS'], 1)),
                step=0.5,
                help='How much the team outscores (positive) or is outscored by (negative) opponents on average.'
            )

        with col2:
            fg3_pct = st.slider(
                '3-Point FG%',
                min_value=0.28, max_value=0.45,
                value=float(round(league_avg['FG3_PCT'], 3)),
                step=0.005,
                help='Fraction of 3-point attempts made.'
            )
            ast = st.slider(
                'Assists Per Game',
                min_value=15.0, max_value=35.0,
                value=float(round(league_avg['AST'], 1)),
                step=0.5,
                help='Average assists per game.'
            )
            tov = st.slider(
                'Turnovers Per Game',
                min_value=8.0, max_value=20.0,
                value=float(round(league_avg['TOV'], 1)),
                step=0.5,
                help='Average turnovers per game. Lower is better.'
            )

        with col3:
            reb = st.slider(
                'Rebounds Per Game',
                min_value=35.0, max_value=55.0,
                value=float(round(league_avg['REB'], 1)),
                step=0.5
            )
            fg_pct = st.slider(
                'Field Goal %',
                min_value=0.40, max_value=0.55,
                value=float(round(league_avg['FG_PCT'], 3)),
                step=0.005
            )

        # Build the input vector using sliders + league avg for remaining features
        user_input = league_avg.copy()
        user_input['W_PCT']     = w_pct
        user_input['PTS']       = pts
        user_input['PLUS_MINUS']= plus_minus
        user_input['FG3_PCT']   = fg3_pct
        user_input['AST']       = ast
        user_input['TOV']       = tov
        user_input['REB']       = reb
        user_input['FG_PCT']    = fg_pct
        # Estimate W and L from W_PCT
        user_input['W'] = round(w_pct * 82)
        user_input['L'] = 82 - user_input['W']

        X_user = pd.DataFrame([user_input[FEATURES]])
        X_user_sc = scaler.transform(X_user)

        # Select model
        if model_choice == 'XGBoost (Best)':
            selected_model = xgb_model
            X_pred_input = X_user
        elif model_choice == 'Random Forest':
            selected_model = rf_model
            X_pred_input = X_user
        else:
            selected_model = lr_model
            X_pred_input = X_user_sc

        # Predict
        raw_prob = selected_model.predict_proba(X_pred_input)[0][1]

        # Normalize against current season to get % probability
        current_X = df_current[[f for f in FEATURES if f in df_current.columns]]
        if model_choice == 'Logistic Regression':
            current_X_input = scaler.transform(current_X)
        else:
            current_X_input = current_X

        all_probs = selected_model.predict_proba(current_X_input)[:, 1]
        combined  = np.append(all_probs, raw_prob)
        norm_prob = raw_prob / combined.sum() * 100

        st.markdown("---")
        st.subheader('Prediction Result')

        res_col1, res_col2, res_col3 = st.columns(3)
        res_col1.metric('Championship Probability', f'{norm_prob:.1f}%')
        res_col2.metric('Raw Model Score', f'{raw_prob:.4f}')
        res_col3.metric('Model Used', model_choice.split(' ')[0])

        # Context
        top3 = df_pred.nlargest(3, 'champ_prob_pct')['champ_prob_pct'].min()
        if norm_prob >= top3:
            st.success('This team profile would rank in the Top 3 championship favorites!')
        elif norm_prob >= df_pred['champ_prob_pct'].quantile(0.75):
            st.info('This team profile would be considered a legitimate contender.')
        else:
            st.warning('This team profile would not be considered a championship favorite.')

        # SHAP waterfall for custom input
        if model_choice != 'Logistic Regression':
            st.markdown("**Why did the model give this probability? (SHAP Waterfall)**")
            exp = shap.TreeExplainer(selected_model)
            sv_user = exp.shap_values(X_user)
            sv_user_use = sv_user[1][0] if isinstance(sv_user, list) else sv_user[0]
            ev_user = exp.expected_value
            if isinstance(ev_user, (list, np.ndarray)): ev_user = float(ev_user[1])

            shap.waterfall_plot(
                shap.Explanation(
                    values=sv_user_use,
                    base_values=ev_user,
                    data=X_user.iloc[0].values,
                    feature_names=FEATURES
                ), show=False
            )
            st.pyplot(plt.gcf())
            plt.clf()
            st.caption("""
            Red bars show which stats are boosting this team's championship probability.
            Blue bars show which stats are dragging it down. The final value on the right
            is the model's raw score before normalization.
            """)

# ═══════════════════════════════════════════════════════════
# TAB 5 — Season Replay
# ═══════════════════════════════════════════════════════════
with tab5:
    st.title('Season Replay — What Would the Model Have Predicted?')
    st.markdown("""
    Select any past season to see what championship probability the model would have
    assigned to every team. The actual champion is highlighted in gold.
    """)
    st.warning(
        "These are in-sample predictions — the model was trained on this data. "
        "Think of this as a sanity check rather than a true backtest."
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
            'Won Title': (df_season['TEAM_NAME'] == actual_champion).astype(int).values
        }).sort_values('champ_prob_pct', ascending=False).reset_index(drop=True)

        predicted_rank = df_season_pred[df_season_pred['team'] == actual_champion].index[0] + 1

        col1, col2, col3 = st.columns(3)
        col1.metric('Season', selected_season)
        col2.metric('Actual Champion', actual_champion)
        col3.metric('Model Ranked Champion',
                    f'#{predicted_rank}',
                    delta='Correct!' if predicted_rank == 1 else f'Missed — ranked #{predicted_rank}',
                    delta_color='normal' if predicted_rank == 1 else 'inverse')

        df_s_sorted = df_season_pred.sort_values('champ_prob_pct', ascending=True)
        bar_colors_s = ['gold' if t == actual_champion else
                        'steelblue' if p >= df_s_sorted['champ_prob_pct'].quantile(0.75) else
                        'lightgray'
                        for t, p in zip(df_s_sorted['team'], df_s_sorted['champ_prob_pct'])]

        fig, ax = plt.subplots(figsize=(10, max(6, len(df_s_sorted) * 0.38)))
        ax.barh(df_s_sorted['team'], df_s_sorted['champ_prob_pct'],
                color=bar_colors_s, edgecolor='black', linewidth=0.5)
        ax.set_xlabel('Model Championship Probability (%)')
        ax.set_title(f'{selected_season} — Model Predictions vs Actual Outcome', fontsize=13)
        ax.legend(handles=[
            mpatches.Patch(color='gold',      label=f'Actual Champion ({actual_champion})'),
            mpatches.Patch(color='steelblue', label='Top quartile'),
            mpatches.Patch(color='lightgray', label='Rest of league'),
        ])
        plt.tight_layout()
        st.pyplot(fig)

        st.dataframe(df_season_pred.rename(columns={
            'team': 'Team', 'W': 'Wins', 'W_PCT': 'Win %',
            'champ_prob_pct': 'Model Probability (%)'
        }))

# ═══════════════════════════════════════════════════════════
# TAB 6 — Head-to-Head
# ═══════════════════════════════════════════════════════════
with tab6:
    st.title('Head-to-Head Team Comparison')
    st.markdown("Select any two current teams to compare their stats and championship probabilities.")

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

        m1, m2 = st.columns(2)
        m1.metric(team_a, f'{prob_a:.1f}%', delta=f'{prob_a - prob_b:+.1f}% vs {team_b}')
        m2.metric(team_b, f'{prob_b:.1f}%', delta=f'{prob_b - prob_a:+.1f}% vs {team_a}')

        if not row_a.empty and not row_b.empty:
            key_stats = ['W_PCT', 'PTS', 'PLUS_MINUS', 'FG3_PCT', 'TOV', 'AST']
            key_stats = [s for s in key_stats if s in row_a.columns]

            vals_a = [float(row_a[s].values[0]) for s in key_stats]
            vals_b = [float(row_b[s].values[0]) for s in key_stats]

            x = np.arange(len(key_stats))
            width = 0.35
            fig, ax = plt.subplots(figsize=(11, 5))
            ax.bar(x - width/2, vals_a, width, label=team_a, color='steelblue', edgecolor='black')
            ax.bar(x + width/2, vals_b, width, label=team_b, color='gold', edgecolor='black')
            ax.set_xticks(x)
            ax.set_xticklabels(key_stats, fontsize=11)
            ax.set_title(f'{team_a} vs {team_b} — Key Stats', fontsize=13)
            ax.legend()
            plt.tight_layout()
            st.pyplot(fig)

            st.subheader('What is driving each team\'s probability?')
            X_a = row_a[[f for f in FEATURES if f in row_a.columns]]
            X_b = row_b[[f for f in FEATURES if f in row_b.columns]]

            exp = shap.TreeExplainer(xgb_model)
            sv_a = exp.shap_values(X_a)
            sv_b = exp.shap_values(X_b)
            sv_a = sv_a[1] if isinstance(sv_a, list) else sv_a
            sv_b = sv_b[1] if isinstance(sv_b, list) else sv_b

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

            plt.suptitle('SHAP Feature Contributions', fontsize=13)
            plt.tight_layout()
            st.pyplot(fig)
