import datetime as dt

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_curve, confusion_matrix, RocCurveDisplay, ConfusionMatrixDisplay, roc_auc_score, PrecisionRecallDisplay, precision_recall_curve
from scipy.stats import norm
from statsmodels.distributions.empirical_distribution import ECDF
import quantstats as qs

from meta_labeling.data_generation import single_regime, dual_regime, prep_data

import matplotlib.pyplot as plt


for z in range(0, 1000, 1):
    # --- Data Prep ---
    # -----------------------------------------------------------------------------------------------
    # Constants
    steps = 10000
    prob_switch = 0.20
    stdev = 0.014543365294448746  # About the same as IBM stdev

    # Create dual data set
    data = dual_regime(total_steps=steps, prob_switch=prob_switch, stdev=stdev)

    # Prep data, add primary model, get meta_labels
    model_data, data = prep_data(data=data, with_flags=True)

    # --- Modeling ---
    # -----------------------------------------------------------------------------------------------
    # Train test split
    train, test = train_test_split(model_data, test_size=0.4, shuffle=False)

    X_train_info = train[['rets', 'rets2', 'rets3']]
    X_test_info = test[['rets', 'rets2', 'rets3']]

    X_train_regime = train[['rets', 'rets2', 'rets3', 'regime']]
    X_test_regime = test[['rets', 'rets2', 'rets3', 'regime']]

    y_train = train['target']
    y_test = test['target']

    # Add standardScalar as a best practice although in this setting its not really needed.
    # Logistic regression is a convex optimisation problem and the global minima is always found.
    scaler = StandardScaler()
    X_train_info_scaled = scaler.fit_transform(X_train_info)
    X_train_regime_scaled = scaler.fit_transform(X_train_regime)
    # Test data
    X_test_info_scaled = scaler.transform(X_test_info)
    X_test_regime_scaled = scaler.transform(X_test_regime)

    # Train 2 models (Info, FP)
    meta_model_info = LogisticRegression(random_state=0, penalty='none')
    meta_model_regime = LogisticRegression(random_state=0, penalty='none')
    meta_model_info.fit(X_train_info_scaled, y_train)
    meta_model_regime.fit(X_train_regime_scaled, y_train)

    # Predict Info and FP
    train_pred_info = meta_model_info.predict(X_train_info_scaled)
    train_pred_regime = meta_model_regime.predict(X_train_regime_scaled)
    # Predictions
    train['pred_info'] = train_pred_info
    train['pred_regime'] = train_pred_regime
    # Probabilities
    train['prob_info'] = meta_model_info.predict_proba(X_train_info_scaled)[:, 1]
    train['prob_regime'] = meta_model_regime.predict_proba(X_train_regime_scaled)[:, 1]

    # --- Prep Strategy Data ---
    # -----------------------------------------------------------------------------------------------
    # Save forecasts to original data
    # Set new columns
    data['pred_info'] = 0
    data['prob_info'] = 0
    data['pred_regime'] = 0
    data['prob_regime'] = 0
    # Assign column values
    data.loc[train.index, 'pred_info'] = train['pred_info']
    data.loc[train.index, 'prob_info'] = train['prob_info']
    data.loc[train.index, 'pred_regime'] = train['pred_regime']
    data.loc[train.index, 'prob_regime'] = train['prob_regime']

    # --- Training Performance ---
    # Subset train data
    data_train_set = data.loc[train.index[0]:train.index[-1]]

    # Create returns for 2 meta models
    meta_rets_info = (data_train_set['pred_info'] * data_train_set['target_rets']).shift(1)
    data_train_set['meta_rets_info'] = meta_rets_info
    meta_rets_regime = (data_train_set['pred_regime'] * data_train_set['target_rets']).shift(1)
    data_train_set['meta_rets_regime'] = meta_rets_regime
    data_train_set.dropna(inplace=True)
    # Train Cumrets
    train_cumrets = pd.DataFrame({'meta_info': ((data_train_set['meta_rets_info'] + 1).cumprod()),
                                  'meta_regime': ((data_train_set['meta_rets_regime'] + 1).cumprod()),
                                  'primary': ((data_train_set['prets'] + 1).cumprod()),
                                  'BAH': ((data_train_set['rets'] + 1).cumprod())})

    # --- Test Performance ---
    test['pred_info'] = meta_model_info.predict(X_test_info_scaled)
    test['prob_info'] = meta_model_info.predict_proba(X_test_info_scaled)[:, 1]
    test['pred_regime'] = meta_model_regime.predict(X_test_regime_scaled)
    test['prob_regime'] = meta_model_regime.predict_proba(X_test_regime_scaled)[:, 1]

    # Add test data to larger data df
    data.loc[test.index, 'pred_info'] = test['pred_info']
    data.loc[test.index, 'prob_info'] = test['prob_info']
    data.loc[test.index, 'pred_regime'] = test['pred_regime']
    data.loc[test.index, 'prob_regime'] = test['prob_regime']

    data_test_set = data.loc[test.index[0]:test.index[-1]]

    # Create returns for meta model
    meta_rets_info = (data_test_set['pred_info'] * data_test_set['target_rets']).shift(1)
    data_test_set['meta_rets_info'] = meta_rets_info
    meta_rets_regime = (data_test_set['pred_regime'] * data_test_set['target_rets']).shift(1)
    data_test_set['meta_rets_regime'] = meta_rets_regime
    data_test_set.dropna(inplace=True)
    # Test Cumrets
    test_cumrets = pd.DataFrame({'meta_info': ((data_test_set['meta_rets_info'] + 1).cumprod()),
                                 'meta_regime': ((data_test_set['meta_rets_regime'] + 1).cumprod()),
                                 'primary': ((data_test_set['prets'] + 1).cumprod()),
                                 'BAH': ((data_test_set['rets'] + 1).cumprod())})
    # --- Bet Sizing ---
    # -----------------------------------------------------------------------------------------------
    # 1) Generate signals from multinomial classification (one-vs-rest)
    prob_train = data_train_set.loc[data_train_set['pred_regime'] == 1, 'prob_regime']
    prob_test = data_test_set.loc[data_test_set['pred_regime'] == 1, 'prob_regime']

    # ECDF Position Sizing
    ecdf = ECDF(prob_train)
    e_bet_sizes = prob_test.apply(lambda x: ecdf(x))
    # Assign position sizes
    data_test_set['e_bet_size'] = 0
    data_test_set.loc[data_test_set['pred_regime'] == 1, 'e_bet_size'] = e_bet_sizes
    # Get daily rets
    data_test_set['bets_e_rets'] = (data_test_set['e_bet_size'] * data_test_set['target_rets']).shift(1)
    data_test_set.dropna(inplace=True)

    # --- Statistics ---
    # -----------------------------------------------------------------------------------------------
    test_cumrets = pd.DataFrame({'e_bet_size': ((data_test_set['bets_e_rets'] + 1).cumprod()),
                                 'meta_info': ((data_test_set['meta_rets_info'] + 1).cumprod()),
                                 'meta_regime': ((data_test_set['meta_rets_regime'] + 1).cumprod()),
                                 'primary': ((data_test_set['prets'] + 1).cumprod()),
                                 'BAH': ((data_test_set['rets'] + 1).cumprod())})

    tc = {'e_bet_size': data_test_set['bets_e_rets'].mean() / data_test_set['bets_e_rets'].std() * np.sqrt(252),
          'BAH': data_test_set['rets'].mean() / data_test_set['rets'].std() * np.sqrt(252),
          'meta_info': data_test_set['meta_rets_info'].mean() / data_test_set['meta_rets_info'].std() * np.sqrt(252),
          'meta_regime': data_test_set['meta_rets_regime'].mean() / data_test_set['meta_rets_regime'].std() * np.sqrt(252),
          'primary': data_test_set['prets'].mean() / data_test_set['prets'].std() * np.sqrt(252),
          }

    sr = {'sr_e': data_test_set['bets_e_rets'].mean() / data_test_set['bets_e_rets'].std() * np.sqrt(252),
          'sr_BAH': data_test_set['rets'].mean() / data_test_set['rets'].std() * np.sqrt(252),
          'sr_mi': data_test_set['meta_rets_info'].mean() / data_test_set['meta_rets_info'].std() * np.sqrt(252),
          'sr_mf': data_test_set['meta_rets_regime'].mean() / data_test_set['meta_rets_regime'].std() * np.sqrt(252),
          'sr_primary': data_test_set['prets'].mean() / data_test_set['prets'].std() * np.sqrt(252),
          }
    m = {'mean_e': (1 + data_test_set['bets_e_rets'].mean()) ** 252 - 1,
         'mean_BAH': (1 + data_test_set['rets'].mean()) ** 252 - 1,
         'mean_mi': (1 + data_test_set['meta_rets_info'].mean()) ** 252 - 1,
         'mean_mf': (1 + data_test_set['meta_rets_regime'].mean()) ** 252 - 1,
         'mean_primary': (1 + data_test_set['prets'].mean()) ** 252 - 1,
         }

    s = {'std_e': data_test_set['bets_e_rets'].std() * np.sqrt(252),
         'std_BAH': data_test_set['rets'].std() * np.sqrt(252),
         'std_mi': data_test_set['meta_rets_info'].std() * np.sqrt(252),
         'std_mf': data_test_set['meta_rets_regime'].std() * np.sqrt(252),
         'std_primary': data_test_set['prets'].std() * np.sqrt(252),
         }

    # Compute Max DDs
    mdds = qs.stats.max_drawdown(test_cumrets)
    # Check for negative values (MDDs) and correct
    for ind, val in mdds.iteritems():
        if val == 0.0:
            md = -(1 - test_cumrets[ind][-1])
            mdds[ind] = md
    mdds = mdds.to_dict()

    final = {**sr, **m, **s, **mdds}
    final_row = pd.DataFrame(final.values(), index=final.keys()).T

    # --- Save Report ---
    # -----------------------------------------------------------------------------------------------
    # Save results to csv
    # final_row.to_csv('hyp3.csv')

    data = pd.read_csv('hyp3.csv', index_col=0)
    concat = pd.concat([data, final_row]).reset_index(drop=True)
    concat.to_csv('hyp3.csv')
    print(z)


















