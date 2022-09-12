import numpy as np
import pandas as pd

from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier

from meta_labeling.data_generation import vol_regime, prep_data, classification_stats, add_strat_metrics

df = pd.read_csv('./gbm2.csv', index_col=0)
df_empty = df[0:0]
df_empty.to_csv('gbm2.csv')

for z in range(0, 1000):
    # --- Data Prep ---
    # -----------------------------------------------------------------------------------------------
    # Constants
    steps = 10000
    prob_switch = 0.20
    stdev = 0.014543365294448746  # About the same as IBM stdev

    # Create dual data set
    data = vol_regime(total_steps=steps, prob_switch=prob_switch, stdev=stdev)

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

    # Add standardScalar as a best practice although in this setting it's not really needed.
    # Logistic regression is a convex optimisation problem and the global minima is always found.
    scaler = StandardScaler()
#    X_train_info_scaled = scaler.fit_transform(X_train_info)
#    X_train_regime_scaled = scaler.fit_transform(X_train_regime)
    # Test data
#    X_test_info_scaled = scaler.transform(X_test_info)
#    X_test_regime_scaled = scaler.transform(X_test_regime)

    # Train 2 models (Info, FP)
#    meta_model_info = LogisticRegression(random_state=0, penalty='none')
    meta_model_info = LGBMClassifier(random_state=0)
#    meta_model_regime = LogisticRegression(random_state=0, penalty='none')
    meta_model_regime = LGBMClassifier(random_state=0)
#    meta_model_info.fit(X_train_info_scaled, y_train)
#    meta_model_regime.fit(X_train_regime_scaled, y_train)
    meta_model_info.fit(X_train_info, y_train)
    meta_model_regime.fit(X_train_regime, y_train)

    # Predict Info and FP
#    train_pred_info = meta_model_info.predict(X_train_info_scaled)
#    train_pred_regime = meta_model_regime.predict(X_train_regime_scaled)
    train_pred_info = meta_model_info.predict(X_train_info)
    train_pred_regime = meta_model_regime.predict(X_train_regime)
    # Predictions
    train['pred_info'] = train_pred_info
    train['pred_regime'] = train_pred_regime
    # Probabilities
#    train['prob_info'] = meta_model_info.predict_proba(X_train_info_scaled)[:, 1]
#    train['prob_regime'] = meta_model_regime.predict_proba(X_train_regime_scaled)[:, 1]
    train['prob_info'] = meta_model_info.predict_proba(X_train_info)[:, 1]
    train['prob_regime'] = meta_model_regime.predict_proba(X_train_regime)[:, 1]

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
#    test['pred_info'] = meta_model_info.predict(X_test_info_scaled)
#    test['prob_info'] = meta_model_info.predict_proba(X_test_info_scaled)[:, 1]
#    test['pred_regime'] = meta_model_regime.predict(X_test_regime_scaled)
#    test['prob_regime'] = meta_model_regime.predict_proba(X_test_regime_scaled)[:, 1]
    test['pred_info'] = meta_model_info.predict(X_test_info)
    test['prob_info'] = meta_model_info.predict_proba(X_test_info)[:, 1]
    test['pred_regime'] = meta_model_regime.predict(X_test_regime)
    test['prob_regime'] = meta_model_regime.predict_proba(X_test_regime)[:, 1]

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
    test_cumrets = pd.DataFrame({'meta_info': ((data_test_set['meta_rets_info']  + 1).cumprod()),
                                 'meta_regime': ((data_test_set['meta_rets_regime']  + 1).cumprod()),
                                 'primary': ((data_test_set['prets']  + 1).cumprod()),
                                 'BAH': ((data_test_set['rets']  + 1).cumprod())})

    # --- Statistics ---
    # -----------------------------------------------------------------------------------------------
    # Primary model stats
    brow = classification_stats(actual=test['target'], predicted=test['pmodel'], prefix='b',
                                get_specificity=False)
    # Information Advantage
    irow = classification_stats(actual=y_test, predicted=test['pred_info'], prefix='mi',
                                get_specificity=True)
    # False Positive Modeling
    fprow = classification_stats(actual=y_test, predicted=test['pred_regime'], prefix='fp',
                                 get_specificity=True)
    # Concat data
    final_row = pd.concat([brow, irow, fprow], axis=1)

    # Add Strategy Metrics
    add_strat_metrics(row=final_row, rets=data_test_set['rets'], prefix='bah')
    add_strat_metrics(row=final_row, rets=data_test_set['prets'], prefix='p')
    add_strat_metrics(row=final_row, rets=data_test_set['meta_rets_info'], prefix='imeta')
    add_strat_metrics(row=final_row, rets=data_test_set['meta_rets_regime'], prefix='fmeta')
    final_row['num_samples'] = y_test.shape[0]

    # Comparison Metrics
    final_row['ip_sr'] = final_row['imeta_sr'] - final_row['p_sr']
    final_row['ip_avg'] = final_row['imeta_mean'] - final_row['p_mean']
    final_row['ip_std'] = final_row['imeta_stdev'] - final_row['p_stdev']

    final_row['fp_sr'] = final_row['fmeta_sr'] - final_row['p_sr']
    final_row['fp_avg'] = final_row['fmeta_mean'] - final_row['p_mean']
    final_row['fp_std'] = final_row['fmeta_stdev'] - final_row['p_stdev']

    # Classification Metrics
    final_row['ip_recall'] = final_row['mi_recall'] - final_row['b_recall']
    final_row['fp_recall'] = final_row['fp_recall'] - final_row['b_recall']

    final_row['ip_prec'] = final_row['mi_precision'] - final_row['b_precision']
    final_row['fp_prec'] = final_row['fp_precision'] - final_row['b_precision']

    final_row['ip_acc'] = final_row['mi_accuracy'] - final_row['b_accuracy']
    final_row['fp_acc'] = final_row['fp_accuracy'] - final_row['b_accuracy']

    final_row['ip_f1'] = final_row['mi_weighted_avg_f1'] - final_row['b_weighted_avg_f1']
    final_row['fp_f1'] = final_row['fp_weighted_avg_f1'] - final_row['b_weighted_avg_f1']

    final_row['ip_auc'] = final_row['mi_auc'] - final_row['b_auc']
    final_row['fp_auc'] = final_row['fp_auc'] - final_row['b_auc']

    # --- Save Report ---
    # -----------------------------------------------------------------------------------------------

    # final_row.to_csv('hyp2.csv')

    # Save results to csv
    data = pd.read_csv('./gbm2.csv', index_col=0)
    concat = pd.concat([data, final_row]).reset_index(drop=True)
    concat.to_csv('gbm2.csv')

    print(z)