import datetime as dt

import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_curve, confusion_matrix, RocCurveDisplay, ConfusionMatrixDisplay, roc_auc_score, PrecisionRecallDisplay, precision_recall_curve
from sklearn.preprocessing import StandardScaler

from meta_labeling.data_generation import single_regime, prep_data, classification_stats, add_strat_metrics

for z in range(0, 1000, 1):
    # --- Data Prep ---
    # -----------------------------------------------------------------------------------------------
    # Constants
    steps = 10000
    stdev = 0.014543365294448746  # About the same as IBM stdev

    # Create single data set
    data = single_regime(steps=steps, stdev=stdev, drift=0.0)

    # Prep data, add primary model, get meta_labels
    model_data, data = prep_data(data=data, with_flags=False)

    # --- Modeling ---
    # -----------------------------------------------------------------------------------------------
    # Train test split
    train, test = train_test_split(model_data, test_size=0.4, shuffle=False)
    # Train data
    X_train = train[['rets', 'rets2', 'rets3']]
    y_train = train['target']
    # Test data
    X_test = test[['rets', 'rets2', 'rets3']]
    y_test = test['target']

    # Scale data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Train model
    meta_model = LogisticRegression(random_state=0, penalty='none')
    meta_model.fit(X_train_scaled, y_train)

    # Predict
    train_pred = meta_model.predict(X_train_scaled)
    train['pred'] = train_pred
    train['prob'] = meta_model.predict_proba(X_train_scaled)[:, 1]

    # --- Prep Strategy Data ---
    # -----------------------------------------------------------------------------------------------
    # Save forecasts to original data
    # Set new columns
    data['pred'] = 0
    data['prob'] = 0
    # Assign column values
    data.loc[train.index, 'pred'] = train['pred']
    data.loc[train.index, 'prob'] = train['prob']

    # --- Training Performance ---
    # Subset for train data
    data_train_set = data.loc[train.index[0]:train.index[-1]]

    # Create returns for 1 meta model
    meta_rets = (data_train_set['pred'] * data_train_set['target_rets']).shift(1)
    data_train_set['meta_rets'] = meta_rets
    data_train_set.dropna(inplace=True)
    # Train Cumrets
    train_cumrets = pd.DataFrame({'meta': ((data_train_set['meta_rets'] + 1).cumprod()),
                                  'primary': ((data_train_set['prets'] + 1).cumprod()),
                                  'BAH': ((data_train_set['rets'] + 1).cumprod())})

    # --- Test Performance ---
    test['pred'] = meta_model.predict(scaler.transform(X_test))
    test['prob'] = meta_model.predict_proba(scaler.transform(X_test))[:, 1]

    # Add test data to larger data df
    data.loc[test.index, 'pred'] = test['pred']
    data.loc[test.index, 'prob'] = test['prob']

    data_test_set = data.loc[test.index[0]:test.index[-1]]

    # Create returns for meta model
    meta_rets = (data_test_set['pred'] * data_test_set['target_rets']).shift(1)
    data_test_set['meta_rets'] = meta_rets
    data_test_set.dropna(inplace=True)
    # Test Cumrets
    test_cumrets = pd.DataFrame({'meta': ((data_test_set['meta_rets'] + 1).cumprod()),
                                 'primary': ((data_test_set['prets'] + 1).cumprod()),
                                 'BAH': ((data_test_set['rets'] + 1).cumprod())})

    # --- Statistics ---
    # -----------------------------------------------------------------------------------------------
    # Primary model stats
    brow = classification_stats(actual=test['target'], predicted=test['pmodel'], prefix='b', get_specificity=False)
    row = classification_stats(actual=y_test, predicted=test['pred'], prefix='m', get_specificity=True)
    # Concat data
    final_row = pd.concat([brow, row], axis=1)

    # ------------------------------------------------------------------
    # Add Strategy Metrics
    add_strat_metrics(row=final_row, rets=data_test_set['rets'], prefix='bah')
    add_strat_metrics(row=final_row, rets=data_test_set['prets'], prefix='p')
    add_strat_metrics(row=final_row, rets=data_test_set['meta_rets'], prefix='meta')
    final_row['num_samples'] = y_test.shape[0]

    # Comparison Metrics
    final_row['ip_sr'] = final_row['meta_sr'] - final_row['p_sr']
    final_row['ip_avg'] = final_row['meta_mean'] - final_row['p_mean']
    final_row['ip_std'] = final_row['meta_stdev'] - final_row['p_stdev']

    # Classification Metrics
    final_row['ip_recall'] = final_row['m_recall'] - final_row['b_recall']
    final_row['ip_prec'] = final_row['m_precision'] - final_row['b_precision']
    final_row['ip_acc'] = final_row['m_accuracy'] - final_row['b_accuracy']
    final_row['ip_f1'] = final_row['m_weighted_avg_f1'] - final_row['b_weighted_avg_f1']
    final_row['ip_auc'] = final_row['m_auc'] - final_row['b_auc']

    # --- Save Report ---
    # -----------------------------------------------------------------------------------------------

    # final_row.to_csv('hyp1.csv')

    data = pd.read_csv('../hyp1.csv', index_col=0)
    concat = pd.concat([data, final_row]).reset_index(drop=True)
    concat.to_csv('hyp1.csv')

    print(z)