# Imports - not filtered yet

# Part 1
import numpy as np
import pandas as pd

# Part 2
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
import quantstats as qs

# Part 3
from scipy.stats import norm
from statsmodels.distributions.empirical_distribution import ECDF

# Other
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from scipy.optimize import minimize

# My imports
from data_generation import single_regime, dual_regime, prep_data

# Silence some warnings
pd.options.mode.chained_assignment = None  # default='warn'

from sklearn.metrics import make_scorer


# Make mean ABS scorer
# TODO: COme back to this
def mean_abs_error(y_true, y_predict):
    return np.abs(np.array(y_true) - np.array(y_predict)).mean()


mean_abs_scorer = make_scorer(mean_abs_error, greater_is_better=False)

# clear results
filename = "results.csv"
# opening the file with w+ mode truncates the file
f = open(filename, "w+")
f.close()

# --- Computation Part ---
# ------------------------------------------

for z in range(0, 1000, 1):

    # --- Data Prep ---
    # -------------------------

    # Constants
    steps = 10000
    prob_switch = 0.20
    stdev = 0.014543365294448746  # About the same as IBM stdev

    # Create dual data set
    data = dual_regime(total_steps=steps, prob_switch=prob_switch, stdev=stdev)

    # Prep data, add primary model, get meta_labels
    model_data, data = prep_data(data=data, with_flags=True)

    # --- Modeling ---
    # --------------------------
    # Train test split
    train, test = train_test_split(model_data, test_size=0.4, shuffle=False)

    X_train_regime = train[['rets', 'rets2', 'rets3', 'regime']]
    X_test_regime = test[['rets', 'rets2', 'rets3', 'regime']]

    y_train = train['target']
    y_test = test['target']

    # Add standardScalar as a best practice although in this setting its not really needed.
    # Logistic regression is a convex optimisation problem and the global minima is always found.
    # We only scale r1, 2, 3 - regime is left unscaled.
    scaler = StandardScaler()
    X_train_regime_scaled = scaler.fit_transform(X_train_regime[['rets', 'rets2', 'rets3']])
    regime = X_train_regime['regime'].values.reshape((-1, 1))
    X_train_regime_scaled = np.append(X_train_regime_scaled, regime, axis=1)

    # Test data
    X_test_regime_scaled = scaler.transform(X_test_regime[['rets', 'rets2', 'rets3']])
    regime = X_test_regime['regime'].values.reshape((-1, 1))
    X_test_regime_scaled = np.append(X_test_regime_scaled, regime, axis=1)

    # =======================================================================================
    # Calibration
    # =======================================================================================

    # Train model (FP)
    meta_model_regime = LogisticRegression(random_state=0, penalty='none')
    # meta_model_regime.fit(X_train_regime_scaled, y_train)

    # Create calibrator which will use the base logistic model from above
    calibrated_model_isotonic = CalibratedClassifierCV(base_estimator=meta_model_regime,
                                                       method='isotonic', cv=5, n_jobs=-1, ensemble=True)
    calibrated_model_isotonic.fit(X_train_regime_scaled, y_train)

    # Get iso model train probs, and return pandas Series with index as date.
    prob_isotonic_train = calibrated_model_isotonic.predict_proba(X_train_regime_scaled)[:, 1]
    prob_isotonic_train = pd.Series(prob_isotonic_train, index=X_train_regime.index)

    # Get unscaled prob from secondary model, i.e., the Logistic Regression.
    # Get base model train scores
    prob_train = np.array([cmodel.base_estimator.predict_proba(X_train_regime_scaled)[:, 1]
                           for cmodel in calibrated_model_isotonic.calibrated_classifiers_]).mean(axis=0)
    prob_train = pd.Series(prob_train, index=X_train_regime.index)

    # Check that the base model and calibrated models align
    assert (prob_train.shape == prob_isotonic_train.shape)

    # Get iso model test scores, and return pandas Series with index as date.
    prob_isotonic_test = calibrated_model_isotonic.predict_proba(X_test_regime_scaled)[:, 1]
    prob_isotonic_test = pd.Series(prob_isotonic_test, index=X_test_regime.index)

    # Get unscaled prob from secondary model, i.e., the Logistic Regression.
    # Get base model train scores
    prob_test = np.array([cmodel.base_estimator.predict_proba(X_test_regime_scaled)[:, 1]
                          for cmodel in calibrated_model_isotonic.calibrated_classifiers_]).mean(axis=0)
    prob_test = pd.Series(prob_test, index=X_test_regime.index)

    # Check that the base model and calibrated models align
    assert (prob_test.shape == prob_isotonic_test.shape)

    # =======================================================================================
    # Add the calibrated and raw probabilities plus the pred to the train and test data sets.

    # # Exp 1: Trade only if P > 0.5
    # # Only take positions with a positive expected payout, i.e., greater than 50% success.
    # prob_train[prob_train<0.5] = 0
    # prob_test[prob_test < 0.5] = 0
    # prob_isotonic_train[prob_isotonic_train < 0.5] = 0
    # prob_isotonic_test[prob_isotonic_test < 0.5] = 0
    # # /End Exp 1

    # Add proba [0, 1]
    train['prob'] = prob_train
    train['prob_iso'] = prob_isotonic_train
    test['prob'] = prob_test
    test['prob_iso'] = prob_isotonic_test

    # Add Predictions {0, 1}
    train['pred'] = 0
    train['pred_iso'] = 0
    train.loc[prob_train > 0.5, 'pred'] = 1
    train.loc[prob_isotonic_train > 0.5, 'pred_iso'] = 1
    test['pred'] = 0
    test['pred_iso'] = 0
    test.loc[prob_test > 0.5, 'pred'] = 1
    test.loc[prob_isotonic_test > 0.5, 'pred_iso'] = 1

    # --- Prep Strategy Data ---
    # ---------------------------------
    # Save forecasts to original data
    # Set new columns
    data['pred'] = 0
    data['prob'] = 0
    data['prob_iso'] = 0
    data['pred_iso'] = 0

    # Assign column values
    data.loc[train.index, 'pred'] = train['pred']
    data.loc[train.index, 'prob'] = train['prob']
    data.loc[train.index, 'pred_iso'] = train['pred_iso']
    data.loc[train.index, 'prob_iso'] = train['prob_iso']
    data.loc[test.index, 'pred'] = test['pred']
    data.loc[test.index, 'prob'] = test['prob']
    data.loc[test.index, 'pred_iso'] = test['pred_iso']
    data.loc[test.index, 'prob_iso'] = test['prob_iso']

    # Subset train data
    data_train_set = data.loc[train.index[0]:train.index[-1]]
    data_test_set = data.loc[test.index[0]:test.index[-1]]

    # Save this to CSV for use in the Kelly analysis notebook
    data_train_set.to_csv('train.csv')
    data_test_set.to_csv('test.csv')

    # ------------------------------------------------------------------------------------------------------------
    #                                               --- Bet Sizing ---
    # ------------------------------------------------------------------------------------------------------------

    # Get target rets series
    target_train = data_train_set['target_rets']
    target_train_p = train['target_rets']
    target_test = data_test_set['target_rets']
    target_test_p = test['target_rets']

    # ----------------------------------------------------------------
    # A7 - All-or-nothing [Checked]
    # ----------------------------------------------------------------

    # Position sizes on test data
    sharpe_r = {}
    mean_r = {}
    std_dev = {}
    mmd = {}

    # normal
    for i in range(35, 70, 2):
        all_or_nothing = prob_test
        all_or_nothing_isotonic = prob_isotonic_test

        threshold = i / 100
        # All or nothing
        all_or_nothing = all_or_nothing[all_or_nothing > threshold]
        all_or_nothing = all_or_nothing.apply(lambda x: 1 if x >= threshold else 0)

        # Assign position sizes
        data_test_set['all_or_nothing_size'] = 0

        data_test_set.loc[prob_test.index, 'all_or_nothing_size'] = all_or_nothing

        # Get daily rets
        data_test_set['all_or_nothing_rets'] = (data_test_set['all_or_nothing_size'] * target_test).shift(1)

        sr = {'{}_aon_sr'.format(threshold): data_test_set['all_or_nothing_rets'].mean() / data_test_set[
            'all_or_nothing_rets'].std() * np.sqrt(
            252)}

        mean = {'{}_aon_avg'.format(threshold): (1 + data_test_set['all_or_nothing_rets'].mean()) ** 252 - 1}

        stdev = {'{}_aon_std'.format(threshold): data_test_set['all_or_nothing_rets'].std() * np.sqrt(252)}

        mm = {'{}_aon_mm'.format(threshold): qs.stats.max_drawdown(
            pd.DataFrame((data_test_set['all_or_nothing_rets'] + 1).cumprod()).dropna())['all_or_nothing_rets']}

        sharpe_r.update(sr)
        mean_r.update(mean)
        std_dev.update(stdev)
        mmd.update(mm)
        # final_row = pd.DataFrame(final.values(), index=final.keys()).T

    # calibrated
    for i in range(35, 70, 2):
        all_or_nothing = prob_test
        all_or_nothing_isotonic = prob_isotonic_test

        threshold = i / 100

        all_or_nothing_isotonic[all_or_nothing_isotonic >= threshold]
        all_or_nothing_isotonic = all_or_nothing_isotonic.apply(lambda x: 1 if x >= threshold else 0)

        # Assign position sizes
        data_test_set['all_or_nothing_iso_size'] = 0
        data_test_set.loc[all_or_nothing_isotonic.index, 'all_or_nothing_iso_size'] = all_or_nothing_isotonic

        # Get daily rets
        data_test_set['all_or_nothing_iso_rets'] = (data_test_set['all_or_nothing_iso_size'] * target_test).shift(1)

        sr = {
            '{}_aon_iso_sr'.format(threshold): data_test_set['all_or_nothing_iso_rets'].mean() / data_test_set[
                'all_or_nothing_iso_rets'].std() * np.sqrt(252)}

        mean = {
            '{}_aon_iso_avg'.format(threshold): (1 + data_test_set['all_or_nothing_iso_rets'].mean()) ** 252 - 1}

        stdev = {
            '{}_aon_iso_std'.format(threshold): data_test_set['all_or_nothing_iso_rets'].std() * np.sqrt(252)}

        mm = {'{}_aon_iso_mm'.format(threshold): qs.stats.max_drawdown(
            pd.DataFrame((data_test_set['all_or_nothing_iso_rets'] + 1).cumprod()))['all_or_nothing_iso_rets']}

        # Compute Max DDs
        # Check for negative values (MDDs) and correct
        sharpe_r.update(sr)
        mean_r.update(mean)
        std_dev.update(stdev)
        mmd.update(mm)


        # final_row = pd.DataFrame(final.values(), index=final.keys()).T

    final = {**sharpe_r, **mean_r, **stdev, **mmd}
    final_row = pd.DataFrame(final.values(), index=final.keys()).T
    # --- Save Report ---
    # ------------------------------------------
    # Save results to csv
    if z == 0:
        final_row.to_csv('aon.csv')
        data_final = pd.read_csv('aon.csv', index_col=0)
    else:
        data_final = pd.read_csv('aon.csv', index_col=0)
        concat = pd.concat([data_final, final_row]).reset_index(drop=True)
        concat.to_csv('aon.csv')

    print('Simulation ',z)
