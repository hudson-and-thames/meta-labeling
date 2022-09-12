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

for z in range(0, 2, 1):

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
    # logistic regression
    meta_model_regime = LogisticRegression(random_state=0, penalty='none')

    # meta_model_regime = svm.SVC(random_state=0,probability=True)
    # meta_model_regime.fit(X_train_regime_scaled, y_train)
    # meta_model_regime = RandomForestClassifier(n_estimators=5)
    # meta_model_regime = LinearDiscriminantAnalysis()


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
    # A1 - Linear Scaling [Checked]
    # ----------------------------------------------------------------

    # Linear scaling: min, max from train, p from test.
    linear_size_test = (prob_test[prob_test > 0.5] - prob_train[prob_train > 0.5].min()) / (
            prob_train[prob_train > 0.5].max() - prob_train[prob_train > 0.5].min())
    linear_size_iso_test = (prob_isotonic_test[prob_isotonic_test > 0.5] - prob_isotonic_train[
        prob_train > 0.5].min()) / (
                                   prob_isotonic_train[prob_train > 0.5].max() -
                                   prob_isotonic_train[prob_train > 0.5].min())

    # Assign position sizes
    data_test_set['lin_size'] = 0
    data_test_set['lin_iso_size'] = 0
    data_test_set.loc[linear_size_test.index, 'lin_size'] = linear_size_test
    data_test_set.loc[linear_size_iso_test.index, 'lin_iso_size'] = linear_size_iso_test

    # Get daily rets of the strategy (vectorised backtest), shifted by 1 to remove lookahead bias
    data_test_set['lin_rets'] = (data_test_set['lin_size'] * target_test).shift(1)
    data_test_set['lin_iso_rets'] = (data_test_set['lin_iso_size'] * target_test).shift(1)


    # ----------------------------------------------------------------
    # A2 - Optimal linear fit [Checked]
    # ----------------------------------------------------------------

    def check_stats(rets):
        if np.std(rets) == 0.0:
            stdev = 10000
        else:
            stdev = np.std(rets)

        if (np.mean(rets) <= 0.00001) and (np.mean(rets) >= -0.00001):
            mean = -10000
        else:
            mean = np.mean(rets)

        return mean, stdev


    def target_linear(x):
        # Linear function
        f = lambda p: min(max(x[0] * p + x[1], 0), 1)
        f = np.vectorize(f)
        # Backtest
        rets = f(prob_train[prob_train > 0.5]) * target_train_p[prob_train > 0.5]
        # Solve for no positions taken
        mean, stdev = check_stats(rets)
        # Sharpe Ratio
        sr = mean / stdev
        return -sr


    def target_linear_iso(x):
        # Linear function
        f = lambda p: min(max(x[0] * p + x[1], 0), 1)
        f = np.vectorize(f)
        # Backtest
        rets = f(prob_isotonic_train[prob_isotonic_train > 0.5]) * target_train_p[prob_isotonic_train > 0.5]
        # Solve for no positions taken
        mean, stdev = check_stats(rets)
        # Sharp Ratio
        sr = mean / stdev
        return -sr


    # Train model on training data
    x0 = np.array([1, 0])
    res = minimize(target_linear, x0, method='nelder-mead', options={'xatol': 1e-8, 'disp': False})
    model = res.x
    # Get test position sizes
    lops_size = model[0] * prob_test + model[1]

    # Scale size [0, 1]
    lops_size[lops_size > 1] = 1
    lops_size[lops_size < 0] = 0

    # Assign position sizes
    data_test_set['lop_size'] = 0
    data_test_set.loc[lops_size.index, 'lop_size'] = lops_size

    # Get daily rets
    data_test_set['lop_rets'] = (data_test_set['lop_size'] * target_test).shift(1)

    # Do ISO version
    # Train model on training data
    x0 = np.array([1, 0])
    res = minimize(target_linear_iso, x0, method='nelder-mead', options={'xatol': 1e-8, 'disp': False})
    model = res.x
    # Get test position sizes
    lops_iso_size = model[0] * prob_isotonic_test + model[1]

    # Scale size [0, 1]
    lops_iso_size[lops_iso_size > 1] = 1
    lops_iso_size[lops_iso_size < 0] = 0

    # Assign position sizes
    data_test_set['lops_iso_size'] = 0
    data_test_set.loc[lops_iso_size.index, 'lops_iso_size'] = lops_iso_size

    # Get daily rets
    data_test_set['lop_iso_rets'] = (data_test_set['lops_iso_size'] * target_test).shift(1)


    # ----------------------------------------------------------------
    # B1 - de Prado's Bet Sizing [Checked] A
    # ----------------------------------------------------------------

    def de_prado_bet_size(prob_series, clip=True):
        # Can't compute for p = 1 or p = 0, leads to inf.
        p = prob_series.copy()
        p[p == 1] = 0.99999
        p[p == 0] = 0.00001

        # Getting max value from training set
        num_classes = 2
        dp_sizes = (p - 1 / num_classes) / ((p * (1 - p)) ** 0.5)

        dp_t_sizes = dp_sizes.apply(lambda s: norm.cdf(s))
        dp_bet_sizes = dp_t_sizes

        # no sigmoid function, only clipping?
        dp_bet_sizes[dp_bet_sizes < 0.5] = 0

        return dp_bet_sizes


    # Get sizes for test data
    dp_size = de_prado_bet_size(prob_test, clip=True)
    dp_size_iso = de_prado_bet_size(prob_isotonic_test, clip=True)

    # Assign position sizes
    data_test_set['dp_size'] = 0
    data_test_set['dp_iso_size'] = 0
    data_test_set.loc[dp_size.index, 'dp_size'] = dp_size
    data_test_set.loc[dp_size_iso.index, 'dp_iso_size'] = dp_size_iso

    # Get daily rets
    data_test_set['dp_rets'] = (data_test_set['dp_size'] * target_test).shift(1)
    data_test_set['dp_iso_rets'] = (data_test_set['dp_iso_size'] * target_test).shift(1)

    if (data_test_set['dp_iso_rets'].std() == 0) or (data_test_set['dp_iso_rets'].mean() == 0):
        print('DP')
        print(dp_size_iso.mean())

    # ----------------------------------------------------------------
    # B2 - ECDF [Checked] A
    # ----------------------------------------------------------------

    # Fit ECDF on training data for prob greater than > 0
    ecdf = ECDF(prob_train[prob_train > 0.5])
    ecdf_iso = ECDF(prob_isotonic_train[prob_isotonic_train > 0.5])

    # ECDF Position Sizing on test data
    ecdf_size = prob_test.apply(lambda x: ecdf(x) if x > 0.5 else 0)
    ecdf_size_iso = prob_isotonic_test.apply(lambda x: ecdf_iso(x) if x > 0.5 else 0)

    # Daily data update with position sizes
    data_test_set['ecdf_size'] = 0
    data_test_set['ecdf_size_iso'] = 0
    data_test_set.loc[ecdf_size.index, 'ecdf_size'] = ecdf_size
    data_test_set.loc[ecdf_size_iso.index, 'ecdf_size_iso'] = ecdf_size_iso

    # Backtest
    data_test_set['ecdf_rets'] = (data_test_set['ecdf_size'] * target_test).shift(1)
    data_test_set['ecdf_iso_rets'] = (data_test_set['ecdf_size_iso'] * target_test).shift(1)


    # ----------------------------------------------------------------
    # B3 - Sigmoid optimal fit [Checked]
    # ----------------------------------------------------------------

    def target_sigmoid(x):
        # Apply sigmoid position sizing
        f = lambda p: min(max(1 / (1 + np.exp(-x[0] * p - x[1])), 0), 1)
        f = np.vectorize(f)

        # Backtest + sharpe ratio
        rets = f(prob_train[prob_train > 0.5]) * target_train_p[prob_train > 0.5]
        # Solve for no positions taken
        mean, stdev = check_stats(rets)
        # Sharpe Ratio
        sharp_ratio = mean / stdev
        return -sharp_ratio


    def target_iso_sigmoid(x):
        # Apply sigmoid position sizing
        f = lambda p: min(max(1 / (1 + np.exp(-x[0] * p - x[1])), 0), 1)
        f = np.vectorize(f)

        # Backtest + sharpe ratio
        rets = f(prob_isotonic_train[prob_isotonic_train > 0.5]) * target_train_p[prob_isotonic_train > 0.5]
        # Solve for no positions taken
        mean, stdev = check_stats(rets)
        # Sharpe Ratio
        sharp_ratio = mean / stdev
        return -sharp_ratio


    # Train model on training data
    x0 = np.array([1, 0])
    res = minimize(target_sigmoid, x0, method='nelder-mead', options={'xatol': 1e-8, 'disp': False})
    model = res.x
    # Get size on test
    sig_size = 1 / (1 + np.exp(-model[0] * prob_test - model[1]))

    sig_size = sig_size[prob_test > 0.5]

    # Train model on training data
    x0 = np.array([1, 0])
    res = minimize(target_iso_sigmoid, x0, method='nelder-mead', options={'xatol': 1e-8, 'disp': False})
    model = res.x
    # Get size on test
    sig_iso_size = 1 / (1 + np.exp(-model[0] * prob_isotonic_test - model[1]))

    sig_iso_size = sig_iso_size[prob_isotonic_test > 0.5]

    # Assign position sizes
    data_test_set['sop_size'] = 0
    data_test_set.loc[sig_size.index, 'sop_size'] = sig_size
    data_test_set['sop_size_iso'] = 0
    data_test_set.loc[sig_iso_size.index, 'sop_size_iso'] = sig_iso_size

    # Get daily rets
    data_test_set['sop_rets'] = (data_test_set['sop_size'] * target_test).shift(1)
    data_test_set['sop_rets_iso'] = (data_test_set['sop_size_iso'] * target_test).shift(1)

    if (data_test_set['sop_rets_iso'].std() == 0) or (data_test_set['sop_rets_iso'].mean() == 0):
        print('SOP_ISO')
        print(sig_iso_size.mean())

    if (data_test_set['sop_rets'].std() == 0) or (data_test_set['sop_rets'].mean() == 0):
        print('SOP')
        print(sig_size.mean())

    # ----------------------------------------------------------------
    # A3 - Model output size
    # ----------------------------------------------------------------
    # Normal bet sizing from probabilities

    # Position sizes on test data
    model_output = prob_test
    model_output_iso = prob_isotonic_test

    # All or nothing
    model_output[model_output >= 0.5] = prob_test
    model_output[model_output < 0.5] = 0

    model_output_iso[model_output_iso >= 0.5] = prob_isotonic_test
    model_output_iso[model_output_iso < 0.5] = 0

    # Assign position sizes
    data_test_set['norm_prob_size'] = 0
    data_test_set['norm_prob_iso_size'] = 0
    data_test_set.loc[model_output.index, 'norm_prob_size'] = model_output
    data_test_set.loc[model_output_iso.index, 'norm_prob_iso_size'] = prob_isotonic_test

    # Get daily rets
    data_test_set['norm_prob_rets'] = (data_test_set['norm_prob_size'] * target_test).shift(1)
    data_test_set['norm_prob_iso_rets'] = (data_test_set['norm_prob_iso_size'] * target_test).shift(1)

    # ----------------------------------------------------------------
    # A4 - All-or-nothing [Checked]
    # ----------------------------------------------------------------

    # Position sizes on test data
    all_or_nothing = prob_test
    all_or_nothing_isotonic = prob_isotonic_test

    # All or nothing
    all_or_nothing[all_or_nothing >= 0.5] = 1
    all_or_nothing[all_or_nothing < 0.5] = 0
    all_or_nothing_isotonic[all_or_nothing_isotonic >= 0.5] = 1
    all_or_nothing_isotonic[all_or_nothing_isotonic < 0.5] = 0

    # Assign position sizes
    data_test_set['all_or_nothing_size'] = 0
    data_test_set['all_or_nothing_iso_size'] = 0
    data_test_set.loc[all_or_nothing.index, 'all_or_nothing_size'] = all_or_nothing
    data_test_set.loc[all_or_nothing_isotonic.index, 'all_or_nothing_iso_size'] = all_or_nothing_isotonic

    # Get daily rets
    data_test_set['all_or_nothing_rets'] = (data_test_set['all_or_nothing_size'] * target_test).shift(1)
    data_test_set['all_or_nothing_iso_rets'] = (data_test_set['all_or_nothing_iso_size'] * target_test).shift(1)

    data_test_set.to_csv('data_test_set.csv')
    # ----------------------------------------------------------------
    # --- Statistics ---
    # -------------------------------------

    # drop Nas
    data_test_set.dropna(inplace=True)

    test_cumrets = pd.DataFrame({'norm': ((data_test_set['norm_prob_rets'] + 1).cumprod()),
                                 'norm_iso': ((data_test_set['norm_prob_iso_rets'] + 1).cumprod()),

                                 'aon': ((data_test_set['all_or_nothing_rets'] + 1).cumprod()),
                                 'aon_iso': ((data_test_set['all_or_nothing_iso_rets'] + 1).cumprod()),

                                 'lin': ((data_test_set['lin_rets'] + 1).cumprod()),
                                 'lin_iso': ((data_test_set['lin_iso_rets'] + 1).cumprod()),

                                 'lop': ((data_test_set['lop_rets'] + 1).cumprod()),
                                 'lop_iso': ((data_test_set['lop_iso_rets'] + 1).cumprod()),

                                 'dp': ((data_test_set['dp_rets'] + 1).cumprod()),
                                 'dp_iso': ((data_test_set['dp_iso_rets'] + 1).cumprod()),

                                 'ecdf': ((data_test_set['ecdf_rets'] + 1).cumprod()),
                                 'ecdf_iso': ((data_test_set['ecdf_iso_rets'] + 1).cumprod()),

                                 'sop': ((data_test_set['sop_rets'] + 1).cumprod()),
                                 'sop_iso': ((data_test_set['sop_rets_iso'] + 1).cumprod()),

                                 'primary': ((data_test_set['prets'] + 1).cumprod()),
                                 'BAH': ((data_test_set['rets'] + 1).cumprod())})

    sr = {'norm_sr': data_test_set['norm_prob_rets'].mean() / data_test_set['norm_prob_rets'].std() * np.sqrt(252),
          'norm_iso_sr': data_test_set['norm_prob_iso_rets'].mean() / data_test_set[
              'norm_prob_iso_rets'].std() * np.sqrt(252),

          'aon_sr': data_test_set['all_or_nothing_rets'].mean() / data_test_set['all_or_nothing_rets'].std() * np.sqrt(
              252),
          'aon_iso_sr': data_test_set['all_or_nothing_iso_rets'].mean() / data_test_set[
              'all_or_nothing_iso_rets'].std() * np.sqrt(252),

          'lin_sr': data_test_set['lin_rets'].mean() / data_test_set['lin_rets'].std() * np.sqrt(252),
          'lin_iso_sr': data_test_set['lin_iso_rets'].mean() / data_test_set['lin_iso_rets'].std() * np.sqrt(252),

          'lop_sr': data_test_set['lop_rets'].mean() / data_test_set['lop_rets'].std() * np.sqrt(252),
          'lop_iso_sr': data_test_set['lop_iso_rets'].mean() / data_test_set['lop_iso_rets'].std() * np.sqrt(252),

          'dp_sr': data_test_set['dp_rets'].mean() / data_test_set['dp_rets'].std() * np.sqrt(252),
          'dp_iso_sr': data_test_set['dp_iso_rets'].mean() / data_test_set['dp_iso_rets'].std() * np.sqrt(252),

          'ecdf_sr': data_test_set['ecdf_rets'].mean() / data_test_set['ecdf_rets'].std() * np.sqrt(252),
          'ecdf_iso_sr': data_test_set['ecdf_iso_rets'].mean() / data_test_set['ecdf_iso_rets'].std() * np.sqrt(252),

          'sop_sr': data_test_set['sop_rets'].mean() / data_test_set['sop_rets'].std() * np.sqrt(252),
          'sop_iso_sr': data_test_set['sop_rets_iso'].mean() / data_test_set['sop_rets_iso'].std() * np.sqrt(252),

          'primary_sr': data_test_set['prets'].mean() / data_test_set['prets'].std() * np.sqrt(252),
          'BAH_sr': data_test_set['rets'].mean() / data_test_set['rets'].std() * np.sqrt(252)}

    mean = {'norm_avg': (1 + data_test_set['norm_prob_rets'].mean()) ** 252 - 1,
            'norm_iso_avg': (1 + data_test_set['norm_prob_iso_rets'].mean()) ** 252 - 1,

            'aon_avg': (1 + data_test_set['all_or_nothing_rets'].mean()) ** 252 - 1,
            'aon_iso_avg': (1 + data_test_set['all_or_nothing_iso_rets'].mean()) ** 252 - 1,

            'lin_avg': (1 + data_test_set['lin_rets'].mean()) ** 252 - 1,
            'lin_iso_avg': (1 + data_test_set['lin_iso_rets'].mean()) ** 252 - 1,

            'lin_avg': (1 + data_test_set['lin_rets'].mean()) ** 252 - 1,
            'lin_iso_avg': (1 + data_test_set['lin_iso_rets'].mean()) ** 252 - 1,

            'lop_avg': (1 + data_test_set['lop_rets'].mean()) ** 252 - 1,
            'lop_iso_avg': (1 + data_test_set['lop_iso_rets'].mean()) ** 252 - 1,

            'dp_avg': (1 + data_test_set['dp_rets'].mean()) ** 252 - 1,
            'dp_iso_avg': (1 + data_test_set['dp_iso_rets'].mean()) ** 252 - 1,

            'ecdf_avg': (1 + data_test_set['ecdf_rets'].mean()) ** 252 - 1,
            'ecdf_iso_avg': (1 + data_test_set['ecdf_iso_rets'].mean()) ** 252 - 1,

            'sop_avg': (1 + data_test_set['sop_rets'].mean()) ** 252 - 1,
            'sop_iso_avg': (1 + data_test_set['sop_rets_iso'].mean()) ** 252 - 1,

            'primary_avg': (1 + data_test_set['prets'].mean()) ** 252 - 1,
            'BAH_avg': (1 + data_test_set['rets'].mean()) ** 252 - 1}

    if (mean['sop_avg'] == 0) or (mean['sop_iso_avg'] == 0):
        print('SOP')
        print(sig_size.mean())

    stdev = {'norm_std': data_test_set['norm_prob_rets'].std() * np.sqrt(252),
             'norm_iso_std': data_test_set['norm_prob_iso_rets'].std() * np.sqrt(252),

             'aon_std': data_test_set['all_or_nothing_rets'].std() * np.sqrt(252),
             'aon_iso_std': data_test_set['all_or_nothing_iso_rets'].std() * np.sqrt(252),

             'lin_std': data_test_set['lin_rets'].std() * np.sqrt(252),
             'lin_iso_std': data_test_set['lin_iso_rets'].std() * np.sqrt(252),

             'lop_std': data_test_set['lop_rets'].std() * np.sqrt(252),
             'lop_iso_std': data_test_set['lop_iso_rets'].std() * np.sqrt(252),

             'dp_std': data_test_set['dp_rets'].std() * np.sqrt(252),
             'dp_iso_std': data_test_set['dp_iso_rets'].std() * np.sqrt(252),

             'ecdf_std': data_test_set['ecdf_rets'].std() * np.sqrt(252),
             'ecdf_iso_std': data_test_set['ecdf_iso_rets'].std() * np.sqrt(252),

             'sop_std': data_test_set['sop_rets'].std() * np.sqrt(252),
             'sop_iso_std': data_test_set['sop_rets_iso'].std() * np.sqrt(252),

             'primary_std': data_test_set['prets'].std() * np.sqrt(252),
             'BAH_std': data_test_set['rets'].std() * np.sqrt(252)}

    # Compute Max DDs
    mdds = qs.stats.max_drawdown(test_cumrets)
    # Check for negative values (MDDs) and correct
    clean_mdds = {}
    for ind, val in mdds.iteritems():
        new_name = ind + '_mdd'

        if val == 0.0:
            val = -(1 - test_cumrets[ind][-1])
            mdds[ind] = val
        clean_mdds[new_name] = val

    final = {**sr, **mean, **stdev, **clean_mdds}
    final_row = pd.DataFrame(final.values(), index=final.keys()).T

    # --- Save Report ---
    # ------------------------------------------
    # Save results to csv
    if z == 0:
        final_row.to_csv('results.csv')
        data_final = pd.read_csv('results.csv', index_col=0)
    else:
        data_final = pd.read_csv('results.csv', index_col=0)
        concat = pd.concat([data_final, final_row]).reset_index(drop=True)
        concat.to_csv('results.csv')

    print(z)
    data_test_set[['pred', 'target_rets']].plot()
