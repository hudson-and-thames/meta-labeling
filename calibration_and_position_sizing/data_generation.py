"""
Module: functions to generate the single or dual regime data using an AR(3) process.
"""
import datetime as dt

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score

# Environment Variables
R1, R2, R3 = 0.032, 0.020, -0.042  # Initial r1, r2, r3
INNER_STEPS = 30
P1, P2, P3 = 0.25, -0.20, 0.35
PN1, PN2, PN3 = -0.25, 0.20, -0.35


def _gen_data(phi1, phi2, phi3, flag, stdev, drift, steps):
    """
    Helper function to generate a time series of returns data using an single AR(3) process.

    :param phi1: (float) Coef for rt_1
    :param phi2: (float) Coef for rt_2
    :param phi3: (float) Coef for rt_3
    :param flag: (int) {0, 1} indicates which regime the data is from.
    :param steps: (int) number of observations in the time series to generate.
    :param stdev: (float) standard deviation in the noise term. (White noise)
    :param drift: (float) Drift component of error term. (White noise drift)
    :return: (2 lists) Returns a list of rets and a list of flags.
    """
    # Initial values for lagged returns
    r1, r2, r3 = R1, R2, R3

    # Create data set based on AR(p)
    rets, flags = [], []
    for _ in range(0, steps):
        a = np.random.normal(loc=0, scale=stdev, size=1)  # white noise component using IBM weekly std
        rt = drift + phi1 * r1 + phi2 * r2 + phi3 * r3 + a
        flags.append(flag)
        rets.append(float(rt))

        # Update lagged returns
        r3, r2, r1 = r2, r1, rt

    return rets, flags


def _gen_dual_regime(steps, inner_steps, prob_switch, stdev):
    """
    Helper function to generate a time series of returns data using a dual AR(3) process.

    :param steps: (int) How many regimes to create. The total number of steps is: steps * inner_steps.
    :param inner_steps: (int) How many steps in a single regime.
    :param prob_switch: (float) Probability of sampling the next n-steps from the negative regime.
    :param stdev: (float) standard deviation in the noise term. (White noise)
    :return: (2 lists) Returns a list of rets and a list of flags.
    """
    rets, flags = [], []
    for _ in range(0, steps):

        rand = np.random.uniform()
        is_regime_two = rand < prob_switch

        if is_regime_two:
            # This negative regime has negative sign coefficients to the original
            rets_regime, flags_regime = _gen_data(phi1=PN1, phi2=PN2, phi3=PN3,
                                                  flag=1, steps=inner_steps,
                                                  stdev=stdev, drift=-0.0001)
        else:
            # Original Regime
            rets_regime, flags_regime = _gen_data(phi1=P1, phi2=P2, phi3=P3,
                                                  flag=0, steps=inner_steps,
                                                  stdev=stdev, drift=0.000)

        # Add to store
        rets.extend(rets_regime)
        flags.extend(flags_regime)
    return rets, flags


def single_regime(steps, stdev, drift):
    """
    Generate a time series sampled from a single regime.

    :param steps: (int) Total number of observations in the time series.
    :param stdev: (float) standard deviation in the noise term. (White noise)
    :param drift: (float) Drift component of error term. (White noise drift)
    :return: (DataFrame) Rets, with date index.
    """
    # Generate returns
    rets, _ = _gen_data(phi1=P1, phi2=P2, phi3=P3,
                        flag=0, steps=steps,
                        stdev=stdev, drift=drift)

    # Convert to DF and add dates
    data = pd.DataFrame({'rets': np.array(rets).flatten()})
    dates = pd.date_range(end=dt.datetime.now(), periods=steps, freq='d', normalize=True)
    data.index = dates

    return data


def dual_regime(total_steps, prob_switch, stdev):
    """
    Generate a time series sampled from a dual regime.

    :param total_steps: (int) Total number of observations in the time series.
    :param prob_switch: (float) Probability of sampling from the negative regime.
    :param stdev: (float) standard deviation in the noise term. (White noise)
    :return: (DataFrame) Rets and flags, with date index.
    """
    # Params
    inner_steps = INNER_STEPS
    steps = int(total_steps / inner_steps)  # Set steps so that total steps is reached

    # Gen dual regime data
    rets, flags = _gen_dual_regime(steps=steps, inner_steps=inner_steps,
                                   prob_switch=prob_switch, stdev=stdev)

    # Convert to DF
    date_range = pd.date_range(end=dt.datetime.now(),
                               periods=steps * inner_steps,
                               freq='d', normalize=True)
    data = pd.DataFrame({'rets': np.array(rets).flatten(), 'flags': flags},
                        index=date_range)
    return data


def prep_data(data, with_flags):
    """
    Create dependant and independent variables, add a primary model's side forecast, and return the
    meta_data for training in the secondary model and the new data DataFrame without the sub setting
    (meta_data).

    :param data: (DataFrame) Data to which to prep.
    :param with_flags: (Bool) If the regime flag should be added or not.
    :return: (2, DataFrames) meta_data and new data.
    """

    # Set target variable
    data['target'] = data['rets'].apply(lambda x: 0 if x < 0 else 1).shift(-1)  # Binary classification

    # Create data set
    data['target_rets'] = data['rets'].shift(-1)  # Add target rets for debugging
    data.dropna(inplace=True)

    # Auto-correlation trading rule: trade sign of previous day.
    data['pmodel'] = data['rets'].apply(lambda x: 1 if x > 0.0 else 0)

    # Strategy daily returns
    data['prets'] = (data['pmodel'] * data['target_rets']).shift(1)  # Lag by 1 to remove look ahead and align dates
    data.dropna(inplace=True)

    # Add lag rets 2 and 3 for Logistic regression
    data['rets2'] = data['rets'].shift(1)
    data['rets3'] = data['rets'].shift(2)

    # Add Regime indicator if with_flags is on
    if with_flags:
        # Add Regime features, lagged by 5 days.
        # We lag it to imitate the lagging nature of rolling statistics.
        data['regime'] = data['flags'].shift(5)

    # Data used to train model
    model_data = data[data['pmodel'] == 1]

    # Apply labels to total data set
    # In this setting the target for the pmodel is the meta_labels when you filter by only pmodel=1
    model_data.dropna(inplace=True)

    return model_data, data


def classification_stats(actual, predicted, prefix, get_specificity):
    # Create Report
    report = classification_report(actual, predicted, output_dict=True,
                                   labels=[0, 1], zero_division=0)
    # Extract (long only) metrics
    report['1'][prefix + '_accuracy'] = report['accuracy']
    report['1'][prefix + '_auc'] = roc_auc_score(actual, predicted)
    report['1'][prefix + '_macro_avg_f1'] = report['macro avg']['f1-score']
    report['1'][prefix + '_weighted_avg'] = report['weighted avg']['f1-score']

    # To DataFrame
    row = pd.DataFrame.from_dict(report['1'], orient='index').T
    row.columns = [prefix + '_precision', prefix + '_recall', prefix + '_f1_score',
                    prefix + '_support', prefix + '_accuracy', prefix + '_auc',
                    prefix + '_macro_avg_f1', prefix + '_weighted_avg_f1']

    # Add Specificity
    if get_specificity:
        row[prefix + '_specificity'] = report['0']['recall']
    else:
        row[prefix + '_specificity'] = 0

    return row


def strat_metrics(rets):
    avg = rets.mean()
    stdev = rets.std()
    sharpe_ratio = avg / stdev * np.sqrt(252)
    return avg, stdev, sharpe_ratio


def add_strat_metrics(row, rets, prefix):
    # Get metrics
    avg, stdev, sharpe_ratio = strat_metrics(rets)
    # Save to row
    row[prefix + '_mean'] = avg
    row[prefix + '_stdev'] = stdev
    row[prefix + '_sr'] = sharpe_ratio
