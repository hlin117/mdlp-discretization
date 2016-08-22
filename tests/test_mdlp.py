from sklearn.utils.testing import assert_array_equal
from sklearn.utils.testing import assert_equal
from sklearn.utils.testing import assert_almost_equal
from sklearn.utils.testing import assert_array_almost_equal

import sys
sys.path.insert(0, "..")

from discretization import MDLP
from _mdlp import slice_entropy
import pandas as pd
import numpy as np

def test_slice_entropy():

    y = np.array([0, 0, 0, 1, 1, 0, 1, 3, 1, 1])

    entropy1, k1 = slice_entropy(y, 0, 3)
    entropy2, k2 = slice_entropy(y, 3, 10)

    assert_equal(entropy1, 0, "Entropy was not calculated correctly.")
    assert_equal(k1, 1, "Incorrect number of classes found.")
    assert_almost_equal(entropy2, 0.796311640173813,
                        err_msg="Entropy was not calculated correctly.")
    assert_equal(k2, 3, "Incorrect number of classes found.")


def test_mdlp_iris():
    from sklearn.datasets import load_iris

    iris = load_iris()
    X = iris.data
    y = iris.target
    mdlp = MDLP(shuffle=False)
    transformed = mdlp.fit_transform(X, y)

    expected = [[ 0.,  1.,  0.,  0.],
        [ 0.,  0.,  0.,  0.],
        [ 0.,  1.,  0.,  0.],
        [ 0.,  1.,  0.,  0.],
        [ 0.,  1.,  0.,  0.],
        [ 0.,  1.,  0.,  0.],
        [ 0.,  1.,  0.,  0.],
        [ 0.,  1.,  0.,  0.],
        [ 0.,  0.,  0.,  0.],
        [ 0.,  1.,  0.,  0.],
        [ 0.,  1.,  0.,  0.],
        [ 0.,  1.,  0.,  0.],
        [ 0.,  0.,  0.,  0.],
        [ 0.,  0.,  0.,  0.],
        [ 1.,  1.,  0.,  0.],
        [ 1.,  1.,  0.,  0.],
        [ 0.,  1.,  0.,  0.],
        [ 0.,  1.,  0.,  0.],
        [ 1.,  1.,  0.,  0.],
        [ 0.,  1.,  0.,  0.],
        [ 0.,  1.,  0.,  0.],
        [ 0.,  1.,  0.,  0.],
        [ 0.,  1.,  0.,  0.],
        [ 0.,  1.,  0.,  0.],
        [ 0.,  1.,  0.,  0.],
        [ 0.,  0.,  0.,  0.],
        [ 0.,  1.,  0.,  0.],
        [ 0.,  1.,  0.,  0.],
        [ 0.,  1.,  0.,  0.],
        [ 0.,  1.,  0.,  0.],
        [ 0.,  1.,  0.,  0.],
        [ 0.,  1.,  0.,  0.],
        [ 0.,  1.,  0.,  0.],
        [ 0.,  1.,  0.,  0.],
        [ 0.,  1.,  0.,  0.],
        [ 0.,  1.,  0.,  0.],
        [ 0.,  1.,  0.,  0.],
        [ 0.,  1.,  0.,  0.],
        [ 0.,  0.,  0.,  0.],
        [ 0.,  1.,  0.,  0.],
        [ 0.,  1.,  0.,  0.],
        [ 0.,  0.,  0.,  0.],
        [ 0.,  1.,  0.,  0.],
        [ 0.,  1.,  0.,  0.],
        [ 0.,  1.,  0.,  0.],
        [ 0.,  0.,  0.,  0.],
        [ 0.,  1.,  0.,  0.],
        [ 0.,  1.,  0.,  0.],
        [ 0.,  1.,  0.,  0.],
        [ 0.,  1.,  0.,  0.],
        [ 1.,  1.,  1.,  1.],
        [ 1.,  1.,  1.,  1.],
        [ 1.,  1.,  1.,  1.],
        [ 0.,  0.,  1.,  1.],
        [ 1.,  0.,  1.,  1.],
        [ 1.,  0.,  1.,  1.],
        [ 1.,  1.,  1.,  1.],
        [ 0.,  0.,  1.,  1.],
        [ 1.,  0.,  1.,  1.],
        [ 0.,  0.,  1.,  1.],
        [ 0.,  0.,  1.,  1.],
        [ 1.,  0.,  1.,  1.],
        [ 1.,  0.,  1.,  1.],
        [ 1.,  0.,  1.,  1.],
        [ 1.,  0.,  1.,  1.],
        [ 1.,  1.,  1.,  1.],
        [ 1.,  0.,  1.,  1.],
        [ 1.,  0.,  1.,  1.],
        [ 1.,  0.,  1.,  1.],
        [ 1.,  0.,  1.,  1.],
        [ 1.,  1.,  1.,  2.],
        [ 1.,  0.,  1.,  1.],
        [ 1.,  0.,  1.,  1.],
        [ 1.,  0.,  1.,  1.],
        [ 1.,  0.,  1.,  1.],
        [ 1.,  0.,  1.,  1.],
        [ 1.,  0.,  1.,  1.],
        [ 1.,  0.,  2.,  1.],
        [ 1.,  0.,  1.,  1.],
        [ 1.,  0.,  1.,  1.],
        [ 0.,  0.,  1.,  1.],
        [ 0.,  0.,  1.,  1.],
        [ 1.,  0.,  1.,  1.],
        [ 1.,  0.,  2.,  1.],
        [ 0.,  0.,  1.,  1.],
        [ 1.,  1.,  1.,  1.],
        [ 1.,  1.,  1.,  1.],
        [ 1.,  0.,  1.,  1.],
        [ 1.,  0.,  1.,  1.],
        [ 0.,  0.,  1.,  1.],
        [ 0.,  0.,  1.,  1.],
        [ 1.,  0.,  1.,  1.],
        [ 1.,  0.,  1.,  1.],
        [ 0.,  0.,  1.,  1.],
        [ 1.,  0.,  1.,  1.],
        [ 1.,  0.,  1.,  1.],
        [ 1.,  0.,  1.,  1.],
        [ 1.,  0.,  1.,  1.],
        [ 0.,  0.,  1.,  1.],
        [ 1.,  0.,  1.,  1.],
        [ 1.,  1.,  2.,  2.],
        [ 1.,  0.,  2.,  2.],
        [ 1.,  0.,  2.,  2.],
        [ 1.,  0.,  2.,  2.],
        [ 1.,  0.,  2.,  2.],
        [ 1.,  0.,  2.,  2.],
        [ 0.,  0.,  1.,  1.],
        [ 1.,  0.,  2.,  2.],
        [ 1.,  0.,  2.,  2.],
        [ 1.,  1.,  2.,  2.],
        [ 1.,  1.,  2.,  2.],
        [ 1.,  0.,  2.,  2.],
        [ 1.,  0.,  2.,  2.],
        [ 1.,  0.,  2.,  2.],
        [ 1.,  0.,  2.,  2.],
        [ 1.,  1.,  2.,  2.],
        [ 1.,  0.,  2.,  2.],
        [ 1.,  1.,  2.,  2.],
        [ 1.,  0.,  2.,  2.],
        [ 1.,  0.,  2.,  1.],
        [ 1.,  1.,  2.,  2.],
        [ 1.,  0.,  1.,  2.],
        [ 1.,  0.,  2.,  2.],
        [ 1.,  0.,  1.,  2.],
        [ 1.,  1.,  2.,  2.],
        [ 1.,  1.,  2.,  2.],
        [ 1.,  0.,  1.,  2.],
        [ 1.,  0.,  1.,  2.],
        [ 1.,  0.,  2.,  2.],
        [ 1.,  0.,  2.,  1.],
        [ 1.,  0.,  2.,  2.],
        [ 1.,  1.,  2.,  2.],
        [ 1.,  0.,  2.,  2.],
        [ 1.,  0.,  2.,  1.],
        [ 1.,  0.,  2.,  1.],
        [ 1.,  0.,  2.,  2.],
        [ 1.,  1.,  2.,  2.],
        [ 1.,  1.,  2.,  2.],
        [ 1.,  0.,  1.,  2.],
        [ 1.,  1.,  2.,  2.],
        [ 1.,  1.,  2.,  2.],
        [ 1.,  1.,  2.,  2.],
        [ 1.,  0.,  2.,  2.],
        [ 1.,  1.,  2.,  2.],
        [ 1.,  1.,  2.,  2.],
        [ 1.,  0.,  2.,  2.],
        [ 1.,  0.,  2.,  2.],
        [ 1.,  0.,  2.,  2.],
        [ 1.,  1.,  2.,  2.],
        [ 1.,  0.,  2.,  2.]]

    assert_array_equal(transformed, expected,
                       err_msg="MDLP output is inconsistent with previous runs.")


def create_IV_tables(dataframe, feature_cutpoints):
    """
    This function returns a new dataframe with Information Value (IV) transformed
    features for a binary case. A list of cutpoints and multiple columns of features
    can be provided for bulk IV table calculations. A column with the name 'y' must
    be provided in the dataframe, which represents the target variable.

        - **parameters**, **types**, **return** and **return types**::

            :param dataframe: dataframe with columns representing features and 'y' as target,
            :param feature_cutpoints: a single list or list of lists of cutpoints,
            :return: IV table dataframe
            :rtype: pandas dataframe

        .. TODO:: This function could be made more convenient if a dictionary of variable names
                  as keys and its associated cutpoints as value of lists.
    """
    if 'y' not in dataframe.columns:
        raise AttributeError("Column name 'y' not in data frame")
    if not isinstance(dataframe, pd.core.frame.DataFrame):
        raise AttributeError('Input dataset should be a pandas dataframe')
    if (len(dataframe.columns.values) > 2) & (not all(isinstance(elem, list) for elem in feature_cutpoints)):
        raise AttributeError('Input cutpoints should be a list of lists')
    if (len(dataframe.columns.values) < 2) & (not isinstance(feature_cutpoints, list)):
        raise AttributeError('Input cutpoints should be a list')
    if (all(isinstance(elem, list) for elem in feature_cutpoints)) & (len(dataframe.columns.values) != len(feature_cutpoints) + 1):
        raise AttributeError('List of cutpoints not the same as the number of features')

    df_temp = dataframe.copy()
    feature_names = df_temp.columns.values.tolist()
    feature_names.remove('y')
    df_iv_final = pd.DataFrame()

    # iterate each feature in dataframe #
    for j, feature in enumerate(feature_names):
        if all(isinstance(elem, list) for elem in feature_cutpoints):
            cutpoints = sorted(feature_cutpoints[j])
        else:
            cutpoints = sorted(feature_cutpoints)
        df_iv = pd.DataFrame(0, index=np.arange(len(cutpoints)), columns=['Cutpoint', 'CntRec', 'CntGood', 'CntBad', 'CntCumRec', 'CntCumGood', 'CntCumBad', 'PctRec', 'GoodRate', 'BadRate', 'Odds', 'LnOdds', 'WoE', 'IV'])

        # Build record counts #
        for i, cutpoint in enumerate(cutpoints):
            df_iv.ix[i, 'Cutpoint'] = str(cutpoint)
            df_iv.ix[i, 'CntCumRec'] = len(df_temp[(df_temp[feature] <= cutpoint) & (df_temp['y'].isin([0, 1]))])
            df_iv.ix[i, 'CntCumGood'] = len(df_temp[(df_temp[feature] <= cutpoint) & (df_temp['y'] == (0))])
            df_iv.ix[i, 'CntCumBad'] = len(df_temp[(df_temp[feature] <= cutpoint) & (df_temp['y'] == 1)])

        # Missing data #
        x_nan = df_temp[feature].isnull().values.sum()

        if x_nan > 0:
            df_iv = df_iv.append(pd.DataFrame([['Missing', np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]], columns=df_iv.columns.values))
            df_iv.index = range(len(df_iv))
            idn = len(df_iv) - 1
            df_iv.ix[idn, 'CntRec'] = x_nan
            df_iv.ix[idn, 'CntGood'] = df_temp[df_temp['y'] == (0)][feature].isnull().values.sum()
            df_iv.ix[idn, 'CntBad'] = df_temp[df_temp['y'] == 1][feature].isnull().values.sum()
        else:
            df_iv = df_iv.append(pd.DataFrame([["Missing", np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]], columns=df_iv.columns.values), ignore_index=True)

        # Total #
        df_iv = df_iv.append(pd.DataFrame([['Total', np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]], columns=df_iv.columns.values), ignore_index=True)

        idn = len(df_iv) - 1
        df_iv.ix[idn, 'CntRec'] = len(df_temp)
        df_iv.ix[idn, 'CntGood'] = len(df_temp[df_temp['y'] == (0)])
        df_iv.ix[idn, 'CntBad'] = len(df_temp[df_temp['y'] == 1])

        # Complete table #
        df_iv.ix[0, 'CntRec'] = df_iv.ix[0, 'CntCumRec']    # Number records #
        df_iv.ix[0, 'CntGood'] = df_iv.ix[0, 'CntCumGood']  # Number goods #
        df_iv.ix[0, 'CntBad'] = df_iv.ix[0, 'CntCumBad']    # Number bads #

        for i in range(1, len(df_iv) - 2):
            df_iv.ix[i, 'CntRec'] = df_iv.ix[i, 'CntCumRec'] - df_iv.ix[i - 1, 'CntCumRec']
            df_iv.ix[i, 'CntGood'] = df_iv.ix[i, 'CntCumGood'] - df_iv.ix[i - 1, 'CntCumGood']
            df_iv.ix[i, 'CntBad'] = df_iv.ix[i, 'CntCumBad'] - df_iv.ix[i - 1, 'CntCumBad']

        df_iv.ix[1, 'CntRec'] = df_iv.ix[1, 'CntCumRec'] - df_iv.ix[0, 'CntCumRec']
        df_iv.ix[1, 'CntGood'] = df_iv.ix[1, 'CntCumGood'] - df_iv.ix[0, 'CntCumGood']
        df_iv.ix[1, 'CntBad'] = df_iv.ix[1, 'CntCumBad'] - df_iv.ix[0, 'CntCumBad']

        if x_nan > 0:
            # Add missing CntCumRec, CntCumGood and CntCumBad #
            missing_idx = int(df_iv[df_iv['Cutpoint'] == 'Missing'].index.values)
            df_iv.ix[missing_idx, 'CntCumRec'] = df_iv.ix[missing_idx + 1, 'CntRec']
            df_iv.ix[missing_idx, 'CntCumGood'] = df_iv.ix[missing_idx + 1, 'CntGood']
            df_iv.ix[missing_idx, 'CntCumBad'] = df_iv.ix[missing_idx + 1, 'CntBad']
        else:
            # Remove missing row if no missing values #
            df_iv = df_iv[df_iv['Cutpoint'] != 'Missing']
            df_iv = df_iv.reset_index(drop=True)

        df_iv['PctRec'] = round(100 * (df_iv['CntRec'].astype(float) / df_iv.ix[(len(df_iv) - 1), 1]).astype(float), 2)      # PctRec #
        df_iv['GoodRate'] = round(100 * (df_iv['CntGood'].astype(float) / df_iv['CntRec']).astype(float), 2)                 # GoodRate #
        df_iv['BadRate'] = round(100 * (df_iv['CntBad'] / df_iv['CntRec']).astype(float), 2)                                 # BadRate #
        df_iv['Odds'] = round((df_iv['CntGood'] / df_iv['CntBad']).astype(float), 4)                                         # Odds #
        df_iv['LnOdds'] = round(np.log(df_iv['Odds']), 4)                                                                    # LnOdds #

        # WOE and IV calculations #
        G = df_iv.ix[(len(df_iv) - 1), 'CntGood']
        B = df_iv.ix[(len(df_iv) - 1), 'CntBad']
        df_iv['WoE'] = round(np.log((df_iv['CntGood'].astype(float) / G) / (df_iv['CntBad'].astype(float) / B)), 4)          # WoE #
        df_iv['IV'] = round(df_iv['WoE'] * ((df_iv['CntGood'].astype(float) / G) - (df_iv['CntBad'].astype(float) / B)), 4)  # IV #
        df_iv.ix[(len(df_iv) - 1), 13] = df_iv['IV'].replace([np.inf, -np.inf], np.nan).sum(skipna=True)

        # create multi index per feature #
        df_iv.index = pd.MultiIndex.from_tuples([(feature, k) for k, v in df_iv.iterrows()])
        df_iv_final = df_iv_final.append(df_iv, ignore_index=False)
    return df_iv_final, df_iv.ix[(len(df_iv) - 1), 13]


def test_cutpoints():
    # mdlp cutpoints produced by Spark mdlp implementation by Sergio Ramírez #
    spark_mdlp_cutpoints = {'x1': [4.1, 19.9, 180.76, 2398.13], 'x2': [-2192.84, -549.56, 2.93, 14.25, 22.09, 336.18, 175492.79], 'x3': [508, 1205, 1681, 2938, 3196], 'x4': [1997, 4057, 7006, 11716, 19665, 1063223]}
    df = pd.read_csv("test.csv")
    for col in df.columns.values[:-1]:
        # drop nan values in column #
        df_temp = df[[col, 'y']].astype(float).dropna(axis=0, how='any')
        X = df_temp.as_matrix()[:, 0:1]
        y = df_temp['y'].astype(int).as_matrix()
        mdlp = MDLP(min_depth=3, shuffle=False)
        conv_X = mdlp.fit(X, y)
        # do IV calculation and compare #
        # print(create_IV_tables(df[[col, 'y']], conv_X.cut_points_[0].tolist())[1])
        # print(create_IV_tables(df[[col, 'y']], spark_mdlp_cutpoints[col])[1])
        assert_almost_equal(create_IV_tables(df[[col, 'y']], conv_X.cut_points_[0].tolist())[1], create_IV_tables(df[[col, 'y']], spark_mdlp_cutpoints[col])[1], decimal=3)


# if __name__ == "__main__":
#     pd.set_option('display.width', 1000)
#     test_cutpoints()
