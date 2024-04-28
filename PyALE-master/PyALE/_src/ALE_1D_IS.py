import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.transforms as mtrans

from ..logfile import logger, log_enable, log_disable

from .lib import quantile_ied, CI_estimate, order_groups

def fn_check(verbose=True):
    if verbose==True:
        log_enable()
    elif verbose==False:
        log_disable()
    else :
        log_disable()

    logger.info("In the fn_check method")
    logger.trace("A trace message.")
    logger.debug("A debug message.")
    logger.info("An info message.")
    logger.success("A success message.")
    logger.warning("A warning message.")
    logger.error("An error message.")
    logger.critical("A critical message.")
    log_disable()
    return None


def aleplot_1D_continuous(X:pd.DataFrame, model, feature:str, grid_size:int=20, include_CI:bool=True, C:float=0.95, verbose:bool=False):
    """Compute the accumulated local effect of a numeric continuous feature.

    This function divides the feature in question into grid_size intervals (bins)
    and computes the difference in prediction between the first and last value
    of each interval and then centers the results.

    Arguments:
    X -- A pandas DataFrame to pass to the model for prediction.
    model -- Any python model with a predict method that accepts X as input.
    feature -- String, the name of the column holding the feature being studied.
    grid_size -- An integer indicating the number of intervals into which the
    feature range is divided.
    include_CI -- A boolean, if True the confidence interval
    of the effect is returned with the results.
    C -- A float the confidence level for which to compute the confidence interval

    Return: A pandas DataFrame containing for each bin: the size of the sample in it
    and the accumulated centered effect of this bin.
    """
    if verbose==True:
        log_enable()
    elif verbose==False:
        log_disable()
    else :
        log_disable()
    
    logger.info(F"Starting of {aleplot_1D_continuous.__qualname__}")
    logger.warning(F"Start: Data types check of the function")
    if type(X)!=type(pd.DataFrame()):
        raise TypeError(f"type of X should be a {type(pd.DataFrame())}")
    
    if type(feature)!=type('str'):
        raise TypeError(f"type of feature should be a {type('str')}")
    
    logger.warning(F"End: Data types check of the function")
    quantiles = np.linspace(0, 1, grid_size + 1, endpoint=True)
    logger.debug(F"quantiles count:{len(quantiles)},quantiles:{quantiles}")
    # use customized quantile function to get the same result as
    # type 1 R quantile (Inverse of empirical distribution function)
    logger.debug(f"type(X[feature]):{type(X[feature])}")
    bins = [X[feature].min()] + quantile_ied(X[feature], quantiles).to_list()
    logger.debug(F"bins count: {len(bins)}, bins:{bins}, bins[:4]:{bins[:4]}")
    bins = np.unique(bins)
    logger.debug(F"unique_bins count: {len(bins)}, unique_bins:{bins}, unique_bins[:4]:{bins[:4]}")
    feat_cut = pd.cut(X[feature], bins, include_lowest=True)
    logger.debug(F"type(X[feature]):{type(X[feature])}, X[feature].shape:{X[feature].shape}, X[feature][:5]:{X[feature][:5]}")
    logger.debug(F"type(feat_cut):{type(feat_cut)},feat_cut.shape:{feat_cut.shape} feat_cut[:5]:{feat_cut[:5]}")

    bin_codes = feat_cut.cat.codes
    logger.debug(f"type(feat_cut.cat):{feat_cut.cat}, type(feat_cut.cat.codes):{type(feat_cut.cat.codes)}")
    logger.debug(F"bin_codes count:{len(bin_codes)}, type(bin_codes):{type(bin_codes)}, bin_codes[:2]:{bin_codes[:2]}")
    bin_codes_unique = np.unique(bin_codes)
    logger.debug(F"type(bin_codes_unique):{type(bin_codes_unique)}, bin_codes_unique[:2]:{bin_codes_unique[:2]}")

    X1 = X.copy()
    X2 = X.copy()

    temp_df = pd.DataFrame({"X1_bin_codes":[i for i in bin_codes],"X1_bin":[bins[i] for i in bin_codes],"X2_bin_codes":[i + 1 for i in bin_codes],"X2_bin":[bins[i + 1] for i in bin_codes]})
    temp_df[feature]=X[feature].copy()
    temp_df["lb_ub"]=feat_cut.copy()
    temp_df = temp_df.loc[:,[feature, "lb_ub", "X1_bin_codes", "X1_bin", "X2_bin_codes", "X2_bin"]]
    logger.debug(F"temp_df.shape:{temp_df.shape}")
    logger.debug(F"temp_df.head(10):{temp_df.head(10)}")
    temp_unique_df = temp_df.drop_duplicates(subset=["X1_bin_codes"], keep="first")
    temp_unique_df.sort_values(by=["X1_bin_codes"], ascending=True, inplace=True)
    temp_unique_df.reset_index(inplace=True)
    temp_unique_df.drop(columns=["index"], inplace=True)
    logger.debug(F"temp_unique_df.shape:{temp_unique_df.shape}")
    logger.debug(F"temp_unique_df.head(grid_size):{temp_unique_df.head(grid_size)}") 

    X1[feature] = [bins[i] for i in bin_codes]
    X2[feature] = [bins[i + 1] for i in bin_codes]

    try:
        y_1 = model.predict(X1).ravel()
        logger.debug(F"type(y_1):{type(y_1)}, y_1.size:{y_1.size}")
        y_2 = model.predict(X2).ravel()
        logger.debug(F"type(y_2):{type(y_2)}, y_2.size:{y_2.size}")
    except Exception as ex:
        raise Exception(
            "Please check that your model is fitted, and accepts X as input."
        )

    local_effects_df = pd.DataFrame({feature: bins[bin_codes + 1], "Local_Effects": y_2 - y_1})
    logger.debug(f"local_effects_df.shape,{local_effects_df.shape}, local_effects_df.head():{local_effects_df.head()}")
    res_df = local_effects_df.groupby([feature], axis=0, observed=False)["Local_Effects"].agg([ "mean", "size"])
    logger.debug(f"Before Renaming:::res_df.shape,{res_df.shape}, res_df.head():{res_df.head()}")
    res_df.columns = ["local_mean_effects", "size"]
    logger.debug(f"After Renaming:::res_df.shape,{res_df.shape}, res_df.head():{res_df.head()}")
    res_df["Accumulated_local_mean_effects"] = res_df["local_mean_effects"].cumsum()
    logger.debug(f"After Cumulative Sum:::res_df.shape,{res_df.shape}, res_df.head():{res_df.head()}")
    logger.debug(f"res_df.shape:{res_df.shape}, res_df.tail():{res_df.tail()}")
    res_df.loc[min(bins), :] = 0
    logger.debug(f"After Adding min(bins) with 0 effect for starting point as the rows started from the upper bound of the first interval but we need to add the first row as the minimum:::res_df.shape,{res_df.shape}, res_df.head():{res_df.tail()}")
    res_df = res_df.sort_index()
    logger.debug(f"After sorting by index values:::res_df.shape:{res_df.shape}, res_df.head():{res_df.head()}")
    logger.debug(f"As we have added another row so either the first bin or the last should have 0 frequency. Here we will take the last bin as 0 frequency. In this way the weighted average will be on the smaller side and this will give us more postitive values after the weighted mean centering")
    res_df.rename(columns={"size":"size_original"}, inplace=True)
    res_df["size"]=res_df["size_original"]
    res_df["Weight"]=res_df["size_original"]
    res_df["Weight"]=res_df["Weight"].shift(-1, fill_value=0)
    # subtract the total average of a moving average of size 2
    res_df["mid_point_of_two_consecutive_acuumulated_local_mean_effects"] = (res_df["Accumulated_local_mean_effects"] + res_df["Accumulated_local_mean_effects"].shift(-1, fill_value=0)) / 2
    res_df["proportional_weight"] = res_df["Weight"]/res_df["Weight"].sum()
    res_df["proportional_weight_times_mid_point_of_two_consecutive_acuumulated_local_mean_effects"] = res_df["proportional_weight"]*res_df["mid_point_of_two_consecutive_acuumulated_local_mean_effects"]
    res_df["weighted_mean_of_accumulated_local_mean_effects"] = res_df["proportional_weight_times_mid_point_of_two_consecutive_acuumulated_local_mean_effects"].sum()
    res_df["Weighted_mean_centered_ALE"]=res_df["Accumulated_local_mean_effects"]-res_df["weighted_mean_of_accumulated_local_mean_effects"]
    logger.debug(f"Weighted_mean_centered_ALE:::res_df.shape:{res_df.shape}, res_df.head():{res_df.head()}")
    res_df["eff"] = res_df["Weighted_mean_centered_ALE"]
    logger.debug(f"added eff=Weighted_mean_centered_ALE for continuity with other dependencies:::res_df.shape:{res_df.shape}, res_df.head():{res_df.head()}")

    if include_CI:
        ci_est = local_effects_df.groupby(feature, axis=0, observed=False)["Local_Effects"].agg(
            [("CI_estimate", lambda x: CI_estimate(x, C=C))]
        )
        logger.debug(f"Before Sorting Index:::ci_est.shape:{ci_est.shape}, ci_est.head(): {ci_est.head()}")
        ci_est = ci_est.sort_index()
        logger.debug(f"After Sorting Index:::ci_est.shape:{ci_est.shape}, ci_est.head(): {ci_est.head()}")
        logger.debug(f"Before Merging Confidence Intervals with ALE table shape:::res_df.shape:{res_df.shape}")
        res_df = res_df.merge(ci_est, how="left", left_index=True, right_index=True, validate="one_to_one")
        logger.debug(f"After Merging Confidence Intervals with ALE table shape:::res_df.shape:{res_df.shape}")
        lowerCI_name_prefix = "lowerCI_" + str(int(C * 100)) + "%"
        upperCI_name_prefix = "upperCI_" + str(int(C * 100)) + "%"
        for colname in ["local_mean_effects", "Accumulated_local_mean_effects", "Weighted_mean_centered_ALE", "eff"]:

            if colname=="eff":
                lowerCI_name = f"{lowerCI_name_prefix}"
                upperCI_name = f"{upperCI_name_prefix}"
            else:
                lowerCI_name = f"{lowerCI_name_prefix}_{colname}"
                upperCI_name = f"{upperCI_name_prefix}_{colname}"           

            res_df[f"{lowerCI_name}"] = res_df[colname]-res_df["CI_estimate"]
            res_df[f"{upperCI_name}"] = res_df[colname]+res_df["CI_estimate"]
    logger.debug(f"Confidence Intervals Added for all relevant columns:::res_df.shape:{res_df.shape}, res_df.head():{res_df.head()}")
    rel_cols = ["eff", "size", lowerCI_name_prefix, upperCI_name_prefix]
    logger.debug(f"Desired output:::res_df.shape:{res_df.shape}, res_df.loc[:,rel_cols].head():{res_df.loc[:,rel_cols].head()}")
    log_disable()
    logger.info(F"Ending of {aleplot_1D_continuous.__qualname__}")
    return res_df

def aleplot_1D_discrete(X:pd.DataFrame, model, feature:str, include_CI:bool=True, C:float=0.95, verbose:bool=False):
    """Compute the accumulated local effect of a numeric discrete feature.

    This function computes the difference in prediction when the value of the feature
    is replaced once with the value before it and once with the value after it, without
    the need to divide into interval like the case of aleplot_1D_continuous.

    Arguments:
    X -- A pandas DataFrame to pass to the model for prediction.
    model -- Any python model with a predict method that accepts X as input.
    feature -- String, the name of the column holding the feature being studied.
    include_CI -- A boolean, if True the confidence interval
    of the effect is returned with the results.
    C -- A float the confidence level for which to compute the confidence interval

    Return: A pandas DataFrame containing for each value of the feature: the size
    of the sample in it and the accumulated centered effect around this value.

    """
    if verbose==True:
        log_enable()
    elif verbose==False:
        log_disable()
    else :
        log_disable()
    
    logger.info(F"Starting of {aleplot_1D_discrete.__qualname__}")
    logger.warning(F"Start: Data types check of the function")
    if type(X)!=type(pd.DataFrame()):
        raise TypeError(f"type of X should be a {type(pd.DataFrame())}")
    
    if type(feature)!=type('str'):
        raise TypeError(f"type of feature should be a {type('str')}")
    
    logger.warning(F"End: Data types check of the function")

    groups = X[feature].unique()
    logger.debug(F"type(groups):{type(groups)}")
    logger.debug(F"Count of unique values: {len(groups)}")
    groups.sort()
    logger.debug(F"len(groups):{len(groups)}, Unique values of groups:{groups}")

    groups_codes = {groups[x]: x for x in range(len(groups))}
    logger.debug(F"len(groups_codes):{len(groups_codes)}, Unique values of groups_codes:{groups_codes}")
    feature_codes = X[feature].replace(groups_codes).astype(int)
    logger.debug(F"type(feature_codes):{type(feature_codes)}, feature_codes.iloc[:5]:{feature_codes.iloc[:5]}")

    groups_counts = X.groupby(feature, observed=False).size()
    logger.debug(F"type(groups_counts):{type(groups_counts)}, groups_counts.iloc[:5]:{groups_counts.iloc[:5]}")
    groups_props = groups_counts / sum(groups_counts)
    logger.debug(F"type(groups_props):{type(groups_props)}, groups_props.iloc[:5]:{groups_props.iloc[:5]}")

    K = len(groups)
    logger.debug(F"len(groups):{len(groups)}")

    # create copies of the dataframe
    X_plus = X.copy()
    X_neg = X.copy()
    # all groups except last one
    last_group = groups[K - 1]
    logger.debug(F"last_group:{last_group}")
    ind_plus = X[feature] != last_group
    # all groups except first one
    first_group = groups[0]
    logger.debug(F"first_group:{first_group}")
    ind_neg = X[feature] != first_group
    # replace once with one level up
    X_plus.loc[ind_plus, feature] = groups[feature_codes[ind_plus] + 1]
    # replace once with one level down
    X_neg.loc[ind_neg, feature] = groups[feature_codes[ind_neg] - 1]
    logger.debug(F"X_neg.loc[:,feature].head():{X_neg.loc[:,feature].head()}")
    logger.debug(F"X.loc[:,feature].head():{X.loc[:,feature].head()}")
    logger.debug(F"X_plus.loc[:,feature].head():{X_plus.loc[:,feature].head()}")
    try:
        # predict with original and with the replaced values
        y_hat = model.predict(X).ravel()
        y_hat_plus = model.predict(X_plus[ind_plus]).ravel()
        y_hat_neg = model.predict(X_neg[ind_neg]).ravel()
        logger.debug(F"type(y_hat_neg):{type(y_hat_neg)}, y_hat_neg.shape:{y_hat_neg.shape}, y_hat_neg[0:5]:{y_hat_neg[0:5]}")
        logger.debug(F"type(y_hat):{type(y_hat)}, y_hat.shape:{y_hat.shape}, y_hat[0:5]:{y_hat[0:5]}")
        logger.debug(F"type(y_hat_plus):{type(y_hat_plus)}, y_hat_plus.shape:{y_hat_plus.shape}, y_hat_plus[0:5]:{y_hat_plus[0:5]}")
        
    except Exception as ex:
        raise Exception(
            "Please check that your model is fitted, and accepts X as input."
        )

    # compute prediction difference
    Delta_plus = y_hat_plus - y_hat[ind_plus]
    logger.debug(F"Delta_plus.shape:{Delta_plus.shape}, y_hat_plus.shape:{y_hat_plus.shape}, y_hat[ind_plus].shape:{y_hat[ind_plus].shape}")
    Delta_neg = y_hat[ind_neg] - y_hat_neg
    logger.debug(F"Delta_neg.shape:{Delta_neg.shape}, y_hat[ind_neg].shape:{y_hat[ind_neg].shape}, y_hat_neg.shape:{y_hat_neg.shape}")

    # compute the mean of the difference per group
    logger.debug(F"type(groups):{type(groups)}, groups.shape:{groups.shape}, groups:{groups}")
    Delta_plus_df = pd.DataFrame({"eff": Delta_plus, feature: groups[feature_codes[ind_plus] + 1]})
    logger.debug(f"len(feature_codes[ind_plus]):{len(feature_codes[ind_plus])}, feature_codes[ind_plus].iloc[0:5]:{feature_codes[ind_plus].iloc[0:5]}")
    logger.debug(f"type(feature_codes[ind_plus] + 1):{type(feature_codes[ind_plus] + 1)}, len(feature_codes[ind_plus] + 1):{len(feature_codes[ind_plus] + 1)}, [feature_codes[ind_plus].to_numpy() + 1][0:5]:{[feature_codes[ind_plus].to_numpy() + 1][0:5]}")
    logger.debug(f"type(groups[feature_codes[ind_plus] + 1]):{type(groups[feature_codes[ind_plus] + 1])}, len(groups[feature_codes[ind_plus] + 1]):{len(groups[feature_codes[ind_plus] + 1])}, groups[feature_codes[ind_plus] + 1][0:5]:{groups[feature_codes[ind_plus] + 1][0:5]}")
    Delta_neg_df = pd.DataFrame({"eff": Delta_neg, feature: groups[feature_codes[ind_neg]]})
    logger.debug(f"len(feature_codes[ind_neg]):{len(feature_codes[ind_neg])}, feature_codes[ind_neg].iloc[0:5]:{feature_codes[ind_neg].iloc[0:5]}")
    logger.debug(f"type(feature_codes[ind_neg]):{type(feature_codes[ind_neg])}, len(feature_codes[ind_neg]):{len(feature_codes[ind_neg])}, [feature_codes[ind_neg].values()][0:5]:{[feature_codes[ind_neg].to_numpy()][0:5]}")
    logger.debug(f"type(groups[feature_codes[ind_neg]]):{type(groups[feature_codes[ind_neg]])}, len(groups[feature_codes[ind_neg]]):{len(groups[feature_codes[ind_neg]])}, groups[feature_codes[ind_neg]][0:5]:{groups[feature_codes[ind_neg]][0:5]}")
    delta_df = pd.concat([Delta_plus_df, Delta_neg_df], axis=0)
    plus_count = Delta_plus_df.groupby([feature], observed=False).size().reset_index().rename(columns={0:"plus_count_min_to_max_minus_1_category"})
    neg_count = Delta_neg_df.groupby([feature], observed=False).size().reset_index().rename(columns={0:"neg_count_min_plus_1_to_max_category"})
    all_count = delta_df.groupby([feature], observed=False).size().reset_index().rename(columns={0:"plus_n_neg_count"})
    logger.debug(F"plus_count.head():{plus_count.head()}")
    logger.debug(F"neg_count.head():{neg_count.head()}")
    logger.debug(F"all_count.head():{all_count.head()}")
    all_count = pd.merge(all_count,plus_count, how="outer", on=feature, validate="one_to_one")
    all_count = pd.merge(all_count,neg_count, how="outer", on=feature, validate="one_to_one")
    logger.debug(F"After merging all the counts--all_count.head():{all_count.head()}")
    res_df = delta_df.groupby([feature], observed=False).mean()
    logger.debug(F"After calculating the local mean effects: res_df.head():{res_df.head()}")
    res_df["eff"] = res_df["eff"].cumsum()
    logger.debug(F"After calculating the cumulative local mean effects: res_df.head():{res_df.head()}")
    res_df.loc[groups[0]] = 0
    logger.debug(F"After adding starting 0 index with 0 values: res_df.head():{res_df.head()}")
    res_df = res_df.sort_index()
    logger.debug(F"After sorting by index: res_df.head():{res_df.head()}")
    res_df["eff"] = res_df["eff"] - sum(res_df["eff"] * groups_props)
    logger.debug(F"After subtracting the weighted mean from cumulative local mean effects: res_df.head():{res_df.head()}")
    res_df["size"] = groups_counts
    logger.debug(F"After including the weight in the dataframe: res_df.head():{res_df.head()}")
    if include_CI:
        ci_est = delta_df.groupby([feature], observed=False).eff.agg(
            [("CI_estimate", lambda x: CI_estimate(x, C=C))]
        )
        lowerCI_name = "lowerCI_" + str(int(C * 100)) + "%"
        upperCI_name = "upperCI_" + str(int(C * 100)) + "%"
        res_df[lowerCI_name] = res_df[["eff"]].subtract(ci_est["CI_estimate"], axis=0)
        res_df[upperCI_name] = upperCI = res_df[["eff"]].add(
            ci_est["CI_estimate"], axis=0
        )
    
    
    log_disable()
    logger.info(F"Ending of {aleplot_1D_discrete.__qualname__}")
    return res_df


def aleplot_1D_categorical(
    X, model, feature, encode_fun, predictors, include_CI=True, C=0.95
):
    """Compute the accumulated local effect of a categorical (or str) feature.

    The function computes the difference in prediction between the different groups
    similar to the function aleplot_1D_discrete.
    This function relies on an ordering of the unique values/groups of the feature,
    if the feature is not an ordered categorical, then an ordering is computed,
    which orders the groups by their similarity based on the distribution of the
    other features in each group.
    The function uses the given encoding function (for example a one-hot-encoding)
    and replaces the feature with the new generated (outputed) feature(s) before
    it calls the function model.predict.

    Arguments:
    X -- A pandas DataFrame containing all columns needed to pass to the model for
    prediction, however it should contain the original feature (before encoding)
    instead of the column(s) encoding it.
    model -- Any python model with a predict method that accepts X as input.
    feature -- String, the name of the column holding the feature being analysed.
    encode_fun -- Function, used to encode the categorical feature, usually a
    one-hot encoder. The function's input and output are as follows
        * input: a DataFrame with one column (the feature)
        * output: a DataFrame with the new column(s) encoding the feature.
        It is also important that this function should be able to handle missing
        categories (for example, a one-hot-encoder applied to a column, in which
        not all categories occur, should add a column of zeros for each missing
        category). Examples of use of this function could be found in the README
        file in github or in the description of the package in PyPI
        https://pypi.org/project/PyALE/.
    predictors -- List or array of strings containing the names of features used
    in the model, and in the right order.
    include_CI -- A boolean, if True the confidence interval of the effect is
    returned with the results.
    C -- A float the confidence level for which to compute the confidence interval

    Return: A pandas DataFrame containing for each value of the feature: the size
    of the sample in it and the accumulated centered effect around this value.
    """
    # if the values of the feature are not ordered, then order them by
    # their similarity to each other, based on the distributions of the other
    # features by group
    if (X[feature].dtype.name != "category") or (not X[feature].cat.ordered):
        X[feature] = X[feature].astype(str)
        groups_order = order_groups(X, feature)
        groups = groups_order.index.values
        X[feature] = X[feature].astype(
            pd.api.types.CategoricalDtype(categories=groups, ordered=True)
        )

    groups = X[feature].unique()
    groups = groups.sort_values()
    feature_codes = X[feature].cat.codes
    groups_counts = X.groupby(feature, observed=False).size()
    groups_props = groups_counts / sum(groups_counts)

    K = len(groups)

    # create copies of the dataframe
    X_plus = X.copy()
    X_neg = X.copy()
    # all groups except last one
    last_group = groups[K - 1]
    ind_plus = X[feature] != last_group
    # all groups except first one
    first_group = groups[0]
    ind_neg = X[feature] != first_group
    # replace once with one level up
    X_plus.loc[ind_plus, feature] = groups[feature_codes[ind_plus] + 1]
    # replace once with one level down
    X_neg.loc[ind_neg, feature] = groups[feature_codes[ind_neg] - 1]
    try:
        # predict with original and with the replaced values
        # encode the categorical feature
        X_coded = pd.concat([X.drop(feature, axis=1), encode_fun(X[[feature]])], axis=1)
        # predict
        y_hat = model.predict(X_coded[predictors]).ravel()

        # encode the categorical feature
        X_plus_coded = pd.concat(
            [X_plus.drop(feature, axis=1), encode_fun(X_plus[[feature]])], axis=1
        )
        # predict
        y_hat_plus = model.predict(X_plus_coded[ind_plus][predictors]).ravel()

        # encode the categorical feature
        X_neg_coded = pd.concat(
            [X_neg.drop(feature, axis=1), encode_fun(X_neg[[feature]])], axis=1
        )
        # predict
        y_hat_neg = model.predict(X_neg_coded[ind_neg][predictors]).ravel()
    except Exception as ex:
        raise Exception(
            """There seems to be a problem when predicting with the model.
            Please check the following: 
                - Your model is fitted.
                - The list of predictors contains the names of all the features"""
            """ used for training the model.
                - The encoding function takes the raw feature and returns the"""
            """ right columns encoding it, including the case of a missing category.
            """
        )

    # compute prediction difference
    Delta_plus = y_hat_plus - y_hat[ind_plus]
    Delta_neg = y_hat[ind_neg] - y_hat_neg

    # compute the mean of the difference per group
    delta_df = pd.concat(
        [
            pd.DataFrame(
                {"eff": Delta_plus, feature: groups[feature_codes[ind_plus] + 1]}
            ),
            pd.DataFrame({"eff": Delta_neg, feature: groups[feature_codes[ind_neg]]}),
        ]
    )
    res_df = delta_df.groupby([feature], observed=False).mean()
    res_df["eff"] = res_df["eff"].cumsum()
    res_df.loc[groups[0]] = 0
    # sort the index (which is at this point an ordered categorical) as a safety measure
    res_df = res_df.sort_index()
    res_df["eff"] = res_df["eff"] - sum(res_df["eff"] * groups_props)
    res_df["size"] = groups_counts
    if include_CI:
        ci_est = delta_df.groupby([feature], observed=False).eff.agg(
            [("CI_estimate", lambda x: CI_estimate(x, C=C))]
        )
        lowerCI_name = "lowerCI_" + str(int(C * 100)) + "%"
        upperCI_name = "upperCI_" + str(int(C * 100)) + "%"
        res_df[lowerCI_name] = res_df[["eff"]].subtract(ci_est["CI_estimate"], axis=0)
        res_df[upperCI_name] = upperCI = res_df[["eff"]].add(
            ci_est["CI_estimate"], axis=0
        )
    return res_df


def plot_1D_continuous_eff(res_df, X, fig=None, ax=None, verbose=False):
    """Plot the 1D ALE plot for a continuous feature.

    Arguments:
    res_df -- A pandas DataFrame containing the computed effects
    (the output of ale_1D_continuous).
    X -- The dataset used to compute the effects.
    fig, ax -- matplotlib figure and axis.
    """
    if verbose==True:
        log_enable()
    elif verbose==False:
        log_disable()
    else :
        log_disable()

    logger.info(F"Starting of {plot_1D_continuous_eff.__qualname__}")
    logger.debug(F"res_df.shape:{res_df.shape}, res_df.head():{res_df.head()}")
    logger.debug(F"X.shape:{X.shape}, X.head():{X.head()}")
    feature_name = res_df.index.name
    logger.debug(F"feature_name:{feature_name}, res_df.index.name:{res_df.index.name}")
    # position: jitter
    # to see the distribution of the data points clearer, each point x will be nudged a random value between
    # -minimum distance between two data points and +
    logger.debug(f"type(X[feature_name]):{type(X[feature_name])}, X[feature_name].iloc[:5]:{X[feature_name].iloc[:5]}")
    sorted_values = X[feature_name].sort_values(ascending=True)
    logger.debug(f"type(sorted_values):{type(sorted_values)}, sorted_values.iloc[:5]:{sorted_values.iloc[:5]}")
    logger.debug(f"type(sorted_values.shift()):{type(sorted_values.shift())}, sorted_values.shift().iloc[:5]:{sorted_values.shift().iloc[:5]}")
    logger.debug(f"type(sorted_values.shift(periods=1)):{type(sorted_values.shift(periods=1))}, sorted_values.shift(periods=1).iloc[:5]:{sorted_values.shift(periods=1).iloc[:5]}")
    values_diff = abs(sorted_values - sorted_values.shift(periods=1))
    logger.debug(f"values_diff = abs(sorted_values - sorted_values.shift(periods=1)): values_diff.describe():{values_diff.describe()}")
    logger.debug(f"values_diff[values_diff > 0].describe():{values_diff[values_diff > 0].describe()}")
    interval = values_diff[values_diff > 0].min()
    logger.debug(F"interval = values_diff[values_diff > 0].min():{values_diff[values_diff > 0].min()}")
    logger.debug(F"lb=-interval / 2:{-interval / 2}, ub=interval / 2:{interval / 2}")
    
    np.random.seed(123)
    rug = X.apply(lambda row: row[feature_name]+ np.random.uniform(low=-interval / 2, high=interval / 2,),axis=1,)
    logger.debug(f"X[feature_name].shape:{X[feature_name].shape}, type(X[feature_name]):{type(X[feature_name])}, X[feature_name].iloc[:5]:{X[feature_name].iloc[:5]}")
    logger.debug(f"rug.shape:{rug.shape}, type(rug):{type(rug)}, rug.iloc[:5]:{rug.iloc[:5]}")

    if fig is None and ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(res_df.loc[:,["eff"]], color="tab:blue")
    # ax.plot(np.arange(res_df.shape[0]), res_df.loc[:,["eff"]])
    logger.debug(f"type(res_df[['eff']]):{type(res_df[['eff']])}")
    logger.debug(f"res_df.drop('size', axis=1).shape:{res_df.drop('size', axis=1).shape}, res_df.drop('size', axis=1).head():{res_df.drop('size', axis=1).head()}")
    logger.debug(f"res_df.drop('size', axis=1).min().min():{res_df.drop('size', axis=1).min().min()}")
    rug_list = [res_df.drop('size', axis=1).min().min()] * len(rug)
    logger.debug(f"len(rug):{len(rug)}, len(rug_list):{len(rug_list)}, rug_list[:5]:{rug_list[:5]}")
    
    tr = mtrans.offset_copy(ax.transData, fig=fig, x=0.0, y=-5, units="points")
    ax.plot(
        rug,
        rug_list,
        "|",
        color="tab:olive",
        alpha=0.2,
        transform=tr,
    )
    lowerCI_name = res_df.columns[res_df.columns.str.contains("lowerCI")]
    logger.debug(F"lowerCI_name[0]:{lowerCI_name[0]}, lowerCI_name:{lowerCI_name},")
    upperCI_name = res_df.columns[res_df.columns.str.contains("upperCI")]
    logger.debug(F"upperCI_name[0]:{upperCI_name[0]}, upperCI_name:{upperCI_name}")
    if (len(lowerCI_name) == 1) and (len(upperCI_name) == 1):
        label = lowerCI_name.str.split("_")[0][1] + " confidence interval"
        logger.debug(f"label:{label}")
        ax.fill_between(
            res_df.index,
            y1=res_df[lowerCI_name[0]],
            y2=res_df[upperCI_name[0]],
            alpha=0.4,
            color="tab:cyan",
            label=label,
        )
        ax.legend()
    ax.set_xlabel(res_df.index.name)
    ax.set_ylabel("Effect on prediction (centered)")
    ax.set_title("1D ALE Plot - Continuous")
    
    logger.info(F"Ending of {plot_1D_continuous_eff.__qualname__}")
    log_disable()
    return fig, ax

def plot_1D_continuous_eff2(res_df, X, gap=0.2, fig=None, ax=None, verbose=False):
    """Plot the 1D ALE plot for a continuous feature.

    Arguments:
    res_df -- A pandas DataFrame containing the computed effects
    (the output of ale_1D_continuous).
    X -- The dataset used to compute the effects.
    fig, ax -- matplotlib figure and axis.
    """
    if verbose==True:
        log_enable()
    elif verbose==False:
        log_disable()
    else :
        log_disable()

    logger.info(F"Starting of {plot_1D_continuous_eff.__qualname__}")
    logger.debug(F"res_df.shape:{res_df.shape}, res_df.head():{res_df.head()}")
    logger.debug(F"X.shape:{X.shape}, X.head():{X.head()}")
    feature_name = res_df.index.name
    logger.debug(F"feature_name:{feature_name}, res_df.index.name:{res_df.index.name}")
    # position: jitter
    # to see the distribution of the data points clearer, each point x will be nudged a random value between
    # -minimum distance between two data points and +
    logger.debug(f"type(X[feature_name]):{type(X[feature_name])}, X[feature_name].iloc[:5]:{X[feature_name].iloc[:5]}")
    sorted_values = X[feature_name].sort_values(ascending=True)
    logger.debug(f"type(sorted_values):{type(sorted_values)}, sorted_values.iloc[:5]:{sorted_values.iloc[:5]}")
    logger.debug(f"type(sorted_values.shift()):{type(sorted_values.shift())}, sorted_values.shift().iloc[:5]:{sorted_values.shift().iloc[:5]}")
    logger.debug(f"type(sorted_values.shift(periods=1)):{type(sorted_values.shift(periods=1))}, sorted_values.shift(periods=1).iloc[:5]:{sorted_values.shift(periods=1).iloc[:5]}")
    values_diff = abs(sorted_values - sorted_values.shift(periods=1))
    logger.debug(f"values_diff = abs(sorted_values - sorted_values.shift(periods=1)): values_diff.describe():{values_diff.describe()}")
    logger.debug(f"values_diff[values_diff > 0].describe():{values_diff[values_diff > 0].describe()}")
    interval = values_diff[values_diff > 0].min()
    logger.debug(F"interval = values_diff[values_diff > 0].min():{values_diff[values_diff > 0].min()}")
    logger.debug(F"lb=-interval / 2:{-interval / 2}, ub=interval / 2:{interval / 2}")
    
    np.random.seed(123)
    rug = X.apply(lambda row: row[feature_name]+ np.random.uniform(low=-interval / 2, high=interval / 2,),axis=1,)
    logger.debug(f"X[feature_name].shape:{X[feature_name].shape}, type(X[feature_name]):{type(X[feature_name])}, X[feature_name].iloc[:5]:{X[feature_name].iloc[:5]}")
    logger.debug(f"rug.shape:{rug.shape}, type(rug):{type(rug)}, rug.iloc[:5]:{rug.iloc[:5]}")

    if fig is None and ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(res_df.loc[:,["eff"]], color="tab:blue")
    # ax.plot(np.arange(res_df.shape[0]), res_df.loc[:,["eff"]])
    logger.debug(f"type(res_df[['eff']]):{type(res_df[['eff']])}")
    logger.debug(f"res_df.drop('size', axis=1).shape:{res_df.drop('size', axis=1).shape}, res_df.drop('size', axis=1).head():{res_df.drop('size', axis=1).head()}")
    logger.debug(f"res_df.drop('size', axis=1).min().min():{res_df.drop('size', axis=1).min().min()}")
    rug_list = [res_df.drop('size', axis=1).min().min()] * len(rug)
    logger.debug(f"len(rug):{len(rug)}, len(rug_list):{len(rug_list)}, rug_list[:5]:{rug_list[:5]}")

    # creating a x,y plot to understand the distribution of data
    # scaling the x-axis as per the grid_size
    rug_scaled = ((rug)/(rug.max()))*(res_df.shape[0]-1)
    sd = res_df.drop('size', axis=1).std(ddof=0)
    min_value = res_df.drop('size', axis=1).min().min()
    mean_value = res_df.drop('size', axis=1).mean().mean()
    delta = abs(abs(mean_value)-abs(min_value)*gap)
    rug_list_reduced = [min_value - delta] * len(rug)
    tr = mtrans.offset_copy(ax.transData, fig=fig, x=0.0, y=-5, units="points")
    ax.plot(
        rug_scaled,
        rug_list_reduced,
        "|",
        color="tab:olive",
        alpha=0.2,
        # transform=tr,
    )
    lowerCI_name = res_df.columns[res_df.columns.str.contains("lowerCI")]
    logger.debug(F"lowerCI_name[0]:{lowerCI_name[0]}, lowerCI_name:{lowerCI_name},")
    upperCI_name = res_df.columns[res_df.columns.str.contains("upperCI")]
    logger.debug(F"upperCI_name[0]:{upperCI_name[0]}, upperCI_name:{upperCI_name}")
    if (len(lowerCI_name) == 1) and (len(upperCI_name) == 1):
        label = lowerCI_name.str.split("_")[0][1] + " confidence interval"
        logger.debug(f"label:{label}")
        ax.fill_between(
            res_df.index,
            y1=res_df[lowerCI_name[0]],
            y2=res_df[upperCI_name[0]],
            alpha=0.4,
            color="tab:cyan",
            label=label,
        )
        ax.legend()
    ax.set_xlabel(res_df.index.name)
    ax.set_ylabel("Effect on prediction (centered)")
    ax.set_title("1D ALE Plot - Continuous")
    
    logger.info(F"Ending of {plot_1D_continuous_eff.__qualname__}")
    log_disable()
    return fig, ax


def plot_1D_discrete_eff(res_df, X, fig=None, ax=None):
    """Plot the 1D ALE plot for a discrete feature.

    Arguments:
    res_df -- A pandas DataFrame with the computed effects
    (the output of ale_1D_discrete).
    X -- The dataset used to compute the effects.
    fig, ax -- matplotlib figure and axis.
    """

    feature_name = res_df.index.name
    if fig is None and ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    ax.set_xlabel(feature_name)
    ax.set_ylabel("Effect on prediction (centered)")
    yerr = 0
    lowerCI_name = res_df.columns[res_df.columns.str.contains("lowerCI")]
    upperCI_name = res_df.columns[res_df.columns.str.contains("upperCI")]
    if (len(lowerCI_name) == 1) and (len(upperCI_name) == 1):
        yerr = res_df[upperCI_name].subtract(res_df["eff"], axis=0).iloc[:, 0]
    ax.errorbar(
        res_df.index.astype(str),
        res_df["eff"],
        yerr=yerr,
        capsize=3,
        marker="o",
        linestyle="dashed",
        color="black",
    )
    ax2 = ax.twinx()
    ax2.set_ylabel("Size", color="lightblue")
    ax2.bar(res_df.index.astype(str), res_df["size"], alpha=0.1, align="center")
    ax2.tick_params(axis="y", labelcolor="lightblue")
    ax2.set_title("1D ALE Plot - Discrete/Categorical")
    fig.tight_layout()
    return fig, ax, ax2
