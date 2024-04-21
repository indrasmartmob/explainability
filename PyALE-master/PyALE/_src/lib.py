import numpy as np
import pandas as pd
from statsmodels.distributions.empirical_distribution import ECDF
from scipy.stats import t

from ..logfile import logger

def cmds(D, k=2):
    """Classical multidimensional scaling

    Theory and code references:
    https://en.wikipedia.org/wiki/Multidimensional_scaling#Classical_multidimensional_scaling
    http://www.nervouscomputer.com/hfs/cmdscale-in-python/

    Arguments:
    D -- A squared matrix-like object (array, DataFrame, ....), usually a distance matrix
    """

    n = D.shape[0]
    if D.shape[0] != D.shape[1]:
        raise Exception("The matrix D should be squared")
    if k > (n - 1):
        raise Exception("k should be an integer <= D.shape[0] - 1")

    # (1) Set up the squared proximity matrix
    D_double = np.square(D)
    # (2) Apply double centering: using the centering matrix
    # centering matrix
    center_mat = np.eye(n) - np.ones((n, n)) / n
    # apply the centering
    B = -(1 / 2) * center_mat.dot(D_double).dot(center_mat)
    # (3) Determine the m largest eigenvalues
    # (where m is the number of dimensions desired for the output)
    # extract the eigenvalues
    eigenvals, eigenvecs = np.linalg.eigh(B)
    # sort descending
    idx = np.argsort(eigenvals)[::-1]
    eigenvals = eigenvals[idx]
    eigenvecs = eigenvecs[:, idx]
    # (4) Now, X=eigenvecs.dot(eigen_sqrt_diag),
    # where eigen_sqrt_diag = diag(sqrt(eigenvals))
    eigen_sqrt_diag = np.diag(np.sqrt(eigenvals[0:k]))
    ret = eigenvecs[:, 0:k].dot(eigen_sqrt_diag)
    return ret


def order_groups(X, feature):
    """Assign an order to the values of a categorical feature.

    The function returns an order to the unique values in X[feature] according to
    their similarity based on the other features.
    The distance between two categories is the sum over the distances of each feature.

    Arguments:
    X -- A pandas DataFrame containing all the features to considering in the ordering
    (including the categorical feature to be ordered).
    feature -- String, the name of the column holding the categorical feature to be ordered.
    """

    features = X.columns
    # groups = X[feature].cat.categories.values
    groups = X[feature].unique()
    D_cumu = pd.DataFrame(0, index=groups, columns=groups)
    K = len(groups)
    for j in set(features) - set([feature]):
        D = pd.DataFrame(index=groups, columns=groups)
        # discrete/factor feature j
        # e.g. j = 'color'
        if (X[j].dtypes.name == "category") | (
            (len(X[j].unique()) <= 10) & ("float" not in X[j].dtypes.name)
        ):
            # counts and proportions of each value in j in each group in 'feature'
            cross_counts = pd.crosstab(X[feature], X[j])
            cross_props = cross_counts.div(np.sum(cross_counts, axis=1), axis=0)
            for i in range(K):
                group = groups[i]
                D_values = abs(cross_props - cross_props.loc[group]).sum(axis=1) / 2
                D.loc[group, :] = D_values
                D[group] = D_values
        else:
            # continuous feature j
            # e.g. j = 'length'
            # extract the 1/100 quantiles of the feature j
            seq = np.arange(0, 1, 1 / 100)
            q_X_j = X[j].quantile(seq).to_list()
            # get the ecdf (empiricial cumulative distribution function)
            # compute the function from the data points in each group
            X_ecdf = X.groupby(feature)[j].agg(ECDF)
            # apply each of the functions on the quantiles
            # i.e. for each quantile value get the probability that j will take
            # a value less than or equal to this value.
            q_ecdf = X_ecdf.apply(lambda x: x(q_X_j))
            for i in range(K):
                group = groups[i]
                D_values = q_ecdf.apply(lambda x: max(abs(x - q_ecdf[group])))
                D.loc[group, :] = D_values
                D[group] = D_values
        D_cumu = D_cumu + D
    # reduce the dimension of the cumulative distance matrix to 1
    D1D = cmds(D_cumu, 1).flatten()
    # order groups based on the values
    order_idx = D1D.argsort()
    groups_ordered = D_cumu.index[D1D.argsort()]
    return pd.Series(range(K), index=groups_ordered)


def quantile_ied(x_vec, q):
    """
    Inverse of empirical distribution function (quantile R type 1).

    More details in
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mstats.mquantiles.html
    https://stat.ethz.ch/R-manual/R-devel/library/stats/html/quantile.html
    https://en.wikipedia.org/wiki/Quantile
    ----------------------------------------------------------------------------------------
    Definitions of sample quantiles
    Suppose that a sample has N observations that are sorted so that x[1] ≤ x[2] ≤ ... ≤ x[N], 
    and suppose that you are interested in estimating the p_th quantile (0 ≤ p ≤ 1) for the population. 
    Intuitively, the data values near x[j], where j = floor(Np) are reasonable values to use to estimate 
    the quantile. For example, if N=10 and you want to estimate the quantile for p=0.64, then j = floor(Np) = 6, 
    so you can use the sixth ordered value (x[6]) and maybe other nearby values to estimate the quantile.

    Hyndman and Fan (henceforth H&F) note that the quantile definitions in statistical software have three properties in common:

    1. The value p and the sample size N are used to determine two adjacent data values, x[j]and x[j+1]. 
    The quantile estimate will be in the closed interval between those data points. For the previous example, 
    the quantile estimate would be in the closed interval between x[6] and x[7].

    2. For many methods, a fractional quantity is used to determine an interpolation parameter, λ. For the previous example, 
    the fraction quantity is (Np - j) = (6.4 - 6) = 0.4. If you use λ = 0.4, then an estimate the 64th percentile would be 
    the value 40% of the way between x[6] and x[7].

    3. Each definition has a parameter m, 0 ≤ m ≤ 1, which determines how the method interpolates between adjacent data points. 
    In general, the methods define the index j by using j = floor(Np + m). The previous example used m=0, but other choices 
    include m=0.5 or values of m that depend on p.

    Thus a general formula for quantile estimates is q = (1 - λ) x[j]+ λ x[j+1], where λ and j depend on the values of p, N, 
    and a method-specific parameter m.
    ----------------------------------------------------------------------------------------
    
    ----------------------------------------------------------------------------------------
    Arguments:
    x_vec -- A pandas series containing the values to compute the quantile for
    q -- An array of probabilities (values between 0 and 1)
    """
    logger.info(F"Starting of {quantile_ied.__qualname__}")
    logger.debug(F"type(x_vec):{type(x_vec)},x_vec[:5]:{x_vec[:5]}")
    x_vec = x_vec.sort_values()
    n = len(x_vec) - 1
    m = 0
    logger.debug(F"Sample size:length of the vector-1: n={n}, m={m}")
    j = (n * q + m).astype(int)  # location of the value
    logger.debug(F"q[0]:{q[0]} & q[-1]:{q[-1]}")
    logger.debug(F"n*q[0]+m:{n*q[0]+m} & n*q[-1]+m:{n*q[-1]+m}")
    logger.debug(F"n*q[0]+m ==>Integer:{int(n*q[0]+m)} & n*q[-1]+m ==> Integer:{int(n*q[-1]+m)}")
    logger.debug(F"j[0]:{j[0]} & j[-1]:{j[-1]}")
    g = n * q + m - j
    logger.debug(F"n*q[0]+m-j[0]:{n*q[0]+m-j[0]} & n*q[-1]+m-j[-1]:{n*q[-1]+m-j[-1]}")
    logger.debug(F"g[0]:{g[0]} & g[-1]:{g[-1]}")

    gamma = (g != 0).astype(int)
    logger.debug(F"g[0]!=0:{g[0]!=0} & g[-1]!=0:{g[-1]!=0}")
    logger.debug(F"g[0]!=0==>Integer:{int(g[0]!=0)} & g[-1]!=0==>Integer:{int(g[-1]!=0)}")
    logger.debug(F"gamma[0]:{gamma[0]} & gamma[-1]:{gamma[-1]}")
    quant_res = (1 - gamma) * x_vec.shift(1, fill_value=0).iloc[j] + gamma * x_vec.iloc[j]

    quant_res.index = q
    logger.debug(F"quant_res.iloc[0]:{quant_res.iloc[0]} & quant_res.iloc[-1]:{quant_res.iloc[-1]}")
    logger.debug(f"x_vec.min():{x_vec.min()}, x_vec.max():{x_vec.max()}")
    # add min at quantile zero and max at quantile one (if needed)
    if 0 in q:
        quant_res.loc[0] = x_vec.min()
    if 1 in q:
        quant_res.loc[1] = x_vec.max()
    logger.debug(F"quant_res.iloc[0:2]:{quant_res.iloc[0:2]} & quant_res.iloc[-2:]:{quant_res.iloc[-2:]}")
    logger.debug(f"type(quant_res):{type(quant_res)}, count quant_res:{len(quant_res)}, quant_res.iloc[0:5],{quant_res.iloc[0:5]}")
    logger.info(F"Ending of {quantile_ied.__qualname__}")
    return quant_res


def CI_estimate(x_vec, C=0.95):
    """Estimate the size of the confidence interval of a data sample.

    The confidence interval of the given data sample (x_vec) is
    [mean(x_vec) - returned value, mean(x_vec) + returned value].
    """
    logger.info(F"Starting of {CI_estimate.__qualname__}")
    logger.debug(f"x_vec shape:{x_vec.shape}")
    logger.debug(F"Here the population variance is not know so t distribution is used for testing.")
    alpha = 1 - C
    n = len(x_vec)
    sample_standard_deviation = x_vec.std(ddof=1)
    logger.debug(f"The standard deviation will be divided by N-1. That is done by ddof:delta degrees of freedom i.e. ddof=1 means N-ddof i.e. N-1")
    stand_err =  sample_standard_deviation/ np.sqrt(n)
    logger.debug(f"Sample Standard Deviation:{sample_standard_deviation}, Standard Error:{stand_err}")
    logger.debug(f"The test statistic is t=(sample_mean-population_mean)/(sample s.d by square root of n) with dof=n-1")
    critical_val = 1 - (alpha / 2)
    logger.debug(F"Two tailed test will be considered as t is symmetric. So Level of significance/2={critical_val}")
    logger.debug(f"t=ppf(cumulativeDistributiveFunction,degreesOfFreedom,loc=0,scale=1): Percent Point Function==> Inverse of Cumulative Distribution Function percentiles")
    t_value = t.ppf(critical_val, n - 1)
    logger.debug(f"For CDF:{critical_val}, the t-statistic value is {t_value} for {n-1} degrees of freedom")
    t_star = stand_err * t_value
    logger.debug(f"t_star:{t_star} value should be subtracted and added to get the {C*100}% Confidence Interval")
    logger.info(F"Ending of {CI_estimate.__qualname__}")
    return t_star
