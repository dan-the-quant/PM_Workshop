import pandas as pd
import numpy as np
from sklearn import linear_model
import math



def wexp(N, half_life):
    c = np.log(0.5)/half_life
    n = np.array(range(N))
    w = np.exp(c*n)
    return w/np.sum(w)

def n_days_nonmiss(returns, tiny_ret=1e-6):
    ix_ret_tiny = np.abs(returns) <= tiny_ret
    return np.sum(~ix_ret_tiny, axis=0)

def calc_capm(returns, market, half_life=0):
    beta_tmp = []
    alpha_tmp = []
    correl_tmp = []
    sigma_tmp = []
    
    X = market.values.reshape(-1,1)
    
    if half_life == 0:
        weights = np.ones(len(returns))
    else:
        weights = len(returns) * wexp(len(returns),half_life)
        
    for asset in returns.columns:
        y = np.array(returns.loc[:,asset])
        if np.sum(y) != 0:
            regr = linear_model.LinearRegression()
            regr.fit(X,y, weights)
            b,a,s,c = regr.coef_[0], regr.intercept_, np.std(regr.predict(X)-y), np.corrcoef(X,y,rowvar=False)[1,0]
        else:
            b,a,s,c = np.nan, np.nan, np.nan, np.nan
        beta_tmp.append(b)
        alpha_tmp.append(a)
        sigma_tmp.append(s)
        correl_tmp.append(c)
        
    capm = pd.DataFrame( list ( zip ( beta_tmp, alpha_tmp, correl_tmp, sigma_tmp)), columns=['hbeta', 'halpha', 'correl', 'hsigma'])
    return capm.T

def calc_rstr(returns, half_life=0, min_obs=100, tiny_ret=1e-6):
    rstr = np.log(1.0 + returns)
    if half_life == 0:
        weights = np.ones_like(rstr)
    else:
        weights = len(returns) * np.mat(wexp(len(returns), half_life)).T
    rstr = np.sum(rstr * weights)
    idx = n_days_nonmiss(returns) < min_obs
    rstr.where(~idx, other=np.nan, inplace=True)
    return pd.DataFrame(rstr, columns=['rstr']).T