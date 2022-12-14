import pywt
from nolitsa import surrogates
import numpy as np
from statsmodels.tsa.stattools import adfuller

def adf_test(values):
    result = adfuller(values)
    print('-' * 10)
    print('Adfuller Test')
    print(f'adf:   \t%s' % str(result[0]))
    print(f'p-value\t%s' % str(result[1]))

def fractional_difference(series, order, cutoff=1e-4):
    def get_weights(d, lags):
        weights = [1]
        for k in range(1, lags):
            weights.append(-weights[-1] * ((d - k + 1)) / k)
        weights = np.array(weights).reshape(-1, 1)
        return weights

    def find_cutoff(cutoff_order, cutoff, start_lags):
        val = np.inf
        lags = start_lags
        while abs(val) > cutoff:
            weight = get_weights(cutoff_order, lags)
            val = weight[len(weight) - 1]
            lags += 1
        return lags

    lag_cutoff = ( find_cutoff(order, cutoff, 1) )
    weights = get_weights(order, lag_cutoff)
    result = 0
    for k in range(lag_cutoff):
        result += weights[k] * series.shift(k).fillna(0)

    return result[lag_cutoff:]

def dwt_denoise(data, level=5):
    #len(c)%(2**n)=0; where n = level; I used len(c)=512
    # original level was 5
    coeffs = pywt.wavedec(data, 'db4', level=level) #returns array of cA5,cD5,cD4,cD3,...
    for i in range(1, len(coeffs)):
        temp = coeffs[i]
        mu = temp.mean()
        sigma = temp.std()
        omega = temp.max()
        kappa = (omega - mu) / sigma  #threshold value
        coeffs[i] = pywt.threshold(temp, kappa, mode='garrote')

    return pywt.waverec(coeffs, 'db4')

# Make sure there is no na
def aaft(returns):
    return surrogates.aaft(returns).cumsum()