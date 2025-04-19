import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import pmdarima 
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller

from scipy.stats import jarque_bera,skew, kurtosis

import LeeCarterModel

########## Inputs ##########
yearsToForecast = 100
targetFields = LeeCarterModel.targetFields
qxMatrix = LeeCarterModel.qxMatrix
aDf = LeeCarterModel.aDf
bDf = LeeCarterModel.bDf
kDf = LeeCarterModel.kDf
yearsPlot = LeeCarterModel.yearsPlot
agesPlot = LeeCarterModel.agesPlot

########## Custom Functions ##########

def get_diagnostics(resid):

    ljung = acorr_ljungbox(resid, lags=[1], return_df=True)
    jb_stat, jb_pval = jarque_bera(resid)
    
    return {
        "LJ_Box_Stat": ljung['lb_stat'].iloc[0],            #Ljung-Box Q-Stat
        "LJung_Box_P-Value": ljung['lb_pvalue'].iloc[0],    #Ljung-Box P-Value
        "JB_Stat": jb_stat,                                 #Jarque-Bera Stat
        "JB_P-Value": jb_pval,                              #Jarque-Bera P-Value
        "ADF_Resid_Stat": adfuller(resid)[0],                        #Augmented Dickey-Fuller Stat
        "ADF_Resid_P-Value": adfuller(resid)[1],                      #Augmented Dickey-Fuller         
        "Skew": skew(resid),                                #Skewness
        "Kurtosis": kurtosis(resid),                        #Kurtosis
    }
     
########## 0. Auto-fitting ARIMA models to kappa (time-varying component) ##########

for field in targetFields:
    y = kDf[kDf["Gender"]==field]["Kappa"]          #Kappa Input 

    kARIMAs = pmdarima.auto_arima(
        y,
        start_p=0, start_q=0,                       #Starting AR(), MA() parameters
        max_p=3, max_q=3,                           #Max AR(), MA() parameters
        seasonal=False,                             #Set to true only for seasonal data
        information_criterion="bic",                #Best model is (1,1,0) under both criteria, aic and bic
        stepwise=False,                             #Complete search iterating through all param combinations
        suppress_warnings=True,                     #Only warnings related to indexing stuff, not relevant
        return_valid_fits=True,                     #Returns every model tried
        trace=True,                                 #See progress on each iteration
        #trend="t"                                  #Linear Time Trend Not statistically significant
    )                  

    ########## 1. Summary of ARIMA models ##########
    
    # 1.1 Getting ARIMA scores to analyze best fit
    kARIMARecords = []

    for k in kARIMAs:                               
        kARIMARecords.append({                                 #Appending validated models 
            "ARIMA Order" : k.order,                           #Model order (p,d,q)
            "AIC" : k.aic(),                                   #Akaike score
            "BIC" : k.bic(),                                   #Bayesian score
            "HQIC": k.hqic(),                                  #Hannan–Quinn score
            "MSE" : k.arima_res_.mse,                          #Mean-Squared Error
            "MAE" : k.arima_res_.mae,                          #Mean-Absolute Error 
            "params" : k.params().to_dict(),                   #Model parameters
            "diags" : get_diagnostics(k.arima_res_.resid)      #Getting model test results
        })           
    
    kARIMAsDf = pd.concat(
        [pd.DataFrame(kARIMARecords).drop(["params","diags"], axis=1),
        pd.json_normalize([m["diags"] for m in kARIMARecords]),
        pd.json_normalize([m["params"] for m in kARIMARecords])],
        axis=1
    )
    
    kARIMAsDf.to_clipboard()
    
    # 1.2 Setting up and fitting ARIMA parameters after selecting best model 
    kARIMA = pmdarima.ARIMA(
        order=(1,1,0),
        #trend="t" Not statistically significant
    ).fit(y)
    
    #assert np.allclose(kARIMA.params(), kARIMA[0].params(), atol=1e-6)

    # 1.3 Getting ARIMA parameters scores 
    kARIMAParam = kARIMA.summary().tables[1].data

    kARIMAParamDf = pd.DataFrame(
        kARIMAParam[1:],
        columns=kARIMAParam[0]
    ).set_index("")

    kARIMAParamDf.to_clipboard()

    ########## 2. Forecast future kappa for n-years ##########
    
    nForecast = yearsToForecast
    kForecast, confIntKForecast = kARIMA.predict(n_periods=nForecast, return_conf_int=True, alpha= 0.05)
    yearsForecast = np.arange(yearsPlot[-1] + 1, yearsPlot[-1] + 1 + nForecast)

    #Plot kappa and forecast
    plt.figure(figsize=(10, 6))
    plt.plot(yearsPlot, y, label="Observed Kappa (κ)", color="green")
    plt.plot(yearsPlot, kARIMA.fittedvalues(), label="Fitted Kappa (κ)", color="purple")
    plt.plot(yearsForecast, kForecast, label="Forecast Kappa (κ)", color="orange")
    plt.fill_between(
        yearsForecast,
        confIntKForecast[:,0],    #Lower conf int Kappa    
        confIntKForecast[:,1],    #Upper conf int Kappa
        color="orange",
        alpha=0.2,
        label="Confidence Interval",
    )

    plt.title("Forecast of Kappa (Time Effect) for " + field)
    plt.xlabel("Year")
    plt.ylabel("Kappa (κ)")
    plt.legend()
    plt.show()
    #"""

    ########## 3. Reconstruct mortality rates for actual and forecast years ##########

    fittedLCMortality = np.exp(
        aDf[aDf["Gender"]==field]["Alpha"].values.reshape(-1,1)
        + bDf[bDf["Gender"]==field]["Beta"].values.reshape(-1,1) 
        @ kARIMA.fittedvalues().values.reshape(1,-1)
    )

    fittedLCMortalityDf = pd.DataFrame(fittedLCMortality)


    """
    #Combine historical and forecasted mortality rates
    all_years = np.concatenate([yearsPlot, yearsForecast])
    all_mortality = np.hstack([qxMatrix.values, forecast_mortality])

    #Plot historical and forecasted mortality rates
    plt.figure(figsize=(12, 6))
    plt.imshow(all_mortality, aspect='auto', cmap='viridis', extent=[all_years[0], all_years[-1], agesPlot[-1], agesPlot[0]])
    plt.colorbar(label='Mortality Rate')
    plt.title("Historical and Forecasted Mortality Rates")
    plt.xlabel("Year")
    plt.ylabel("Age")
    plt.show()
    """