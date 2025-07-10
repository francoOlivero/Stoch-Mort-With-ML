import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import pmdarima 

import RunParameters as rp
import UserDefinedFunctions as udf
import LC_01_BaseModel as lc1

########## 0. Inputs ##########
targetFields = rp.genders
yearsToForecast = rp.yearsToForecast

aDf = lc1.aDf
bDf = lc1.bDf
kDf = lc1.kDf
yearsPlot = lc1.yearsPlot
agesPlot = lc1.agesPlot

########## 1. Auto-fitting ARIMA models to kappa (time-varying component) ##########
mxLC = []

for field in targetFields:
    y = kDf[kDf["Gender"]==field]["Kappa"]          #Kappa Input 

    kARIMAs = pmdarima.auto_arima(
        y,
        start_p=0, start_q=0,                       #Starting AR(), MA() parameters
        max_p=3, max_q=3,                           #Max AR(), MA() parameters
        seasonal=False,                             #Set to true only for seasonal data
        information_criterion="bic",                #AIC or BIC
        stepwise=False,                             #Complete search iterating through all param combinations
        suppress_warnings=True,                     #Only warnings related to indexing stuff, not relevant
        return_valid_fits=True,                     #Returns every model tried
        trace=True,                                 #See progress on each iteration
        #trend="t"                                  #Linear Time Trend Not statistically significant
    )                  

    ########## 2. Summary of ARIMA models ##########
    kARIMAsDf = udf.ARIMAsGrid(kARIMAs)
    kARIMAsDf.insert(0, "Gender", field)

    #Testing   
    kARIMAsDf.to_clipboard()
    #"""
    
    # 2.1 Setting up and fitting ARIMA parameters after selecting best model for both Male and Females
    kARIMA = pmdarima.ARIMA(order=(1,1,0)).fit(y) #trend="t" Not statistically significant
    
    # 2.2 Getting ARIMA parameters scores 
    kARIMAParamByGender= kARIMA.summary().tables[1].data
    kARIMAParamByGender[0][0] = "Parameter"
    kARIMAParamDfByGender = pd.DataFrame(kARIMAParamByGender[1:], columns=kARIMAParamByGender[0] )
    kARIMAParamDfByGender.insert(0, "Gender", field)

    #Testing   
    #kARIMAParamDfByGender.to_clipboard()
    #"""

    ########## 3. Forecast future kappa for n-years ##########
    
    """#Testing
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

    ########## 4. Reconstruct mortality rates for actual and forecast years ##########

    mxLCByGender = np.exp(
        aDf[aDf["Gender"]==field]["Alpha"].values.reshape(-1,1)
        + bDf[bDf["Gender"]==field]["Beta"].values.reshape(-1,1) 
        @ kARIMA.fittedvalues().values.reshape(1,-1)
    )

    mxLCByGenderDf = pd.DataFrame(mxLCByGender, index=agesPlot, columns=yearsPlot).rename_axis(index="Age", columns="Year")
    mxLCByGenderDf["Gender"] = field
    mxLCByGenderDf = mxLCByGenderDf.melt(id_vars="Gender", var_name="Year", value_name="mx_LC", ignore_index=False)
  
    mxLC.append(mxLCByGenderDf)
    
    """
    mxMatrix = lc.mxMatrix    
    #Combine historical and forecasted mortality rates
    all_years = np.concatenate([yearsPlot, yearsForecast])
    all_mortality = np.hstack([mxMatrix.values, forecast_mortality])

    #Plot historical and forecasted mortality rates
    plt.figure(figsize=(12, 6))
    plt.imshow(all_mortality, aspect='auto', cmap='viridis', extent=[all_years[0], all_years[-1], agesPlot[-1], agesPlot[0]])
    plt.colorbar(label='Mortality Rate')
    plt.title("Historical and Forecasted Mortality Rates")
    plt.xlabel("Year")
    plt.ylabel("Age")
    plt.show()
    """
mxLC_Base_Df = pd.concat(mxLC)

"""#Testing
mxLCFittedDf.to_clipboard()
#"""