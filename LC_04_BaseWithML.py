import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import RunParameters as rp
import UserDefinedFunctions as udf

import LC_01_BaseModel as lc1
import LC_02_kARIMA as lc2
import LC_03_ML as lc3

########## Inputs ##########
targetFields = rp.genders
yearsToForecast = rp.yearsToForecast

aDf = lc1.aDf
bDf = lc1.bDf
yearsPlot = lc1.yearsPlot
agesPlot = lc1.agesPlot

########## 2. Forecast future kappa for n-years ##########
    
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

########## 3. Reconstruct mortality rates for actual and forecast years ##########

mxLC_DT = []

for field in targetFields:
    mxLCByGender = np.exp(
        aDf[aDf["Gender"]==field]["Alpha"].values.reshape(-1,1)
        + bDf[bDf["Gender"]==field]["Beta"].values.reshape(-1,1) 
        @ lc2.kARIMA.fittedvalues().values.reshape(1,-1)
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
mxLCDf = pd.concat(mxLC)

"""#Testing
mxLCFittedDf.to_clipboard()
#"""