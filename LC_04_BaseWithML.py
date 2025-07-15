import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import RunParameters as rp
import UserDefinedFunctions as udf

import LC_01_BaseModel as lc1
import LC_02_kARIMA as lc2
import LC_03_ML as lc3

########## 0. Inputs ##########
targetFields = rp.genders
yearsToForecast = rp.yearsToForecast

aDf = lc1.aDf
bDf = lc1.bDf
yearsPlot = lc1.yearsPlot
agesPlot = lc1.agesPlot
mY_DT_Df = lc3.mY_DT_Df
kARIMA = lc2.kARIMA

########## 1. Apply Lee Carter model to mortality adjustments from ML model ##########
alphaAgg = []
betaAgg = []
kappaAgg = []
agesAgg = []
gendersAgg = []
yearsAgg = []
kappaGendersAgg = []

for field in targetFields:
    # 1.1 Preparing mx matrix for SVD process. 
    mxMatrix = mY_DT_Df[mY_DT_Df["Gender"]==field].pivot_table(values="mx_Y_DT", index="Age", columns="Year")
    mxMatrix.to_clipboard()
    # 1.2 LC params
    alpha_x, beta_x, kappa_t = udf.LeeCarterSVD(mxMatrix)

    # 1.3 Extract and aggregate Lee-Carter components
    alphaAgg.extend(alpha_x)
    betaAgg.extend(beta_x)
    kappaAgg.extend(kappa_t)

    gendersAgg.extend([field]*len(alpha_x))
    agesAgg.extend(mxMatrix.index.values)
    yearsAgg.extend(mxMatrix.columns.values)
    kappaGendersAgg.extend([field]*len(kappa_t))

########## 3. Preparing summary of LC model parameters and Df indexes-columns ##########
a_DT_Df = pd.DataFrame({"Age":agesAgg, "Gender":gendersAgg, "Alpha_DT":alphaAgg})
b_DT_Df = pd.DataFrame({"Age":agesAgg, "Gender":gendersAgg, "Beta_DT":betaAgg})
k_DT_Df = pd.DataFrame({"Year":yearsAgg, "Gender": kappaGendersAgg, "Kappa_DT":kappaAgg})

a_DT_Df.to_clipboard()
b_DT_Df.to_clipboard()
k_DT_Df.to_clipboard()

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

########## 3. Adjusting Kappa ##########

# 2.1 Setting up and fitting ARIMA parameters after selecting best model for both Male and Females
#kARIMA = pmdarima.ARIMA(order=(1,1,0)).fit(y) #trend="t" Not statistically significant
########## 3. Reconstruct mortality rates for actual and forecast years ##########

mx_LC_DT = []

for field in targetFields:
    mxLCByGender = np.exp(
        a_DT_Df[a_DT_Df["Gender"]==field]["Alpha_DT"].values.reshape(-1,1)
        + aDf[aDf["Gender"]==field]["Alpha"].values.reshape(-1,1)
        + bDf[bDf["Gender"]==field]["Beta"].values.reshape(-1,1) 
        @ kARIMA.fittedvalues().values.reshape(1,-1)        
        + b_DT_Df[b_DT_Df["Gender"]==field]["Beta_DT"].values.reshape(-1,1) 
        @ kARIMA.fittedvalues().values.reshape(1,-1)
        
    )

    mxLCByGenderDf = pd.DataFrame(mxLCByGender, index=agesPlot, columns=yearsPlot).rename_axis(index="Age", columns="Year")
    mxLCByGenderDf["Gender"] = field
    mxLCByGenderDf = mxLCByGenderDf.melt(id_vars="Gender", var_name="Year", value_name="mx_LC", ignore_index=False)
  
    mx_LC_DT.append(mxLCByGenderDf)
    
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
mx_LC_DT_Df = pd.concat(mx_LC_DT)
mx_LC_DT_Df.to_clipboard()

"""#Testing
mxLCFittedDf.to_clipboard()
#"""