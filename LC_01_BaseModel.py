import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import RunParameters as rp
import UserDefinedFunctions as udf

########## 0. Inputs ##########
mxRatesPath = rp.mxRates #Mortality matrix from HMD
initCalendarYear = rp.initCalendarYear
maxAge = rp.maxAge
targetFields = rp.genders
targetIndexes = rp.headers

########## 1. Preparing Data ##########
mxRates = pd.read_csv(mxRatesPath, sep="\s+", header=1)

# 1.1 Cleaning up and defining formats, setting zero to NaN and filtering
mxRates["Age"] = mxRates["Age"].replace("110+", 110).astype(int)

mxRates[targetFields] = (
    mxRates[targetFields]
    .astype(str)
    .replace(r"[^\d.]+", "", regex=True)
    .apply(pd.to_numeric, errors="coerce")
    .replace(0.0, np.nan)
)   

mxRates = mxRates[mxRates["Year"]>=initCalendarYear] 
mxRates = mxRates[mxRates["Age"]<= maxAge]
mxRates = mxRates[(targetIndexes + targetFields)]

# 1.2 Setting Output for ML feature
mxBEDf = mxRates.melt(id_vars=("Year", "Age"), var_name="Gender", value_name="mx_BE").set_index("Age")

########## 2. LC parameter estimations usign LC SVD ##########
alphaAgg = []
betaAgg = []
kappaAgg = []
agesAgg = []
gendersAgg = []
yearsAgg = []
kappaGendersAgg = []

for field in targetFields:
    # 2.1 Preparing mx matrix for SVD process. For NaN, interpolate function repeats the last value. 
    mxMatrix = mxRates.pivot_table(values=field, index="Age", columns="Year")
    mxMatrix = mxMatrix.interpolate(axis=0, method="linear")    #Axis=0 stands for rows *it may impact the final values.
    
    # 2.2 LC params
    alpha_x, beta_x, kappa_t = udf.LeeCarterSVD(mxMatrix)

    # 2.3 Extract and aggregate Lee-Carter components
    alphaAgg.extend(alpha_x)
    betaAgg.extend(beta_x)
    kappaAgg.extend(kappa_t)

    gendersAgg.extend([field]*len(alpha_x))
    agesAgg.extend(mxMatrix.index.values)
    yearsAgg.extend(mxMatrix.columns.values)
    kappaGendersAgg.extend([field]*len(kappa_t))

########## 3. Preparing summary of LC model parameters and Df indexes-columns ##########
yearsPlot = mxMatrix.columns.tolist()
agesPlot = mxMatrix.index.to_list()

aDf = pd.DataFrame({"Age":agesAgg, "Gender":gendersAgg, "Alpha":alphaAgg})
bDf = pd.DataFrame({"Age":agesAgg, "Gender":gendersAgg, "Beta":betaAgg})
kDf = pd.DataFrame({"Year":yearsAgg, "Gender": kappaGendersAgg, "Kappa":kappaAgg})

"""#Plot LC parameters
sns.relplot(x="Age", y="Alpha", data=aDf, hue= "Gender")
sns.relplot(x="Age", y="Beta", data=bDf,  hue= "Gender")
sns.relplot(x="Year", y="Kappa", data=kDf,  hue= "Gender")
plt.show()
#"""
