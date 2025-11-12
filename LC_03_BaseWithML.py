import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import os

import RunParameters as rp
import UserDefinedFunctions as udf

import LC_01_BaseModel as lc1
import LC_02_ML as lc2

########## 0. Inputs ##########
targetFields = rp.genders
yearsToForecast = rp.yearsToForecast

aDf = lc1.aDf
bDf = lc1.bDf
kDf = lc1.kDf

yearsPlot = lc1.yearsPlot

agesPlot = lc1.agesPlot
mY_ML_Df = lc2.mY_ML_Df

# ML targets to decompose via Lee-Carter
ml_targets = {
    "Y_DT": "Y_DT",
    "Y_RF": "Y_RF",
    "Y_GB": "Y_GB",
}

########## 1. Apply Lee-Carter to each ML field (by gender) ##########
# Containers per model for LC components
a_ml = {}
b_ml = {}
k_ml = {}

for key, col in ml_targets.items():
    alphaAgg = []
    betaAgg = []
    kappaAgg = []
    agesAgg = []
    gendersAgg = []
    yearsAgg = []
    kappaGendersAgg = []
    
    for gender in targetFields:
        # Prepare matrix (ages x years) for the ML adjustment surface
        mxMatrix = (
            mY_ML_Df[mY_ML_Df["Gender"] == gender]
            .pivot_table(values=col, index="Age", columns="Year")
        )

        # Lee-Carter on the adjustment surface
        alpha_x, beta_x, kappa_t = udf.LeeCarterSVD(mxMatrix)

        # Aggregate components and identifiers
        alphaAgg.extend(alpha_x)
        betaAgg.extend(beta_x)
        kappaAgg.extend(kappa_t)

        gendersAgg.extend([gender] * len(alpha_x))
        agesAgg.extend(mxMatrix.index.to_numpy())
        yearsAgg.extend(mxMatrix.columns.to_numpy())
        kappaGendersAgg.extend([gender] * len(kappa_t))

    # Component DataFrames for this ML model
    a_ml[key] = pd.DataFrame({"Age": agesAgg, "Gender": gendersAgg, f"Alpha_{key}": alphaAgg})
    b_ml[key] = pd.DataFrame({"Age": agesAgg, "Gender": gendersAgg, f"Beta_{key}": betaAgg})
    k_ml[key] = pd.DataFrame({"Year": yearsAgg, "Gender": kappaGendersAgg, f"Kappa_{key}": kappaAgg})

########## 2. Consolidate kappa_t from all ML models for ARIMA analysis ##########
a_list = []
b_list = []
k_list = []

for key, df in a_ml.items():
    # rename la columna alpha y agregar el identificador del modelo
    df_temp = df.copy()
    df_temp = df_temp.rename(columns={f"Alpha_{key}": "alpha_x"})
    df_temp["Model"] = key.replace("Y_", "")
    a_list.append(df_temp)

# Unir todo en un único DataFrame
df_a_lc = lc1.aDf.assign(Model="LC")
df_a_lc = df_a_lc.rename(columns={"Alpha":"alpha_x"})
a_list.append(df_a_lc)
df_a_all = pd.concat(a_list, ignore_index=True)

for key, df in b_ml.items():
    # rename la columna alpha y agregar el identificador del modelo
    df_temp = df.copy()
    df_temp = df_temp.rename(columns={f"Beta_{key}": "beta_x"})
    df_temp["Model"] = key.replace("Y_", "")
    b_list.append(df_temp)

# Unir todo en un único DataFrame
df_b_lc = lc1.bDf.assign(Model="LC")
df_b_lc = df_b_lc.rename(columns={"Beta":"beta_x"})
b_list.append(df_b_lc)
df_b_all = pd.concat(b_list, ignore_index=True)

for key, df in k_ml.items():
    # rename la columna kappa y agregar el identificador del modelo
    df_temp = df.copy()
    df_temp = df_temp.rename(columns={f"Kappa_{key}": "kappa_t"})
    df_temp["Model"] = key.replace("Y_", "")
    k_list.append(df_temp)

# Unir todo en un único DataFrame
df_kappa_lc = lc1.kDf.assign(Model="LC")
df_kappa_lc = df_kappa_lc.rename(columns={"Kappa":"kappa_t"})
k_list.append(df_kappa_lc)
df_k_all = pd.concat(k_list, ignore_index=True)

udf.save_df_to_excel(rp.summaryFile,df_a_all, f"3.LC_Ax_{rp.minTrainYr}-{rp.maxTrainYr}")
udf.save_df_to_excel(rp.summaryFile,df_b_all, f"4.LC_Bx_{rp.minTrainYr}-{rp.maxTrainYr}")
udf.save_df_to_excel(rp.summaryFile,df_k_all, f"5.LC_Kt_{rp.minTrainYr}-{rp.maxTrainYr}")

