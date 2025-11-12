import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import os

import RunParameters as rp
import UserDefinedFunctions as udf

import LC_01_BaseModel as lc1
import LC_02_ML as lc2
import LC_03_BaseWithML as lc3
import LC_04_kARIMA as lc4

########## 0. Inputs ##########
targetFields = rp.genders
yearsToForecast = rp.yearsToForecast
minTestYr = rp.minTestYr
maxTestYr = rp.maxTestYr

yearsPlot = lc1.yearsPlot
agesPlot = lc1.agesPlot

# Model targets
model_targets = {
    "LC": "mx_LC",
    "DT": "mx_Y_DT",
    "RF": "mx_Y_RF",
    "GB": "mx_Y_GB",
}

df_a_all = lc3.df_a_all
df_b_all = lc3.df_b_all
df_k_combined_all = lc4.df_k_combined_all

########## 1. Reconstruct mortality surfaces (LC base + ML LC, con κ_t proyectados) ##########

mx_LC_by_model = {}

for key in model_targets.keys():
    mx_list = []

    for gender in targetFields:
        # === Filtrar kappas observadas + proyectadas para el modelo y género ===
        k_lc = (
            df_k_combined_all[(df_k_combined_all["Gender"] == gender) & (df_k_combined_all["Model"] == "LC")]
            .sort_values("Year")["kappa_t"].to_numpy()
        )   
        
        k_Y = (
            df_k_combined_all[(df_k_combined_all["Gender"] == gender) & (df_k_combined_all["Model"] == key)]
            .sort_values("Year")["kappa_t"].to_numpy()
        )   

        # === Extender los años si hay forecast === #
        years_full = np.arange(yearsPlot[0], yearsPlot[0] + len(k_Y))
    
        # === Recalcular mx = e^ (a + a(y) + b·κ + b(y)·κ(y)) para LC base + ML === #
        a_lc = df_a_all[(df_a_all["Gender"] == gender) & (df_a_all["Model"] == "LC")]["alpha_x"].values.reshape(-1, 1)
        b_lc = df_b_all[(df_b_all["Gender"] == gender) & (df_b_all["Model"] == "LC")]["beta_x"].values.reshape(-1, 1)

        a_Y = df_a_all[(df_a_all["Gender"] == gender) & (df_a_all["Model"] == key)]["alpha_x"].values.reshape(-1, 1)
        b_Y = df_b_all[(df_b_all["Gender"] == gender) & (df_b_all["Model"] == key)]["beta_x"].values.reshape(-1, 1)

        if key == "LC":
            mxLCByGender = np.exp(
                a_lc
                + b_lc @ k_lc.reshape(1, -1)
            )
        else:
            mxLCByGender = np.exp(
                a_lc 
                + a_Y
                + b_lc @ k_lc.reshape(1, -1) 
                + b_Y @ k_Y.reshape(1, -1)
            )

        # === Construir DataFrame de superficie === #
        mxLCByGenderDf = (
            pd.DataFrame(mxLCByGender, index=agesPlot, columns=years_full)
            .rename_axis(index="Age", columns="Year")
        )
        mxLCByGenderDf["Gender"] = gender

        mxLCByGenderDf = mxLCByGenderDf.melt(
            id_vars="Gender", var_name="Year", value_name="mx", ignore_index=False
        )
        mx_list.append(mxLCByGenderDf)

    mx_LC_by_model[key] = pd.concat(mx_list)

# === Superficies por modelo ===
LC_mx_Df = mx_LC_by_model["LC"]
LC_DT_Df = mx_LC_by_model["DT"]
LC_RF_Df = mx_LC_by_model["RF"]
LC_GB_Df = mx_LC_by_model["GB"]

########## 3b. Preparar DataFrame combinado de todas las superficies ##########
BE_mx_DF = lc1.mxBEDfAll.reset_index().assign(Model="BE")
BE_mx_DF.rename(columns={"mx_BE": "mx"}, inplace=True)

LC_mx_Df = LC_mx_Df.reset_index().assign(Model="LC")
LC_DT_Df = LC_DT_Df.reset_index().assign(Model="DT")
LC_RF_Df = LC_RF_Df.reset_index().assign(Model="RF")
LC_GB_Df = LC_GB_Df.reset_index().assign(Model="GB")

# Concatenar todas las superficies
mx_LC_All_Df = pd.concat(
    [BE_mx_DF, LC_mx_Df, LC_DT_Df, LC_RF_Df, LC_GB_Df],
    ignore_index=True
)

mx_LC_All_Df["log_mx"] = np.log(mx_LC_All_Df["mx"])

# Definición de modelos y columnas a usar
models = [
    ("LC", "mx", "log_mx"),
    ("DT", "mx", "log_mx"),
    ("RF", "mx", "log_mx"),
    ("GB", "mx", "log_mx"),
]
# Cuadro resumen por género para métricas de poder predictivo
metrics_by_gender = []

mx_forecasted = mx_LC_All_Df[(mx_LC_All_Df["Year"] >= minTestYr) & (mx_LC_All_Df["Year"] <= maxTestYr)]
    
for model_key, col_val, col_log in models:
    for gender in sorted(mx_forecasted["Gender"].unique()):
        sub = mx_forecasted[mx_forecasted["Gender"] == gender]
        
        y_true = sub[sub["Model"] == "BE"][col_val]
        y_pred = sub[sub["Model"] == model_key][col_val]
        y_true_log = sub[sub["Model"] == "BE"][col_log]
        y_pred_log = sub[sub["Model"] == model_key][col_log]

        metrics_by_gender.append({
            "Model": model_key,
            "Gender": gender,
            "MAPE_pct": udf.mape(y_true, y_pred),
            "RMSE": udf.rmse(y_true, y_pred),
            "RMSLE": udf.rmsle_from_logs(y_true_log, y_pred_log),
        })

forecasting_metrics_by_gender_df = pd.DataFrame(metrics_by_gender).sort_values(["Model", "Gender"])

udf.save_df_to_excel(rp.summaryFile,mx_LC_All_Df, f"9.LC_mx_ML_{rp.minTrainYr}-{rp.maxTrainYr}")
udf.save_df_to_excel(rp.summaryFile,forecasting_metrics_by_gender_df, f"10.LC_Forecasting_{rp.minTrainYr}-{rp.maxTrainYr}")