import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pmdarima
import os

import RunParameters as rp
import UserDefinedFunctions as udf
import LC_01_BaseModel as lc1
import LC_03_BaseWithML as lc3


########## 0. Inputs ##########
targetFields = rp.genders
yearsToForecast = rp.yearsToForecast

yearsPlot = lc1.yearsPlot
agesPlot = lc1.agesPlot
df_k_all = lc3.df_k_all

########## 1. Configuración de modelos ARIMA ##########
kARIMA_param_config = {
    ("Male", "LC"): {"final_order": (0, 1, 0)},
    ("Male", "DT"): {"final_order": (0, 1, 0)},
    ("Male", "RF"): {"final_order": (0, 1, 0)},
    ("Male", "GB"): {"final_order": (0, 1, 0)},
    ("Female", "LC"): {"final_order": (0, 1, 0)},
    ("Female", "DT"): {"final_order": (0, 1, 0)},
    ("Female", "RF"): {"final_order": (0, 1, 0)},
    ("Female", "GB"): {"final_order": (0, 1, 0)},
}


########## 2. Ajuste y proyección ##########
all_kARIMAs = []
all_kARIMA_params = []
all_kappa_series = []

for model in df_k_all["Model"].unique():
    for field in targetFields:
        y = (
            df_k_all[
                (df_k_all["Gender"] == field) &
                (df_k_all["Model"] == model)
            ]
            .sort_values("Year")["kappa_t"]
        )

        if y.empty:
            continue

        # Auto-fit ARIMA
        kARIMAs = pmdarima.auto_arima(
            y,
            start_p=0, start_q=0,
            max_p=3, max_q=3,
            seasonal=False,
            information_criterion="bic",
            stepwise=False,
            suppress_warnings=True,
            return_valid_fits=True,
            trace=False,
            with_intercept=True 
        )

        # === 2.1 Resumen de modelos probados ===
        kARIMAsDf = udf.ARIMAsGrid(kARIMAs)
        kARIMAsDf.insert(0, "Gender", field)
        kARIMAsDf.insert(1, "Model", model)
        all_kARIMAs.append(kARIMAsDf)
       
        # === 2.2 Ajuste del modelo final según configuración ===
        kARIMA = pmdarima.ARIMA(order=kARIMA_param_config[(field, model)]["final_order"]).fit(y)

        # === 2.3 Guardar parámetros ===
        kARIMAParamByGender = kARIMA.summary().tables[1].data
        kARIMAParamByGender[0][0] = "Parameter"
        kARIMAParamDfByGender = pd.DataFrame(kARIMAParamByGender[1:], columns=kARIMAParamByGender[0])
        kARIMAParamDfByGender.insert(0, "Gender", field)
        kARIMAParamDfByGender.insert(1, "Model", model)
        all_kARIMA_params.append(kARIMAParamDfByGender)

        ########## 3. Forecast futuro ##########
        nForecast = yearsToForecast
        kForecast, confIntKForecast = kARIMA.predict(
            n_periods=nForecast,
            return_conf_int=True,
            alpha=0.05
        )
        yearsForecast = np.arange(yearsPlot[-1] + 1, yearsPlot[-1] + 1 + nForecast)

        # --- Observado ---
        df_kappa_obs = pd.DataFrame({
            "Year": yearsPlot,
            "Gender": field,
            "kappa_t": y.values,
            "Model": model,
            "Series_Type": "Observed",
            "Conf-Interval-ARIMA": [np.nan] * len(y),
            "Conf_Lower": [np.nan] * len(y),
            "Conf_Upper": [np.nan] * len(y)
        })

        # --- Proyectado (con intervalos de confianza) ---
        df_kappa_proj = pd.DataFrame({
            "Year": yearsForecast,
            "Gender": field,
            "kappa_t": kForecast,
            "Model": model,
            "Series_Type": "Projected",
            "Conf-Interval-ARIMA": [
                f"[{round(ci[0], 6)}, {round(ci[1], 6)}]" for ci in confIntKForecast
            ],
            "Conf_Lower": confIntKForecast[:, 0],
            "Conf_Upper": confIntKForecast[:, 1]
        })

        # --- Combinar ---
        df_kappa_combined = pd.concat([df_kappa_obs, df_kappa_proj], ignore_index=True)
        all_kappa_series.append(df_kappa_combined)


########## 4. Consolidar resultados ##########
kARIMAsDf_All = pd.concat(all_kARIMAs, ignore_index=True)
kARIMAParamDfByGender_All = pd.concat(all_kARIMA_params, ignore_index=True)
df_k_combined_all = pd.concat(all_kappa_series, ignore_index=True)

########## 5. Gráficos de líneas por modelo y género ##########

# Ordenar modelos: primero Lee-Carter
model_labels = sorted(df_k_combined_all["Model"].unique(), key=lambda x: 0 if x == "LC" else 1)

# Nombres descriptivos para títulos
model_name_map = {
    "LC": r"$\kappa_t$" + " — Lee–Carter (1)",
    "DT": r"$\kappa_t^{\psi}$" + " — Decision Tree (2)",
    "RF": r"$\kappa_t^{\psi}$" + " — Random Forest (3)",
    "GB": r"$\kappa_t^{\psi}$" + " — Gradient Boosting (4)"
}

gender_map = {"Male": "Hombres", "Female": "Mujeres"}

# === Crear figura (4x2) ===
fig, axes = plt.subplots(4, 2, figsize=(8, 10), sharex=False, sharey=False)

# Ejes globales
all_years = np.sort(df_k_combined_all["Year"].unique())
year_ticks = all_years[::10] if len(all_years) > 10 else all_years

# === Loop modelos x géneros ===
for i, model in enumerate(model_labels):
    for j, (gender, genero_str) in enumerate(gender_map.items()):
        ax = axes[i, j]

        subset = df_k_combined_all[
            (df_k_combined_all["Model"] == model) &
            (df_k_combined_all["Gender"] == gender)
        ]

        # --- Línea Observada (negra) ---
        sns.lineplot(
            data=subset[subset["Series_Type"] == "Observed"],
            x="Year", y="kappa_t",
            label="Observado",
            color="black",
            ax=ax,
            linewidth=1.0
        )

        # --- Línea Proyectada (azul eléctrico, continua) ---
        sns.lineplot(
            data=subset[subset["Series_Type"] == "Projected"],
            x="Year", y="kappa_t",
            label="Proyectado",
            color="#0077FF",
            ax=ax,
            linewidth=1.0
        )

        # --- Área de intervalo de confianza ---
        projected = subset[subset["Series_Type"] == "Projected"]
        if not projected.empty and projected["Conf_Lower"].notna().any():
            ax.fill_between(
                projected["Year"],
                projected["Conf_Lower"],
                projected["Conf_Upper"],
                color="#0077FF",
                alpha=0.2,
                label="IC 95%"
            )

        # === Etiquetas y formato ===
        ax.set_xlabel("Año", fontsize=9)
        ax.set_ylabel(r"$\kappa_t$" if model == "k_LC" else r"$\kappa_t^{\psi}$", fontsize=9)

        model_fullname = model_name_map.get(model, model)
        ax.set_title(f"{model_fullname} — {genero_str}", fontsize=9)

        ax.set_xticks(year_ticks)
        ax.set_xticklabels(year_ticks, rotation=45, ha="right", fontsize=8)
        ax.yaxis.set_major_locator(plt.MaxNLocator(8))
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(fontsize=7, loc="best", frameon=False)

# === Título general ===
fig.suptitle("Evolución de " + r"$\kappa_t$" + " y " + r"$\kappa_t^{\psi}$" +
             " — Observado, Proyectado e Intervalos de Confianza (95%)", fontsize=11)

# === Ajuste de espaciado ===
plt.subplots_adjust(
    wspace=0.35,
    hspace=0.8,
    right=0.92,
    top=0.93,
    bottom=0.07
)

# === Guardar imagen ===
downloads_folder = os.path.join(os.path.expanduser("~"), "Downloads")
output_path = os.path.join(downloads_folder, f"Kappa_Series_Modelos_{rp.minTrainYr}-{rp.maxTrainYr}.png")

fig.savefig(output_path, dpi=300, bbox_inches="tight")
print(f"\n✅ Imagen guardada correctamente en:\n{output_path}")

udf.save_df_to_excel(rp.summaryFile,kARIMAsDf_All, f"6.kARIMA_Models_{rp.minTrainYr}-{rp.maxTrainYr}")
udf.save_df_to_excel(rp.summaryFile,kARIMAParamDfByGender_All, f"7.kARIMA_Params_{rp.minTrainYr}-{rp.maxTrainYr}")
udf.save_df_to_excel(rp.summaryFile,df_k_combined_all, f"8.kARIMA_Kappa_{rp.minTrainYr}-{rp.maxTrainYr}")