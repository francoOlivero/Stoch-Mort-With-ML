import pandas as pd
import numpy as np
import os

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import GridSearchCV, train_test_split

import seaborn as sns
import matplotlib.pyplot as plt

import RunParameters as rp
import UserDefinedFunctions as udf
import LC_01_BaseModel as lc1

########## 0. Inputs ##########
gDict = rp.gDict
gDictInv = rp.gDictInv
minTrainYr = rp.minTrainYr
maxTrainYr = rp.maxTrainYr
minTestYr = rp.minTestYr
maxTestYr = rp.maxTestYr
mxBEDfTrain = lc1.mxBEDfTrain
mxLC_Base_Df = lc1.mxLC_Base_Df

########## 1. Setting up ML feature ##########
mx_X = mxBEDfTrain.merge(mxLC_Base_Df, on= ["Age", "Year", "Gender"], how="inner").reset_index()
mx_X["Gender"] = mx_X["Gender"].map(gDict)
mx_X.insert(loc=3, column='Cohort', value= mx_X["Year"] - mx_X["Age"])
mx_X["Y_LC"] = mx_X["mx_BE"]/mx_X["mx_LC"]

########## 2.Defining Training and Testing data ##########

X_train = mx_X[(mx_X["Year"] >= minTrainYr) & (mx_X["Year"] <= maxTrainYr)][[ "Gender","Age","Year","Cohort",]] #DF
y_train = mx_X[(mx_X["Year"] >= minTrainYr) & (mx_X["Year"] <= maxTrainYr)]["Y_LC"] #Series

X_test = mx_X[(mx_X["Year"] >= minTestYr) & (mx_X["Year"] <= maxTestYr)][["Gender","Age","Year","Cohort"]] #DF
y_test = mx_X[(mx_X["Year"] >= minTestYr) & (mx_X["Year"] <= maxTestYr)]["Y_LC"] #Series

########## 3. Tuneo de hiperparametros usando GridSearchCV ##########
if rp.tunningFlag == True:
    # --- Decision Tree ---
    param_DT = {
        'max_depth': [5, 10, 20, 40],
        'min_samples_leaf': [1, 5, 10, 20]
    }

    grid_DT = GridSearchCV(
        estimator=DecisionTreeRegressor(random_state=1),
        param_grid=param_DT,
        scoring='neg_root_mean_squared_error',
        cv=5,
        n_jobs=-1,
        verbose=1
    )

    # --- Random Forest ---
    param_RF = {
        'n_estimators': [100, 200, 500],
        'max_depth': [3, 5, 6, 10, 20],
        'min_samples_leaf': [1, 2, 5]
    }

    grid_RF = GridSearchCV(
        estimator=RandomForestRegressor(random_state=1),
        param_grid=param_RF,
        scoring='neg_root_mean_squared_error',
        cv=5,
        n_jobs=-1,
        verbose=1
    )

    # --- Gradient Boosting ---
    param_GB = {
        'n_estimators': [200, 500, 1000],
        'learning_rate': [0.01, 0.005, 0.001],
        'max_depth': [3, 5, 6, 10 ,20]
    }

    grid_GB = GridSearchCV(
        estimator=GradientBoostingRegressor(random_state=1),
        param_grid=param_GB,
        scoring='neg_root_mean_squared_error',
        cv=5,
        n_jobs=-1,
        verbose=1
    ) 

    # Ajuste de hiperparametros y resumenes k-fold CV por modelo
    grid_DT.fit(X_train, y_train)
    cv_results_DT = pd.DataFrame(grid_DT.cv_results_).sort_values(by="mean_test_score", ascending=False)

    grid_RF.fit(X_train, y_train)
    cv_results_RF = pd.DataFrame(grid_RF.cv_results_).sort_values(by="mean_test_score", ascending=False)

    grid_GB.fit(X_train, y_train)
    cv_results_GB = pd.DataFrame(grid_GB.cv_results_).sort_values(by="mean_test_score", ascending=False)

    # Modelos ya entrenados con mejores hiperparametros
    mY_DT = grid_DT.best_estimator_
    mY_RF = grid_RF.best_estimator_
    mY_GB = grid_GB.best_estimator_
    
    # 3.1 Resumen de resultados de optimización CV 
    models_best = {
        "Decision Tree": grid_DT,
        "Random Forest": grid_RF,
        "Gradient Boosting": grid_GB
    }
    results_summary = []
    for name, model in models_best.items():
        y_pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = root_mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Seleccionar el objeto grid correspondiente
        if name == "Decision Tree":
            grid = grid_DT
        elif name == "Random Forest":
            grid = grid_RF
        else:
            grid = grid_GB
        
        best_params = grid.best_params_
        best_score_cv = -grid.best_score_  # Negativo porque GridSearch usa "neg_root_mean_squared_error"

        # Imprimir resumen individual
        print(f"\n{name} (Mejores hiperparámetros: {best_params})")
        print(f"MAE: {mae:.6f} | RMSE: {rmse:.6f} | R²: {r2:.6f} | RMSE(CV): {best_score_cv:.6f}")
        
        # Agregar fila al resumen
        results_summary.append({
            "Modelo": name,
            "Best_Params": best_params,
            "CV_RMSE": best_score_cv,
            "Test_MAE": mae,
            "Test_RMSE": rmse,
            "Test_R2": r2
        })

    tunning_df = pd.DataFrame(results_summary)  

else:    #Modelos tuneados período 1915-2010
    mY_DT = DecisionTreeRegressor(
        max_depth=5,
        min_samples_leaf=5,
        random_state=1
        )
    
    mY_RF = RandomForestRegressor(
        n_estimators=200,
        max_depth=5,
        min_samples_leaf=1,
        random_state=1
        )

    mY_GB = GradientBoostingRegressor(
        n_estimators=500,
        learning_rate=0.001,
        max_depth=10,
        random_state=1
        )
    mY_DT.fit(X_train, y_train)
    mY_RF.fit(X_train, y_train)
    mY_GB.fit(X_train, y_train)
    
# 3.2 Resumen de predicciones ML sobre todo el conjunto de datos

mY_ML_Df = mx_X[(mx_X["Year"] >= minTrainYr) & (mx_X["Year"] <= maxTrainYr)].copy()
mY_ML_Df["Gender"] = mY_ML_Df["Gender"].map(gDictInv)
mY_ML_Df["Y_DT"] = mY_DT.predict(X_train)
mY_ML_Df["Y_RF"] = mY_RF.predict(X_train)
mY_ML_Df["Y_GB"] = mY_GB.predict(X_train)

# 3.3 adding deltas to measure mort adjustments  
mY_ML_Df = udf.add_transformed_cols(
    df= mY_ML_Df,
    targetCols=["Y_LC", "Y_DT", "Y_RF", "Y_GB"], 
    function= lambda x: x-1, 
    prefix="delta_"
    )

# 3.4 Calculating LeeCarter-ML mx´s 
mY_ML_Df = udf.add_transformed_cols(
    df= mY_ML_Df,
    targetCols=["Y_DT", "Y_RF", "Y_GB"], 
    function= lambda x: x * mY_ML_Df["mx_LC"], 
    prefix="lc_mx_"
    )

# 3.5 Calculating LeeCarter-ML log mx´s 
mY_ML_Df = udf.add_transformed_cols(
    df= mY_ML_Df,
    targetCols=["Y_LC", "Y_DT", "Y_RF", "Y_GB", "mx_BE", "mx_LC", "lc_mx_Y_DT", "lc_mx_Y_RF", "lc_mx_Y_GB"], 
    function= np.log, 
    prefix="log_"
    )

########## 4.Analyzing ML results ##########

# === 2) Configuración general ===
#Original cmap = sns.diverging_palette(240, 10, as_cmap=True)
cmap = sns.diverging_palette(240, 10, s=85, l=50, n=20, as_cmap=True)
deltas = ["delta_Y_LC", "delta_Y_DT", "delta_Y_RF", "delta_Y_GB"]
model_labels = ["Lee-Carter (1)", "Decision Tree (2)", "Random Forest (3)", "Gradient Boost (4)"]
gender_map = {"Male": "Hombres", "Female": "Mujeres"}

# === 3) Crear figura (4 filas x 2 columnas) ===
# Tamaño adecuado para insertar en Word (aprox 20x25 cm)
fig, axes = plt.subplots(4, 2, figsize=(8, 10), sharex=False, sharey=False)

# === 4) Calcular límites globales del color (robustos) ===
vals = mY_ML_Df[deltas].values.flatten()
vals = vals[np.isfinite(vals)]
lim = np.nanpercentile(np.abs(vals), 98)
vmin, vmax = -lim, lim

# Rango total de años y edades
all_years = np.sort(mY_ML_Df["Year"].unique())
all_ages = np.sort(mY_ML_Df["Age"].unique())

# Definir ticks cada 15 unidades
year_ticks = all_years[::15] if len(all_years) > 15 else all_years
age_ticks = np.arange(0, 100, 15)  # edades cada 15 años

# === 5) Loop para generar heatmaps ===
for i, delta in enumerate(deltas):
    for j, gender in enumerate(gender_map.keys()):
        ax = axes[i, j]

        # Filtrar por género
        filtered = mY_ML_Df[mY_ML_Df["Gender"] == gender]

        # Pivot: filas = Edad, columnas = Año
        pivot_data = (
            filtered
            .pivot(index="Age", columns="Year", values=delta)
            .reindex(index=all_ages, columns=all_years)
        )

        # Crear heatmap
        sns.heatmap(
            pivot_data,
            ax=ax,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            cbar=False,
        )

        # === Ejes ===
        ax.invert_yaxis()
        ax.set_xlabel("Año", fontsize=9)
        ax.set_ylabel("Edad", fontsize=9)

        modelo = model_labels[i]
        genero = gender_map[gender]
        ax.set_title(f"Δmx — {modelo} — {genero}", fontsize=9)

        # Eje X: ticks cada 15 años
        tick_positions_x = [np.where(all_years == y)[0][0] + 0.5 for y in year_ticks]
        ax.set_xticks(tick_positions_x)
        ax.set_xticklabels(year_ticks, rotation=45, ha="right", fontsize=8)

        # Eje Y: ticks cada 15 edades
        valid_age_ticks = [a for a in age_ticks if a in all_ages]
        tick_positions_y = [np.where(all_ages == a)[0][0] + 0.5 for a in valid_age_ticks]
        ax.set_yticks(tick_positions_y)
        ax.set_yticklabels(valid_age_ticks, fontsize=8)

# === 6) Colorbar común ===
cbar_ax = fig.add_axes([0.93, 0.25, 0.02, 0.5])
norm = plt.Normalize(vmin=vmin, vmax=vmax)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, cax=cbar_ax)
cbar.ax.set_ylabel("Δ (ψ − 1)", rotation=270, labelpad=15, fontsize=9)

# === 7) Título general ===
fig.suptitle("Mapas de calor (Δmx = ψ − 1) por modelo y género", fontsize=11)

# === 8) Ajustar espaciado ===
plt.subplots_adjust(
    wspace=0.45,   # espacio horizontal
    hspace=0.8,    # espacio vertical
    right=0.9,
    top=0.93,
    bottom=0.07
)

# === 9) Guardar imagen automáticamente en Descargas ===
downloads_folder = os.path.join(os.path.expanduser("~"), "Downloads")
output_path = os.path.join(downloads_folder, f"Heatmaps_DeltaMx_Modelos_T{rp.minTrainYr}-{rp.maxTrainYr}_F{rp.minOOByr}-{rp.maxOOByr}.png")
fig.savefig(output_path, dpi=300, bbox_inches="tight")

print(f"\n✅ Imagen guardada correctamente en:\n{output_path}")

########## 5. metricas de bondad de ajuste ##########
# Definición de modelos y columnas a usar
models = [
    ("LC", "mx_LC",      "log_mx_LC"),
    ("DT", "lc_mx_Y_DT", "log_lc_mx_Y_DT"),
    ("RF", "lc_mx_Y_RF", "log_lc_mx_Y_RF"),
    ("GB", "lc_mx_Y_GB", "log_lc_mx_Y_GB"),
]
# Cuadro resumen por género
metrics_by_gender = []
for model_key, col_val, col_log in models:
    for gender in sorted(mY_ML_Df["Gender"].unique()):
        sub = mY_ML_Df[mY_ML_Df["Gender"] == gender]
        y_true = sub["mx_BE"]
        y_pred = sub[col_val]
        y_true_log = sub["log_mx_BE"]
        y_pred_log = sub[col_log]

        metrics_by_gender.append({
            "Model": model_key,
            "Gender": gender,
            "MAPE_pct": udf.mape(y_true, y_pred),
            "RMSE": udf.rmse(y_true, y_pred),
            "RMSLE": udf.rmsle_from_logs(y_true_log, y_pred_log),
        })

fitting_metrics_by_gender_df = pd.DataFrame(metrics_by_gender).sort_values(["Model", "Gender"])

if rp.tunningFlag == True: 
    udf.save_df_to_excel(rp.summaryFile, tunning_df, f"0.Tuning_T{rp.minTrainYr}-{rp.maxTrainYr}_F{rp.minOOByr}-{rp.maxOOByr}")
    udf.save_df_to_excel(rp.summaryFile, cv_results_DT, f"0.kCV-DT_T{rp.minTrainYr}-{rp.maxTrainYr}_F{rp.minOOByr}-{rp.maxOOByr}")
    udf.save_df_to_excel(rp.summaryFile, cv_results_RF, f"0.kCV-RF_T{rp.minTrainYr}-{rp.maxTrainYr}_F{rp.minOOByr}-{rp.maxOOByr}")
    udf.save_df_to_excel(rp.summaryFile, cv_results_GB, f"0.kCV-GB_T{rp.minTrainYr}-{rp.maxTrainYr}_F{rp.minOOByr}-{rp.maxOOByr}")
udf.save_df_to_excel(rp.summaryFile, mY_ML_Df, f"1.Y_ML_T{rp.minTrainYr}-{rp.maxTrainYr}_F{rp.minOOByr}-{rp.maxOOByr}")
udf.save_df_to_excel(rp.summaryFile, fitting_metrics_by_gender_df, f"2.Fitting_T{rp.minTrainYr}-{rp.maxTrainYr}_F{rp.minOOByr}-{rp.maxOOByr}")
