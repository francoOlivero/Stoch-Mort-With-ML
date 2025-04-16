import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from scipy.linalg import svd

#Mortality matrix from HMD

#qx_rates_path = r"C:\Users\frank\Downloads\PyUtilities\Stoch-Mort-With-ML\Docs\ITA\STATS\Deaths_1x1.txt"
#qx_rates_path = r"C:\Users\frank\Downloads\PyUtilities\Stoch-Mort-With-ML\Docs\ITA\STATS\Population.txt"

#Inputs
qxRatesPath = r"C:\Users\frank\Downloads\PyUtilities\Stoch-Mort-With-ML\Docs\ITA\STATS\Mx_1x1.txt"
initYear = 1910

#Step 0: Preparing Data
qxRates = pd.read_csv(qxRatesPath, sep="\s+", header=1)
qxRates.to_clipboard()

qxRates["Age"] = qxRates["Age"].replace("110+", 110).astype(int)
qxRates[["Female", "Male", "Total"]] = (
    qxRates[["Female", "Male", "Total"]]
    .astype(str)
    .replace(r"[^\d.]+", "", regex=True)
    .apply(pd.to_numeric, errors="coerce")
    .replace(0.0, np.nan)
)


qxRates = qxRates[qxRates["Year"]>=initYear] 
qxRates.to_clipboard()

print(qxRates.info())

qxRates = qxRates.pivot_table(values="Total", index="Age", columns="Year")
qxRates = qxRates.interpolate(axis=0, method="linear") #For NaN, this function repeats the last value. *it may impact the final values.

qxRates.to_clipboard()

#Step 1: Log-transform mortality rates
qxLog = np.log(qxRates)

qxLogCentered = qxLog - qxLog.mean(axis=1).values[:, None]

#Step 2: Singular Value Decomposition (SVD) for Lee-Carter decomposition
U, S, Vt = svd(qxLogCentered)

# Extract Lee-Carter components
alpha_x = qxLogCentered.mean(axis=1)  # Average mortality across time
beta_x = U[:, 0]  # Age effect
kappa_t = S[0] * Vt[0, :]  # Time-varying component

#print(type(alpha_x), alpha_x, alpha_x.shape)
#print(type(beta_x), beta_x, beta_x.shape)
#print(type(kappa_t), kappa_t, kappa_t.shape)

yearsPlot = qxRates.columns.tolist()
agesPlot = qxRates.index.to_list()

alphaDf = pd.DataFrame(alpha_x, index=agesPlot, columns=["Alpha"]).rename_axis(index="Age")
betaDf  = pd.DataFrame(beta_x, index=agesPlot, columns=["Beta"]).rename_axis(index="Age")
kappaDf = pd.DataFrame(kappa_t, index=yearsPlot, columns=["Kappa"]).rename_axis(index= "Year")

alphaDf.to_clipboard()

print(alphaDf)
print(betaDf)
print(kappaDf)

# Plot components
"""sns.relplot(x="Age", y="Alpha", data=alphaDf)
sns.relplot(x="Age", y="Beta", data=betaDf)
sns.relplot(x="Year", y="Kappa", data=kappaDf)
plt.show()"""


#Step 3: Fit ARIMA model to kappa_t (time-varying component)
p, d, q = 1, 1, 1  # Example ARIMA parameters
LC_ARIMA = ARIMA(kappa_t, order=(p, d, q))
LC_ARIMA_fitted = LC_ARIMA.fit()

print(LC_ARIMA_fitted.summary())

LC_ARIMA_fitted.summary()

# Forecast future kappa_t
forecast_steps = 10
forecast_kappa = LC_ARIMA_fitted.get_forecast(steps=forecast_steps)
forecast_index = np.arange(yearsPlot[-1] + 1, yearsPlot[-1] + 1 + forecast_steps)
forecast_mean = forecast_kappa.predicted_mean
forecast_conf_int = forecast_kappa.conf_int()

# Plot kappa_t and forecast
"""plt.figure(figsize=(10, 6))
plt.plot(yearsPlot, kappa_t, label="Observed Kappa (κ)", color="green")
plt.plot(forecast_index, forecast_mean, label="Forecast Kappa (κ)", color="orange")
plt.fill_between(
    forecast_index,
    forecast_conf_int[:, 0],
    forecast_conf_int[:, 1],
    color="orange",
    alpha=0.2,
    label="Confidence Interval",
)

plt.title("Forecast of Kappa (Time Effect)")
plt.xlabel("Year")
plt.ylabel("Kappa (κ)")
plt.legend()
plt.show()"""

#Step 4: Reconstruct mortality rates for forecast years
forecast_log_mortality = alphaDf["Alpha"].values.reshape(-1,1) + betaDf["Beta"].values.reshape(-1,1) @ forecast_mean.reshape(1,-1)
forecast_mortality = np.exp(forecast_log_mortality)

# Combine historical and forecasted mortality rates
all_years = np.concatenate([yearsPlot, forecast_index])
all_mortality = np.hstack([qxRates.values, forecast_mortality])

# Plot historical and forecasted mortality rates
plt.figure(figsize=(12, 6))
plt.imshow(all_mortality, aspect='auto', cmap='viridis', extent=[all_years[0], all_years[-1], agesPlot[-1], agesPlot[0]])
plt.colorbar(label='Mortality Rate')
plt.title("Historical and Forecasted Mortality Rates")
plt.xlabel("Year")
plt.ylabel("Age")
plt.show()
