import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from scipy.linalg import svd

#Inputs
qxRatesPath = r"C:\Users\frank\Downloads\PyUtilities\Stoch-Mort-With-ML\Docs\ITA\STATS\Mx_1x1.txt" #Mortality matrix from HMD
initCalendarYear = 1910

#Step 0: Preparing Data
qxRates = pd.read_csv(qxRatesPath, sep="\s+", header=1)

#Cleaning up and defining formats, setting zero to NaN
qxRates["Age"] = qxRates["Age"].replace("110+", 110).astype(int)
qxRates[["Female", "Male", "Total"]] = (
    qxRates[["Female", "Male", "Total"]]
    .astype(str)
    .replace(r"[^\d.]+", "", regex=True)
    .apply(pd.to_numeric, errors="coerce")
    .replace(0.0, np.nan)
)   

#Filtering relevant years
qxRates = qxRates[qxRates["Year"]>=initCalendarYear] 

#Preparing qx matrix for SVD process 
qxRatesPivot = qxRates.pivot_table(values="Total", index="Age", columns="Year")

#Cleaning up qx matrix. For NaN, this function repeats the last value. Axis=0 stands for rows *it may impact the final values.
qxRatesPivot = qxRatesPivot.interpolate(axis=0, method="linear") 

"""#Testing
qxRatesPivot.to_clipboard()
#"""

#Step 1: Log-transform mortality rates
qxLog = np.log(qxRatesPivot)

qxLogCentered = qxLog - qxLog.mean(axis=1).values.reshape(-1,1) #Axis=1 stands for average of all columns by row.

#Step 2: Singular Value Decomposition (SVD) for Lee-Carter decomposition
U, S, Vt = svd(qxLogCentered, full_matrices=False)

"""#Testing
print("U, S, Vt Shapes: ", U.shape, S.shape, Vt.shape)
qxLogCenteredReplica = U @ np.diag(S) @ Vt
testSVD = np.allclose(qxLogCentered, qxLogCenteredReplica) #Check if arrays are equal
if testSVD: print("SVD Test Succesful")
#"""

# Extract Lee-Carter components
alpha_x = qxLog.mean(axis=1)  # Average mortality across time
beta_x = U[:, 0]/sum(U[:, 0])  # Age effect, Beta is normalized to get the unique model solution, it does not impact forecasted results though.
kappa_t = sum(U[:, 0]) * S[0] * Vt[0, :]  # Time-varying component, adjusted by Beta normalization factor.

"""#Testing
print(type(alpha_x), alpha_x, alpha_x.shape)
print(type(beta_x), beta_x, beta_x.shape)
print(type(kappa_t), kappa_t, kappa_t.shape)
#"""

yearsPlot = qxRatesPivot.columns.tolist()
agesPlot = qxRatesPivot.index.to_list()

yearsPlotDf = pd.DataFrame(yearsPlot)
agesPlotDf = pd.DataFrame(agesPlot)

alphaDf = pd.DataFrame(alpha_x, index=agesPlot, columns=["Alpha"]).rename_axis(index="Age")
betaDf  = pd.DataFrame(beta_x, index=agesPlot, columns=["Beta"]).rename_axis(index="Age")
kappaDf = pd.DataFrame(kappa_t, index=yearsPlot, columns=["Kappa"]).rename_axis(index= "Year")

"""#Testing
yearsPlotDf.to_clipboard()
agesPlotDf.to_clipboard()
alphaDf.to_clipboard()
betaDf.to_clipboard()
kappaDf.to_clipboard()
#"""

#Plot components
sns.relplot(x="Age", y="Alpha", data=alphaDf)
sns.relplot(x="Age", y="Beta", data=betaDf)
sns.relplot(x="Year", y="Kappa", data=kappaDf)
plt.show()
#"""


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

#Plot kappa_t and forecast
plt.figure(figsize=(10, 6))
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
plt.show()
#"""

#Step 4: Reconstruct mortality rates for forecast years
forecast_log_mortality = alphaDf["Alpha"].values.reshape(-1,1) + betaDf["Beta"].values.reshape(-1,1) @ forecast_mean.reshape(1,-1)
forecast_mortality = np.exp(forecast_log_mortality)

# Combine historical and forecasted mortality rates
all_years = np.concatenate([yearsPlot, forecast_index])
all_mortality = np.hstack([qxRatesPivot.values, forecast_mortality])

# Plot historical and forecasted mortality rates
plt.figure(figsize=(12, 6))
plt.imshow(all_mortality, aspect='auto', cmap='viridis', extent=[all_years[0], all_years[-1], agesPlot[-1], agesPlot[0]])
plt.colorbar(label='Mortality Rate')
plt.title("Historical and Forecasted Mortality Rates")
plt.xlabel("Year")
plt.ylabel("Age")
plt.show()
