import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import LeeCarterModel
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima

########## Fitting ARIMA model to kappa_t (time-varying component) ##########

qxMatrix = LeeCarterModel.qxMatrix
alphaDf = LeeCarterModel.alphaDf
betaDf = LeeCarterModel.betaDf
kappa_t = LeeCarterModel.kappa_t
kappaDf = LeeCarterModel.kappaDf
yearsPlot = LeeCarterModel.yearsPlot
agesPlot = LeeCarterModel.agesPlot

p, d, q = 1, 0, 0  # ARIMA parameters

from pmdarima import auto_arima
print(kappaDf[kappaDf["Gender"]=="Female"]["Kappa"])

# Let's say 'series' is your time series data (Pandas Series or array-like)
model = auto_arima(kappaDf[kappaDf["Gender"]=="Female"]["Kappa"], 
                   start_p=0, start_q=0,
                   max_p=3, max_q=3,
                   seasonal=False,       # set to True for seasonal data
                   stepwise=False,        # faster search
                   trace=True)           # see progress

print(model.summary())


LC_ARIMA = ARIMA(kappaDf[kappaDf["Gender"]=="Female"]["Kappa"], 
                 order=(p, d, q), 
                 enforce_stationarity=False,
                 trend="ct")

LC_ARIMA_fitted = LC_ARIMA.fit()

print(LC_ARIMA_fitted.summary())

LC_ARIMA_fitted.summary()

########## Forecast future kappa_t ##########

forecast_steps = 50
forecast_kappa = LC_ARIMA_fitted.get_forecast(steps=forecast_steps)
forecast_index = np.arange(yearsPlot[-1] + 1, yearsPlot[-1] + 1 + forecast_steps)
forecast_mean = forecast_kappa.predicted_mean
forecast_conf_int = forecast_kappa.conf_int()

#Plot kappa_t and forecast
plt.figure(figsize=(10, 6))
plt.plot(yearsPlot, kappaDf[kappaDf["Gender"]=="Female"]["Kappa"], label="Observed Kappa (κ)", color="green")
plt.plot(forecast_index, forecast_mean, label="Forecast Kappa (κ)", color="orange")
"""plt.fill_between(
    forecast_index,
    forecast_conf_int[:, 0],
    forecast_conf_int[:, 1],
    color="orange",
    alpha=0.2,
    label="Confidence Interval",
)"""

plt.title("Forecast of Kappa (Time Effect)")
plt.xlabel("Year")
plt.ylabel("Kappa (κ)")
plt.legend()
plt.show()
#"""

########## Reconstruct mortality rates for forecast years ##########
"""
forecast_log_mortality = alphaDf[alphaDf["Gender"]=="Female"]["Alpha"].values.reshape(-1,1) + betaDf[betaDf["Gender"]=="Female"]["Beta"].values.reshape(-1,1) @ forecast_mean.reshape(1,-1)
forecast_mortality = np.exp(forecast_log_mortality)

#Combine historical and forecasted mortality rates
all_years = np.concatenate([yearsPlot, forecast_index])
all_mortality = np.hstack([qxMatrix.values, forecast_mortality])

#Plot historical and forecasted mortality rates
plt.figure(figsize=(12, 6))
plt.imshow(all_mortality, aspect='auto', cmap='viridis', extent=[all_years[0], all_years[-1], agesPlot[-1], agesPlot[0]])
plt.colorbar(label='Mortality Rate')
plt.title("Historical and Forecasted Mortality Rates")
plt.xlabel("Year")
plt.ylabel("Age")
plt.show()
"""