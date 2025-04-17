import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import LeeCarterModel
from statsmodels.tsa.arima.model import ARIMA


########## Fitting ARIMA model to kappa_t (time-varying component) ##########

qxRatesPivot = LeeCarterModel.qxRatesPivot
alphaDf = LeeCarterModel.alphaDf
betaDf = LeeCarterModel.betaDf
kappa_t = LeeCarterModel.kappa_t
yearsPlot = LeeCarterModel.yearsPlot
agesPlot = LeeCarterModel.agesPlot

p, d, q = 1, 1, 1  # ARIMA parameters
LC_ARIMA = ARIMA(kappa_t, order=(p, d, q))
LC_ARIMA_fitted = LC_ARIMA.fit()

print(LC_ARIMA_fitted.summary())

LC_ARIMA_fitted.summary()

########## Forecast future kappa_t ##########

forecast_steps = 10
forecast_kappa = LC_ARIMA_fitted.get_forecast(steps=forecast_steps)
forecast_index = np.arange(yearsPlot[-1] + 1, yearsPlot[-1] + 1 + forecast_steps)
forecast_mean = forecast_kappa.predicted_mean
forecast_conf_int = forecast_kappa.conf_int()

"""#Plot kappa_t and forecast
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

########## Reconstruct mortality rates for forecast years ##########

forecast_log_mortality = alphaDf["Alpha"].values.reshape(-1,1) + betaDf["Beta"].values.reshape(-1,1) @ forecast_mean.reshape(1,-1)
forecast_mortality = np.exp(forecast_log_mortality)

#Combine historical and forecasted mortality rates
all_years = np.concatenate([yearsPlot, forecast_index])
all_mortality = np.hstack([qxRatesPivot.values, forecast_mortality])

#Plot historical and forecasted mortality rates
plt.figure(figsize=(12, 6))
plt.imshow(all_mortality, aspect='auto', cmap='viridis', extent=[all_years[0], all_years[-1], agesPlot[-1], agesPlot[0]])
plt.colorbar(label='Mortality Rate')
plt.title("Historical and Forecasted Mortality Rates")
plt.xlabel("Year")
plt.ylabel("Age")
plt.show()
